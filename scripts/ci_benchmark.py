"""Probe a CI Mac, then compare committed code on one worker with an A/A control."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = 6
STEPS = 100
WARMUP = 20


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def probe(output):
    import mlx.core as mx
    import numpy as np

    available = mx.metal.is_available()
    details = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "system": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "numpy": np.__version__,
        "metal_available": available,
        "device": mx.device_info(mx.gpu) if available else {},
        "runner": {
            key: os.environ.get(key)
            for key in (
                "BENCHMARK_RUNNER_LABEL",
                "RUNNER_NAME",
                "RUNNER_ARCH",
                "ImageOS",
                "ImageVersion",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
            )
        },
    }
    if available:
        # Force actual Metal execution before recording a usable host.
        with mx.stream(mx.gpu):
            value = mx.sum(mx.arange(1024, dtype=mx.float32))
            mx.eval(value)
        details["gpu_check"] = float(value.item())
        if details["gpu_check"] != 523776:
            raise ValueError("Metal preflight returned an unexpected sum")
    save(output / "host.json", details)
    print(json.dumps(details, indent=2))
    return details


def resolve_commit(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Use a full lowercase commit hash")
    actual = command(["git", "rev-parse", "--verify", f"{value}^{{commit}}"])
    if actual != value:
        raise ValueError("Commit hash disagrees with the checkout")
    return actual


def paired_summary(pairs):
    ratios = [pair["b"] / pair["a"] for pair in pairs]
    if len(ratios) < 2 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("A comparison needs at least two finite, positive pairs")
    changes = [(ratio - 1) * 100 for ratio in ratios]
    return {
        "pairs": len(pairs),
        "median_change_percent": statistics.median(changes),
        "min_change_percent": min(changes),
        "max_change_percent": max(changes),
        "median_absolute_change_percent": statistics.median(map(abs, changes)),
        "b_faster_pairs": sum(ratio > 1 for ratio in ratios),
        "ratios": ratios,
    }


def check_receipt(receipt, commit, preset):
    source = receipt["source"]
    if source.get("commit") != commit or source.get("source_dirty") is not False:
        raise ValueError("Worker loaded a different or changed source tree")
    if receipt["environment"]["device"] != "gpu":
        raise ValueError("The comparison requires Metal GPU samples")
    method = receipt["method"]
    if (
        method.get("dtype"),
        method.get("execution"),
        method.get("queue_depth"),
        method.get("seed"),
        method.get("memory_limit_gib"),
        method.get("cache_limit_gib"),
    ) != ("float32", "pipelined", 2, 1337, 2, 2):
        raise ValueError("Worker precision, seed, queue, or memory limit changed")
    if (method["name"], method["steps"], method["warmup_steps"], method["repeats"]) != (
        "training-loop-v2",
        STEPS,
        WARMUP,
        1,
    ):
        raise ValueError("Worker timing protocol differs from the CI protocol")
    (result,) = receipt["results"]
    (trial,) = result["trials"]
    intervals = trial["step_seconds"]
    if result["preset"] != preset or len(intervals) != STEPS:
        raise ValueError("Worker returned a different preset or step count")
    if any(not math.isfinite(value) or value <= 0 for value in intervals):
        raise ValueError("Worker returned invalid step durations")
    rate = result["train_config"]["batch_size"] * result["model_config"]["context"]
    rate *= STEPS / sum(intervals)
    if not math.isclose(rate, trial["bytes_per_second"], rel_tol=1e-9):
        raise ValueError("Worker throughput disagrees with its raw timings")
    return rate


def sample(checkout, commit, preset, output):
    env = {**os.environ, "PYTHONPATH": str(checkout / "src")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nanogpt_macbook",
            "benchmark",
            "--preset",
            preset,
            "--out",
            str(output),
            "--device",
            "gpu",
            "--steps",
            str(STEPS),
            "--warmup",
            str(WARMUP),
            "--repeats",
            "1",
            "--seed",
            "1337",
            "--memory-gb",
            "2",
        ],
        cwd=checkout,
        env=env,
        check=True,
        timeout=300,
    )
    return json.loads(output.read_text())


def compare(args):
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        raise ValueError("Choose a new directory for each comparison")
    host = probe(output)
    if not host["metal_available"]:
        raise ValueError("Use a runner with Metal GPU access; see benchmarks/CI.md")
    commits = {name: resolve_commit(getattr(args, name)) for name in ("baseline", "candidate")}
    lock_hashes = {
        key: hashlib.sha256(command(["git", "show", f"{value}:uv.lock"]).encode()).hexdigest()
        for key, value in commits.items()
    }
    installed_lock = hashlib.sha256((ROOT / "uv.lock").read_text().strip().encode()).hexdigest()
    if set(lock_hashes.values()) != {installed_lock}:
        raise ValueError("Use commits with the workflow's uv.lock for a fixed software stack")
    report = {
        "format": "ci-paired-v1",
        "host": host,
        "commits": commits,
        "harness_commit": command(["git", "rev-parse", "HEAD"]),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": {
            "pairs_per_phase": PAIRS,
            "steps": STEPS,
            "warmup": WARMUP,
            "seed": 1337,
            "memory_gib": 2,
            "order": "A/B on odd pairs, B/A on even pairs; fresh process for each sample",
            "calibration": "A and B use the baseline commit before each preset comparison",
        },
        "complete": False,
        "samples": [],
        "summaries": {},
    }
    save(output / "comparison.json", report)
    with tempfile.TemporaryDirectory(prefix="nanogpt-ci-") as directory:
        checkouts = {}
        try:
            for name, commit in commits.items():
                checkout = Path(directory) / name
                command(["git", "worktree", "add", "--detach", str(checkout), commit])
                checkouts[name] = checkout
            presets = ("tiny", "small", "medium") if args.preset == "all" else (args.preset,)
            for preset in presets:
                settings = None
                for phase in ("calibration", "comparison"):
                    pairs = []
                    for pair in range(1, PAIRS + 1):
                        rates = {}
                        for side in ("a", "b") if pair % 2 else ("b", "a"):
                            name = (
                                "candidate" if phase == "comparison" and side == "b" else "baseline"
                            )
                            file = output / f"{preset}-{phase}-{pair:02}-{side}.json"
                            receipt = sample(checkouts[name], commits[name], preset, file)
                            rate = check_receipt(receipt, commits[name], preset)
                            row = receipt["results"][0]
                            current = (
                                row["model_config"],
                                row["train_config"],
                                receipt["environment"],
                            )
                            if settings is not None and current != settings:
                                raise ValueError(
                                    "Model, training, or host settings changed between samples"
                                )
                            settings = current
                            rates[side] = rate
                            report["samples"].append(
                                {
                                    "preset": preset,
                                    "phase": phase,
                                    "pair": pair,
                                    "side": side,
                                    "commit": commits[name],
                                    "file": file.name,
                                    "bytes_per_second": rate,
                                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                                }
                            )
                            save(output / "comparison.json", report)
                        pairs.append(rates)
                    report["summaries"].setdefault(preset, {})[phase] = paired_summary(pairs)
                    save(output / "comparison.json", report)
            report["complete"] = True
            save(output / "comparison.json", report)
        finally:
            for checkout in checkouts.values():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(checkout)], cwd=ROOT, check=True
                )
    report["timing"] = {
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "probe, checkouts, all fresh processes, warmup, A/A, A/B, receipts, cleanup",
    }
    save(output / "comparison.json", report)
    lines = [
        "# Mac GPU benchmark",
        "",
        f"Baseline: `{commits['baseline']}`",
        "",
        f"Candidate: `{commits['candidate']}`",
        "",
        "| Preset | A/A median change | A/A range | A/B median change | A/B range |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for preset, phases in report["summaries"].items():
        a, b = phases["calibration"], phases["comparison"]
        lines.append(
            f"| {preset} | {a['median_change_percent']:+.2f}% | "
            f"{a['min_change_percent']:+.2f}% to {a['max_change_percent']:+.2f}% | "
            f"{b['median_change_percent']:+.2f}% | "
            f"{b['min_change_percent']:+.2f}% to {b['max_change_percent']:+.2f}% |"
        )
    lines += [
        "",
        "Compare each gain with its A/A spread. Keep each hardware and software stack "
        "in a separate history.",
        "",
    ]
    summary = "\n".join(lines)
    (output / "summary.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(summary)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "compare"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--preset", choices=("tiny", "small", "medium", "all"), default="tiny")
    args = parser.parse_args()
    if args.mode == "probe":
        probe(args.out)
    else:
        if not args.baseline or not args.candidate:
            parser.error("compare requires --baseline and --candidate")
        compare(args)


if __name__ == "__main__":
    main()
