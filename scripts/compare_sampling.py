"""Compare sampling calls with model and engine code from a Git ref."""

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import mlx.core as mx

from nanogpt_macbook import engine
from nanogpt_macbook.benchmark import source_info
from nanogpt_macbook.checkpoint import write_json
from nanogpt_macbook.config import PRESETS
from nanogpt_macbook.data import decode
from nanogpt_macbook.model import GPT


def reference_modules(ref):
    root = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=root,
        text=True,
    ).strip()
    modules, hashes = {}, {}
    for name in ["model", "engine"]:
        path = f"src/nanogpt_macbook/{name}.py"
        source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
        module = ModuleType(f"reference_{name}")
        module.__package__ = "nanogpt_macbook"
        exec(compile(source, f"{commit}:{path}", "exec"), module.__dict__)
        modules[name] = module
        hashes[path] = hashlib.sha256(source).hexdigest()
    return modules, {
        "commit": commit,
        "files_sha256": hashes,
        "scope": "current shared operators with model and engine code from this commit",
    }


def sample(model_type, module, config, prompt, args):
    gc.collect()
    mx.clear_cache()
    mx.random.seed(1337)
    model = model_type(config)
    mx.eval(model.parameters())
    generated = []

    def capture(tokens):
        generated.extend(tokens)
        return decode(tokens)

    mx.reset_peak_memory()
    with patch.object(module, "decode", capture):
        started = time.perf_counter()
        text = module.generate(
            model, prompt, args.tokens, args.temperature, args.top_k, seed=args.seed
        )
        mx.synchronize()
        elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "bytes_per_second": args.tokens / elapsed,
        "peak_memory_mib": mx.get_peak_memory() / 1024**2,
        "generated_byte_count": len(generated),
        "generated_bytes_sha256": hashlib.sha256(bytes(generated)).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def compare(config, preset, prompt, args, reference, record):
    implementations = {
        "reference": (reference["model"].GPT, reference["engine"]),
        "updated": (GPT, engine),
    }
    result = {
        "preset": preset,
        "model_config": asdict(config),
        "prompt": prompt,
        "prompt_bytes": len(prompt.encode()),
        "samples": [],
        "updated_over_reference_ratios": [],
    }
    record["results"].append(result)
    matches = 0
    for pair in range(args.pairs):
        order = ["reference", "updated"] if pair % 2 == 0 else ["updated", "reference"]
        measured = {}
        for path in order:
            measured[path] = sample(*implementations[path], config, prompt, args)
            assert measured[path]["generated_byte_count"] == args.tokens
            result["samples"].append({"pair": pair + 1, "path": path, **measured[path]})
        ratio = measured["reference"]["elapsed_seconds"] / measured["updated"]["elapsed_seconds"]
        result["updated_over_reference_ratios"].append(ratio)
        matches += (
            measured["reference"]["generated_bytes_sha256"]
            == measured["updated"]["generated_bytes_sha256"]
        )
        result["matching_byte_pairs"] = matches
        write_json(args.out, record)
        if (pair + 1) % max(1, args.pairs // 5) == 0:
            print(
                f"{preset} prompt {len(prompt)} pair {pair + 1}: "
                f"median ratio {statistics.median(result['updated_over_reference_ratios']):.4f}",
                flush=True,
            )
    ratios = result["updated_over_reference_ratios"]
    durations = {
        path: [row["elapsed_seconds"] for row in result["samples"] if row["path"] == path]
        for path in implementations
    }
    result.update(
        median_ratio=statistics.median(ratios),
        aggregate_ratio=sum(durations["reference"]) / sum(durations["updated"]),
        faster_pairs=sum(ratio > 1 for ratio in ratios),
        median_bytes_per_second={
            path: args.tokens / statistics.median(times) for path, times in durations.items()
        },
    )
    write_json(args.out, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output path")
    if args.pairs < 1 or args.tokens < 1 or args.seed < 0:
        parser.error("Pairs and tokens must be positive; seed must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 <= args.top_k <= 256:
        parser.error("Temperature must be finite and nonnegative; top-k must be between 0 and 256")
    reference, reference_info = reference_modules(args.baseline_ref)
    engine.select_device(args.device, 2)
    receipt = {
        "format": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": source_info(),
        "reference": reference_info,
        "mlx": mx.__version__,
        "chip": mx.device_info(mx.gpu)["device_name"] if mx.metal.is_available() else "CPU",
        "method": {
            "name": "alternating-sampling-fresh-v1",
            "device": args.device,
            "memory_limit_gib": 2 if args.device == "gpu" else None,
            "pairs": args.pairs,
            "generated_bytes": args.tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "sampling_seed": args.seed,
            "weight_seed": 1337,
            "weights": "matching freshly initialized float32 models",
            "timed_work": "complete generate call and final GPU wait, including compilation",
            "warmup": "model weights are evaluated before timing; sampling begins fresh",
            "order": "reference then updated on odd pairs; updated then reference on even pairs",
            "model_lifetime": "one fresh model per path per pair",
            "memory_scope": "peak active MLX allocation during sampling, including weights",
        },
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS if args.preset == "all" else [args.preset]:
        config = PRESETS[preset][0]
        for length in [16, config.context]:
            prompt = ("After the rain, Mira opened the window. " * 20)[:length]
            compare(config, preset, prompt, args, reference, receipt)
    print(json.dumps({"saved": str(args.out), "results": len(receipt["results"])}))


if __name__ == "__main__":
    main()
