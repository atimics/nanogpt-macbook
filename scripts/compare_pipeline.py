"""Compare queued training with a trusted local reference in alternating blocks."""

import argparse
import gc
import hashlib
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from nanogpt_macbook.benchmark import _batches, _timed_steps, source_info
from nanogpt_macbook.checkpoint import write_json
from nanogpt_macbook.config import PRESETS
from nanogpt_macbook.engine import make_train_step, run_steps, select_device
from nanogpt_macbook.model import GPT


def compare(preset, reference, pairs, steps, warmup):
    config, training = PRESETS[preset]
    records = []
    for name, builder, pipeline in (
        ("reference", reference, False),
        ("queued", make_train_step, True),
    ):
        mx.random.seed(training.seed)
        rng = np.random.default_rng(training.seed)
        model = GPT(config)
        optimizer = optim.AdamW(
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            betas=[0.9, 0.95],
            bias_correction=True,
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        step = builder(model, optimizer, training.accumulation, training.grad_clip)
        for _ in run_steps(step, _batches(rng, config, training, warmup), pipeline=pipeline):
            pass
        mx.synchronize()
        records.append((name, step, rng, pipeline))
    samples, ratios = [], []
    for pair in range(pairs):
        rates = {}
        for name, step, rng, pipeline in records if pair % 2 == 0 else reversed(records):
            durations = _timed_steps(step, rng, config, training, steps, pipeline)
            elapsed = sum(durations)
            rates[name] = training.batch_size * config.context * steps / elapsed
            samples.append(
                {
                    "pair": pair + 1,
                    "path": name,
                    "completion_seconds": durations,
                    "elapsed_seconds": elapsed,
                    "bytes_per_second": rates[name],
                }
            )
        ratios.append(rates["queued"] / rates["reference"])
        if (pair + 1) % max(1, pairs // 10) == 0 or pair + 1 == pairs:
            print(
                f"{preset} pair {pair + 1:3}: median queued/reference "
                f"{statistics.median(ratios):.4f}",
                flush=True,
            )
    return {
        "preset": preset,
        "model_config": asdict(config),
        "train_config": asdict(training),
        "samples": samples,
        "queued_over_reference_ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "aggregate_ratio": sum(s["elapsed_seconds"] for s in samples if s["path"] == "reference")
        / sum(s["elapsed_seconds"] for s in samples if s["path"] == "queued"),
        "faster_pairs": sum(ratio > 1 for ratio in ratios),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-ref", required=True, help="Trusted local Git ref")
    parser.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if min(args.pairs, args.steps, args.warmup) < 1:
        parser.error("Pairs, steps, and warmup must be positive")
    if args.out.exists():
        parser.error("Choose a new output path")
    root = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.baseline_ref}^{{commit}}"],
        cwd=root,
        text=True,
    ).strip()
    path = "src/nanogpt_macbook/engine.py"
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
    module = ModuleType("nanogpt_macbook.reference_engine")
    module.__package__ = "nanogpt_macbook"
    exec(compile(source, f"{commit}:{path}", "exec"), module.__dict__)
    select_device("gpu", 4)
    receipt = {
        "format": 1,
        "source": source_info(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "chip": mx.device_info(mx.gpu)["device_name"],
        "mlx": mx.__version__,
        "reference": {
            "commit": commit,
            "path": path,
            "sha256": hashlib.sha256(source).hexdigest(),
            "scope": "Reference compiled step with current model math; per-step device waits",
        },
        "method": {
            "name": "alternating-training-queue-v1",
            "pairs": args.pairs,
            "steps_per_path_per_pair": args.steps,
            "warmup_steps_per_path": args.warmup,
            "order": "reference then queued on odd pairs; queued then reference on even pairs",
            "queue_depth": 2,
            "memory_limit_gib": 4,
            "cache_limit_gib": 4,
            "timed_work": "host batch creation, forward, loss, backward, clipping, AdamW, sync",
            "timing_semantics": (
                "Reference intervals each finish a complete step. Queued intervals measure "
                "checked loss completion, including a full device wait at each block's end."
            ),
            "summary": "median queued/reference throughput ratio across adjacent blocks",
        },
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS if args.preset == "all" else [args.preset]:
        gc.collect()
        mx.clear_cache()
        receipt["results"].append(
            compare(preset, module.make_train_step, args.pairs, args.steps, args.warmup)
        )
        write_json(args.out, receipt)
    print(f"Saved comparison to {args.out}")


if __name__ == "__main__":
    main()
