"""Compare native and grouped LayerNorm with alternating complete training steps."""

import argparse
import gc
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from nanogpt_macbook import engine
from nanogpt_macbook.benchmark import _step, source_info
from nanogpt_macbook.checkpoint import write_json
from nanogpt_macbook.config import PRESETS
from nanogpt_macbook.engine import make_train_step, select_device
from nanogpt_macbook.model import GPT
from nanogpt_macbook.normalization import LayerNorm


def compare(preset, pairs, steps, warmup, component="normalization"):
    config, training = PRESETS[preset]
    records = []
    if component == "clipping":
        target, attribute = engine, "clip_grad_norm"
        native, grouped = optim.clip_grad_norm, engine.clip_grad_norm
    else:
        target, attribute = LayerNorm, "__call__"
        native, grouped = nn.LayerNorm.__call__, LayerNorm.__call__
    for name, implementation in (("native", native), ("grouped", grouped)):
        with patch.object(target, attribute, implementation):
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
            train_step = make_train_step(
                model, optimizer, training.accumulation, training.grad_clip
            )
            # Compile each path while its implementation is selected.
            for _ in range(warmup):
                _step(train_step, rng, config, training)
            mx.synchronize()
            records.append((name, train_step, rng))

    samples = []
    ratios = []
    for pair in range(pairs):
        order = records if pair % 2 == 0 else list(reversed(records))
        rates = {}
        for name, train_step, rng in order:
            timings = []
            for _ in range(steps):
                start = time.perf_counter()
                _step(train_step, rng, config, training)
                mx.synchronize()
                timings.append(time.perf_counter() - start)
            rates[name] = training.batch_size * config.context * steps / sum(timings)
            samples.append(
                {
                    "pair": pair + 1,
                    "path": name,
                    "step_seconds": timings,
                    "bytes_per_second": rates[name],
                }
            )
        ratios.append(rates["grouped"] / rates["native"])
        if (pair + 1) % max(1, pairs // 10) == 0 or pair + 1 == pairs:
            median = statistics.median(ratios)
            print(f"{preset} pair {pair + 1:3}: median grouped / native {median:.4f}", flush=True)
    return {
        "preset": preset,
        "model_config": asdict(config),
        "train_config": asdict(training),
        "samples": samples,
        "grouped_over_native_ratios": ratios,
        "median_ratio": statistics.median(ratios),
    }


def main(component="normalization"):
    parser = argparse.ArgumentParser(
        description=f"Compare native and grouped {component} in complete training steps."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if min(args.pairs, args.steps, args.warmup) < 1:
        parser.error("Pairs, steps, and warmup must be positive")
    if args.out.exists():
        parser.error("Choose a new output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Both compiled models stay alive during each comparison.
    select_device("gpu", 4)
    receipt = {
        "format": 1,
        "source": source_info(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "chip": mx.device_info(mx.gpu)["device_name"],
        "mlx": mx.__version__,
        "method": {
            "name": "alternating-clipping-v1"
            if component == "clipping"
            else "alternating-layernorm-v1",
            "pairs": args.pairs,
            "steps_per_path_per_pair": args.steps,
            "warmup_steps_per_path": args.warmup,
            "order": "native then grouped on odd pairs; grouped then native on even pairs",
            "memory_limit_gib": 4,
            "cache_limit_gib": 4,
            "timed_work": "host batch creation, forward, loss, backward, clipping, AdamW, sync",
            "summary": "median of grouped/native throughput ratios across adjacent pairs",
        },
        "results": [],
    }
    for preset in PRESETS if args.preset == "all" else [args.preset]:
        gc.collect()
        mx.clear_cache()
        receipt["results"].append(compare(preset, args.pairs, args.steps, args.warmup, component))
        write_json(args.out, receipt)
    print(f"Saved comparison to {args.out}")


if __name__ == "__main__":
    main()
