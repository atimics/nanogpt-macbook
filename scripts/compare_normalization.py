"""Compare training implementations with alternating complete training steps."""

import argparse
import gc
import hashlib
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from nanogpt_macbook import engine
from nanogpt_macbook import model as model_module
from nanogpt_macbook.benchmark import _batches, _timed_steps, source_info
from nanogpt_macbook.checkpoint import write_json
from nanogpt_macbook.config import PRESETS
from nanogpt_macbook.engine import make_train_step, run_steps, select_device
from nanogpt_macbook.normalization import LayerNorm


def compare(preset, pairs, steps, warmup, component="normalization", reference=None, queued=False):
    config, training = PRESETS[preset]
    records = []
    if component == "clipping":
        target, attribute = engine, "clip_grad_norm"
        native, grouped = optim.clip_grad_norm, engine.clip_grad_norm
    elif component == "attention":
        target, attribute = model_module, "training_attention"
        native, grouped = reference, model_module.training_attention
    elif component == "model":
        target, attribute = model_module, "GPT"
        native, grouped = reference, model_module.GPT
    else:
        target, attribute = LayerNorm, "__call__"
        native, grouped = reference or nn.LayerNorm.__call__, LayerNorm.__call__
    before, after = ("reference", "updated") if reference else ("native", "grouped")
    for name, implementation in ((before, native), (after, grouped)):
        with patch.object(target, attribute, implementation):
            mx.random.seed(training.seed)
            rng = np.random.default_rng(training.seed)
            model = model_module.GPT(config)
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
            for _ in run_steps(
                train_step, _batches(rng, config, training, warmup), pipeline=queued
            ):
                pass
            mx.synchronize()
            records.append((name, train_step, rng))

    samples = []
    ratios = []
    for pair in range(pairs):
        order = records if pair % 2 == 0 else list(reversed(records))
        rates = {}
        for name, train_step, rng in order:
            timings = _timed_steps(train_step, rng, config, training, steps, queued)
            rates[name] = training.batch_size * config.context * steps / sum(timings)
            samples.append(
                {
                    "pair": pair + 1,
                    "path": name,
                    "step_seconds": timings,
                    "bytes_per_second": rates[name],
                }
            )
        ratios.append(rates[after] / rates[before])
        if (pair + 1) % max(1, pairs // 10) == 0 or pair + 1 == pairs:
            median = statistics.median(ratios)
            print(f"{preset} pair {pair + 1:3}: median {after} / {before} {median:.4f}", flush=True)
    return {
        "preset": preset,
        "model_config": asdict(config),
        "train_config": asdict(training),
        "samples": samples,
        f"{after}_over_{before}_ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "aggregate_ratio": sum(
            sum(sample["step_seconds"]) for sample in samples if sample["path"] == before
        )
        / sum(sum(sample["step_seconds"]) for sample in samples if sample["path"] == after),
        "faster_pairs": sum(ratio > 1 for ratio in ratios),
    }


def main(component="normalization"):
    parser = argparse.ArgumentParser(
        description=f"Compare {component} implementations in complete training steps."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--baseline-ref", help=f"Use {component} from this trusted local Git ref")
    parser.add_argument("--queued", action="store_true", help="Use the two-step GPU queue")
    parser.add_argument(
        "--memory-gb", type=float, default=4, help="MLX memory/cache budget for both live models"
    )
    args = parser.parse_args()
    if min(args.pairs, args.steps, args.warmup) < 1:
        parser.error("Pairs, steps, and warmup must be positive")
    if args.out.exists():
        parser.error("Choose a new output path")
    if args.baseline_ref and component == "clipping":
        parser.error("A baseline ref applies to normalization, attention, or model")
    if component in {"attention", "model"} and not args.baseline_ref:
        parser.error(f"Choose a baseline ref for the {component} comparison")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reference = None
    reference_info = None
    if args.baseline_ref:
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{args.baseline_ref}^{{commit}}"],
            cwd=root,
            text=True,
        ).strip()
        path = f"src/nanogpt_macbook/{component}.py"
        source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
        module = ModuleType(f"reference_{component}")
        module.__package__ = "nanogpt_macbook"
        exec(compile(source, f"{commit}:{path}", "exec"), module.__dict__)
        if component == "model":
            reference = module.GPT
        elif component == "attention":
            reference = module.training_attention
        else:
            reference = module.LayerNorm.__call__
        reference_info = {
            "commit": commit,
            "path": path,
            "sha256": hashlib.sha256(source).hexdigest(),
            "scope": f"Current training code with {component} from this commit",
        }
    # Both compiled models stay alive during each comparison.
    select_device("gpu", args.memory_gb)
    operation = "layernorm" if component == "normalization" else component
    receipt = {
        "format": 1,
        "source": source_info(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "chip": mx.device_info(mx.gpu)["device_name"],
        "mlx": mx.__version__,
        "method": {
            "name": f"alternating-{operation}-v1",
            "pairs": args.pairs,
            "steps_per_path_per_pair": args.steps,
            "warmup_steps_per_path": args.warmup,
            "order": "native then grouped on odd pairs; grouped then native on even pairs",
            "memory_limit_gib": args.memory_gb,
            "cache_limit_gib": args.memory_gb,
            "timed_work": "host batch creation, forward, loss, backward, clipping, AdamW, sync",
            "summary": "median of grouped/native throughput ratios across adjacent pairs",
        },
        "results": [],
    }
    if reference_info:
        receipt["reference"] = reference_info
        receipt["method"].update(
            name=f"alternating-{operation}-reference-v1",
            order="reference then updated on odd pairs; updated then reference on even pairs",
            summary="median of updated/reference throughput ratios across adjacent pairs",
        )
    if args.queued:
        receipt["method"].update(
            name=receipt["method"]["name"].replace("-v1", "-queued-v1"),
            queue_depth=2,
            timing_semantics=(
                "Checked loss-completion intervals, including all GPU work "
                "through the final device wait in each block"
            ),
        )
    for preset in PRESETS if args.preset == "all" else [args.preset]:
        gc.collect()
        mx.clear_cache()
        receipt["results"].append(
            compare(preset, args.pairs, args.steps, args.warmup, component, reference, args.queued)
        )
        write_json(args.out, receipt)
    print(f"Saved comparison to {args.out}")


if __name__ == "__main__":
    main()
