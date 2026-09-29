"""Repeatable training throughput measurements with raw timing receipts."""

import gc
import hashlib
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from . import __version__
from .checkpoint import write_json
from .config import PRESETS
from .engine import make_train_step, select_device
from .model import GPT


def source_info() -> dict:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for file in sorted(package.glob("*.py")):
        digest.update(file.name.encode() + b"\0" + file.read_bytes())
    result = {"package_version": __version__, "python_source_sha256": digest.hexdigest()}
    root = package.parent.parent
    if (root / "pyproject.toml").exists():
        try:
            result["commit"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            result["source_dirty"] = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain", "--", "src/nanogpt_macbook"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return result


def summarize(trials: list[dict]) -> dict:
    rates = [trial["bytes_per_second"] for trial in trials]
    return {
        "median_bytes_per_second": statistics.median(rates),
        "min_bytes_per_second": min(rates),
        "max_bytes_per_second": max(rates),
        "median_step_ms": statistics.median(
            statistics.mean(trial["step_seconds"]) * 1000 for trial in trials
        ),
        "peak_memory_mib": max(trial["peak_memory_mib"] for trial in trials),
    }


def _step(train_step, rng, config, training):
    block = rng.integers(0, 256, (training.batch_size, config.context + 1), dtype=np.int32)
    train_step(mx.array(block[None, :, :-1]), mx.array(block[None, :, 1:]), training.learning_rate)


def benchmark(
    output: Path,
    presets: list[str],
    *,
    device: str = "gpu",
    memory_gb: float = 2,
    steps: int = 100,
    warmup: int = 20,
    repeats: int = 3,
    seed: int = 1337,
    report=print,
) -> dict:
    if output.exists():
        raise ValueError(f"Choose a new result path; {output} already exists")
    if min(steps, warmup, repeats) < 1 or not 0 <= seed < 2**64:
        raise ValueError("Steps, warmup, and repeats must be positive; seed must fit uint64")
    if not presets or len(set(presets)) != len(presets) or any(p not in PRESETS for p in presets):
        raise ValueError("Choose distinct presets from tiny, small, and medium")
    selected = select_device(device, memory_gb)
    info = mx.device_info(mx.gpu) if mx.metal.is_available() else {}
    receipt = {
        "format": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": source_info(),
        "environment": {
            "chip": info.get("device_name", platform.processor() or platform.machine()),
            "system": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "architecture": platform.machine(),
            "memory_gib": info.get("memory_size", 0) / 1024**3 or None,
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "numpy": np.__version__,
            "device": selected,
        },
        "method": {
            "name": "training-step-v1",
            "data": "seeded uniform random UTF-8 byte token IDs (0 to 255)",
            "dtype": "float32",
            "execution": "compiled",
            "timed_work": "host batch creation, forward, loss, backward, clipping, AdamW, sync",
            "steps": steps,
            "warmup_steps": warmup,
            "repeats": repeats,
            "seed": seed,
            "memory_limit_gib": memory_gb if selected == "gpu" else None,
            "cache_limit_gib": memory_gb if selected == "gpu" else None,
            "attention": "fused causal softmax on Metal; MLX attention on CPU",
            "activation": "explicit stable GELU derivative on Metal; MLX derivative on CPU",
            "memory_scope": "peak active MLX allocation during measured steps, including weights",
            "summary": "median of per-trial throughput; range is minimum to maximum trial",
        },
        "results": [],
    }
    for preset in presets:
        config, training = PRESETS[preset]
        trials = []
        parameter_count = None
        for repeat in range(repeats):
            gc.collect()
            mx.clear_cache()
            mx.random.seed(seed)
            rng = np.random.default_rng(seed)
            model = GPT(config)
            parameter_count = model.parameter_count
            optimizer = optim.AdamW(
                learning_rate=training.learning_rate,
                betas=[0.9, 0.95],
                weight_decay=training.weight_decay,
                bias_correction=True,
            )
            optimizer.init(model.trainable_parameters())
            mx.eval(model.parameters(), optimizer.state)
            train_step = make_train_step(
                model, optimizer, training.accumulation, training.grad_clip
            )

            for _ in range(warmup):
                _step(train_step, rng, config, training)
            mx.synchronize()
            mx.reset_peak_memory()
            durations = []
            for _ in range(steps):
                start = time.perf_counter()
                _step(train_step, rng, config, training)
                mx.synchronize()
                durations.append(time.perf_counter() - start)
            elapsed = sum(durations)
            rate = training.batch_size * config.context * steps / elapsed
            trials.append(
                {
                    "trial": repeat + 1,
                    "step_seconds": durations,
                    "elapsed_seconds": elapsed,
                    "bytes_per_second": rate,
                    "peak_memory_mib": mx.get_peak_memory() / 1024**2,
                }
            )
            report(
                f"{preset:6} {selected} trial {repeat + 1}/{repeats}: "
                f"{rate:,.0f} bytes/s | {trials[-1]['peak_memory_mib']:.1f} MiB"
            )
            del train_step, optimizer, model
        receipt["results"].append(
            {
                "preset": preset,
                "parameters": parameter_count,
                "model_config": asdict(config),
                "train_config": asdict(training),
                "trials": trials,
                "summary": summarize(trials),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    report(f"Saved benchmark results to {output}")
    return receipt
