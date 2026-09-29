"""Training, evaluation, and generation on MLX."""

import json
import math
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from . import checkpoint
from .config import ModelConfig, TrainConfig
from .data import Dataset, decode, encode
from .model import GPT, loss_fn


def select_device(device: str = "auto", memory_gb: float = 2.0) -> str:
    if not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("memory-gb must be finite and positive")
    metal = mx.metal.is_available()
    if device == "gpu" and not metal:
        raise ValueError("Metal is unavailable; use --device cpu or a native Apple Silicon Python")
    selected = "gpu" if device == "gpu" or (device == "auto" and metal) else "cpu"
    mx.set_default_device(mx.gpu if selected == "gpu" else mx.cpu)
    if selected == "gpu":
        mx.set_memory_limit(int(memory_gb * 1024**3))
        mx.set_cache_limit(int(memory_gb * 1024**3 / 4))
    return selected


def evaluate(model: GPT, dataset: Dataset, batch_size: int, batches: int = 10) -> dict:
    if batches < 1 or batch_size < 1:
        raise ValueError("Evaluation batches and batch size must be positive")
    # A separate NumPy generator keeps evaluation out of the training RNG stream.
    rng = np.random.default_rng(2026)
    model.eval()
    values = []
    for _ in range(batches):
        x, y = dataset.batch("val", batch_size, rng)
        values.append(float(loss_fn(model, mx.array(x), mx.array(y)).item()))
    model.train()
    loss = sum(values) / len(values)
    if not math.isfinite(loss):
        raise ValueError("Validation loss became non-finite; use a lower learning rate")
    return {"val_loss": loss, "bits_per_byte": loss / math.log(2)}


def train(
    data: Path,
    run: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    steps: int,
    *,
    resume: bool = False,
    eval_every: int = 100,
    eval_batches: int = 10,
    log_every: int = 10,
    time_limit: float | None = None,
    report=print,
) -> dict:
    if min(steps, eval_every, eval_batches, log_every) < 1:
        raise ValueError("Steps and report intervals must be positive")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time-limit must be finite and positive")
    with checkpoint.run_lock(run):
        if resume:
            path, state = checkpoint.read(run)
            model_config = ModelConfig(**state["model_config"])
            train_config = TrainConfig(**state["train_config"])
            if steps <= state["step"]:
                raise ValueError(f"Set --steps above the saved step {state['step']}")
        elif any(item.name != ".lock" for item in run.iterdir()):
            raise ValueError(f"Use resume for {run}, or choose a new run folder")
        dataset = Dataset(data, model_config.context)
        if resume and dataset.identity != state["dataset"]:
            raise ValueError("Resume needs the same prepared data as the saved run")
        mx.random.seed(train_config.seed)
        rng = np.random.default_rng(train_config.seed)
        model = GPT(model_config)
        optimizer = optim.AdamW(
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=[0.9, 0.95],
            bias_correction=True,
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        if resume:
            checkpoint.restore(path, model, optimizer, rng, state)
        else:
            state = {
                "format": 1,
                "step": 0,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "dataset": dataset.identity,
                "data_path": str(dataset.path),
                "best_val_loss": None,
            }
        state["data_path"] = str(dataset.path)
        checkpoint.write_json(run / "run.json", {**state, "target_steps": steps})
        grad_fn = nn.value_and_grad(model, loss_fn)
        report(
            f"{model.parameter_count:,} parameters | {mx.default_device()} | "
            f"{train_config.batch_size * train_config.accumulation * model_config.context:,} "
            "bytes per step"
        )
        stop = False

        def request_stop(_signum, _frame):
            nonlocal stop
            stop = True

        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, request_stop)
        started = time.monotonic()
        last_log = started
        interval_tokens = 0
        last_saved = state["step"] if resume else -1
        last_loss = None
        metrics_path = run / "metrics.jsonl"
        # A crash can leave reports after the latest complete checkpoint.
        if resume and metrics_path.exists():
            rows = []
            for line in metrics_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["step"] <= state["step"]:
                    rows.append(json.dumps(row, allow_nan=False))
            metrics_path.write_text("\n".join(rows) + ("\n" if rows else ""))

        def validate_and_save():
            nonlocal last_saved
            metrics = evaluate(model, dataset, train_config.batch_size, eval_batches)
            best = state["best_val_loss"]
            improved = best is None or metrics["val_loss"] < best
            if improved:
                state["best_val_loss"] = metrics["val_loss"]
            state["validation"] = metrics
            checkpoint.save(run, model, optimizer, state, rng, improved)
            last_saved = state["step"]
            with metrics_path.open("a") as stream:
                stream.write(json.dumps({"step": state["step"], **metrics}) + "\n")
            report(
                f"step {state['step']:>6} | val {metrics['val_loss']:.4f} | "
                f"{metrics['bits_per_byte']:.3f} bits/byte | saved"
            )

        try:
            if not resume:
                validate_and_save()
            while state["step"] < steps and not stop:
                if time_limit is not None and time.monotonic() - started >= time_limit:
                    break
                accumulated = None
                losses = []
                for _ in range(train_config.accumulation):
                    x, y = dataset.batch("train", train_config.batch_size, rng)
                    loss, grads = grad_fn(model, mx.array(x), mx.array(y))
                    accumulated = (
                        grads
                        if accumulated is None
                        else tree_map(lambda a, b: a + b, accumulated, grads)
                    )
                    mx.eval(loss, accumulated)
                    losses.append(loss.item())
                accumulated = tree_map(lambda g: g / train_config.accumulation, accumulated)
                accumulated, norm = optim.clip_grad_norm(accumulated, train_config.grad_clip)
                last_loss = sum(losses) / len(losses)
                if not math.isfinite(last_loss) or not math.isfinite(norm.item()):
                    raise ValueError("Training became non-finite; resume with the last checkpoint")
                rate = train_config.rate(state["step"])
                optimizer.learning_rate = rate
                optimizer.update(model, accumulated)
                mx.eval(model.parameters(), optimizer.state)
                state["step"] += 1
                interval_tokens += (
                    train_config.batch_size * train_config.accumulation * model_config.context
                )
                if state["step"] % log_every == 0 or state["step"] == steps:
                    now = time.monotonic()
                    tokens_per_second = interval_tokens / max(now - last_log, 1e-6)
                    row = {
                        "step": state["step"],
                        "train_loss": last_loss,
                        "learning_rate": rate,
                        "bytes_per_second": tokens_per_second,
                        "peak_memory_mib": mx.get_peak_memory() / 1024**2,
                    }
                    with metrics_path.open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    report(
                        f"step {state['step']:>6} | train {last_loss:.4f} | "
                        f"{tokens_per_second:,.0f} bytes/s | {row['peak_memory_mib']:.0f} MiB"
                    )
                    interval_tokens = 0
                    last_log = now
                if state["step"] % eval_every == 0:
                    validate_and_save()
            if last_saved != state["step"]:
                validate_and_save()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        report(f"Saved step {state['step']} to {run}")
        return {**state, "train_loss": last_loss, "elapsed_seconds": time.monotonic() - started}


def load_model(run: Path, which: str = "best"):
    with checkpoint.run_lock(run):
        path, state = checkpoint.read(run, which)
        model = GPT(ModelConfig(**state["model_config"]))
        model.load_weights(str(path / "model.safetensors"))
        mx.eval(model.parameters())
    model.eval()
    return model, state


def generate(
    model: GPT,
    prompt: str,
    tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 42,
) -> str:
    if tokens < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("tokens must be positive and temperature finite and nonnegative")
    if not 0 <= top_k <= 256:
        raise ValueError("top-k must be between 0 and 256")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    mx.random.seed(seed)
    model.eval()
    sequence = encode(prompt) or [10]
    generated = []
    for _ in range(tokens):
        logits = model(mx.array([sequence[-model.config.context :]], dtype=mx.int32))[0, -1]
        if temperature == 0:
            token = int(mx.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k:
                cutoff = mx.sort(logits)[-top_k]
                logits = mx.where(logits >= cutoff, logits, -float("inf"))
            token = int(mx.random.categorical(logits).item())
        sequence.append(token)
        generated.append(token)
    return prompt + decode(generated)
