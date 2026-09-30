"""Training, evaluation, and generation on MLX."""

import json
import math
import signal
import threading
import time
from collections import deque
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from . import checkpoint
from .config import ModelConfig, TrainConfig
from .data import Dataset, decode, encode
from .gradients import clip_grad_norm
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
        mx.set_cache_limit(int(memory_gb * 1024**3))
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


def _checked_loss(loss, norm):
    loss_value, norm_value = loss.item(), norm.item()
    if not math.isfinite(loss_value) or not math.isfinite(norm_value):
        raise ValueError("Training became non-finite; resume with the last checkpoint")
    return loss_value


class TrainStep:
    """One compiled update with synchronous and queued execution."""

    def __init__(self, compiled, state):
        self.compiled = compiled
        self.state = state

    def __call__(self, inputs, targets, rate):
        loss, norm = self.compiled(inputs, targets, mx.array(rate))
        mx.eval(loss, norm, self.state)
        return _checked_loss(loss, norm)

    def enqueue(self, inputs, targets, rate):
        loss, norm = self.compiled(inputs, targets, mx.array(rate))
        mx.async_eval(loss, norm, self.state)
        return loss, norm


def run_steps(step, batches, *, pipeline=False):
    """Yield checked losses in order and finish every update before returning.

    Each batch contains inputs, targets, and its learning rate. Queued execution
    overlaps host work with GPU work and keeps at most two updates in flight.
    The batch iterator controls stop and checkpoint boundaries.
    """
    if not pipeline:
        for inputs, targets, rate in batches:
            yield step(inputs, targets, rate), rate
        return
    pending = deque()
    drained = False
    try:
        for inputs, targets, rate in batches:
            pending.append((step.enqueue(inputs, targets, rate), rate))
            if len(pending) == 2:
                (loss, norm), rate = pending.popleft()
                yield _checked_loss(loss, norm), rate
        # The last yielded result is also a boundary for evaluation, reporting,
        # and saving. All parameter and optimizer updates finish here.
        mx.eval(step.state)
        drained = True
        while pending:
            (loss, norm), rate = pending.popleft()
            yield _checked_loss(loss, norm), rate
    finally:
        # Complete submitted work when an input error or early close ends a run.
        if not drained:
            mx.eval(step.state)


def make_train_step(model: GPT, optimizer, accumulation: int, grad_clip: float):
    """Compile the full gradient and optimizer update for a fixed batch shape."""
    state = [model.state, optimizer.state]
    grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(
        mx.compile,
        inputs=state,
        outputs=state,
    )
    def compiled_step(inputs, targets, rate):
        total_loss = None
        total_grads = None
        for index in range(accumulation):
            loss, grads = grad_fn(model, inputs[index], targets[index])
            total_loss = loss if total_loss is None else total_loss + loss
            total_grads = (
                grads if total_grads is None else tree_map(lambda a, b: a + b, total_grads, grads)
            )
        grads = tree_map(lambda grad: grad / accumulation, total_grads)
        grads, norm = clip_grad_norm(grads, grad_clip)
        optimizer.learning_rate = rate
        optimizer.update(model, grads)
        return total_loss / accumulation, norm

    return TrainStep(compiled_step, state)


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
    pipeline: bool | None = None,
    report=print,
) -> dict:
    if min(steps, eval_every, eval_batches, log_every) < 1:
        raise ValueError("Steps and report intervals must be positive")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time-limit must be finite and positive")
    if pipeline is None:
        pipeline = mx.default_device() == mx.gpu
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
        state["execution"] = "pipelined" if pipeline else "compiled"
        checkpoint.write_json(run / "run.json", {**state, "target_steps": steps})
        train_step = make_train_step(
            model, optimizer, train_config.accumulation, train_config.grad_clip
        )
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

            def should_stop():
                return stop or (time_limit is not None and time.monotonic() - started >= time_limit)

            while state["step"] < steps and not should_stop():
                first = state["step"]
                boundary = min(
                    steps,
                    (first // log_every + 1) * log_every,
                    (first // eval_every + 1) * eval_every,
                )

                def batches(first=first, boundary=boundary):
                    for index in range(first, boundary):
                        if should_stop():
                            return
                        inputs, targets = [], []
                        for _ in range(train_config.accumulation):
                            x, y = dataset.batch("train", train_config.batch_size, rng)
                            inputs.append(x)
                            targets.append(y)
                        yield (
                            mx.array(np.stack(inputs)),
                            mx.array(np.stack(targets)),
                            train_config.rate(index),
                        )

                for result in run_steps(train_step, batches(), pipeline=pipeline):
                    last_loss, rate = result
                    state["step"] += 1
                    interval_tokens += (
                        train_config.batch_size * train_config.accumulation * model_config.context
                    )
                if state["step"] == first:
                    break
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


def _sample_token(logits, temperature, top_k):
    if temperature == 0:
        return mx.argmax(logits)
    logits = logits / temperature
    if top_k:
        cutoff = mx.sort(logits)[-top_k]
        logits = mx.where(logits >= cutoff, logits, -float("inf"))
    return mx.random.categorical(logits)


def _sample_prefix(model, window, count, temperature, top_k):
    logits, cache = model.cached_logits(mx.array([window], dtype=mx.int32))
    token = _sample_token(logits[0, -1], temperature, top_k)
    generated = [token]
    mx.async_eval(token, cache)

    def step(token, state, position):
        logits, state = model.cached_logits(token.reshape(1, 1), state, position)
        return _sample_token(logits[0, -1], temperature, top_k), state

    if mx.default_device() == mx.gpu and count >= 16:
        step = mx.compile(step, inputs=mx.random.state, outputs=mx.random.state)
    for index in range(1, count):
        position = mx.array([len(window) + index - 1], dtype=mx.int32)
        token, cache = step(token, cache, position)
        generated.append(token)
        mx.async_eval(token, cache)
        # Bound queued work while each sampled byte stays on the device.
        if index % 8 == 0:
            mx.eval(generated[-8])
    return mx.stack(generated).tolist()


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

    def last_logits(inputs):
        return model(inputs, last_token_only=True)[0, -1]

    if len(sequence) < model.config.context and tokens > 1:
        count = min(tokens, model.config.context - len(sequence) + 1)
        generated = _sample_prefix(model, sequence, count, temperature, top_k)
        sequence.extend(generated)
    compiled_logits = None
    compile_windows = mx.default_device() == mx.gpu
    for index in range(len(generated), tokens):
        window = sequence[-model.config.context :]
        # A full window keeps a fixed shape as generation continues. Sixteen
        # remaining bytes amortize compilation in the Metal measurements.
        # CPU sampling measured faster with the eager last-token path.
        if (
            compile_windows
            and compiled_logits is None
            and len(window) == model.config.context
            and tokens - index >= 16
        ):
            compiled_logits = mx.compile(last_logits)
        forward = compiled_logits or last_logits
        logits = forward(mx.array([window], dtype=mx.int32))
        token = int(_sample_token(logits, temperature, top_k).item())
        sequence.append(token)
        generated.append(token)
    return prompt + decode(generated)
