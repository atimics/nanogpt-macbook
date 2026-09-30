import json
import signal

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import checkpoint, engine
from nanogpt_macbook.config import ModelConfig, TrainConfig
from nanogpt_macbook.data import Dataset
from nanogpt_macbook.model import GPT


def device(name):
    if name == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if name == "gpu" else mx.cpu)


def same_tree(a, b):
    for (ak, av), (bk, bv) in zip(tree_flatten(a), tree_flatten(b), strict=True):
        assert ak == bk
        np.testing.assert_allclose(np.array(av), np.array(bv), atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("context,width", [(17, 32), (256, 64)])
def test_queued_updates_match_synchronous_steps(backend, accumulation, context, width):
    device(backend)
    rng = np.random.default_rng(71)
    blocks = [
        rng.integers(0, 256, (accumulation, 2, context + 1), dtype=np.int32) for _ in range(7)
    ]
    rates = [0.0001 * (index + 1) for index in range(len(blocks))]
    records = []
    for pipeline in (False, True):
        mx.random.seed(23)
        model = GPT(ModelConfig(context=context, width=width, heads=2, layers=2))
        optimizer = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.95], bias_correction=True)
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        step = engine.make_train_step(model, optimizer, accumulation, 0.1)
        batches = (
            (mx.array(block[..., :-1]), mx.array(block[..., 1:]), rate)
            for block, rate in zip(blocks, rates, strict=True)
        )
        results = list(engine.run_steps(step, batches, pipeline=pipeline))
        assert [rate for _, rate in results] == rates
        records.append((results, model.parameters(), optimizer.state))
    same_tree(records[0], records[1])


def test_queue_is_bounded_and_drains_when_closed(monkeypatch):
    submitted, completed, drains = [], [], []

    class Scalar:
        def __init__(self, index):
            self.index = index

        def item(self):
            completed.append(self.index)
            return 1.0

    class Step:
        state = []

        def enqueue(self, inputs, targets, rate):
            submitted.append(inputs)
            assert len(submitted) - len(completed) <= 2
            return Scalar(inputs), mx.array(1.0)

    monkeypatch.setattr(engine.mx, "eval", lambda state: drains.append(state))
    iterator = engine.run_steps(Step(), ((i, i, 0.01) for i in range(5)), pipeline=True)
    assert next(iterator) == (1.0, 0.01)
    assert submitted == [0, 1]
    iterator.close()
    assert len(drains) == 1


@pytest.mark.parametrize("bad_loss,bad_norm", [(float("nan"), 1), (1, float("inf"))])
def test_queue_checks_each_loss_and_norm_and_drains_errors(bad_loss, bad_norm, monkeypatch):
    drains = []

    class Step:
        state = []

        def enqueue(self, inputs, targets, rate):
            return mx.array(bad_loss if inputs == 1 else 1), mx.array(
                bad_norm if inputs == 1 else 1
            )

    monkeypatch.setattr(engine.mx, "eval", lambda state: drains.append(state))
    with pytest.raises(ValueError, match="non-finite"):
        list(engine.run_steps(Step(), ((i, i, 0.01) for i in range(4)), pipeline=True))
    assert len(drains) == 1


def train_run(corpus, run, steps, pipeline=True, **kwargs):
    return engine.train(
        corpus,
        run,
        ModelConfig(context=16, layers=1, heads=2, width=16),
        TrainConfig(batch_size=2, accumulation=2, warmup_steps=2, decay_steps=20),
        steps,
        eval_every=4,
        log_every=3,
        eval_batches=1,
        pipeline=pipeline,
        report=lambda _: None,
        **kwargs,
    )


def same_checkpoint(a, b):
    ap, ast = checkpoint.read(a)
    bp, bst = checkpoint.read(b)
    assert ast["step"] == bst["step"]
    assert ast["numpy_rng"] == bst["numpy_rng"]
    for name in ("model.safetensors", "optimizer.safetensors"):
        same_tree(mx.load(str(ap / name)), mx.load(str(bp / name)))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_reports_checkpoints_and_resume_match_synchronous_training(backend, corpus, tmp_path):
    device(backend)
    sync, queued, split = (tmp_path / name for name in ("sync", "queued", "split"))
    train_run(corpus, sync, 9, pipeline=False)
    train_run(corpus, queued, 9)
    train_run(corpus, split, 5)
    train_run(corpus, split, 9, resume=True)
    same_checkpoint(sync, queued)
    same_checkpoint(queued, split)
    rows = [json.loads(line) for line in (queued / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if "train_loss" in r] == [3, 6, 9]
    assert [r["step"] for r in rows if "val_loss" in r] == [0, 4, 8, 9]
    expected = [json.loads(line) for line in (sync / "metrics.jsonl").read_text().splitlines()]
    np.testing.assert_allclose(
        [r["val_loss"] for r in rows if "val_loss" in r],
        [r["val_loss"] for r in expected if "val_loss" in r],
        atol=1e-6,
    )


@pytest.mark.parametrize("stop_mode", ["SIGINT", "SIGTERM", "time"])
def test_stopping_drains_submitted_steps_and_preserves_resume(
    stop_mode, corpus, tmp_path, monkeypatch
):
    original = Dataset.batch
    count = 0

    def batch(self, split, size, rng):
        nonlocal count
        result = original(self, split, size, rng)
        if split == "train":
            count += 1
            if count == 5 and stop_mode != "time":
                signal.raise_signal(getattr(signal, stop_mode))
        return result

    with monkeypatch.context() as changes:
        changes.setattr(Dataset, "batch", batch)
        options = {}
        if stop_mode == "time":
            changes.setattr(engine.time, "monotonic", lambda: 2.0 if count >= 5 else 0.0)
            options["time_limit"] = 1.0
        state = train_run(corpus, tmp_path / "stopped", 9, **options)
    assert state["step"] == 3
    assert count == 6
    train_run(corpus, tmp_path / "stopped", 9, resume=True)
    train_run(corpus, tmp_path / "full", 9)
    same_checkpoint(tmp_path / "full", tmp_path / "stopped")


@pytest.mark.parametrize("failure", ["loss", "data"])
def test_failed_queued_training_preserves_last_checkpoint(failure, corpus, tmp_path, monkeypatch):
    original_batch, original_loss = Dataset.batch, engine.loss_fn
    count = 0

    def batch(self, split, size, rng):
        nonlocal count
        x, y = original_batch(self, split, size, rng)
        if split == "train":
            count += 1
            if count == 3:
                if failure == "data":
                    raise OSError("data read failed")
                x.fill(0)
        return x, y

    def loss(model, x, y):
        return mx.where(mx.all(x == 0), mx.array(float("nan")), original_loss(model, x, y))

    monkeypatch.setattr(Dataset, "batch", batch)
    monkeypatch.setattr(engine, "loss_fn", loss)
    with pytest.raises((ValueError, OSError), match="non-finite|data read failed"):
        train_run(corpus, tmp_path / "failed", 9)
    _, saved = checkpoint.read(tmp_path / "failed")
    assert saved["step"] == 0
    assert saved["execution"] == "pipelined"
    model, state = engine.load_model(tmp_path / "failed", "latest")
    assert state["step"] == 0
    assert all(bool(mx.all(mx.isfinite(value))) for _, value in tree_flatten(model.parameters()))
