import json
from dataclasses import replace

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import checkpoint
from nanogpt_macbook.config import PRESETS, ModelConfig, TrainConfig
from nanogpt_macbook.data import Dataset
from nanogpt_macbook.engine import evaluate, generate, load_model, train
from nanogpt_macbook.model import GPT

MODEL = ModelConfig(context=16, layers=1, heads=2, width=16)
TRAIN = TrainConfig(batch_size=2, warmup_steps=0, decay_steps=100, learning_rate=0.01)


def run_training(data, run, steps, **kwargs):
    return train(
        data,
        run,
        MODEL,
        TRAIN,
        steps,
        eval_every=2,
        eval_batches=2,
        log_every=2,
        report=lambda _: None,
        **kwargs,
    )


def weights(run):
    path, _ = checkpoint.read(run)
    return mx.load(str(path / "model.safetensors"))


def test_future_tokens_cannot_change_past_logits():
    model = GPT(MODEL)
    a = mx.array([[1, 2, 3, 4, 5]])
    b = mx.array([[1, 2, 3, 90, 91]])
    np.testing.assert_allclose(np.array(model(a)[:, :3]), np.array(model(b)[:, :3]), atol=1e-6)


def test_training_learns_and_keeps_best_and_latest(corpus, tmp_path):
    run = tmp_path / "learning"
    result = run_training(corpus, run, 20)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    validation = [row["val_loss"] for row in rows if "val_loss" in row]
    assert result["step"] == 20
    assert validation[-1] < validation[0] - 0.5
    assert 1 <= len(list((run / "checkpoints").glob("step-*"))) <= 2
    assert checkpoint.read(run, "best")[1]["best_val_loss"] == min(validation)


def test_resume_matches_uninterrupted_training(corpus, tmp_path):
    full = tmp_path / "full"
    split = tmp_path / "split"
    run_training(corpus, full, 6)
    run_training(corpus, split, 2)
    run_training(corpus, split, 6, resume=True)
    for key, value in weights(full).items():
        np.testing.assert_allclose(np.array(value), np.array(weights(split)[key]), atol=1e-7)
    full_path, full_state = checkpoint.read(full)
    split_path, split_state = checkpoint.read(split)
    assert full_state["numpy_rng"] == split_state["numpy_rng"]
    for key, value in mx.load(str(full_path / "optimizer.safetensors")).items():
        restored = mx.load(str(split_path / "optimizer.safetensors"))[key]
        np.testing.assert_allclose(np.array(value), np.array(restored), atol=1e-7)


def test_checkpoint_restores_the_next_random_draw(tmp_path):
    run = tmp_path / "random-state"
    run.mkdir()
    model = GPT(MODEL)
    optimizer = optim.AdamW(learning_rate=0.01)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    rng = np.random.default_rng(67)
    mx.eval(mx.random.normal((13,)))
    checkpoint.save(run, model, optimizer, {"format": 1, "step": 0}, rng, True)
    expected_mlx = np.array(mx.random.normal((8,)))
    expected_numpy = rng.integers(0, 1000, size=8)
    mx.random.seed(987654321)
    rng = np.random.default_rng(987654321)
    path, state = checkpoint.read(run)
    checkpoint.restore(path, model, optimizer, rng, state)
    np.testing.assert_array_equal(np.array(mx.random.normal((8,))), expected_mlx)
    np.testing.assert_array_equal(rng.integers(0, 1000, size=8), expected_numpy)


def test_accumulation_matches_a_larger_batch(corpus, tmp_path):
    for name, config in (
        ("large", replace(TRAIN, batch_size=4)),
        ("micro", replace(TRAIN, accumulation=2)),
    ):
        train(corpus, tmp_path / name, MODEL, config, 1, eval_batches=1, report=lambda _: None)
    for key, value in weights(tmp_path / "large").items():
        np.testing.assert_allclose(
            np.array(value), np.array(weights(tmp_path / "micro")[key]), atol=2e-6, rtol=2e-5
        )


def test_evaluation_keeps_training_random_state(corpus):
    model = GPT(MODEL)
    mx.eval(model.parameters(), mx.random.state)
    before = [np.array(value) for value in mx.random.state]
    evaluate(model, Dataset(corpus, MODEL.context), 2, 2)
    for a, b in zip(before, mx.random.state, strict=True):
        np.testing.assert_array_equal(a, np.array(b))


def test_sampling_is_seeded_and_supports_long_unicode_prompts(corpus, tmp_path):
    run = tmp_path / "sample"
    run_training(corpus, run, 2)
    model, _ = load_model(run)
    prompt = "世界 🌊 " * 10
    a = generate(model, prompt, tokens=12, seed=9)
    b = generate(model, prompt, tokens=12, seed=9)
    assert a == b
    assert a.startswith(prompt)
    assert generate(model, "", tokens=2, temperature=0) == generate(
        model, "", tokens=2, temperature=0, seed=11
    )


def test_run_lock_and_existing_run_protect_saved_work(corpus, tmp_path):
    run = tmp_path / "run"
    run_training(corpus, run, 2)
    latest = (run / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="Use resume"):
        run_training(corpus, run, 4)
    with checkpoint.run_lock(run), pytest.raises(ValueError, match="already owns"):
        run_training(corpus, run, 4, resume=True)
    assert (run / "latest.json").read_bytes() == latest


def test_resume_checks_dataset_identity(corpus, tmp_path):
    run = tmp_path / "run"
    run_training(corpus, run, 2)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same prepared data"):
        run_training(corpus, run, 4, resume=True)


def test_failed_save_preserves_last_pointer(corpus, tmp_path, monkeypatch):
    run = tmp_path / "run"
    run_training(corpus, run, 2)
    before = (run / "latest.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(GPT, "save_weights", fail)
    with pytest.raises(OSError, match="disk full"):
        run_training(corpus, run, 4, resume=True)
    assert (run / "latest.json").read_bytes() == before
    assert checkpoint.read(run)[1]["step"] == 2


def test_time_limit_saves_a_valid_checkpoint(corpus, tmp_path):
    state = run_training(corpus, tmp_path / "bounded", 100, time_limit=1e-9)
    assert state["step"] == 0
    assert checkpoint.read(tmp_path / "bounded")[1]["step"] == 0


def test_schedule_and_config_validation():
    config = TrainConfig()
    assert config.rate(0) == config.learning_rate / config.warmup_steps
    assert config.rate(config.warmup_steps) == config.learning_rate
    assert config.rate(10000) == config.min_learning_rate
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(width=17, heads=2)
    with pytest.raises(ValueError, match="finite"):
        TrainConfig(learning_rate=float("nan"))


@pytest.mark.parametrize("name", PRESETS)
def test_preset_parameter_counts(name):
    config, _ = PRESETS[name]
    model = GPT(config)
    width = config.width
    expected = config.layers * (12 * width**2 + 4 * width)
    expected += (config.vocab_size + config.context + 2) * width
    assert model.parameter_count == expected
    assert all(value.dtype == mx.float32 for _, value in tree_flatten(model.parameters()))
