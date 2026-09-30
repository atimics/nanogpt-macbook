import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from nanogpt_macbook import checkpoint, engine
from nanogpt_macbook.config import ModelConfig, TrainConfig
from nanogpt_macbook.updates import update_parameters


class Parameters(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.leaves = list(values)


def compare_trees(actual, expected):
    for (a_key, a), (e_key, e) in zip(tree_flatten(actual), tree_flatten(expected), strict=True):
        assert a_key == e_key
        np.testing.assert_array_equal(np.array(a), np.array(e))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bias_correction", [False, True])
def test_packed_updates_match_adamw_and_restore(compiled, bias_correction, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu)
    mx.random.seed(41)
    shapes = [(17,), (3, 11), (128,), ()] * 8 + [(32, 64), (0,)]
    initial = [mx.random.normal(shape) for shape in shapes]
    actual, expected = Parameters(initial), Parameters(initial)
    options = dict(
        learning_rate=lambda step: 0.003 / (step + 1),
        betas=[0.9, 0.95],
        weight_decay=0.1,
        bias_correction=bias_correction,
    )
    a_opt, e_opt = optim.AdamW(**options), optim.AdamW(**options)
    a_opt.init(actual.trainable_parameters())
    e_opt.init(expected.trainable_parameters())
    shapes_seen = []
    apply_single = a_opt.apply_single

    def record_update(gradient, parameter, state):
        shapes_seen.append(parameter.shape)
        return apply_single(gradient, parameter, state)

    monkeypatch.setattr(a_opt, "apply_single", record_update)

    def make_step(model, optimizer, packed):
        def step(grads):
            if packed:
                update_parameters(model, optimizer, grads)
            else:
                optimizer.update(model, grads)

        if compiled:
            state = [model.state, optimizer.state]
            step = mx.compile(step, inputs=state, outputs=state)
        return step

    a_step = make_step(actual, a_opt, True)
    e_step = make_step(expected, e_opt, False)
    for index in range(12):
        grads = {"leaves": [mx.random.normal(tuple(reversed(s))).T for s in shapes]}
        a_step(grads)
        e_step(grads)
        compare_trees([actual.parameters(), a_opt.state], [expected.parameters(), e_opt.state])
        if index == 5:
            # The same tree operation restores optimizer.safetensors data.
            a_opt.state = tree_unflatten(tree_flatten(a_opt.state))
            a_step = make_step(actual, a_opt, True)
    assert (1432,) in shapes_seen


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("kind", ["small_group", "float16", "other_optimizer"])
def test_regular_updates_keep_the_optimizer_path(device, kind):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    count = 31 if kind == "small_group" else 34
    dtype = mx.float16 if kind == "float16" else mx.float32
    values = [mx.ones((17,), dtype=dtype) for _ in range(count)]
    actual, expected = Parameters(values), Parameters(values)
    cls = optim.SGD if kind == "other_optimizer" else optim.AdamW
    a_opt, e_opt = cls(0.01), cls(0.01)
    gradients = {"leaves": [mx.full((17,), 0.2, dtype=dtype) for _ in range(count)]}
    update_parameters(actual, a_opt, gradients)
    e_opt.update(expected, gradients)
    compare_trees([actual.parameters(), a_opt.state], [expected.parameters(), e_opt.state])


def test_packed_training_checkpoint_matches_reference_and_resume(corpus, tmp_path, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu)
    model = ModelConfig(context=17, layers=8, heads=2, width=64)
    training = TrainConfig(batch_size=2, learning_rate=0.0001, warmup_steps=2, decay_steps=20)

    def train(name, steps, resume=False):
        engine.train(
            corpus,
            tmp_path / name,
            model,
            training,
            steps,
            resume=resume,
            eval_every=2,
            eval_batches=1,
            log_every=2,
            report=lambda _: None,
        )

    train("packed", 6)
    train("resumed", 2)
    train("resumed", 6, resume=True)
    monkeypatch.setattr(engine, "update_parameters", lambda m, o, g: o.update(m, g))
    train("reference", 6)
    packed_path, _ = checkpoint.read(tmp_path / "packed")
    for name in ("resumed", "reference"):
        path, _ = checkpoint.read(tmp_path / name)
        for file in ("model.safetensors", "optimizer.safetensors", "random.safetensors"):
            packed = mx.load(str(packed_path / file))
            compared = mx.load(str(path / file))
            assert packed.keys() == compared.keys()
            for key in packed:
                np.testing.assert_allclose(
                    np.array(packed[key]), np.array(compared[key]), atol=2e-7, rtol=2e-6
                )
