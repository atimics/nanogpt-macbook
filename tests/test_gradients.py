import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import engine, gradients
from nanogpt_macbook.config import ModelConfig
from nanogpt_macbook.gradients import clip_grad_norm
from nanogpt_macbook.model import GPT


def select(device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mx.random.seed(18)


def assert_same(actual, expected):
    for (a_key, a), (e_key, e) in zip(tree_flatten(actual), tree_flatten(expected), strict=True):
        assert a_key == e_key
        np.testing.assert_allclose(np.array(a), np.array(e), atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("limit", [0, 0.1, 1e6])
def test_global_clipping_matches_mlx_for_strided_nested_gradients(device, compiled, limit):
    select(device)
    grads = {
        "scalar": mx.array(-3.0),
        "empty": mx.zeros((0, 3)),
        "matrix": mx.random.normal((65, 129)).T,
        "slice": mx.random.normal((133,))[::3],
        "cube": mx.random.normal((3, 17, 9)).swapaxes(0, 2),
        "broadcast": mx.broadcast_to(mx.random.normal((17,)), (5, 17)),
        "layers": [{"bias": mx.random.normal((i + 1,))} for i in range(19)],
    }

    def actual(g):
        return clip_grad_norm(g, limit)

    def expected(g):
        return optim.clip_grad_norm(g, limit)

    if compiled:
        actual, expected = mx.compile(actual), mx.compile(expected)
    assert_same(actual(grads), expected(grads))


def test_large_gradient_reduction_matches_float64_reference():
    select("gpu")
    # Cross the larger-block threshold with a transposed matrix and an odd tail.
    grads = {"large": mx.random.normal((1024, 4097)).T, "tail": mx.random.normal((17,))}
    clipped, norm = mx.compile(lambda g: clip_grad_norm(g, 0.3))(grads)
    arrays = {key: np.array(value) for key, value in grads.items()}
    reference = np.sqrt(sum(np.square(a.astype(np.float64)).sum() for a in arrays.values()))
    np.testing.assert_allclose(norm.item(), reference, rtol=2e-6)
    scale = min(0.3 / (reference + 1e-6), 1.0)
    for key, actual in clipped.items():
        np.testing.assert_allclose(np.array(actual), arrays[key] * scale, atol=1e-9, rtol=2e-6)


@pytest.mark.parametrize("value", [0.0, 1e10, 1e20, float("inf"), float("nan")])
def test_zero_large_and_nonfinite_gradients_match_mlx(value):
    select("gpu")
    grads = {"values": mx.full((35,), value)}
    assert_same(
        mx.compile(lambda g: clip_grad_norm(g, 1.0))(grads), optim.clip_grad_norm(grads, 1.0)
    )


@pytest.mark.parametrize("device,dtype", [("cpu", mx.float32), ("gpu", mx.float16)])
def test_other_backends_and_dtypes_use_native_mlx(device, dtype, monkeypatch):
    select(device)

    def unexpected_kernel(*_args):
        raise AssertionError("This input should use native MLX")

    monkeypatch.setattr(gradients, "_reduction", unexpected_kernel)
    grads = {"value": mx.arange(17).astype(dtype)}
    assert_same(clip_grad_norm(grads, 1.0), optim.clip_grad_norm(grads, 1.0))


def test_empty_tree_empty_arrays_and_negative_limit():
    select("gpu")
    for grads in ({}, {"empty": mx.zeros((0,))}):
        assert_same(clip_grad_norm(grads, 1.0), optim.clip_grad_norm(grads, 1.0))
    with pytest.raises(ValueError, match="max_norm"):
        clip_grad_norm({"value": mx.ones((3,))}, -1)


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("width", [32, 512])
def test_complete_updates_match_native_clipping(accumulation, width, monkeypatch):
    select("gpu")
    saved = []
    for implementation in (optim.clip_grad_norm, clip_grad_norm):
        monkeypatch.setattr(engine, "clip_grad_norm", implementation)
        mx.random.seed(16)
        model = GPT(ModelConfig(context=17, layers=2, heads=2, width=width))
        optimizer = optim.AdamW(
            learning_rate=0.001, weight_decay=0.1, betas=[0.9, 0.95], bias_correction=True
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        step = engine.make_train_step(model, optimizer, accumulation, 0.01)
        rng = np.random.default_rng(16)
        for _ in range(5):
            block = rng.integers(0, 256, (accumulation, 3, 18), dtype=np.int32)
            step(mx.array(block[:, :, :-1]), mx.array(block[:, :, 1:]), 0.001)
        saved.append((model.parameters(), optimizer.state))
    assert_same(saved[0], saved[1])
