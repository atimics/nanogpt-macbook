import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import model as model_module
from nanogpt_macbook.activations import gelu_approx
from nanogpt_macbook.config import ModelConfig
from nanogpt_macbook.model import GPT, loss_fn


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("compiled", [False, True])
def test_gelu_values_and_gradients_match_mlx(device, compiled):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    # Every batch has different values. This also exercises noncontiguous input.
    x = mx.linspace(-10, 10, 2 * 3 * 129).reshape(2, 3, 129).swapaxes(1, 2)
    weights = mx.random.normal(x.shape)
    actual = mx.value_and_grad(lambda x: (gelu_approx(x) * weights).sum())
    expected = mx.value_and_grad(lambda x: (nn.gelu_approx(x) * weights).sum())
    if compiled:
        actual, expected = mx.compile(actual), mx.compile(expected)
    for a, b in zip(actual(x), expected(x), strict=True):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_gelu_gradient_matches_float64_finite_difference(device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    x = np.linspace(-10, 10, 10003, dtype=np.float32)
    actual = np.array(mx.compile(mx.grad(lambda x: gelu_approx(x).sum()))(mx.array(x)))

    def reference(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    values = x.astype(np.float64)
    delta = 1e-4
    expected = (reference(values + delta) - reference(values - delta)) / (2 * delta)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-5)
    tails = mx.array([-100.0, 0.0, 100.0])
    np.testing.assert_allclose(
        np.array(mx.grad(lambda x: gelu_approx(x).sum())(tails)), [0, 0.5, 1], atol=1e-7
    )


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_full_model_gradients_match_mlx_activation(device, monkeypatch):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    model = GPT(ModelConfig(context=33, layers=2, heads=2, width=32))
    tokens = mx.random.randint(0, 256, (3, 34))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def gradients():
        grad_fn = nn.value_and_grad(model, lambda x, y: loss_fn(model, x, y))
        return mx.compile(grad_fn, inputs=model.state, outputs=model.state)(inputs, targets)

    actual = gradients()
    mx.eval(actual)
    monkeypatch.setattr(model_module, "gelu_approx", nn.gelu_approx)
    expected = gradients()
    mx.eval(expected)
    actual_leaves, expected_leaves = tree_flatten(actual), tree_flatten(expected)
    assert [key for key, _ in actual_leaves] == [key for key, _ in expected_leaves]
    for (_, a), (_, b) in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=2e-6, rtol=2e-5)
