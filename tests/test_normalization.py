import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import normalization
from nanogpt_macbook.config import ModelConfig
from nanogpt_macbook.model import GPT, loss_fn
from nanogpt_macbook.normalization import LayerNorm


def select(device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mx.random.seed(52)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("shape", [(17,), (3, 5, 33), (2, 7, 128), (3, 256), (2, 5, 384), (7, 512)])
def test_values_and_all_gradients_match_mlx(device, compiled, shape):
    select(device)
    # Strided inputs, weights, and output gradients also cover the copy boundary.
    x = mx.random.normal((*shape, 2))[..., 0]
    w = mx.random.normal((shape[-1] * 2,))[::2]
    b = mx.random.normal((shape[-1],))
    cotangent = mx.random.normal((*shape, 2))[..., 1]
    layer = LayerNorm(shape[-1], eps=0.002)

    def loss(x, w, b):
        layer.weight, layer.bias = w, b
        return (layer(x) * cotangent).sum()

    actual = mx.value_and_grad(loss, argnums=(0, 1, 2))
    expected = mx.value_and_grad(
        lambda x, w, b: (mx.fast.layer_norm(x, w, b, layer.eps) * cotangent).sum(),
        argnums=(0, 1, 2),
    )
    if compiled:
        actual, expected = mx.compile(actual), mx.compile(expected)
    a_value, a_grads = actual(x, w, b)
    e_value, e_grads = expected(x, w, b)
    for a, e in zip((a_value, *a_grads), (e_value, *e_grads), strict=True):
        np.testing.assert_allclose(np.array(a), np.array(e), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("constant", [False, True])
def test_gradients_match_float64_finite_differences(constant):
    select("gpu")
    rng = np.random.default_rng(16)
    x = (rng.normal(size=(3, 17)) * 0.5 + 4).astype(np.float32)
    if constant:
        x.fill(4)
    w = rng.normal(size=(17,)).astype(np.float32)
    b = rng.normal(size=(17,)).astype(np.float32)
    cotangent = rng.normal(size=x.shape).astype(np.float32)
    layer = LayerNorm(17)

    def actual(x, w, b):
        layer.weight, layer.bias = w, b
        return (layer(x) * mx.array(cotangent)).sum()

    gradients = mx.compile(mx.grad(actual, argnums=(0, 1, 2)))(
        mx.array(x), mx.array(w), mx.array(b)
    )

    def reference(x, w, b):
        centered = x - x.mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
        return ((normalized * w + b) * cotangent).sum()

    inputs = [value.astype(np.float64) for value in (x, w, b)]
    delta = 1e-6
    for index, grad in enumerate(gradients):
        expected = np.empty_like(inputs[index])
        for position in np.ndindex(expected.shape):
            plus, minus = [a.copy() for a in inputs], [a.copy() for a in inputs]
            plus[index][position] += delta
            minus[index][position] -= delta
            expected[position] = (reference(*plus) - reference(*minus)) / (2 * delta)
        np.testing.assert_allclose(np.array(grad), expected, atol=2e-5, rtol=3e-5)


@pytest.mark.parametrize("width", [32, 128, 256, 384])
def test_full_model_gradients_match_native_normalization(width, monkeypatch):
    select("gpu")
    model = GPT(ModelConfig(context=17, layers=2, heads=2, width=width))
    tokens = mx.random.randint(0, 256, (3, 18))

    def gradients():
        grad = nn.value_and_grad(model, lambda x, y: loss_fn(model, x, y))
        out = mx.compile(grad, inputs=model.state, outputs=model.state)(
            tokens[:, :-1], tokens[:, 1:]
        )
        mx.eval(out)
        return out

    actual = gradients()
    monkeypatch.setattr(LayerNorm, "__call__", nn.LayerNorm.__call__)
    expected = gradients()
    for (a_key, a), (e_key, e) in zip(tree_flatten(actual), tree_flatten(expected), strict=True):
        assert a_key == e_key
        np.testing.assert_allclose(np.array(a), np.array(e), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "device,width,options,dtype",
    [
        ("cpu", 128, {}, mx.float32),
        ("gpu", 513, {}, mx.float32),
        ("gpu", 128, {"affine": False}, mx.float32),
        ("gpu", 128, {"bias": False}, mx.float32),
        ("gpu", 128, {}, mx.float16),
    ],
)
def test_other_layouts_use_native_mlx(device, width, options, dtype, monkeypatch):
    select(device)

    def unexpected_kernel(_eps):
        raise AssertionError("This input should use native MLX")

    monkeypatch.setattr(normalization, "_normalize", unexpected_kernel)
    layer = LayerNorm(width, **options)
    layer.set_dtype(dtype)
    x = mx.random.normal((3, width)).astype(dtype)
    actual = mx.value_and_grad(lambda x: layer(x).sum())(x)
    expected = mx.value_and_grad(lambda x: nn.LayerNorm.__call__(layer, x).sum())(x)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.array(a), np.array(e))
