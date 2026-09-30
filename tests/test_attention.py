from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import attention, checkpoint
from nanogpt_macbook import model as model_module
from nanogpt_macbook.attention import training_attention
from nanogpt_macbook.config import ModelConfig, TrainConfig
from nanogpt_macbook.engine import make_train_step, train
from nanogpt_macbook.model import GPT, loss_fn


def use_device(device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)


def reference(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "length,width",
    [(1, 8), (5, 8), (33, 16), (128, 32), (256, 32), (288, 40), (480, 64), (512, 48), (513, 16)],
)
def test_attention_values_and_gradients_match_mlx(device, compiled, length, width):
    use_device(device)
    # Slices also cover strided input arrays and rows beyond a 32-lane boundary.
    packed = mx.random.normal((1, 2, length + 1, 3 * width))
    inputs = [part[:, :, :length] for part in mx.split(packed, 3, axis=-1)]
    weights = mx.random.normal((1, 2, length, 2 * width))[..., ::2]

    def forward(q, k, v):
        return training_attention(q, k, v)

    if compiled:
        forward = mx.compile(forward)
    actual = forward(*inputs)
    expected = reference(*inputs)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5)

    actual_grad_fn = mx.grad(
        lambda q, k, v: mx.sum(training_attention(q, k, v) * weights), argnums=(0, 1, 2)
    )
    if compiled:
        actual_grad_fn = mx.compile(actual_grad_fn)
    actual_grads = actual_grad_fn(*inputs)
    expected_grads = mx.grad(
        lambda q, k, v: mx.sum(reference(q, k, v) * weights), argnums=(0, 1, 2)
    )(*inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(
            np.array(actual_grad), np.array(expected_grad), atol=5e-6, rtol=5e-5
        )


@pytest.mark.parametrize("length,width", [(33, 16), (128, 32), (256, 32), (512, 48)])
def test_metal_attention_respects_the_causal_boundary(length, width):
    use_device("gpu")
    boundary = length // 2 + 1
    q, k, v = [mx.random.normal((1, 2, length, width)) for _ in range(3)]
    changed_k = mx.concatenate([k[:, :, :boundary], k[:, :, boundary:] * 20], axis=2)
    changed_v = mx.concatenate([v[:, :, :boundary], v[:, :, boundary:] + 20], axis=2)
    np.testing.assert_allclose(
        np.array(training_attention(q, k, v)[:, :, :boundary]),
        np.array(training_attention(q, changed_k, changed_v)[:, :, :boundary]),
        atol=1e-7,
    )
    grads = mx.grad(
        lambda keys, values: training_attention(q, keys, values)[:, :, :boundary].sum(),
        argnums=(0, 1),
    )(k, v)
    for grad in grads:
        np.testing.assert_array_equal(np.array(grad[:, :, boundary:]), 0)


@pytest.mark.parametrize("length,width", [(33, 16), (128, 32), (256, 32), (512, 48)])
def test_metal_softmax_is_stable_for_large_scores(length, width):
    use_device("gpu")
    q = mx.full((1, 1, length, width), 100.0)
    k = mx.full((1, 1, length, width), 100.0)
    v = mx.random.normal((1, 1, length, width))
    expected = mx.cumsum(v, axis=2) / mx.arange(1, length + 1)[None, None, :, None]
    np.testing.assert_allclose(np.array(training_attention(q, k, v)), np.array(expected), atol=1e-6)


@pytest.mark.parametrize("length,width", [(128, 32), (256, 32), (288, 40), (480, 64), (512, 48)])
@pytest.mark.parametrize("scale", [0, 1, 5])
def test_fused_probabilities_and_values_match_float64(length, width, scale):
    use_device("gpu")
    rng = np.random.default_rng(93)
    q, k = [(rng.normal(size=(2, 3, length, width)) * scale).astype(np.float32) for _ in range(2)]
    v = rng.normal(size=q.shape).astype(np.float32)
    scores = q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2) / np.sqrt(width)
    causal = np.tri(length, dtype=bool)
    scores = np.where(causal, scores, -np.inf)
    expected = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    probabilities, output = attention._score_output(mx.array(q), mx.array(k), mx.array(v))
    actual = np.array(probabilities)
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)
    np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=3e-7)
    np.testing.assert_array_equal(actual[..., ~causal], 0)
    # Check value accumulation with the rounded float32 probabilities. Large
    # scores amplify the probability rounding already checked above.
    expected_output = actual.astype(np.float64) @ v.astype(np.float64)
    np.testing.assert_allclose(np.array(output), expected_output, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("length,width", [(128, 32), (256, 32), (512, 48)])
@pytest.mark.parametrize("uniform", [False, True])
def test_blocked_gradients_match_float64_reference(length, width, uniform):
    use_device("gpu")
    rng = np.random.default_rng(12)
    arrays = [rng.normal(size=(2, 2, length, width)).astype(np.float32) for _ in range(4)]
    if uniform:
        arrays[0].fill(1)
        arrays[1].fill(1)
    q, k, v, weight = (array.astype(np.float64) for array in arrays)
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(width)
    scores = np.where(np.tri(length, dtype=bool), scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    grad_probs = weight @ v.swapaxes(-1, -2)
    grad_scores = probs * (grad_probs - (probs * grad_probs).sum(axis=-1, keepdims=True))
    grad_scores /= np.sqrt(width)
    expected = (
        grad_scores @ k,
        grad_scores.swapaxes(-1, -2) @ q,
        probs.swapaxes(-1, -2) @ weight,
    )
    cotangent = mx.array(arrays[3])
    actual = mx.compile(
        mx.grad(lambda q, k, v: (training_attention(q, k, v) * cotangent).sum(), argnums=(0, 1, 2))
    )(*(mx.array(array) for array in arrays[:3]))
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.array(a), e, atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize("length,width", [(128, 64), (256, 64), (512, 96)])
def test_compiled_model_gradients_and_updates_match_mlx(length, width, monkeypatch):
    use_device("gpu")
    config = ModelConfig(context=length, layers=2, heads=2, width=width)
    tokens = mx.random.randint(0, 256, (2, length + 1))
    records = []
    for implementation in (training_attention, reference):
        monkeypatch.setattr(model_module, "training_attention", implementation)
        mx.random.seed(19)
        model = GPT(config)
        optimizer = optim.AdamW(learning_rate=1e-4)
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        grad_fn = nn.value_and_grad(model, partial(loss_fn, model))
        value, gradients = mx.compile(grad_fn, inputs=model.state, outputs=model.state)(
            tokens[:, :-1], tokens[:, 1:]
        )
        mx.eval(value, gradients)
        step = make_train_step(model, optimizer, accumulation=2, grad_clip=1.0)
        x = mx.stack([tokens[:, :-1], tokens[:, 1:]])
        y = mx.stack([tokens[:, 1:], tokens[:, :-1]])
        for _ in range(3):
            step(x, y, 1e-4)
        records.append((value, gradients, model.parameters(), optimizer.state))
    for (actual_key, actual), (expected_key, expected) in zip(
        tree_flatten(records[0]), tree_flatten(records[1]), strict=True
    ):
        assert actual_key == expected_key
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "length,width,value_width,dtype",
    [
        (96, 32, 32, mx.float32),
        (160, 32, 32, mx.float32),
        (257, 32, 32, mx.float32),
        (256, 33, 33, mx.float32),
        (256, 72, 72, mx.float32),
        (256, 32, 16, mx.float32),
        (256, 32, 32, mx.float16),
    ],
)
def test_other_shapes_keep_the_general_attention_path(
    length, width, value_width, dtype, monkeypatch
):
    use_device("gpu")

    def unexpected_kernel(_block_size):
        raise AssertionError("This shape uses the general attention path")

    monkeypatch.setattr(attention, "_blocked_attention", unexpected_kernel)
    q, k = [mx.random.normal((1, 2, length, width)).astype(dtype) for _ in range(2)]
    v = mx.random.normal((1, 2, length, value_width)).astype(dtype)
    np.testing.assert_allclose(
        np.array(training_attention(q, k, v)),
        np.array(reference(q, k, v)),
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.parametrize("context,width", [(16, 16), (128, 64), (256, 64)])
def test_metal_training_resume_with_accumulation(corpus, tmp_path, context, width):
    use_device("gpu")
    model = ModelConfig(context=context, layers=1, heads=2, width=width)
    config = TrainConfig(batch_size=2, accumulation=2, warmup_steps=2, decay_steps=20)

    def run(name, steps, resume=False):
        return train(
            corpus,
            tmp_path / name,
            model,
            config,
            steps,
            resume=resume,
            eval_every=2,
            eval_batches=1,
            report=lambda _: None,
        )

    full = run("full", 6)
    run("split", 2)
    resumed = run("split", 6, resume=True)
    assert full["validation"]["val_loss"] == pytest.approx(
        resumed["validation"]["val_loss"], abs=1e-6
    )
    full_path, full_state = checkpoint.read(tmp_path / "full")
    split_path, split_state = checkpoint.read(tmp_path / "split")
    assert full_state["numpy_rng"] == split_state["numpy_rng"]
    for filename in ("model.safetensors", "optimizer.safetensors"):
        expected = mx.load(str(full_path / filename))
        actual = mx.load(str(split_path / filename))
        for name in expected:
            np.testing.assert_allclose(
                np.array(actual[name]), np.array(expected[name]), atol=1e-7, rtol=1e-5
            )


@pytest.mark.parametrize("length,width", [(128, 32), (256, 32), (512, 48)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("layout", ["projection", "sliced", "reversed", "broadcast"])
def test_blocked_attention_handles_input_and_gradient_layouts(length, width, compiled, layout):
    use_device("gpu")
    mx.random.seed(91)
    shape = (2, 3, length, width)
    arrays = []
    for _ in range(4):
        if layout == "projection":
            value = mx.random.normal((2, length, 3, 3 * width))
            value = value[..., width : 2 * width].transpose(0, 2, 1, 3)
        elif layout == "sliced":
            value = mx.random.normal((2, 3, length, 2 * width))[..., ::2]
        elif layout == "reversed":
            value = mx.random.normal(shape)[:, :, ::-1, ::-1]
        else:
            value = mx.broadcast_to(mx.random.normal((1, 1, length, width)), shape)
        arrays.append(value)
    q, k, v, cotangent = arrays

    def actual(q, k, v):
        return mx.vjp(training_attention, [q, k, v], [cotangent])

    # Contiguous reference operands also isolate the layout check from MLX's
    # own copies inside the native attention implementation.
    def expected(q, k, v):
        return mx.vjp(reference, [q, k, v], [mx.contiguous(cotangent)])

    if compiled:
        actual, expected = mx.compile(actual), mx.compile(expected)
    values, gradients = actual(q, k, v)
    reference_values, reference_gradients = expected(*(mx.contiguous(a) for a in (q, k, v)))
    for a, e in zip((*values, *gradients), (*reference_values, *reference_gradients), strict=True):
        np.testing.assert_allclose(np.array(a), np.array(e), atol=5e-6, rtol=5e-5)
