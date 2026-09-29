import mlx.core as mx
import numpy as np
import pytest

from nanogpt_macbook import checkpoint
from nanogpt_macbook.attention import training_attention
from nanogpt_macbook.config import ModelConfig, TrainConfig
from nanogpt_macbook.engine import train


def use_device(device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)


def reference(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "length,width", [(1, 8), (5, 8), (33, 16), (128, 32), (256, 32), (512, 48), (513, 16)]
)
def test_attention_values_and_gradients_match_mlx(device, length, width):
    use_device(device)
    # Slices also cover strided input arrays and rows beyond a 32-lane boundary.
    packed = mx.random.normal((1, 2, length + 1, 3 * width))
    inputs = [part[:, :, :length] for part in mx.split(packed, 3, axis=-1)]
    weights = mx.random.normal((1, 2, length, width))
    actual = training_attention(*inputs)
    expected = reference(*inputs)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5)

    actual_grads = mx.grad(
        lambda q, k, v: mx.sum(training_attention(q, k, v) * weights), argnums=(0, 1, 2)
    )(*inputs)
    expected_grads = mx.grad(
        lambda q, k, v: mx.sum(reference(q, k, v) * weights), argnums=(0, 1, 2)
    )(*inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(
            np.array(actual_grad), np.array(expected_grad), atol=5e-6, rtol=5e-5
        )


def test_metal_attention_respects_the_causal_boundary():
    use_device("gpu")
    q, k, v = [mx.random.normal((1, 2, 33, 16)) for _ in range(3)]
    changed_k = mx.concatenate([k[:, :, :17], k[:, :, 17:] * 20], axis=2)
    changed_v = mx.concatenate([v[:, :, :17], v[:, :, 17:] + 20], axis=2)
    np.testing.assert_allclose(
        np.array(training_attention(q, k, v)[:, :, :17]),
        np.array(training_attention(q, changed_k, changed_v)[:, :, :17]),
        atol=1e-7,
    )
    grads = mx.grad(
        lambda keys, values: training_attention(q, keys, values)[:, :, :17].sum(),
        argnums=(0, 1),
    )(k, v)
    for grad in grads:
        np.testing.assert_array_equal(np.array(grad[:, :, 17:]), 0)


def test_metal_softmax_is_stable_for_large_scores():
    use_device("gpu")
    q = mx.full((1, 1, 33, 16), 100.0)
    k = mx.full((1, 1, 33, 16), 100.0)
    v = mx.random.normal((1, 1, 33, 16))
    expected = mx.cumsum(v, axis=2) / mx.arange(1, 34)[None, None, :, None]
    np.testing.assert_allclose(np.array(training_attention(q, k, v)), np.array(expected), atol=1e-6)


def test_metal_training_resume_with_accumulation(corpus, tmp_path):
    use_device("gpu")
    model = ModelConfig(context=16, layers=1, heads=2, width=16)
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
