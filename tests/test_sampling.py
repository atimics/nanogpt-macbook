import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook import engine
from nanogpt_macbook.config import ModelConfig
from nanogpt_macbook.data import decode, encode
from nanogpt_macbook.model import GPT


def select(device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("layers,length", [(1, 1), (2, 17), (2, 128), (2, 256), (2, 512)])
def test_last_token_logits_and_gradients_match_full_output(device, layers, length):
    select(device)
    model = GPT(ModelConfig(context=max(16, length), layers=layers, width=32, heads=2))
    model.eval()
    rng = np.random.default_rng(91)
    inputs = mx.array(rng.integers(0, 256, (2, 2 * length), dtype=np.int32))[:, ::2]
    expected = model(inputs)[:, -1:]
    actual = mx.compile(lambda ids: model(ids, last_token_only=True))(inputs)
    assert actual.shape == (2, 1, 256)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5)

    weight = mx.array(rng.normal(size=(2, 1, 256)).astype(np.float32))
    _, expected_grads = nn.value_and_grad(model, lambda net: (net(inputs)[:, -1:] * weight).sum())(
        model
    )
    _, actual_grads = nn.value_and_grad(
        model, lambda net: (net(inputs, last_token_only=True) * weight).sum()
    )(model)
    for (name, before), (other, after) in zip(
        tree_flatten(expected_grads), tree_flatten(actual_grads), strict=True
    ):
        assert name == other
        np.testing.assert_allclose(
            np.array(after), np.array(before), atol=5e-6, rtol=5e-5, err_msg=name
        )


def full_output_tokens(model, prompt, count, temperature, top_k, seed):
    mx.random.seed(seed)
    model.eval()
    sequence = encode(prompt) or [10]
    generated = []
    for _ in range(count):
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
    return generated


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("prompt", ["", "Mira ", "世界 🌊 " * 3])
@pytest.mark.parametrize("temperature,top_k", [(0, 40), (0.8, 0), (0.8, 7), (0.8, 256)])
def test_sampling_bytes_match_full_output_across_growing_and_sliding_windows(
    device, prompt, temperature, top_k, monkeypatch
):
    select(device)
    model = GPT(ModelConfig(context=16, layers=2, heads=2, width=16))
    expected = full_output_tokens(model, prompt, 40, temperature, top_k, 9)
    captured = []

    def capture(tokens):
        captured.extend(tokens)
        return decode(tokens)

    monkeypatch.setattr(engine, "decode", capture)
    actual = engine.generate(model, prompt, 40, temperature, top_k, seed=9)
    assert captured == expected
    assert actual == prompt + decode(expected)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_sampling_compiles_full_gpu_windows_and_keeps_short_tails_eager(device, monkeypatch):
    select(device)
    model = GPT(ModelConfig(context=16, layers=1, heads=2, width=16))
    original = mx.compile
    compiled_shapes = []

    def checked_compile(function, **kwargs):
        compiled = original(function, **kwargs)

        def checked(inputs):
            compiled_shapes.append(inputs.shape)
            return compiled(inputs)

        return checked

    monkeypatch.setattr(mx, "compile", checked_compile)
    engine.generate(model, "Mira ", 20, temperature=0)
    assert compiled_shapes == []
    engine.generate(model, "Mira ", 40, temperature=0)
    if device == "gpu":
        assert compiled_shapes and set(compiled_shapes) == {(1, 16)}
    else:
        assert compiled_shapes == []


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_each_sampling_call_uses_the_current_weights(device, monkeypatch):
    select(device)
    model = GPT(ModelConfig(context=16, layers=1, heads=2, width=16))
    prompt = "A long prompt for a short context."
    engine.generate(model, prompt, 20, seed=8)
    mx.random.seed(55)
    model.tokens.weight = mx.random.normal(model.tokens.weight.shape) * 0.2
    expected = full_output_tokens(model, prompt, 20, 0.8, 40, 8)
    captured = []

    def capture(tokens):
        captured.extend(tokens)
        return decode(tokens)

    monkeypatch.setattr(engine, "decode", capture)
    engine.generate(model, prompt, 20, seed=8)
    assert captured == expected


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("layers,prefix", [(1, 1), (2, 7), (2, 16)])
def test_prefix_cache_logits_match_rebuilt_context_and_ignore_unused_slots(device, layers, prefix):
    select(device)
    model = GPT(ModelConfig(context=32, layers=layers, heads=2, width=32))
    model.eval()
    parameter_names = [name for name, _ in tree_flatten(model.parameters())]
    rng = np.random.default_rng(56)
    inputs = mx.array(rng.integers(0, 256, (2, 64), dtype=np.int32))[:, ::2]
    logits, cache = model.cached_logits(inputs[:, :prefix])
    expected = model(inputs[:, :prefix], last_token_only=True)
    np.testing.assert_allclose(np.array(logits), np.array(expected), atol=3e-6, rtol=3e-5)
    assert logits.shape == (2, 1, 256)
    # Future slots must remain masked even when their values are large.
    cache = [
        tuple(
            mx.concatenate((a[:, :, :prefix], mx.full_like(a[:, :, prefix:], 1000)), axis=2)
            for a in state
        )
        for state in cache
    ]
    step = mx.compile(model.cached_logits)
    for position in range(prefix, model.config.context):
        logits, cache = step(inputs[:, position : position + 1], cache, mx.array([position]))
        expected = model(inputs[:, : position + 1], last_token_only=True)
        np.testing.assert_allclose(np.array(logits), np.array(expected), atol=3e-6, rtol=3e-5)
        assert all(a.shape == (2, 2, 32, 16) for state in cache for a in state)
    assert [name for name, _ in tree_flatten(model.parameters())] == parameter_names


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("temperature,top_k", [(0, 40), (0.8, 0), (0.8, 7), (0.8, 256)])
def test_cached_sampling_preserves_bytes_and_random_state_across_full_context(
    device, temperature, top_k, monkeypatch
):
    select(device)
    model = GPT(ModelConfig(context=64, layers=2, heads=2, width=32))
    expected = full_output_tokens(model, "Mira ", 100, temperature, top_k, 56)
    expected_random = np.array(mx.random.uniform(shape=(10,)))
    captured = []
    monkeypatch.setattr(engine, "decode", lambda tokens: captured.extend(tokens) or decode(tokens))
    engine.generate(model, "Mira ", 100, temperature, top_k, 56)
    assert captured == expected
    np.testing.assert_array_equal(np.array(mx.random.uniform(shape=(10,))), expected_random)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("count", [1, 2, 4, 8, 15, 16, 17, 32])
def test_cached_sampling_short_outputs_and_repeated_calls(device, count, monkeypatch):
    select(device)
    model = GPT(ModelConfig(context=64, layers=1, heads=2, width=32))
    engine.generate(model, "Mira ", count, seed=55)
    mx.random.seed(100)
    model.tokens.weight = mx.random.normal(model.tokens.weight.shape) * 0.2
    expected = full_output_tokens(model, "Mira ", count, 0.8, 40, 55)
    captured = []
    monkeypatch.setattr(engine, "decode", lambda tokens: captured.extend(tokens) or decode(tokens))
    engine.generate(model, "Mira ", count, seed=55)
    assert captured == expected


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("context", [1, 16, 64])
@pytest.mark.parametrize("count", [15, 16, 17, 65])
def test_full_windows_preserve_bytes_and_random_state_at_queue_boundary(
    device, context, count, monkeypatch
):
    select(device)
    model = GPT(ModelConfig(context=context, layers=1, heads=2, width=32))
    prompt = "世界 🌊 " * 12
    expected = full_output_tokens(model, prompt, count, 0.8, 40, 101)
    expected_random = np.array(mx.random.uniform(shape=(10,)))
    captured = []
    monkeypatch.setattr(engine, "decode", lambda tokens: captured.extend(tokens) or decode(tokens))
    actual = engine.generate(model, prompt, count, 0.8, 40, 101)
    assert captured == expected
    assert actual == prompt + decode(expected)
    np.testing.assert_array_equal(np.array(mx.random.uniform(shape=(10,))), expected_random)
