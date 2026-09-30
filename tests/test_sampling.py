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
def test_sampling_compiles_a_full_window_and_keeps_short_tails_eager(device, monkeypatch):
    select(device)
    model = GPT(ModelConfig(context=16, layers=1, heads=2, width=16))
    original = mx.compile
    compiled_shapes = []

    def checked_compile(function):
        compiled = original(function)

        def checked(inputs):
            compiled_shapes.append(inputs.shape)
            return compiled(inputs)

        return checked

    monkeypatch.setattr(mx, "compile", checked_compile)
    engine.generate(model, "Mira ", 20, temperature=0)
    assert compiled_shapes == []
    engine.generate(model, "Mira ", 40, temperature=0)
    assert compiled_shapes and set(compiled_shapes) == {(1, 16)}


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
