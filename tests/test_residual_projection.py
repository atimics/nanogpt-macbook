from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from nanogpt_macbook.activations import gelu_approx
from nanogpt_macbook.config import ModelConfig
from nanogpt_macbook.engine import make_train_step, run_steps
from nanogpt_macbook.model import GPT, Block, loss_fn


def select(backend):
    if backend == "gpu" and not mx.metal.is_available():
        pytest.skip("This check needs Apple Metal")
    mx.set_default_device(mx.gpu if backend == "gpu" else mx.cpu)


def separate_block(self, x):
    x = x + self.attention(self.attention_norm(x))
    return x + self.down(gelu_approx(self.up(self.mlp_norm(x))))


def same_tree(before, after, atol=3e-6):
    for (name, expected), (other, actual) in zip(
        tree_flatten(before), tree_flatten(after), strict=True
    ):
        assert name == other
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), atol=atol, rtol=5e-5, err_msg=name
        )


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    "context,width,heads", [(17, 32, 4), (128, 128, 4), (256, 256, 8), (512, 384, 8)]
)
def test_model_outputs_and_gradients_match_separate_additions(
    backend, training, context, width, heads
):
    select(backend)
    config = ModelConfig(context=max(128, context), layers=2, heads=heads, width=width)
    rng = np.random.default_rng(71)
    # Strided token views cover partial sequences and non-contiguous inputs.
    inputs = mx.array(rng.integers(0, 256, (2, 2 * context), dtype=np.int32))[:, ::2]
    targets = mx.array(rng.integers(0, 256, (2, context), dtype=np.int32))
    results = []
    for implementation in (separate_block, Block.__call__):
        with patch.object(Block, "__call__", implementation):
            mx.random.seed(23)
            model = GPT(config)
            model.train(training)
            logits = model(inputs)
            loss, gradients = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
            mx.eval(logits, loss, gradients)
            results.append((logits, loss, gradients))
    same_tree(*results)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("context,width", [(17, 32), (256, 64)])
def test_queued_optimizer_updates_match_separate_additions(backend, accumulation, context, width):
    select(backend)
    config = ModelConfig(context=context, layers=2, heads=2, width=width)
    rng = np.random.default_rng(59)
    blocks = [
        rng.integers(0, 256, (accumulation, 2, context + 1), dtype=np.int32) for _ in range(6)
    ]
    records = []
    for implementation in (separate_block, Block.__call__):
        with patch.object(Block, "__call__", implementation):
            mx.random.seed(23)
            model = GPT(config)
            optimizer = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.95], bias_correction=True)
            optimizer.init(model.trainable_parameters())
            mx.eval(model.parameters(), optimizer.state)
            step = make_train_step(model, optimizer, accumulation, 0.1)
            batches = (
                (mx.array(block[..., :-1]), mx.array(block[..., 1:]), 0.0001 * (index + 1))
                for index, block in enumerate(blocks)
            )
            results = list(run_steps(step, batches, pipeline=backend == "gpu"))
            records.append((results, model.parameters(), optimizer.state))
    same_tree(*records)
