"""A pre-norm, decoder-only GPT with tied embeddings and causal attention."""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .activations import gelu_approx
from .attention import training_attention
from .config import ModelConfig
from .normalization import LayerNorm


def _residual_projection(residual, hidden, projection):
    """Combine a bias-free projection and residual addition on Metal."""
    if (
        mx.default_device() == mx.gpu
        and residual.dtype == hidden.dtype == projection.weight.dtype == mx.float32
        # Wider training MLPs measured faster with separate operations.
        and (hidden.shape[-1] <= 1024 or not projection.training)
    ):
        return mx.addmm(residual, hidden, projection.weight.T)
    return residual + projection(hidden)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, config.width * 3, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)

    def _project(self, x):
        batch, length, width = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        return [
            item.reshape(batch, length, self.heads, width // self.heads).transpose(0, 2, 1, 3)
            for item in (q, k, v)
        ]

    def _output(self, attended, residual):
        batch, heads, length, head_width = attended.shape
        hidden = attended.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_width)
        if residual is None:
            return self.proj(hidden)
        return _residual_projection(residual, hidden, self.proj)

    def __call__(self, x, residual=None, *, last_token_only=False):
        q, k, v = self._project(x)
        if last_token_only:
            # The final query attends to the whole prefix. Its output is the
            # only row needed for the last block's projection and MLP.
            q = q[:, :, -1:, :]
            if residual is not None:
                residual = residual[:, -1:, :]
        if self.training and not last_token_only:
            attended = training_attention(q, k, v)
        else:
            attended = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=q.shape[-1] ** -0.5,
                mask=None if last_token_only else "causal",
            )
        return self._output(attended, residual)

    def cached(self, x, residual, cache, position, context, *, last_token_only=False):
        q, k, v = self._project(x)
        if cache is None:
            padding = [(0, 0), (0, 0), (0, context - k.shape[2]), (0, 0)]
            updated = (mx.pad(k, padding), mx.pad(v, padding))
            mask = None if last_token_only else "causal"
        else:
            k, v = [
                mx.slice_update(old, new, position, axes=[2])
                for old, new in zip(cache, (k, v), strict=True)
            ]
            updated = (k, v)
            mask = mx.arange(context) <= position
        if last_token_only:
            q, residual = q[:, :, -1:, :], residual[:, -1:, :]
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=q.shape[-1] ** -0.5, mask=mask
        )
        return self._output(attended, residual), updated


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = LayerNorm(config.width)
        self.attention = Attention(config)
        self.mlp_norm = LayerNorm(config.width)
        self.up = nn.Linear(config.width, 4 * config.width, bias=False)
        self.down = nn.Linear(4 * config.width, config.width, bias=False)

    def __call__(self, x, *, last_token_only=False):
        if last_token_only:
            x = self.attention(self.attention_norm(x), x, last_token_only=True)
        else:
            x = self.attention(self.attention_norm(x), x)
        return self._mlp(x)

    def _mlp(self, x):
        hidden = gelu_approx(self.up(self.mlp_norm(x)))
        return _residual_projection(x, hidden, self.down)

    def cached(self, x, cache, position, context, *, last_token_only=False):
        x, updated = self.attention.cached(
            self.attention_norm(x), x, cache, position, context, last_token_only=last_token_only
        )
        return self._mlp(x), updated


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm = LayerNorm(config.width)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                scale = 0.02
                if name.endswith(("attention.proj", "down")):
                    scale /= math.sqrt(2 * config.layers)
                module.weight = mx.random.normal(module.weight.shape) * scale

    def __call__(self, tokens, *, last_token_only=False):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError(f"Use a batch with 1 to {self.config.context} tokens per sequence")
        x = self.tokens(tokens) + self.positions(mx.arange(tokens.shape[1]))
        for index, block in enumerate(self.blocks):
            if last_token_only and index == len(self.blocks) - 1:
                x = block(x, last_token_only=True)
            else:
                x = block(x)
        return self.tokens.as_linear(self.norm(x))

    def cached_logits(self, tokens, cache=None, position=None):
        """Fill a prefix cache, or append one token at its absolute position.

        The caller owns this temporary state and rebuilds the full window once
        positions shift. Cache arrays have a fixed context capacity so a decode
        step can share one compiled graph across positions.
        """
        positions = mx.arange(tokens.shape[1]) if cache is None else position
        x = self.tokens(tokens) + self.positions(positions)
        updated = []
        for index, block in enumerate(self.blocks):
            x, state = block.cached(
                x,
                None if cache is None else cache[index],
                position,
                self.config.context,
                last_token_only=index == len(self.blocks) - 1,
            )
            updated.append(state)
        return self.tokens.as_linear(self.norm(x)), updated

    @property
    def parameter_count(self):
        return sum(value.size for _, value in tree_flatten(self.parameters()))


def loss_fn(model: GPT, inputs, targets):
    logits = model(inputs).astype(mx.float32)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")
