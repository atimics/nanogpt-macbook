"""A pre-norm, decoder-only GPT with tied embeddings and causal attention."""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, config.width * 3, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)

    def __call__(self, x):
        batch, length, width = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = [
            item.reshape(batch, length, self.heads, width // self.heads).transpose(0, 2, 1, 3)
            for item in (q, k, v)
        ]
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(width // self.heads) ** -0.5, mask="causal"
        )
        return self.proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.up = nn.Linear(config.width, 4 * config.width, bias=False)
        self.down = nn.Linear(4 * config.width, config.width, bias=False)

    def __call__(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.down(nn.gelu_approx(self.up(self.mlp_norm(x))))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm = nn.LayerNorm(config.width)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                scale = 0.02
                if name.endswith(("attention.proj", "down")):
                    scale /= math.sqrt(2 * config.layers)
                module.weight = mx.random.normal(module.weight.shape) * scale

    def __call__(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError(f"Use a batch with 1 to {self.config.context} tokens per sequence")
        x = self.tokens(tokens) + self.positions(mx.arange(tokens.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.tokens.as_linear(self.norm(x))

    @property
    def parameter_count(self):
        return sum(value.size for _, value in tree_flatten(self.parameters()))


def loss_fn(model: GPT, inputs, targets):
    logits = model(inputs).astype(mx.float32)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")
