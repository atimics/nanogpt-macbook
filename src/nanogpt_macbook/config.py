"""Plain, serializable settings for models and training."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    context: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    vocab_size: int = 256

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.vocab_size != 256:
            raise ValueError("The UTF-8 byte tokenizer needs a vocabulary of 256")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    accumulation: int = 1
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 20
    decay_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    def __post_init__(self):
        for name in ("batch_size", "accumulation", "decay_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.decay_steps <= self.warmup_steps:
            raise ValueError("decay_steps must exceed warmup_steps")
        for name in ("learning_rate", "min_learning_rate", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be at most learning_rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")

    def rate(self, completed_steps: int) -> float:
        if completed_steps < self.warmup_steps:
            return self.learning_rate * (completed_steps + 1) / self.warmup_steps
        progress = min(
            1.0,
            (completed_steps - self.warmup_steps) / (self.decay_steps - self.warmup_steps),
        )
        return self.min_learning_rate + 0.5 * (self.learning_rate - self.min_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )


PRESETS = {
    "tiny": (ModelConfig(), TrainConfig()),
    "small": (ModelConfig(context=256, layers=6, heads=8, width=256), TrainConfig()),
    "medium": (
        ModelConfig(context=512, layers=8, heads=8, width=384),
        TrainConfig(batch_size=4),
    ),
}
