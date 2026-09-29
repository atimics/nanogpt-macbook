"""GELU with an explicit derivative for the compiled training step."""

import math

import mlx.core as mx
import mlx.nn as nn


@mx.custom_function
def _metal_gelu(x):
    return nn.gelu_approx(x)


@_metal_gelu.vjp
def _gelu_vjp(x, cotangent, _output):
    # Writing the derivative directly lets MLX fuse its elementwise work.
    # The cosh form keeps the small derivative in the saturated tails accurate.
    scale = math.sqrt(2 / math.pi)
    inner = scale * (x + 0.044715 * x * x * x)
    slope = 0.5 * (1 + mx.tanh(inner))
    slope += 0.5 * x * scale * (1 + 0.134145 * x * x) / mx.square(mx.cosh(inner))
    return cotangent * slope


def gelu_approx(x):
    """Choose the derivative suited to each backend's compiled execution."""
    if mx.default_device() == mx.gpu and x.dtype == mx.float32:
        return _metal_gelu(x)
    return nn.gelu_approx(x)
