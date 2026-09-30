"""LayerNorm with saved row statistics and grouped gradients on Metal."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=1)
def _forward_kernel():
    return mx.fast.metal_kernel(
        name="nanogpt_layer_norm_forward",
        input_names=["x", "weight", "bias", "epsilon"],
        output_names=["out", "stats"],
        source="""
            uint row = thread_position_in_grid.x / 32;
            uint lane = thread_index_in_simdgroup;
            if (row >= N) return;
            // Shift before summing to preserve small differences in nearly
            // constant rows. Save the origin and shifted mean separately.
            float origin = x[row * D];
            float values[C], mean = 0;
            for (uint i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                values[i] = col < D ? x[row * D + col] - origin : 0;
                mean += values[i];
            }
            mean = simd_sum(mean) / D;
            float variance = 0;
            for (uint i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                values[i] = col < D ? values[i] - mean : 0;
                variance += values[i] * values[i];
            }
            float inv = metal::precise::rsqrt(simd_sum(variance) / D + epsilon);
            if (lane == 0) {
                stats[row * 3] = origin;
                stats[row * 3 + 1] = mean;
                stats[row * 3 + 2] = inv;
            }
            for (uint i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                if (col < D) {
                    out[row * D + col] = values[i] * inv * weight[col] + bias[col];
                }
            }
        """,
    )


@lru_cache(maxsize=1)
def _backward_kernel():
    return mx.fast.metal_kernel(
        name="nanogpt_layer_norm_saved_vjp",
        input_names=["x", "weight", "grad", "stats"],
        output_names=["dx", "dw", "db"],
        source="""
            uint group = threadgroup_position_in_grid.x;
            uint simd = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint row = group * R + simd;
            float origin = row < N ? stats[row * 3] : 0;
            float mean = row < N ? stats[row * 3 + 1] : 0;
            float inv = row < N ? stats[row * 3 + 2] : 0;
            float values[C], gradients[C], weights[C];
            float sum_g = 0, sum_gx = 0;
            for (uint i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                values[i] = (row < N && col < D) ? (x[row * D + col] - origin) - mean : 0;
                gradients[i] = (row < N && col < D) ? grad[row * D + col] : 0;
                weights[i] = col < D ? weight[col] : 0;
                sum_g += gradients[i] * weights[i];
                sum_gx += gradients[i] * weights[i] * values[i];
            }
            sum_g = simd_sum(sum_g) / D;
            sum_gx = simd_sum(sum_gx) / D * inv * inv;
            threadgroup float partial_w[R * D];
            threadgroup float partial_b[R * D];
            for (uint i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                if (col < D) {
                    if (row < N) {
                        dx[row * D + col] = inv * (
                            gradients[i] * weights[i] - sum_g - values[i] * sum_gx);
                    }
                    partial_w[simd * D + col] = gradients[i] * values[i] * inv;
                    partial_b[simd * D + col] = gradients[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint col = thread_position_in_threadgroup.x; col < D; col += R * 32) {
                float w = 0, b = 0;
                for (uint r = 0; r < R; ++r) {
                    w += partial_w[r * D + col];
                    b += partial_b[r * D + col];
                }
                dw[group * D + col] = w;
                db[group * D + col] = b;
            }
        """,
    )


@lru_cache(maxsize=32)
def _normalize(eps):
    forward = _forward_kernel()
    backward = _backward_kernel()

    @mx.custom_function
    def apply(x, weight, bias):
        width = x.shape[-1]
        count = x.size // width
        return tuple(
            forward(
                inputs=[x, weight, bias, mx.array(eps)],
                template=[("N", count), ("D", width), ("C", (width + 31) // 32)],
                output_shapes=[x.shape, (count, 3)],
                output_dtypes=[mx.float32] * 2,
                grid=(((count + 3) // 4) * 128, 1, 1),
                threadgroup=(128, 1, 1),
            )
        )

    @apply.vjp
    def vjp(primals, cotangent, output):
        x, weight, _bias = primals
        width = x.shape[-1]
        count = x.size // width
        # Keep the two shared partial buffers within 32 KiB. More rows per
        # group reduce the weight and bias gradients written to device memory.
        rows = 16 if width <= 256 else 8
        groups = (count + rows - 1) // rows
        dx, dw, db = backward(
            inputs=[x, weight, cotangent[0], output[1]],
            template=[("N", count), ("D", width), ("C", (width + 31) // 32), ("R", rows)],
            output_shapes=[x.shape, (groups, width), (groups, width)],
            output_dtypes=[mx.float32] * 3,
            grid=(groups * rows * 32, 1, 1),
            threadgroup=(rows * 32, 1, 1),
        )
        return dx, mx.sum(dw, axis=0), mx.sum(db, axis=0)

    # Row statistics are private outputs used only by the backward pass.
    return lambda x, weight, bias: apply(x, weight, bias)[0]


class LayerNorm(nn.LayerNorm):
    """Normalize float32 rows on Metal while keeping MLX's checkpoint layout."""

    def __call__(self, x):
        weight, bias = self.get("weight"), self.get("bias")
        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and x.ndim > 0
            and x.size > 0
            and 1 <= x.shape[-1] <= 512
            and weight is not None
            and bias is not None
            and x.dtype == weight.dtype == bias.dtype == mx.float32
        ):
            return _normalize(self.eps)(x, weight, bias)
        return super().__call__(x)
