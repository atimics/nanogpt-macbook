"""Causal training attention with fused float32 softmax on Metal."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernels():
    # Each group of 32 GPU lanes handles one row. Keep the row in registers
    # while applying the scale, causal mask, and stable softmax.
    forward = mx.fast.metal_kernel(
        name="nanogpt_causal_softmax",
        input_names=["scores"],
        output_names=["out"],
        source="""
            uint row = thread_position_in_grid.x / 32;
            uint lane = thread_index_in_simdgroup;
            uint query = row % N;
            float values[C];
            float peak = -INFINITY;
            for (int i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                values[i] = (col < N && col <= query)
                    ? scores[row * N + col] * rsqrt(float(D)) : -INFINITY;
                peak = max(peak, values[i]);
            }
            peak = simd_max(peak);
            float total = 0;
            for (int i = 0; i < C; ++i) {
                values[i] = exp(values[i] - peak);
                total += values[i];
            }
            total = simd_sum(total);
            for (int i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                if (col < N) out[row * N + col] = values[i] / total;
            }
        """,
    )
    backward = mx.fast.metal_kernel(
        name="nanogpt_causal_softmax_vjp",
        input_names=["probs", "cotangent"],
        output_names=["out"],
        source="""
            uint row = thread_position_in_grid.x / 32;
            uint lane = thread_index_in_simdgroup;
            float p[C], g[C];
            float dot = 0;
            for (int i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                p[i] = col < N ? probs[row * N + col] : 0;
                g[i] = col < N ? cotangent[row * N + col] : 0;
                dot += p[i] * g[i];
            }
            dot = simd_sum(dot);
            for (int i = 0; i < C; ++i) {
                uint col = lane + i * 32;
                if (col < N) {
                    out[row * N + col] = p[i] * (g[i] - dot) * rsqrt(float(D));
                }
            }
        """,
    )
    return forward, backward


def _run(kernel, arrays, head_dim):
    shape = arrays[0].shape
    length = shape[-1]
    threads = arrays[0].size // length * 32
    return kernel(
        inputs=arrays,
        template=[("N", length), ("C", (length + 31) // 32), ("D", head_dim)],
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        grid=(threads, 1, 1),
        threadgroup=(min(128, threads), 1, 1),
    )[0]


@lru_cache(maxsize=32)
def _softmax(head_dim):
    forward, backward = _kernels()

    @mx.custom_function
    def apply(scores):
        return _run(forward, [scores], head_dim)

    @apply.vjp
    def vjp(_primals, cotangent, output):
        return (_run(backward, [output, cotangent], head_dim),)

    return apply


def training_attention(q, k, v):
    """Use fused softmax for the short, square float32 attention in our presets."""
    if (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and q.dtype == k.dtype == v.dtype == mx.float32
        and 1 < q.shape[-2] == k.shape[-2] == v.shape[-2] <= 512
    ):
        probabilities = _softmax(q.shape[-1])(q @ k.swapaxes(-1, -2))
        return probabilities @ v
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")
