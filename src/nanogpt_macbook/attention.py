"""Causal training attention with fused float32 kernels on Metal."""

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


@lru_cache(maxsize=1)
def _score_gradient_kernel():
    # Four SIMD groups form an 8-query by 32-key tile. Matrix instructions
    # calculate grad_output @ value.T, then each lane writes adjacent scores.
    return mx.fast.metal_kernel(
        name="nanogpt_attention_score_gradient",
        input_names=["probs", "grad", "value", "delta"],
        output_names=["scores_grad"],
        header="#include <metal_simdgroup_matrix>\n",
        source="""
            uint simd = simdgroup_index_in_threadgroup;
            uint query = threadgroup_position_in_grid.y * 8;
            uint key = threadgroup_position_in_grid.x * 32 + simd * 8;
            uint head = threadgroup_position_in_grid.z;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
            if (key <= query + 7) {
                for (uint d = 0; d < D; d += 8) {
                    simdgroup_float8x8 a, b;
                    simdgroup_load(a, grad + (head * N + query) * D + d, D);
                    simdgroup_load(b, value + (head * N + key) * D + d,
                                   D, ulong2(0), true);
                    simdgroup_multiply_accumulate(acc, a, b, acc);
                }
            }
            threadgroup float tile[4 * 64];
            simdgroup_store(acc, tile + simd * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = thread_position_in_threadgroup.x; i < 256; i += 128) {
                uint q = query + i / 32;
                uint k = threadgroup_position_in_grid.x * 32 + i % 32;
                uint index = (head * N + q) * N + k;
                float dot = tile[(i % 32 / 8) * 64 + (i / 32) * 8 + i % 8];
                scores_grad[index] = k <= q
                    ? probs[index] * (dot - delta[head * N + q]) * rsqrt(float(D))
                    : 0.f;
            }
        """,
    )


def _block_mask(length, block_size):
    blocks = mx.arange((length + block_size - 1) // block_size)
    return blocks[:, None] >= blocks[None, :]


@lru_cache(maxsize=2)
def _blocked_attention(block_size):
    kernel = _score_gradient_kernel()

    @mx.custom_function
    def apply(q, k, v):
        mask = _block_mask(q.shape[-2], block_size)
        scores = mx.block_masked_mm(q, k.swapaxes(-1, -2), block_size=block_size, mask_out=mask)
        probs = _softmax(q.shape[-1])(scores)
        output = mx.block_masked_mm(probs, v, block_size=block_size, mask_lhs=mask)
        # Keep probabilities as a private saved value for the gradient.
        return output, probs

    @apply.vjp
    def vjp(primals, cotangents, outputs):
        q, k, v = primals
        grad = mx.contiguous(cotangents[0])
        output, probs = outputs
        length, width = q.shape[-2:]
        # sum(grad_output * output) equals sum(grad_probs * probs) per row.
        # This reduction uses the smaller head dimension.
        delta = mx.sum(grad * output, axis=-1)
        scores_grad = kernel(
            inputs=[probs, grad, v, delta],
            template=[("N", length), ("D", width)],
            output_shapes=[probs.shape],
            output_dtypes=[mx.float32],
            grid=(length // 32 * 128, length // 8, q.size // (length * width)),
            threadgroup=(128, 1, 1),
        )[0]
        mask = _block_mask(length, block_size)

        def product(left, right, active):
            return mx.block_masked_mm(left, right, block_size=block_size, mask_lhs=active)

        return (
            product(scores_grad, k, mask),
            product(scores_grad.swapaxes(-1, -2), q, mask.T),
            product(probs.swapaxes(-1, -2), grad, mask.T),
        )

    return lambda q, k, v: apply(q, k, v)[0]


def training_attention(q, k, v):
    """Use fused softmax for the short, square float32 attention in our presets."""
    if (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and q.dtype == k.dtype == v.dtype == mx.float32
        and 1 < q.shape[-2] == k.shape[-2] == v.shape[-2] <= 512
    ):
        length, width = q.shape[-2:]
        # The matrix tiles cover complete rows and groups of eight head values.
        # Longer preset contexts benefit from the masked matrix products.
        if (
            q.ndim == 4
            and q.shape == k.shape == v.shape
            and length >= 256
            and length % 32 == 0
            and 32 <= width <= 64
            and width % 8 == 0
        ):
            # MLX 0.32.3 masked products can retain the original batch strides
            # after an internal copy. Pack here so every product uses the same
            # layout, including sliced, reversed, and broadcast inputs.
            q, k, v = (mx.contiguous(item) for item in (q, k, v))
            return _blocked_attention(32 if length == 256 else 64)(q, k, v)
        probabilities = _softmax(q.shape[-1])(q @ k.swapaxes(-1, -2))
        return probabilities @ v
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")
