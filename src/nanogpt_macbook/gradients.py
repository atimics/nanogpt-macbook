"""Global gradient clipping with grouped float32 reductions on Metal."""

from functools import lru_cache
from math import prod

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map


@lru_cache(maxsize=128)
def _reduction(shapes, chunk):
    names = [f"g{i}" for i in range(len(shapes))]
    lines = [
        "uint block = threadgroup_position_in_grid.x;",
        "uint lane = thread_position_in_threadgroup.x;",
        "float value = 0;",
    ]
    start = 0
    for i, shape in enumerate(shapes):
        size = prod(shape)
        blocks = (size + chunk - 1) // chunk
        dense = " && ".join(
            f"g{i}_strides[{axis}] == {prod(shape[axis + 1 :])}" for axis in range(len(shape))
        )
        if len(shape) == 2:
            dense = f"({dense}) || (g{i}_strides[0] == 1 && g{i}_strides[1] == {shape[0]})"
        location = " + ".join(
            f"((index / {prod(shape[axis + 1 :])}) % {size}) * g{i}_strides[{axis}]"
            for axis, size in enumerate(shape)
        )
        lines.append(
            f"""
            {"if" if i == 0 else "else if"} (block < {start + blocks}) {{
                uint offset = (block - {start}) * {chunk} + lane;
                bool dense = {dense};
                for (uint j = 0; j < {chunk // 256}; ++j) {{
                    uint index = offset + j * 256;
                    if (index < {size}) {{
                        size_t loc = dense ? index : {location};
                        float g = g{i}[loc];
                        value += g * g;
                    }}
                }}
            }}
            """
        )
        start += blocks
    lines.append(
        """
        value = simd_sum(value);
        threadgroup float partial[8];
        if (thread_index_in_simdgroup == 0) {
            partial[simdgroup_index_in_threadgroup] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup_index_in_threadgroup == 0) {
            value = thread_index_in_simdgroup < 8 ? partial[thread_index_in_simdgroup] : 0;
            value = simd_sum(value);
            if (thread_index_in_simdgroup == 0) out[block] = value;
        }
        """
    )
    kernel = mx.fast.metal_kernel(
        name="nanogpt_gradient_norm",
        input_names=names,
        output_names=["out"],
        source="\n".join(lines),
        ensure_row_contiguous=False,
    )
    return kernel, start


def clip_grad_norm(grads, max_norm):
    """Clip by the global L2 norm while keeping MLX's scale and epsilon."""
    leaves = [g for _, g in tree_flatten(grads)]
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or max_norm < 0
        or any(g.dtype != mx.float32 or g.size >= 2**32 for g in leaves)
    ):
        return optim.clip_grad_norm(grads, max_norm)
    leaves = [g.reshape((1,)) if g.ndim == 0 else g for g in leaves if g.size]
    if not leaves:
        return optim.clip_grad_norm(grads, max_norm)

    # Small models need more blocks to fill the GPU. Larger models can reduce
    # more values per block and put their largest arrays first.
    chunk = 1024
    if sum(g.size for g in leaves) >= 2**22:
        chunk = 4096
        leaves.sort(key=lambda g: g.size, reverse=True)
    partials = []
    # Each input uses a data buffer and a stride buffer. Fourteen inputs plus
    # the output fit within Metal's 31 buffer slots.
    for start in range(0, len(leaves), 14):
        group = leaves[start : start + 14]
        kernel, blocks = _reduction(tuple(g.shape for g in group), chunk)
        partials.append(
            kernel(
                inputs=group,
                output_shapes=[(blocks,)],
                output_dtypes=[mx.float32],
                grid=(blocks * 256, 1, 1),
                threadgroup=(256, 1, 1),
            )[0]
        )
    norm = mx.sqrt(mx.sum(mx.concatenate(partials)))
    scale = mx.minimum(max_norm / (norm + 1e-6), 1.0)
    return tree_map(lambda g: g * scale, grads), norm
