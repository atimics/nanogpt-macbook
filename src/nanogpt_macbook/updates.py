"""Batch small AdamW updates while preserving MLX's parameter state tree."""

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map


def update_parameters(model, optimizer, gradients):
    """Pack small float32 arrays when enough updates share the Metal launch."""
    if mx.default_device() != mx.gpu or type(optimizer) is not optim.AdamW:
        optimizer.update(model, gradients)
        return
    if not optimizer._initialized:
        optimizer.init(gradients)

    rows = []
    tree_map(lambda g, p, s: rows.append((g, p, s)), gradients, model, optimizer.state)
    small = [
        i
        for i, (g, p, s) in enumerate(rows)
        if 0 < p.size <= 1024 and g.dtype == p.dtype == s["m"].dtype == s["v"].dtype == mx.float32
    ]
    # Larger groups amortize the four packing operations. Medium has 34
    # eligible arrays; tiny and small retain their per-parameter updates.
    if len(small) < 32:
        optimizer.update(model, gradients)
        return

    # Match Optimizer.apply_gradients: schedules use the previous step, and
    # Adam's bias correction uses the incremented step.
    for name, schedule in optimizer._schedulers.items():
        optimizer.state[name] = schedule(optimizer.step)
    optimizer.state["step"] = optimizer.step + 1

    packed_gradients = mx.concatenate([rows[i][0].reshape(-1) for i in small])
    packed_parameters = mx.concatenate([rows[i][1].reshape(-1) for i in small])
    packed_state = {
        key: mx.concatenate([rows[i][2][key].reshape(-1) for i in small]) for key in ("m", "v")
    }
    packed = optimizer.apply_single(packed_gradients, packed_parameters, packed_state)
    results = [None] * len(rows)
    start = 0
    for i in small:
        _, parameter, state = rows[i]
        end = start + parameter.size
        results[i] = packed[start:end].reshape(parameter.shape)
        for key in ("m", "v"):
            state[key] = packed_state[key][start:end].reshape(parameter.shape)
        start = end
    for i, (gradient, parameter, state) in enumerate(rows):
        if results[i] is None:
            results[i] = optimizer.apply_single(gradient, parameter, state)
    result = iter(results)
    model.update(tree_map(lambda _: next(result), gradients))
