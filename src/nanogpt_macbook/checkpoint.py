"""Immutable checkpoints with atomic pointers and a single writer per run."""

import fcntl
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def write_json(path: Path, value):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def run_lock(run: Path):
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"A training process already owns {run}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def save(run: Path, model, optimizer, state: dict, rng, is_best: bool):
    root = run / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step-{state['step']:08d}-{uuid.uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=root))
    try:
        model.save_weights(str(staging / "model.safetensors"))
        mx.save_safetensors(
            str(staging / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
        )
        mx.save_safetensors(
            str(staging / "random.safetensors"),
            {str(index): value for index, value in enumerate(mx.random.state)},
        )
        write_json(staging / "state.json", {**state, "numpy_rng": rng.bit_generator.state})
        # Finish all checkpoint files before publishing either pointer.
        for file in staging.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(staging, root / name)
        pointer = {"format": 1, "checkpoint": name}
        write_json(run / "latest.json", pointer)
        if is_best:
            write_json(run / "best.json", pointer)
        # Keep latest and best. Inference uses the run lock while loading weights.
        keep = {name}
        if (run / "best.json").exists():
            keep.add(json.loads((run / "best.json").read_text())["checkpoint"])
        for old in root.glob("step-*"):
            if old.is_dir() and old.name not in keep:
                shutil.rmtree(old)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read(run: Path, which: str = "latest"):
    if which not in ("latest", "best"):
        raise ValueError("Choose latest or best")
    pointer = json.loads((run / f"{which}.json").read_text())
    name = pointer["checkpoint"]
    if pointer.get("format") != 1 or Path(name).name != name or not name.startswith("step-"):
        raise ValueError("Checkpoint pointer is invalid")
    path = run / "checkpoints" / name
    state = json.loads((path / "state.json").read_text())
    if state.get("format") != 1:
        raise ValueError("Use a checkpoint saved with this version of nanogpt")
    return path, state


def restore(path: Path, model, optimizer, rng, state):
    model.load_weights(str(path / "model.safetensors"))
    optimizer.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    rng.bit_generator.state = state["numpy_rng"]
    saved_random = mx.load(str(path / "random.safetensors"))
    # MLX 0.32 exposes a read-only state sentinel. Its key is the two uint32
    # words of the public uint64 seed, so seed() restores it exactly.
    high, low = saved_random["0"].tolist()
    mx.random.seed((int(high) << 32) | int(low))
    mx.eval(model.parameters(), optimizer.state, mx.random.state)
