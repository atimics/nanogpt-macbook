"""Byte tokens, disjoint data splits, and memory-mapped batches."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def prepare(source: Path, output: Path, validation_fraction: float = 0.1) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if output.exists():
        raise ValueError(f"Choose a new data folder; {output} already exists")
    # Validate UTF-8 incrementally so a large corpus stays off the Python heap.
    with source.open(encoding="utf-8") as stream:
        while stream.read(1024 * 1024):
            pass
    size = source.stat().st_size
    split = int(size * (1 - validation_fraction))
    if min(split, size - split) < 2:
        raise ValueError("Provide enough text for at least two bytes in each split")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        source_hash = hashlib.sha256()
        with source.open("rb") as stream:
            for name, count in (("train", split), ("val", size - split)):
                with (staging / f"{name}.bin").open("wb") as target:
                    remaining = count
                    while remaining:
                        chunk = stream.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError("The source changed during preparation; try again")
                        target.write(chunk)
                        source_hash.update(chunk)
                        remaining -= len(chunk)
            if stream.read(1):
                raise ValueError("The source changed during preparation; try again")
        manifest = {
            "format": 1,
            "tokenizer": "utf8-bytes-v1",
            "source_name": source.name,
            "source_sha256": source_hash.hexdigest(),
            "vocab_size": 256,
            "train_tokens": split,
            "val_tokens": size - split,
            "train_sha256": digest(staging / "train.bin"),
            "val_sha256": digest(staging / "val.bin"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class Dataset:
    def __init__(self, path: Path, context: int):
        self.path = path.resolve()
        self.manifest = json.loads((path / "manifest.json").read_text())
        if self.manifest.get("format") != 1 or self.manifest.get("tokenizer") != "utf8-bytes-v1":
            raise ValueError("Prepare the text with this version of nanogpt")
        self.context = context
        self.splits = {}
        for name in ("train", "val"):
            file = path / f"{name}.bin"
            size = file.stat().st_size
            if (
                size != self.manifest[f"{name}_tokens"]
                or digest(file) != self.manifest[f"{name}_sha256"]
            ):
                raise ValueError(f"{name} data failed its integrity check; prepare it again")
            if size <= context:
                raise ValueError(
                    f"{name} has {size} bytes; add more text or use --context below {size}"
                )
            self.splits[name] = np.memmap(file, mode="r", dtype=np.uint8)

    @property
    def identity(self) -> dict:
        return {
            key: self.manifest[key]
            for key in ("tokenizer", "source_sha256", "train_sha256", "val_sha256")
        }

    def batch(self, split: str, size: int, rng: np.random.Generator):
        tokens = self.splits[split]
        starts = rng.integers(0, len(tokens) - self.context, size=size)
        indices = starts[:, None] + np.arange(self.context + 1)
        block = np.asarray(tokens[indices], dtype=np.int32)
        return block[:, :-1].copy(), block[:, 1:].copy()
