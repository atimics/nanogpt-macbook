import json

import numpy as np
import pytest

from nanogpt_macbook.data import Dataset, decode, encode, prepare


def test_unicode_round_trip():
    text = "Hello, 世界 — café 🌊\n"
    assert decode(encode(text)) == text


def test_preparation_preserves_all_bytes_in_disjoint_splits(tmp_path):
    source = tmp_path / "text.txt"
    content = "abcdef 世界\n" * 100
    source.write_text(content)
    out = tmp_path / "data"
    manifest = prepare(source, out, 0.2)
    assert (out / "train.bin").read_bytes() + (out / "val.bin").read_bytes() == content.encode()
    assert manifest["train_tokens"] == int(len(content.encode()) * 0.8)
    assert manifest["val_tokens"] == len(content.encode()) - manifest["train_tokens"]
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, out)


def test_batch_targets_shift_by_one(corpus):
    dataset = Dataset(corpus, 16)
    x, y = dataset.batch("train", 3, np.random.default_rng(10))
    assert x.shape == y.shape == (3, 16)
    assert x.dtype == np.int32
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_tampered_data_is_rejected(corpus):
    file = corpus / "train.bin"
    value = bytearray(file.read_bytes())
    value[0] ^= 1
    file.write_bytes(value)
    with pytest.raises(ValueError, match="integrity"):
        Dataset(corpus, 16)


def test_small_validation_split_has_useful_error(corpus):
    size = json.loads((corpus / "manifest.json").read_text())["val_tokens"]
    with pytest.raises(ValueError, match="context"):
        Dataset(corpus, size)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_bad_split_is_rejected(tmp_path, fraction):
    with pytest.raises(ValueError, match="between"):
        prepare(tmp_path / "absent.txt", tmp_path / "data", fraction)
