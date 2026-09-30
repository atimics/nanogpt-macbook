"""Check CI pairing, provenance, and preservation of partial results."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("ci_benchmark", ROOT / "scripts/ci_benchmark.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
BASE = "a" * 40
HEAD = "b" * 40


def receipt(commit=BASE):
    return {
        "source": {"commit": commit, "source_dirty": False},
        "environment": {"device": "gpu"},
        "method": {
            "dtype": "float32",
            "execution": "pipelined",
            "queue_depth": 2,
            "seed": 1337,
            "memory_limit_gib": 2,
            "cache_limit_gib": 2,
            "name": "training-loop-v2",
            "steps": 100,
            "warmup_steps": 20,
            "repeats": 1,
        },
        "results": [
            {
                "preset": "tiny",
                "model_config": {"context": 128},
                "train_config": {"batch_size": 8},
                "trials": [{"step_seconds": [0.1] * 100, "bytes_per_second": 10240}],
            }
        ],
    }


def test_pair_summary_uses_paired_changes():
    summary = ci.paired_summary([{"a": 100, "b": 110}, {"a": 200, "b": 180}])
    assert summary["median_change_percent"] == pytest.approx(0)
    assert summary["median_absolute_change_percent"] == pytest.approx(10)
    assert summary["b_faster_pairs"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["source"].update(commit=HEAD),
        lambda r: r["source"].update(source_dirty=True),
        lambda r: r["environment"].update(device="cpu"),
        lambda r: r["method"].update(steps=99),
        lambda r: r["results"][0]["trials"][0].update(bytes_per_second=12000),
        lambda r: r["results"][0]["trials"][0]["step_seconds"].__setitem__(0, float("nan")),
    ],
)
def test_rejects_wrong_source_or_timing(change):
    value = receipt()
    assert ci.check_receipt(value, BASE, "tiny") == pytest.approx(10240)
    change(value)
    with pytest.raises(ValueError):
        ci.check_receipt(value, BASE, "tiny")


def harness(monkeypatch, tmp_path, fail_at=None):
    calls = []
    monkeypatch.setattr(ci, "PAIRS", 2)
    monkeypatch.setattr(ci, "probe", lambda _: {"metal_available": True})
    monkeypatch.setattr(ci, "resolve_commit", lambda commit: commit)

    def command(args, **kwargs):
        if args[:2] == ["git", "show"]:
            return (ROOT / "uv.lock").read_text().strip()
        return HEAD

    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(ci.subprocess, "run", lambda *args, **kwargs: None)

    def sample(checkout, commit, preset, output):
        calls.append((commit, output.name))
        if len(calls) == fail_at:
            raise RuntimeError("Worker stopped")
        value = copy.deepcopy(receipt(commit))
        ci.save(output, value)
        return value

    monkeypatch.setattr(ci, "sample", sample)
    args = SimpleNamespace(out=tmp_path, baseline=BASE, candidate=HEAD, preset="tiny")
    return args, calls


def test_alternates_order_and_calibrates_on_the_same_commit(monkeypatch, tmp_path):
    args, calls = harness(monkeypatch, tmp_path)
    ci.compare(args)
    assert [commit for commit, _ in calls] == [BASE] * 4 + [BASE, HEAD, HEAD, BASE]
    assert [name[-6] for _, name in calls] == ["a", "b", "b", "a"] * 2
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["complete"] is True
    assert len(report["samples"]) == 8
    assert (tmp_path / "summary.md").is_file()


def test_keeps_finished_samples_after_a_failure(monkeypatch, tmp_path):
    args, _ = harness(monkeypatch, tmp_path, fail_at=3)
    with pytest.raises(RuntimeError, match="Worker stopped"):
        ci.compare(args)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["complete"] is False
    assert len(report["samples"]) == 2
    assert len(list(tmp_path.glob("tiny-*.json"))) == 2


@pytest.mark.parametrize("value", ["main", "--all", "a" * 39, "A" * 40])
def test_commit_input_is_a_full_hash(value):
    with pytest.raises(ValueError, match="full lowercase"):
        ci.resolve_commit(value)
