import json
import statistics

import pytest

from nanogpt_macbook import benchmark as module
from nanogpt_macbook.config import ModelConfig, TrainConfig


@pytest.mark.parametrize("pipeline", [False, True])
def test_benchmark_records_recomputable_timings(tmp_path, monkeypatch, pipeline):
    monkeypatch.setitem(
        module.PRESETS,
        "tiny",
        (ModelConfig(context=8, layers=1, heads=2, width=8), TrainConfig(batch_size=2)),
    )
    output = tmp_path / "result.json"
    result = module.benchmark(
        output,
        ["tiny"],
        device="cpu",
        steps=3,
        warmup=1,
        repeats=2,
        pipeline=pipeline,
        report=lambda _: None,
    )
    assert json.loads(output.read_text()) == result
    assert result["environment"]["device"] == "cpu"
    assert result["method"]["queue_depth"] == (2 if pipeline else 1)
    assert result["method"]["execution"] == ("pipelined" if pipeline else "compiled")
    assert len(result["source"]["python_source_sha256"]) == 64
    row = result["results"][0]
    rates = []
    for trial in row["trials"]:
        assert len(trial["step_seconds"]) == 3
        assert all(duration > 0 for duration in trial["step_seconds"])
        rate = 16 * 3 / sum(trial["step_seconds"])
        assert trial["bytes_per_second"] == pytest.approx(rate)
        rates.append(rate)
    assert row["summary"]["median_bytes_per_second"] == pytest.approx(statistics.median(rates))
    with pytest.raises(ValueError, match="already exists"):
        module.benchmark(output, ["tiny"])


@pytest.mark.parametrize("option", [{"steps": 0}, {"warmup": 0}, {"repeats": 0}, {"seed": -1}])
def test_invalid_benchmark_settings_fail_early(tmp_path, option):
    with pytest.raises(ValueError, match="must"):
        module.benchmark(tmp_path / "receipt.json", ["tiny"], **option)
