"""Keep experiment capacity and spending limits separate from timing claims."""

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUDGET = runpy.run_path(str(ROOT / "scripts/experiment_budget.py"))


def plan(*argv):
    return BUDGET["plan"](BUDGET["parser"]().parse_args(argv))


def test_reproduces_the_day_plan_from_workflow_wall_times():
    result = plan()
    assert result["timing_basis"]["reference_seconds"] == 962
    assert result["timing_basis"]["kind"] == "projection_from_hosted_ci"
    assert result["available_minutes"] == 1320
    assert result["minutes_per_iteration"] == pytest.approx(39.05)
    assert result["planned_iterations"] == 33
    assert result["raw_benchmark_samples"] == 2376
    assert result["hardware_usd"] == pytest.approx(21.072)
    assert result["total_usd"] == pytest.approx(54.072)


def test_counts_both_controls_and_comparisons_per_preset():
    result = plan("--preset", "medium")
    assert result["planned_iterations"] == 44
    assert result["raw_benchmark_samples"] == 44 * 24
    assert result["benchmark_minutes_per_iteration"] == pytest.approx(14.65)


@pytest.mark.parametrize(("multiplier", "count"), [("1", 42), ("1.5", 33), ("2", 28)])
def test_hardware_speed_scenarios_are_explicit(multiplier, count):
    assert plan("--slowdown", multiplier)["planned_iterations"] == count


def test_twenty_dollars_is_below_the_m2_day_minimum():
    result = plan("--hardware-budget-usd", "20")
    assert result["fixed_cost_fits_budget"] is False
    assert result["time_capacity_iterations"] == 33
    assert result["planned_iterations"] == 0


def test_short_session_still_pays_for_a_full_day():
    result = plan("--hours", "3")
    assert result["hardware_billed_hours"] == 24
    assert result["planned_iterations"] == 1


def test_longer_session_includes_extra_host_time():
    assert plan("--hours", "25")["hardware_usd"] == pytest.approx(21.95)


def test_money_rounding_preserves_exact_decimal_budget_boundaries():
    result = plan("--inference-budget-usd", "0.3", "--inference-usd-per-iteration", "0.1")
    assert result["planned_iterations"] == 3
    result = plan("--total-budget-usd", "21.372", "--inference-usd-per-iteration", "0.1")
    assert result["planned_iterations"] == 3


def test_total_cap_reserves_hardware_and_extra_cost_first():
    result = plan("--total-budget-usd", "40", "--extra-cost-usd", "3")
    assert result["planned_iterations"] == 15
    assert result["total_usd"] == pytest.approx(39.072)


def test_zero_inference_price_and_zero_time_are_supported():
    assert (
        plan("--inference-usd-per-iteration", "0", "--inference-budget-usd", "0")[
            "planned_iterations"
        ]
        == 33
    )
    result = plan("--setup-minutes", "1440")
    assert result["planned_iterations"] == 0
    assert result["hardware_usd_per_iteration"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ("--agent-minutes", "nan"),
        ("--inference-usd-per-iteration", "-1"),
        ("--hours", "0"),
        ("--slowdown", "0"),
        ("--total-budget-usd", "inf"),
    ],
)
def test_invalid_inputs_fail_before_cost_math(argv):
    with pytest.raises(ValueError, match="must be finite"):
        plan(*argv)


def calibration(tmp_path):
    report = json.loads((ROOT / "benchmarks/ci/36750498161/comparison.json").read_text())
    report["host"]["device"]["device_name"] = "Apple M2"
    report["timing"] = {"elapsed_seconds": 300}
    for row in report["samples"]:
        receipt = json.loads((ROOT / "benchmarks/ci/36750498161" / row["file"]).read_text())
        receipt["environment"]["chip"] = "Apple M2"
        data = json.dumps(receipt).encode()
        (tmp_path / row["file"]).write_bytes(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(report))
    return report, path


def test_target_measurement_replaces_the_hardware_multiplier(tmp_path):
    _, path = calibration(tmp_path)
    result = plan("--preset", "tiny", "--calibration", str(path), "--slowdown", "10")
    assert result["benchmark_minutes_per_iteration"] == 5
    assert result["planned_iterations"] == 66
    assert result["timing_basis"]["kind"] == "measured_comparison"
    assert (
        result["timing_basis"]["comparison_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("kind", ["partial", "preset", "sample", "clock", "host", "steps"])
def test_calibration_requires_a_complete_matching_m2_run(tmp_path, kind):
    report, path = calibration(tmp_path)
    report = copy.deepcopy(report)
    if kind == "partial":
        report["complete"] = False
    elif kind == "preset":
        report["samples"].pop()
    elif kind == "sample":
        (tmp_path / report["samples"][0]["file"]).write_text("{}")
    elif kind == "clock":
        report["timing"]["elapsed_seconds"] = -1
    elif kind == "host":
        report["host"]["device"]["device_name"] = "Apple M2 Pro"
    else:
        report["method"]["steps"] = 1000
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        plan("--preset", "tiny", "--calibration", str(path))


def test_cli_saves_a_replayable_plan(tmp_path):
    output = tmp_path / "plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/experiment_budget.py"),
            "--json",
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == json.loads(output.read_text())
