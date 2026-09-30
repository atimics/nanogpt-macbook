"""Plan a serial AI edit, check, and benchmark loop on an EC2 M2 Mac."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "benchmarks/planning/m2-24h-inputs.json"
PRESETS = ("tiny", "small", "medium")


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def comparison_seconds(path, presets):
    """Read complete comparisons timed by the current harness on the target Mac."""
    report = json.loads(path.read_text())
    if report.get("format") != "ci-paired-v1" or report.get("complete") is not True:
        raise ValueError("Calibration needs a complete ci-paired-v1 comparison")
    host = report["host"]
    if host.get("metal_available") is not True or host["device"]["device_name"] != "Apple M2":
        raise ValueError("M2 cost planning needs calibration from an Apple M2 Metal device")
    method = report["method"]
    if (method["pairs_per_phase"], method["steps"], method["warmup"]) != (6, 100, 20):
        raise ValueError("Calibration needs six pairs, 100 steps, and 20 warmup steps")
    expected = {
        (preset, phase, pair, side)
        for preset in presets
        for phase in ("calibration", "comparison")
        for pair in range(1, 7)
        for side in ("a", "b")
    }
    samples = report["samples"]
    actual = {(r["preset"], r["phase"], r["pair"], r["side"]) for r in samples}
    if actual != expected or len(samples) != len(expected):
        raise ValueError("Calibration needs every sample for the selected presets")
    for row in samples:
        name = row["file"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Calibration sample names must be local filenames")
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Calibration sample hash differs from the saved receipt")
        receipt = json.loads((path.parent / name).read_text())
        if (
            receipt["environment"]["chip"] != "Apple M2"
            or receipt["environment"]["device"] != "gpu"
        ):
            raise ValueError("Calibration samples must use the same Apple M2 Metal device")
    seconds = number(report["timing"]["elapsed_seconds"], "calibration time", positive=True)
    return seconds, host


def plan(args):
    inputs = json.loads(INPUTS.read_text())
    hardware = inputs["hardware"]
    for name in (
        "hours",
        "setup_minutes",
        "cleanup_minutes",
        "agent_minutes",
        "check_minutes",
        "inference_usd_per_iteration",
        "extra_cost_usd",
        "slowdown",
    ):
        number(getattr(args, name), name, positive=name in ("hours", "slowdown"))
    for name in ("inference_budget_usd", "hardware_budget_usd", "total_budget_usd"):
        if getattr(args, name) is not None:
            number(getattr(args, name), name)
    presets = PRESETS if args.preset == "all" else (args.preset,)
    sources = []
    seconds = 0
    for row in inputs["reference_runs"]:
        if row["preset"] in presets:
            elapsed = (
                datetime.fromisoformat(row["completed_at"])
                - datetime.fromisoformat(row["started_at"])
            ).total_seconds()
            if elapsed != row["seconds"] or elapsed <= 0:
                raise ValueError("Reference duration differs from its workflow timestamps")
            seconds += elapsed
            sources.append(row)
    basis = {
        "kind": "projection_from_hosted_ci",
        "host": inputs["reference_host"],
        "reference_seconds": seconds,
        "duration_multiplier": args.slowdown,
        "sources": sources,
    }
    if args.calibration:
        seconds, host = comparison_seconds(args.calibration, presets)
        basis = {
            "kind": "measured_comparison",
            "host": host,
            "comparison_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
            "reference_seconds": seconds,
            "duration_multiplier": 1,
            "source": str(args.calibration),
        }
    benchmark_minutes = seconds * basis["duration_multiplier"] / 60
    cycle = args.agent_minutes + args.check_minutes + benchmark_minutes
    available = max(0, args.hours * 60 - args.setup_minutes - args.cleanup_minutes)
    time_capacity = math.floor(available / cycle)
    billed_hours = max(hardware["minimum_hours"], args.hours)
    hardware_decimal = Decimal(str(billed_hours)) * Decimal(str(hardware["hourly_usd"]))
    fixed_decimal = hardware_decimal + Decimal(str(args.extra_cost_usd))
    hardware_cost = float(hardware_decimal)
    fixed_cost = float(fixed_decimal)
    feasible = (args.hardware_budget_usd is None or hardware_cost <= args.hardware_budget_usd) and (
        args.total_budget_usd is None or fixed_cost <= args.total_budget_usd
    )
    iterations = time_capacity if feasible else 0
    if args.inference_usd_per_iteration > 0:
        caps = []
        if args.inference_budget_usd is not None:
            caps.append(Decimal(str(args.inference_budget_usd)))
        if args.total_budget_usd is not None:
            caps.append(max(Decimal(0), Decimal(str(args.total_budget_usd)) - fixed_decimal))
        for cap in caps:
            count = (cap / Decimal(str(args.inference_usd_per_iteration))).to_integral_value(
                rounding=ROUND_FLOOR
            )
            iterations = min(iterations, int(count))
    inference_decimal = iterations * Decimal(str(args.inference_usd_per_iteration))
    return {
        "format": "experiment-budget-v1",
        "status": "plan",
        "hardware": hardware,
        "timing_basis": basis,
        "assumptions": {
            "hours": args.hours,
            "setup_minutes": args.setup_minutes,
            "cleanup_minutes": args.cleanup_minutes,
            "agent_minutes": args.agent_minutes,
            "check_and_pr_minutes": args.check_minutes,
            "inference_usd_per_iteration": args.inference_usd_per_iteration,
            "extra_cost_usd": args.extra_cost_usd,
            "inference_budget_usd": args.inference_budget_usd,
            "hardware_budget_usd": args.hardware_budget_usd,
            "total_budget_usd": args.total_budget_usd,
        },
        "presets": list(presets),
        "available_minutes": available,
        "benchmark_minutes_per_iteration": benchmark_minutes,
        "minutes_per_iteration": cycle,
        "time_capacity_iterations": time_capacity,
        "fixed_cost_fits_budget": feasible,
        "planned_iterations": iterations,
        "raw_benchmark_samples": iterations * 24 * len(presets),
        "hardware_billed_hours": billed_hours,
        "hardware_usd": hardware_cost,
        "inference_usd": float(inference_decimal),
        "total_usd": float(fixed_decimal + inference_decimal),
        "hardware_usd_per_iteration": hardware_cost / iterations if iterations else None,
        "notes": [
            "One iteration includes an AI edit, checks, a PR, and fresh A/A and A/B comparisons.",
            "AI time and inference cost are planning inputs; record every call and retry.",
            "Attempted iterations and accepted improvements are separate counts.",
            "Storage, network, tax, and host release delay belong in extra cost.",
            "The current timing protocol uses short samples; longer trials need new calibration.",
        ],
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--setup-minutes", type=float, default=60)
    p.add_argument("--cleanup-minutes", type=float, default=60)
    p.add_argument("--agent-minutes", type=float, default=10)
    p.add_argument("--check-minutes", type=float, default=5)
    p.add_argument(
        "--slowdown",
        type=float,
        default=1.5,
        help="assumed benchmark duration relative to hosted CI (default: 1.5)",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        help="complete timed comparison.json; replaces the hosted-CI projection",
    )
    p.add_argument(
        "--inference-usd-per-iteration",
        type=float,
        default=1,
        help="assumed total cost of all AI calls per iteration (default: $1)",
    )
    p.add_argument("--inference-budget-usd", type=float)
    p.add_argument("--hardware-budget-usd", type=float)
    p.add_argument("--total-budget-usd", type=float)
    p.add_argument("--extra-cost-usd", type=float, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, help="save the full plan as JSON")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        result = plan(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    if args.json:
        print(encoded, end="")
    else:
        print(f"Timing basis: {result['timing_basis']['kind']}")
        print(f"Available loop time: {result['available_minutes']:g} minutes")
        print(f"One iteration: {result['minutes_per_iteration']:.2f} minutes")
        print(f"Planned iterations: {result['planned_iterations']}")
        print(f"Raw benchmark samples: {result['raw_benchmark_samples']}")
        print(f"M2 host: ${result['hardware_usd']:.2f} USD")
        print(f"AI inference assumption: ${result['inference_usd']:.2f} USD")
        print(f"Extra cost allowance: ${args.extra_cost_usd:.2f} USD")
        print(f"Planned total: ${result['total_usd']:.2f} USD")
        if not result["fixed_cost_fits_budget"]:
            print("The fixed host cost exceeds the supplied budget. Planned iterations: 0.")
        for note in result["notes"]:
            print(note)


if __name__ == "__main__":
    main()
