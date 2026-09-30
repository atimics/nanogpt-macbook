"""Build a small static benchmark site from checked measurement receipts."""

import argparse
import hashlib
import html
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/atimics/nanogpt-macbook"


def close(actual, expected):
    return (
        isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    )


def validate(receipt: dict):
    """Recompute all displayed measurements from the raw timing samples."""
    if receipt.get("format") != 1:
        raise ValueError("Unsupported benchmark receipt format")
    datetime.fromisoformat(receipt["recorded_at"])
    source = receipt["source"]
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")):
        raise ValueError("Published results need a source commit")
    if source.get("source_dirty") is not False:
        raise ValueError("Published results need committed source")
    if not re.fullmatch(r"[0-9a-f]{64}", source.get("python_source_sha256", "")):
        raise ValueError("Published results need a source hash")
    method = receipt["method"]
    protocol = (method["name"], method["execution"])
    if method["dtype"] != "float32" or protocol not in (
        ("training-step-v1", "eager"),
        ("training-step-v1", "compiled"),
        ("training-loop-v2", "pipelined"),
    ):
        raise ValueError("Use the documented float32 benchmark protocol")
    if method["execution"] == "pipelined" and method.get("queue_depth") != 2:
        raise ValueError("The pipelined protocol uses two queued steps")
    # The page states these counts. Require them before publishing a comparison.
    if (method["steps"], method["warmup_steps"], method["repeats"]) != (100, 20, 3):
        raise ValueError("The site protocol uses 100 steps, 20 warmup steps, and 3 trials")
    if receipt["environment"]["device"] not in ("gpu", "cpu"):
        raise ValueError("Unknown benchmark device")
    if not receipt["results"]:
        raise ValueError("A benchmark receipt needs measured results")
    for row in receipt["results"]:
        if row["preset"] not in ("tiny", "small", "medium"):
            raise ValueError("Unknown model preset")
        config, training = row["model_config"], row["train_config"]
        width, layers = config["width"], config["layers"]
        parameters = layers * (12 * width**2 + 4 * width)
        parameters += (config["vocab_size"] + config["context"] + 2) * width
        if row["parameters"] != parameters or training["accumulation"] != 1:
            raise ValueError("Model settings disagree with the benchmark protocol")
        trials = row["trials"]
        if len(trials) != method["repeats"]:
            raise ValueError("Trial count disagrees with the method")
        rates, peaks, times = [], [], []
        for index, trial in enumerate(trials, 1):
            values = trial["step_seconds"]
            if trial["trial"] != index or len(values) != method["steps"]:
                raise ValueError("Raw step count disagrees with the method")
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Step durations must be finite and positive")
            elapsed = sum(values)
            rate = training["batch_size"] * config["context"] * len(values) / elapsed
            if not close(trial["elapsed_seconds"], elapsed) or not close(
                trial["bytes_per_second"], rate
            ):
                raise ValueError("Trial throughput disagrees with raw timings")
            peak = trial["peak_memory_mib"]
            if not math.isfinite(peak) or peak < 0:
                raise ValueError("Memory must be finite and nonnegative")
            rates.append(rate)
            peaks.append(peak)
            times.append(statistics.mean(values) * 1000)
        expected = {
            "median_bytes_per_second": statistics.median(rates),
            "min_bytes_per_second": min(rates),
            "max_bytes_per_second": max(rates),
            "median_step_ms": statistics.median(times),
            "peak_memory_mib": max(peaks),
        }
        if any(not close(row["summary"][key], value) for key, value in expected.items()):
            raise ValueError("Summary disagrees with raw timings")


def load_results(root: Path):
    receipts = []
    rows = []
    for file in sorted((root / "benchmarks/results").glob("*.json")):
        receipt = json.loads(file.read_text())
        validate(receipt)
        receipts.append({**receipt, "file": file.name})
        for result in receipt["results"]:
            rows.append(
                {
                    **result,
                    "environment": receipt["environment"],
                    "source": receipt["source"],
                    "recorded_at": receipt["recorded_at"],
                    "execution": receipt["method"]["execution"],
                    "file": file.name,
                }
            )
    if not receipts:
        raise ValueError("Add measured benchmark JSON files before building the site")
    # Compare source changes on one machine and one software stack.
    signatures = {
        (
            r["environment"]["chip"],
            r["environment"]["memory_gib"],
            r["environment"]["mlx"],
            r["environment"]["python"],
            r["environment"]["os_version"],
            r["method"]["seed"],
        )
        for r in receipts
    }
    if len(signatures) != 1:
        raise ValueError("This report compares one machine and software stack")
    source_hashes = {}
    for receipt in receipts:
        commit = receipt["source"]["commit"]
        digest = receipt["source"]["python_source_sha256"]
        if commit in source_hashes and source_hashes[commit] != digest:
            raise ValueError("One commit has conflicting source hashes")
        source_hashes[commit] = digest
    model_settings = {}
    for row in rows:
        identity = (row["preset"], row["environment"]["device"])
        settings = (row["parameters"], row["model_config"], row["train_config"])
        if identity in model_settings and model_settings[identity] != settings:
            raise ValueError("A timeline series needs matching model and training settings")
        model_settings[identity] = settings
    rows.sort(
        key=lambda row: (
            row["recorded_at"],
            row["environment"]["device"] == "cpu",
            ("tiny", "small", "medium").index(row["preset"]),
        )
    )
    identities = [
        (row["source"]["commit"], row["preset"], row["environment"]["device"]) for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Keep one receipt per commit, model, and device")
    return receipts, rows


def timeline_cards(rows, commits):
    cards = []
    for preset, device in sorted(
        {(r["preset"], r["environment"]["device"]) for r in rows},
        key=lambda item: (item[1] == "cpu", ("tiny", "small", "medium").index(item[0])),
    ):
        series = sorted(
            (r for r in rows if r["preset"] == preset and r["environment"]["device"] == device),
            key=lambda row: commits.index(row["source"]["commit"]),
        )
        left, right = 9, 91
        top, bottom = 30, 165
        maximum = max(r["summary"]["median_bytes_per_second"] for r in series) * 1.2
        positions = []
        for row in series:
            index = commits.index(row["source"]["commit"])
            x = left + index * (right - left) / max(1, len(commits) - 1)
            y = bottom - row["summary"]["median_bytes_per_second"] / maximum * (bottom - top)
            positions.append((x, y))
        chart = [
            '<svg width="100%" height="215" role="img" '
            f'aria-label="{preset} {device} '
            f'throughput across {len(series)} measured commits">',
            f'<line x1="{left}%" y1="{bottom}" x2="{right}%" y2="{bottom}" stroke="#ccd5c0"/>',
            '<text x="8" y="21" font-size="10" fill="#5f6e64">BYTES / SECOND</text>',
        ]
        if len(positions) > 1:
            for (x1, y1), (x2, y2) in zip(positions[:-1], positions[1:], strict=True):
                chart.append(
                    f'<line x1="{x1:.2f}%" y1="{y1:.1f}" x2="{x2:.2f}%" y2="{y2:.1f}" '
                    'stroke="#187556" stroke-width="3"/>'
                )
        for index, (row, (x, y)) in enumerate(zip(series, positions, strict=True)):
            value = row["summary"]["median_bytes_per_second"]
            sha = row["source"]["commit"]
            anchor = "start" if index == 0 else "end" if index == len(series) - 1 else "middle"
            chart.append(
                f'<a href="{REPO}/commit/{sha}" data-position="{x:.2f}" '
                f'data-label="{str(index in (0, len(series) - 1)).lower()}" '
                f'aria-label="Commit {sha[:7]}: '
                f'{value:,.0f} bytes per second">'
                f"<title>{sha[:7]}: {value:,.0f} bytes per second</title>"
                f'<circle cx="{x:.2f}%" cy="{y:.1f}" r="5" fill="#187556"/>'
                f'<text class="timeline-label" x="{x:.2f}%" y="{y - 11:.1f}" '
                f'text-anchor="{anchor}" '
                f'font-size="10" fill="#172b27">{value / 1000:.1f}k</text>'
                f'<text class="timeline-label" x="{x:.2f}%" y="190" text-anchor="{anchor}" '
                f'font-size="10" fill="#187556">{sha[:7]}</text></a>'
            )
        chart.append("</svg>")
        previous = series[-2 if len(series) > 1 else 0]["summary"]["median_bytes_per_second"]
        last = series[-1]["summary"]["median_bytes_per_second"]
        baseline = series[0]["summary"]["median_bytes_per_second"]
        baseline_change = (last / baseline - 1) * 100
        change = (last / previous - 1) * 100
        change_text = f"{change:+.1f}% vs previous" if len(series) > 1 else "First measurement"
        label = "Metal GPU" if device == "gpu" else "CPU"
        cards.append(
            f'<article class="timeline-card"><div class="timeline-head"><h3>{preset} / {label}</h3>'
            f'<div class="timeline-changes"><span>{baseline_change:+.1f}% vs baseline</span>'
            f'<span>{change_text}</span></div></div><div class="timeline-plot">'
            f"{''.join(chart)}</div></article>"
        )
    return "\n".join(cards)


def bars(rows, memory=False):
    key = "peak_memory_mib" if memory else "median_bytes_per_second"
    largest = max(row["summary"][key] for row in rows) or 1
    parts = []
    for row in rows:
        summary, device = row["summary"], row["environment"]["device"]
        value = summary[key]
        label = "Metal GPU" if device == "gpu" else "CPU"
        unit = "MiB" if memory else "bytes/s"
        detail = (
            "peak allocation"
            if memory
            else (
                f"{summary['min_bytes_per_second'] / 1000:.1f}–"
                f"{summary['max_bytes_per_second'] / 1000:.1f}k range"
            )
        )
        number = f"{value:,.1f}" if memory else f"{value:,.0f}"
        parts.append(
            f'<div class="bar-row" data-device="{device}"><div class="bar-label">'
            f'{row["preset"]}<span>{label}</span></div><div class="bar-track" aria-hidden="true">'
            f'<div class="bar-fill" style="width:{value / largest * 100:.4f}%"></div></div>'
            f'<div class="bar-number" aria-label="{number} {unit}">{number}'
            f"<small>{detail}</small></div></div>"
        )
    return "\n".join(parts)


def table(rows):
    parts = []
    for row in reversed(rows):
        summary, config, device = row["summary"], row["model_config"], row["environment"]["device"]
        label = "Metal" if device == "gpu" else "CPU"
        sha = row["source"]["commit"]
        parts.append(
            f'<tr data-device="{device}"><td data-label="Preset / device">'
            f"{row['preset']}<small>{label}</small></td>"
            f'<td data-label="Source commit"><a href="{REPO}/commit/{sha}">{sha[:7]}</a>'
            f"<small>{row['execution']}</small></td>"
            f'<td data-label="Parameters">{row["parameters"]:,}</td>'
            f'<td data-label="Context × batch">{config["context"]} × '
            f"{row['train_config']['batch_size']}</td>"
            f'<td data-label="Bytes / second">{summary["median_bytes_per_second"]:,.0f}</td>'
            f'<td data-label="Step time">{summary["median_step_ms"]:.2f} ms</td>'
            f'<td data-label="Peak MLX">{summary["peak_memory_mib"]:,.1f} MiB</td>'
            f'<td data-label="Receipt"><a href="./data/{html.escape(row["file"])}" download '
            f'aria-label="Download {row["preset"]} {label} receipt">JSON ↗</a></td></tr>'
        )
    return "\n".join(parts)


def learning_chart(record):
    points = [m for m in record["metrics"] if "val_loss" in m]
    if [point["step"] for point in points] != [0, 100, 200, 300]:
        raise ValueError("Learning report needs the four measured validation points")
    if any(not math.isfinite(p["val_loss"]) or p["val_loss"] <= 0 for p in points):
        raise ValueError("Learning losses must be finite and positive")
    coords = [(50 + p["step"] / 300 * 390, 215 - (p["val_loss"] - 2) / 4 * 160) for p in points]
    path = " ".join(f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}" for i, (x, y) in enumerate(coords))
    svg = [
        '<svg viewBox="0 0 490 270" role="img" aria-labelledby="loss-title loss-desc">',
        '<title id="loss-title">Held-out loss over 300 training steps</title>',
        '<desc id="loss-desc">'
        + "; ".join(f"step {p['step']}: {p['val_loss']:.4f} nats" for p in points)
        + "</desc>",
        '<g font-family="ui-monospace, monospace" font-size="10" fill="#5f6e64">',
    ]
    for value in (2, 3, 4, 5, 6):
        y = 215 - (value - 2) / 4 * 160
        svg.append(f'<path d="M50 {y}H440" stroke="#ccd5c0" stroke-dasharray="3 5"/>')
        svg.append(f'<text x="22" y="{y + 4}">{value}.0</text>')
    for p, (x, _) in zip(points, coords, strict=True):
        svg.append(f'<text x="{x}" y="240" text-anchor="middle">{p["step"]}</text>')
    svg += [
        '<text x="50" y="29" font-size="9">VALIDATION LOSS / NATS</text>',
        '<text x="440" y="263" text-anchor="end" font-size="9">TRAINING STEPS</text></g>',
        f'<path d="{path}" fill="none" stroke="#187556" stroke-width="3"/>',
    ]
    for x, y in coords:
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
            'fill="#187556" stroke="#e9eddf" stroke-width="2"/>'
        )
    svg.append("</svg>")
    return "".join(svg), points


def build(output: Path, root: Path = ROOT):
    receipts, rows = load_results(root)
    commits = sorted(
        {r["source"]["commit"] for r in receipts},
        key=lambda sha: min(r["recorded_at"] for r in receipts if r["source"]["commit"] == sha),
    )
    latest = {}
    for row in rows:
        latest[(row["preset"], row["environment"]["device"])] = row
    current_rows = sorted(
        latest.values(),
        key=lambda row: (
            row["environment"]["device"] == "cpu",
            ("tiny", "small", "medium").index(row["preset"]),
        ),
    )
    learning = json.loads((root / "benchmarks/learning/demo.json").read_text())
    curve, losses = learning_chart(learning)
    tiny = latest[("tiny", "gpu")]
    env = tiny["environment"]
    summary = tiny["summary"]
    stats = [
        (
            "Tiny / Metal throughput",
            f"{summary['median_bytes_per_second'] / 1000:.1f}k",
            "bytes / second",
            "median of 3 measured trials",
        ),
        (
            "Tiny / peak MLX memory",
            f"{summary['peak_memory_mib']:.0f}",
            "MiB",
            "float32 · context 128 · batch 8",
        ),
        (
            "Model sizes measured",
            str(len({r["preset"] for r in current_rows})),
            "GPT presets",
            "0.84M → 14.46M parameters",
        ),
    ]
    highlights = "".join(
        f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">'
        f'{value}<small>{unit}</small></div><div class="stat-note">{note}</div></div>'
        for label, value, unit, note in stats
    )
    replacements = {
        "MACHINE": html.escape(
            f"{env['chip']} · {env['memory_gib']:g} GiB · "
            f"macOS {env['os_version']} · MLX {env['mlx']}"
        ),
        "DATE": max(r["recorded_at"][:10] for r in receipts),
        "HIGHLIGHTS": highlights,
        "TIMELINE_CARDS": timeline_cards(rows, commits),
        "COMMIT_COUNT": str(len(commits)),
        "LATEST_COMMIT": " · ".join(
            dict.fromkeys(row["source"]["commit"][:7] for row in current_rows)
        ),
        "SPEED_BARS": bars(current_rows),
        "MEMORY_BARS": bars(current_rows, memory=True),
        "TABLE_ROWS": table(rows),
        "ROW_COUNT": str(len(current_rows)),
        "SCOPE_NOTE": "One M4 Max machine across measured commits. Results vary with chip, "
        "model settings, power state, temperature, and other running apps.",
        "FIRST_LOSS": f"{losses[0]['val_loss']:.2f}",
        "LAST_LOSS": f"{losses[-1]['val_loss']:.2f}",
        "LOSS_CHART": curve,
        "SOURCE_LINKS": " · ".join(
            f'<a href="{REPO}/commit/{sha}">{sha[:7]} ↗</a>' for sha in commits
        ),
    }
    page = (root / "site/index.html").read_text()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("style.css", "app.js", "favicon.svg"):
        asset = (root / "site" / name).read_bytes()
        digest = hashlib.sha256(asset).hexdigest()[:12]
        path = Path(name)
        versioned = f"{path.stem}.{digest}{path.suffix}"
        (output / versioned).write_bytes(asset)
        page = page.replace(f'"./{name}"', f'"./{versioned}"')
    for key, value in replacements.items():
        page = page.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", page):
        raise ValueError("Unfilled site template value")
    (output / "data").mkdir(exist_ok=True)
    (output / "index.html").write_text(page)
    (output / ".nojekyll").touch()
    for receipt in receipts:
        (output / "data" / receipt["file"]).write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "data/benchmarks.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (output / "data/demo-learning.json").write_text(json.dumps(learning, indent=2) + "\n")
    print(f"Built {len(rows)} measured commit/configuration points into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    build(args.out)
