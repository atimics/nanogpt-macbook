import copy
import json
import re
import runpy
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILDER = runpy.run_path(str(ROOT / "scripts/build_site.py"))


def receipt():
    return json.loads((ROOT / "benchmarks/results/m4-max-gpu.json").read_text())


def test_site_recomputes_raw_measurements():
    BUILDER["validate"](receipt())
    altered = receipt()
    altered["results"][0]["summary"]["median_bytes_per_second"] *= 2
    with pytest.raises(ValueError, match="Summary"):
        BUILDER["validate"](altered)


@pytest.mark.parametrize("kind", ["timing", "count", "source", "warmup"])
def test_site_rejects_inconsistent_receipts(kind):
    altered = copy.deepcopy(receipt())
    if kind == "timing":
        altered["results"][0]["trials"][0]["step_seconds"][0] = float("nan")
    elif kind == "count":
        altered["results"][0]["trials"].pop()
    elif kind == "source":
        altered["source"]["source_dirty"] = True
    else:
        altered["method"]["warmup_steps"] = 0
    with pytest.raises(ValueError):
        BUILDER["validate"](altered)


def test_built_site_has_real_figures_and_resolving_local_links(tmp_path):
    output = tmp_path / "site"
    BUILDER["build"](output)
    page = (output / "index.html").read_text()
    assert "{{" not in page
    assert "128.9k" in page
    assert "165.8k" in page
    assert "be1802a" in page and "d1877d0" in page
    assert "Training speed by commit" in page
    assert "5.52" in page and "2.17" in page
    links = []
    ids = set()

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                assert attrs["id"] not in ids
                ids.add(attrs["id"])
            for name in ("href", "src"):
                if name in attrs:
                    links.append(attrs[name])

    Links().feed(page)
    for link in links:
        if link.startswith("#"):
            assert link[1:] in ids, link
        elif not link.startswith("https://"):
            assert (output / link).exists(), link
    published = json.loads((output / "data/benchmarks.json").read_text())
    assert len(published) == len(list((ROOT / "benchmarks/results").glob("*.json")))
    for result in published:
        assert result["source"]["commit"][:7] in page


def test_timeline_uses_distinct_commits_and_matching_workloads(tmp_path):
    target = tmp_path / "benchmarks/results"
    target.mkdir(parents=True)
    first = receipt()
    (target / "first.json").write_text(json.dumps(first))
    second = copy.deepcopy(first)
    second["source"]["commit"] = "f" * 40
    second["source"]["python_source_sha256"] = "e" * 64
    second["recorded_at"] = "2026-09-30T00:00:00+00:00"
    (target / "second.json").write_text(json.dumps(second))
    receipts, rows = BUILDER["load_results"](tmp_path)
    assert len({r["source"]["commit"] for r in receipts}) == 2
    assert len(rows) == 6

    (target / "duplicate.json").write_text(json.dumps(first))
    with pytest.raises(ValueError, match="one receipt per commit"):
        BUILDER["load_results"](tmp_path)
    (target / "duplicate.json").unlink()

    second["results"][0]["train_config"]["weight_decay"] = 0.2
    (target / "second.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="matching model and training settings"):
        BUILDER["load_results"](tmp_path)


def test_asset_urls_change_with_content_for_returning_visitors(tmp_path):
    root = tmp_path / "source"
    for name in ("site", "benchmarks"):
        shutil.copytree(ROOT / name, root / name)

    def asset_links(output):
        BUILDER["build"](output, root)
        page = (output / "index.html").read_text()
        return set(re.findall(r'(?:src|href)="\./([^"/]+\.(?:css|js|svg))"', page))

    first = asset_links(tmp_path / "first")
    assert len(first) == 3
    assert asset_links(tmp_path / "repeat") == first
    for name in ("style.css", "app.js"):
        source = root / "site" / name
        source.write_text(source.read_text() + "\n/* New layout */\n")
    second = asset_links(tmp_path / "second")
    assert len(first & second) == 1  # The unchanged favicon keeps its URL.
    for name in second:
        original = name.split(".")[0] + "." + name.split(".")[-1]
        assert (tmp_path / "second" / name).read_bytes() == (root / "site" / original).read_bytes()


def test_site_checks_the_queued_timing_protocol():
    updated = receipt()
    updated["method"].update(name="training-loop-v2", execution="pipelined", queue_depth=2)
    BUILDER["validate"](updated)
    for changes in (
        {"queue_depth": 3},
        {"name": "training-step-v1"},
        {"execution": "compiled"},
        {"dtype": "bfloat16"},
    ):
        altered = copy.deepcopy(updated)
        altered["method"].update(changes)
        with pytest.raises(ValueError):
            BUILDER["validate"](altered)
