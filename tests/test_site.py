import copy
import json
import runpy
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
    assert "128,865" in page
    assert "18,765" in page
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
    assert len(json.loads((output / "data/benchmarks.json").read_text())) == 2
