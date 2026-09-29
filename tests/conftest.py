import mlx.core as mx
import pytest

from nanogpt_macbook.data import prepare


@pytest.fixture(autouse=True)
def cpu_backend():
    mx.set_default_device(mx.cpu)
    mx.random.seed(7)


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "story.txt"
    source.write_text("the sun rose over the small town. the moon rose over the quiet sea.\n" * 80)
    data = tmp_path / "data"
    prepare(source, data)
    return data
