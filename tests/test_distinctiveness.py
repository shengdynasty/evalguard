"""Distinctiveness study harness sanity (single-form calibration refinement)."""
import importlib.util
from pathlib import Path
import pytest

REAL_DIR = Path(__file__).resolve().parents[1] / "demo" / "real_data"
pytestmark = pytest.mark.skipif(
    not (REAL_DIR / "ag_news.jsonl").exists(), reason="real_data cache not built")


def _load():
    p = Path(__file__).resolve().parents[1] / "demo" / "run_distinctiveness_study.py"
    spec = importlib.util.spec_from_file_location("run_distinctiveness_study", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_terciles_partition_and_order():
    m = _load()
    bench = m.load_bench(); corpus = m.load_corpus()
    dist = m.answer_distinctiveness(bench, corpus)
    buckets = m.terciles(bench, dist)
    assert [b[0] for b in buckets] == ["low", "mid", "high"]
    # mean distinctiveness strictly increases across terciles
    means = [b[2] for b in buckets]
    assert means[0] < means[1] < means[2]


def test_metrics_deterministic():
    m = _load()
    bench = m.load_bench(); corpus = m.load_corpus()
    dist = m.answer_distinctiveness(bench, corpus)
    _, bk, _ = m.terciles(bench, dist)[0]
    tr, te = m.split(bk, m.SPLIT_SEED)
    a = m.answer_only_metrics(150, tr, te, corpus)
    b = m.answer_only_metrics(150, tr, te, corpus)
    assert a == b
