"""Corpus-size scaling harness sanity (design sections 6 & 7).

Loose, non-brittle assertions that document the scaling BEHAVIOUR:
  * the harness is deterministic given fixed seeds,
  * strong forms (verbatim/paraphrase/format_shift) stay perfectly detected as
    the corpus grows, while answer_only collapses past a small corpus size,
  * rho/Delta are recovered (small corpus) then conservatively UNDER-estimated
    (large corpus) -- never fabricated (precision stays high).
"""
import importlib.util
from pathlib import Path

import pytest

REAL_DIR = Path(__file__).resolve().parents[1] / "demo" / "real_data"

pytestmark = pytest.mark.skipif(
    not (REAL_DIR / "ag_news.jsonl").exists(),
    reason="real_data cache not built",
)


def _load():
    path = Path(__file__).resolve().parents[1] / "demo" / "run_scaling_study.py"
    spec = importlib.util.spec_from_file_location("run_scaling_study", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scaling_is_deterministic():
    m = _load()
    bench = m.load_real_benchmark()
    corpus = m.load_real_corpus()
    train, test = m.split(bench, m.SPLIT_SEED)
    r1 = m.run_one(160, train, test, corpus)
    r2 = m.run_one(160, train, test, corpus)
    assert r1 == r2  # fixed seeds -> identical results


def test_strong_forms_hold_answer_only_collapses():
    m = _load()
    bench = m.load_real_benchmark()
    corpus = m.load_real_corpus()
    train, test = m.split(bench, m.SPLIT_SEED)

    small = m.run_one(100, train, test, corpus)
    large = m.run_one(500, train, test, corpus)

    # strong forms detected regardless of corpus size
    for r in (small, large):
        assert r["recall_verbatim"] == 1.0
        assert r["recall_paraphrase"] == 1.0
        assert r["recall_format_shift"] == 1.0

    # answer_only degrades as the corpus grows (phase transition)
    assert large["recall_answer_only"] < small["recall_answer_only"]

    # honest failure mode: precision never collapses, so rho/Delta are
    # UNDER-estimated at large corpus, not inflated
    assert large["precision"] >= 0.9
    assert large["rho_hat"] <= large["rho_true"] + 1e-9
    assert large["delta_hat"] <= large["delta_true"] + 1e-9
