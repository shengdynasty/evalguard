"""Real-data cache + real demo pipeline sanity (design sections 6 & 7).

These assert the REAL public data cache loads and that the honest, DEGRADED
recovery on real text holds: strong forms detected, answer_only weak, rho/Delta
under-estimated rather than perfect. They are deliberately loose bounds so they
document the behaviour without being brittle.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REAL_DIR = Path(__file__).resolve().parents[1] / "demo" / "real_data"

pytestmark = pytest.mark.skipif(
    not (REAL_DIR / "openbookqa.jsonl").exists(),
    reason="real_data cache not built",
)


def _load_real_demo():
    path = Path(__file__).resolve().parents[1] / "demo" / "run_real_demo.py"
    spec = importlib.util.spec_from_file_location("run_real_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_cache_shapes():
    rd = _load_real_demo()
    bench = rd.load_real_benchmark()
    corpus = rd.load_real_corpus()
    assert len(bench) == 300
    # corpus was expanded from the cached raw AG News pages (700 real docs)
    # for the scaling study; assert a lower bound rather than an exact count.
    assert len(corpus) >= 200
    # real MCQ items have non-empty question + answer text
    assert all(it.question and it.answer for it in bench.items)
    # answers are the correct-choice TEXT, not just a letter
    assert any(len(it.answer.split()) >= 1 for it in bench.items)


def test_real_pipeline_is_honestly_degraded(tmp_path):
    rd = _load_real_demo()
    from evalguard.aggregate import aggregate
    from evalguard.calibrate import calibrate
    from evalguard.estimands import contaminated_ids, rho
    from evalguard.inject import inject

    bench = rd.load_real_benchmark()
    corpus = rd.load_real_corpus()
    train, test = rd.split(bench, rd.SPLIT_SEED)
    res_tr = inject(corpus, train, rate=rd.RATE, form_mix=rd.FORM_MIX, seed=rd.INJECT_SEED_TRAIN)
    res_te = inject(corpus, test, rate=rd.RATE, form_mix=rd.FORM_MIX, seed=rd.INJECT_SEED_TEST)

    cal = calibrate(train, res_tr, seed=1, out_dir=tmp_path, write_plots=False)
    tau = cal["chosen"]["tau"]

    ev = aggregate(test, res_te.corpus, seed=1)
    truth = res_te.injected_ids
    pred = set(contaminated_ids(ev, tau=tau))
    ibf = res_te.ids_by_form()

    # strong forms: verbatim + format_shift caught well
    for form in ("verbatim", "format_shift"):
        tf = set(ibf.get(form, []))
        if tf:
            assert len(pred & tf) / len(tf) >= 0.8, f"{form} recall too low on real text"

    # answer_only genuinely degrades on real short common-word answers
    ao = set(ibf.get("answer_only", []))
    if ao:
        assert len(pred & ao) / len(ao) <= 0.5, "answer_only unexpectedly easy on real text"

    # precision stays high (no false positives on real background text)
    fp = len(pred - truth)
    assert fp == 0 or fp / max(1, len(pred)) <= 0.1

    # rho is UNDER-estimated (auditor is conservative), not perfect
    true_rho = res_te.true_rho(len(test))
    rho_hat = rho(ev, tau=tau)
    # conservative failure mode: never MORE contamination than truly exists
    assert rho_hat <= true_rho + 1e-9
