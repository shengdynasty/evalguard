"""Per-form (OR-gate) decision layer + the fix it delivers over global-tau."""
import random
from pathlib import Path

import pytest

from evalguard import decision
from evalguard.aggregate import aggregate
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids as contam_global
from evalguard.inject import inject
from evalguard.synth import make_benchmark, make_clean_corpus

FORM_MIX = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}


def _setup(n_docs, seed=1):
    bench = make_benchmark(n=120, seed=7)
    clean = make_clean_corpus(n_docs=n_docs, seed=11)
    tr, te = bench, bench  # calibrate + eval on same synthetic set for a unit test
    res = inject(clean, bench, rate=0.2, form_mix=FORM_MIX, seed=seed)
    return bench, res


def test_matcher_thresholds_resolved_from_chosen():
    chosen = {"ngram": 0.1, "embedding": 0.2, "paraphrase": 0.3, "answer": 0.4, "tau": 0.5}
    thr = decision.matcher_thresholds(chosen)
    assert thr == {"ngram": 0.1, "embedding": 0.2, "paraphrase": 0.3, "answer": 0.4}
    assert "tau" not in thr


def test_perform_recovers_answer_only_where_global_drops_it(tmp_path):
    # A corpus large enough that a single global tau abandons answer_only.
    bench = make_benchmark(n=150, seed=7)
    clean = make_clean_corpus(n_docs=600, seed=11)
    res = inject(clean, bench, rate=0.2, form_mix=FORM_MIX, seed=42)
    cal = calibrate(bench, res, seed=1, out_dir=tmp_path, write_plots=False)
    chosen = cal["chosen"]; tau = float(chosen["tau"])

    ev = aggregate(bench, res.corpus, seed=1)
    truth = res.injected_ids
    ao = set(res.ids_by_form().get("answer_only", []))

    global_pred = set(contam_global(ev, tau=tau))
    perform_pred = set(decision.contaminated_ids(ev, chosen))

    global_ao = len(global_pred & ao) / len(ao)
    perform_ao = len(perform_pred & ao) / len(ao)

    # per-form recovers the answer_only form that global-tau discards
    assert perform_ao >= global_ao
    assert perform_ao >= 0.5


def test_perform_rho_and_delta_run(tmp_path):
    bench = make_benchmark(n=100, seed=3)
    clean = make_clean_corpus(n_docs=200, seed=5)
    res = inject(clean, bench, rate=0.2, form_mix=FORM_MIX, seed=9)
    cal = calibrate(bench, res, seed=1, out_dir=tmp_path, write_plots=False)
    ev = aggregate(bench, res.corpus, seed=1)
    scores = {it.id: 1.0 for it in bench.items}
    r = decision.rho(ev, cal["chosen"])
    d = decision.delta(ev, scores, cal["chosen"])
    assert 0.0 <= r <= 1.0
    assert d.n_full == len(bench)
