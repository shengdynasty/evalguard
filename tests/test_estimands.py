"""Estimators recover rho within tolerance and Delta within tolerance."""
import random

from evalguard.aggregate import aggregate
from evalguard.estimands import delta, rho
from evalguard.inject import inject
from evalguard.synth import make_benchmark, make_clean_corpus

TAU = 0.5


def _simulate_scores(bench, injected_ids, p_clean=0.4, p_contam=0.95, seed=0):
    rng = random.Random(seed)
    return {
        it.id: 1.0 if rng.random() < (p_contam if it.id in injected_ids else p_clean) else 0.0
        for it in bench.items
    }


def test_rho_recovered_within_tolerance():
    bench = make_benchmark(n=200, seed=7)
    clean = make_clean_corpus(n_docs=600, seed=11)
    for true_rate in (0.05, 0.1, 0.2):
        res = inject(
            clean, bench, rate=true_rate,
            form_mix={"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1},
            seed=42,
        )
        ev = aggregate(bench, res.corpus, seed=1)
        rho_hat = rho(ev, tau=TAU)
        true_rho = res.true_rho(len(bench))
        assert abs(rho_hat - true_rho) <= 0.03, (
            f"rate {true_rate}: rho_hat {rho_hat} vs true {true_rho}"
        )


def test_delta_recovered_within_tolerance():
    bench = make_benchmark(n=200, seed=7)
    clean = make_clean_corpus(n_docs=600, seed=11)
    res = inject(
        clean, bench, rate=0.2,
        form_mix={"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1},
        seed=42,
    )
    injected = res.injected_ids
    scores = _simulate_scores(bench, injected, seed=0)

    # TRUE delta from ground-truth clean subset
    all_s = [scores[i.id] for i in bench.items]
    clean_s = [scores[i.id] for i in bench.items if i.id not in injected]
    true_delta = sum(all_s) / len(all_s) - sum(clean_s) / len(clean_s)

    ev = aggregate(bench, res.corpus, seed=1)
    dres = delta(ev, scores, tau=TAU)
    assert abs(dres.delta - true_delta) <= 0.03, (
        f"Delta_hat {dres.delta} vs true {true_delta}"
    )
    assert dres.delta > 0  # contamination inflates the score


def test_delta_zero_when_no_contamination():
    bench = make_benchmark(n=100, seed=7)
    clean = make_clean_corpus(n_docs=300, seed=11)
    res = inject(clean, bench, rate=0.0, form="verbatim", seed=0)
    scores = _simulate_scores(bench, set(), seed=0)
    ev = aggregate(bench, res.corpus, seed=1)
    dres = delta(ev, scores, tau=TAU)
    assert abs(dres.delta) < 1e-9
    assert dres.n_dropped == 0


def test_rho_monotonic_in_rate():
    bench = make_benchmark(n=200, seed=7)
    clean = make_clean_corpus(n_docs=600, seed=11)
    rhos = []
    for rate in (0.0, 0.05, 0.2):
        res = inject(clean, bench, rate=rate, form="verbatim", seed=42)
        ev = aggregate(bench, res.corpus, seed=1)
        rhos.append(rho(ev, tau=TAU))
    assert rhos[0] <= rhos[1] <= rhos[2]
