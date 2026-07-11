"""Matcher behaviour: n-gram catches verbatim with high recall; paraphrase
matcher lifts recall on reworded items; precision stays high on clean items."""
from evalguard.aggregate import aggregate
from evalguard.estimands import contaminated_ids
from evalguard.inject import inject
from evalguard.matchers import (
    EmbeddingMatcher,
    NGramMatcher,
    ParaphraseMatcher,
)
from evalguard.synth import make_benchmark, make_clean_corpus

TAU = 0.5


def _recall(matcher, form, rate=0.3, thresh=0.5):
    bench = make_benchmark(n=120, seed=7)
    clean = make_clean_corpus(n_docs=300, seed=11)
    res = inject(clean, bench, rate=rate, form=form, seed=4)
    truth = res.injected_ids
    m = matcher.fit(res.corpus)
    caught = set()
    for item in bench.items:
        if m.match(item).score >= thresh:
            caught.add(item.id)
    tp = len(caught & truth)
    fp = len(caught - truth)
    rec = tp / len(truth) if truth else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    return rec, prec


def test_ngram_catches_verbatim_high_recall():
    rec, prec = _recall(NGramMatcher(), "verbatim", thresh=0.8)
    assert rec >= 0.95, f"ngram verbatim recall {rec}"
    assert prec >= 0.95, f"ngram verbatim precision {prec}"


def test_ngram_weak_on_paraphrase():
    # The whole point of the ensemble: n-gram alone MISSES rewordings.
    rec_v, _ = _recall(NGramMatcher(), "verbatim", thresh=0.8)
    rec_p, _ = _recall(NGramMatcher(), "paraphrase", thresh=0.8)
    assert rec_p < rec_v, "n-gram should lose recall on paraphrase vs verbatim"


def test_paraphrase_matcher_lifts_paraphrase_recall():
    _, ng_prec = _recall(NGramMatcher(), "paraphrase", thresh=0.8)
    ng_rec, _ = _recall(NGramMatcher(), "paraphrase", thresh=0.8)
    pp_rec, pp_prec = _recall(ParaphraseMatcher(), "paraphrase", thresh=0.5)
    assert pp_rec > ng_rec, f"paraphrase matcher {pp_rec} should beat ngram {ng_rec}"
    assert pp_rec >= 0.8
    assert pp_prec >= 0.9


def test_embedding_catches_format_shift():
    rec, prec = _recall(EmbeddingMatcher(), "format_shift", thresh=0.4)
    assert rec >= 0.8, f"embedding format_shift recall {rec}"
    assert prec >= 0.9


def test_ensemble_precision_recall_on_mixed_forms():
    bench = make_benchmark(n=200, seed=7)
    clean = make_clean_corpus(n_docs=600, seed=11)
    res = inject(
        clean, bench, rate=0.2,
        form_mix={"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1},
        seed=42,
    )
    truth = res.injected_ids
    ev = aggregate(bench, res.corpus, seed=1)
    pred = set(contaminated_ids(ev, tau=TAU))
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    assert prec >= 0.9, f"ensemble precision {prec}"
    assert rec >= 0.9, f"ensemble recall {rec}"


def test_no_contamination_gives_no_flags():
    bench = make_benchmark(n=80, seed=7)
    clean = make_clean_corpus(n_docs=200, seed=11)
    # rate 0 -> corpus unchanged; nothing should be flagged
    res = inject(clean, bench, rate=0.0, form="verbatim", seed=0)
    ev = aggregate(bench, res.corpus, seed=1)
    pred = contaminated_ids(ev, tau=TAU)
    assert pred == [], f"false positives on clean corpus: {pred}"


def test_evidence_trail_present():
    bench = make_benchmark(n=40, seed=7)
    clean = make_clean_corpus(n_docs=100, seed=11)
    res = inject(clean, bench, rate=0.5, form="verbatim", seed=1)
    ev = aggregate(bench, res.corpus, seed=1)
    flagged = [e for e in ev if e.c > TAU]
    assert flagged
    for e in flagged:
        # every flagged item carries a non-empty contamination path with a
        # matched doc id (the receipt).
        path = e.path()
        assert path, "flagged item missing contamination path"
        assert any(x.matched_doc_id for x in e.evidence)
