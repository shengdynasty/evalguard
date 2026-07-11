"""Injection harness produces correct ground-truth counts and is deterministic."""
import pytest

from evalguard.inject import FORMS, inject
from evalguard.synth import make_benchmark, make_clean_corpus


def _fixtures(n=100, docs=200):
    return make_benchmark(n=n, seed=7), make_clean_corpus(n_docs=docs, seed=11)


def test_rate_controls_injected_count():
    bench, clean = _fixtures(n=100)
    for rate in (0.0, 0.01, 0.05, 0.2):
        res = inject(clean, bench, rate=rate, form="verbatim", seed=3)
        expected = int(round(rate * len(bench)))
        assert len(res.injected_ids) == expected
        assert abs(res.true_rho(len(bench)) - expected / len(bench)) < 1e-9


def test_corpus_grows_by_injected_count():
    bench, clean = _fixtures(n=100)
    res = inject(clean, bench, rate=0.2, form="verbatim", seed=1)
    assert len(res.corpus) == len(clean) + len(res.injected_ids)
    # clean corpus is not mutated
    assert len(clean) == 200


def test_labels_match_injected_docs():
    bench, clean = _fixtures(n=100)
    res = inject(clean, bench, rate=0.1, form="paraphrase", seed=5)
    for lab in res.labels:
        assert lab.form == "paraphrase"
        assert lab.injected_doc_id in res.corpus.doc_ids


def test_all_forms_supported():
    bench, clean = _fixtures(n=50)
    for form in FORMS:
        res = inject(clean, bench, rate=0.2, form=form, seed=2)
        assert len(res.injected_ids) == 10
        assert all(lab.form == form for lab in res.labels)


def test_deterministic_given_seed():
    bench, clean = _fixtures(n=100)
    a = inject(clean, bench, rate=0.2, form="verbatim", seed=42)
    b = inject(clean, bench, rate=0.2, form="verbatim", seed=42)
    assert a.injected_ids == b.injected_ids
    assert a.corpus.docs == b.corpus.docs
    c = inject(clean, bench, rate=0.2, form="verbatim", seed=43)
    assert a.injected_ids != c.injected_ids  # different seed -> different pick


def test_form_mix_distributes_forms():
    bench, clean = _fixtures(n=200)
    mix = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}
    res = inject(clean, bench, rate=0.5, form_mix=mix, seed=9)
    by = res.ids_by_form()
    assert all(len(by[f]) > 0 for f in mix)
    assert sum(len(v) for v in by.values()) == len(res.injected_ids)


def test_answer_only_omits_question():
    bench, clean = _fixtures(n=20)
    res = inject(clean, bench, rate=1.0, form="answer_only", seed=0)
    for lab in res.labels:
        item = bench.by_id(lab.item_id)
        doc = res.corpus.docs[res.corpus.doc_ids.index(lab.injected_doc_id)]
        assert item.answer in doc
        # the question text should NOT be present verbatim
        assert item.question not in doc


def test_translation_preserves_answer_content():
    bench, clean = _fixtures(n=20)
    res = inject(clean, bench, rate=1.0, form="translation", seed=0)
    for lab in res.labels:
        item = bench.by_id(lab.item_id)
        doc = res.corpus.docs[res.corpus.doc_ids.index(lab.injected_doc_id)]
        assert lab.form == "translation"
        assert "Pregunta:" in doc
        assert "Respuesta:" in doc


def test_bad_rate_and_form_rejected():
    bench, clean = _fixtures(n=10)
    with pytest.raises(ValueError):
        inject(clean, bench, rate=1.5)
    with pytest.raises(ValueError):
        inject(clean, bench, rate=0.1, form="nonsense")
