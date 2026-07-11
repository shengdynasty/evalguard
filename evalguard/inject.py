"""The injection harness - the core research artifact (design doc section 6).

Given a clean corpus D and a benchmark B, inject benchmark items into D at a
controlled RATE and in a controlled FORM, returning:
  - the contaminated corpus, and
  - the ground-truth label set (which item ids were injected, in what form).

Because we manufacture the contamination, we KNOW the true contamination rate
rho and the true injected-item set by construction. That is what lets the demo
prove the estimands are recoverable.

Supported forms (design section 6):
  verbatim     - exact copy of the item text
  paraphrase   - light deterministic reword (synonyms + word-order perturbation)
  format_shift - Q/A reformatted, whitespace / casing changed
  answer_only  - only the answer leaks, not the question

Everything is deterministic given a seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from .benchmark import Benchmark, BenchmarkItem
from .corpus import Corpus

FORMS = ("verbatim", "paraphrase", "format_shift", "answer_only")

# A tiny deterministic synonym table for the paraphrase transform.
# SWAP: a real paraphrase attack would use an LLM rephrasing; this rule-based
# reword is enough to defeat exact n-gram match while preserving meaning, which
# is exactly the "rephrased samples" problem (Yang et al.) we want to test.
_SYNONYMS: Dict[str, str] = {
    "capital": "seat of government",
    "largest": "biggest",
    "smallest": "tiniest",
    "compute": "calculate",
    "value": "quantity",
    "number": "figure",
    "author": "writer",
    "wrote": "penned",
    "began": "started",
    "ended": "finished",
    "year": "annum",
    "country": "nation",
    "city": "town",
    "river": "waterway",
    "planet": "world",
    "element": "substance",
    "what": "which",
    "who": "which person",
    "the": "the",
    "of": "of",
}


@dataclass
class InjectionLabel:
    """Ground-truth record for one injected item."""

    item_id: str
    form: str
    injected_doc_id: str


@dataclass
class InjectionResult:
    corpus: Corpus
    labels: List[InjectionLabel] = field(default_factory=list)

    @property
    def injected_ids(self) -> set:
        return {lab.item_id for lab in self.labels}

    def form_of(self, item_id: str) -> str | None:
        for lab in self.labels:
            if lab.item_id == item_id:
                return lab.form
        return None

    def true_rho(self, n_benchmark: int) -> float:
        if n_benchmark == 0:
            return 0.0
        return len(self.injected_ids) / n_benchmark

    def ids_by_form(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {f: [] for f in FORMS}
        for lab in self.labels:
            out.setdefault(lab.form, []).append(lab.item_id)
        return out


def _paraphrase(item: BenchmarkItem, rng: random.Random) -> str:
    """Light deterministic reword: synonym swap + mild word-order perturbation."""
    def reword(sentence: str) -> str:
        words = sentence.split()
        out = []
        for w in words:
            stripped = w.strip("?.,").lower()
            repl = _SYNONYMS.get(stripped)
            if repl is not None and repl != stripped:
                # preserve trailing punctuation
                tail = "".join(ch for ch in w if ch in "?.,")
                out.append(repl + tail)
            else:
                out.append(w)
        # mild local word-order perturbation: swap two adjacent interior words
        if len(out) > 4:
            i = 1 + rng.randrange(max(1, len(out) - 3))
            out[i], out[i + 1] = out[i + 1], out[i]
        return " ".join(out)

    q = reword(item.question)
    a = reword(item.answer)
    return f"Question: {q}\nCorrect response: {a}"


def _format_shift(item: BenchmarkItem) -> str:
    """Same content, reformatted: different labels, whitespace, casing."""
    return (
        f"[PROMPT]  {item.question.upper()}\n\n"
        f"[EXPECTED OUTPUT]  ->  {item.answer}\n"
    )


def inject(
    corpus: Corpus,
    benchmark: Benchmark,
    rate: float = 0.05,
    form: str = "verbatim",
    seed: int = 0,
    form_mix: Dict[str, float] | None = None,
) -> InjectionResult:
    """Inject benchmark items into a copy of `corpus`.

    Args:
        corpus: clean training corpus (not mutated).
        benchmark: benchmark whose items may leak.
        rate: fraction of benchmark items to inject (0.0 .. 1.0).
        form: contamination form to use for every injected item, unless
            `form_mix` is given.
        seed: RNG seed for deterministic selection & perturbation.
        form_mix: optional {form: weight} distribution. When provided, each
            injected item's form is sampled from this distribution instead of
            using the single `form`. Weights need not sum to 1.

    Returns:
        InjectionResult(contaminated_corpus, ground_truth_labels).
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    if form_mix is None and form not in FORMS:
        raise ValueError(f"unknown form {form!r}; choose from {FORMS}")
    if form_mix is not None:
        bad = set(form_mix) - set(FORMS)
        if bad:
            raise ValueError(f"unknown form(s) in form_mix: {bad}")

    rng = random.Random(seed)
    out = corpus.copy()

    n = len(benchmark)
    k = int(round(rate * n))
    order = list(range(n))
    rng.shuffle(order)
    chosen = sorted(order[:k])

    forms_pool = None
    weights = None
    if form_mix is not None:
        forms_pool = list(form_mix.keys())
        weights = [form_mix[f] for f in forms_pool]

    labels: List[InjectionLabel] = []
    for idx in chosen:
        item = benchmark.items[idx]
        this_form = form if form_mix is None else rng.choices(forms_pool, weights)[0]

        if this_form == "verbatim":
            text = item.text()
        elif this_form == "paraphrase":
            text = _paraphrase(item, rng)
        elif this_form == "format_shift":
            text = _format_shift(item)
        elif this_form == "answer_only":
            # Only the answer leaks - the question is NOT present. This is the
            # hardest case: there is no question text in D to retrieve against.
            text = f"Note: the answer is {item.answer}."
        else:  # pragma: no cover - guarded above
            raise ValueError(this_form)

        doc_id = f"injected::{item.id}::{this_form}"
        out.add(text, doc_id=doc_id)
        labels.append(InjectionLabel(item_id=item.id, form=this_form, injected_doc_id=doc_id))

    return InjectionResult(corpus=out, labels=labels)
