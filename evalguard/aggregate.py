"""Aggregator (design section 5.3): combine matcher evidence into c_i in [0,1].

Trust comes from showing the receipt, not from a black-box score: every c_i is
returned WITH the contributing evidence (which matcher fired, matched doc,
detail) - the "contamination path". This is also where the honesty principle of
section 4.1 lives: this open-data build reports per-item membership evidence and
never claims more than the corpus evidence supports.

The combiner is a simple, transparent weighted-max-style rule rather than an
opaque learned stack, so the aggregation itself is auditable. (A learned
combiner can be dropped in later; the design allows it, but for a trustworthy
v1 the transparent rule is preferable.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .benchmark import Benchmark, BenchmarkItem
from .corpus import Corpus
from .matchers import (
    Evidence,
    EmbeddingMatcher,
    NGramMatcher,
    ParaphraseMatcher,
    AnswerMatcher,
)

# Default matcher weights. n-gram is high-precision so it dominates when it
# fires strongly; embedding + paraphrase lift recall on reworded / shifted items.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ngram": 1.0,
    "embedding": 0.9,
    "paraphrase": 0.95,
    "answer": 0.85,
}


@dataclass
class ItemEvidence:
    """The per-item contamination score c_i plus its full evidence trail."""

    item_id: str
    c: float
    evidence: List[Evidence] = field(default_factory=list)

    def is_contaminated(self, tau: float) -> bool:
        return self.c > tau

    def path(self) -> List[str]:
        """Human-readable contamination path (only matchers that contributed)."""
        out = []
        for ev in sorted(self.evidence, key=lambda e: e.score, reverse=True):
            if ev.score > 0.01:
                out.append(f"[{ev.matcher} {ev.score:.2f}] {ev.detail}")
        return out


def build_matchers(seed: int = 1) -> List:
    """Construct the default open-data matcher bank."""
    return [
        NGramMatcher(seed=seed),
        EmbeddingMatcher(),
        ParaphraseMatcher(),
        AnswerMatcher(),
    ]


def _combine(evs: List[Evidence], weights: Dict[str, float]) -> float:
    """Transparent combiner: weighted noisy-OR of the per-matcher signals.

    noisy-OR rewards agreement (two weak matchers firing raise confidence) while
    staying in [0,1] and never exceeding what any single strong matcher asserts
    by much. This keeps c_i interpretable as 'probability the item is in D'.
    """
    prod = 1.0
    for ev in evs:
        w = weights.get(ev.matcher, 0.5)
        p = max(0.0, min(1.0, ev.score)) * w
        prod *= (1.0 - p)
    return 1.0 - prod


def aggregate(
    benchmark: Benchmark,
    corpus: Corpus,
    matchers: List | None = None,
    weights: Dict[str, float] | None = None,
    seed: int = 1,
) -> List[ItemEvidence]:
    """Run every matcher over the corpus and combine into per-item c_i.

    Returns one ItemEvidence per benchmark item, each carrying its evidence
    trail.
    """
    matchers = matchers if matchers is not None else build_matchers(seed=seed)
    weights = weights if weights is not None else DEFAULT_WEIGHTS

    for m in matchers:
        m.fit(corpus)

    results: List[ItemEvidence] = []
    for item in benchmark.items:
        evs = [m.match(item) for m in matchers]
        c = _combine(evs, weights)
        results.append(ItemEvidence(item_id=item.id, c=c, evidence=evs))
    return results
