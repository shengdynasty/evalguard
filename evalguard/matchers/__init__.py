"""Matcher bank (design section 5.2, open-data matchers).

A Matcher scans the corpus D for evidence that a benchmark item leaked into it.
Each matcher returns an Evidence object carrying a normalized score in [0, 1]
and a human-readable trail (which doc matched, what was matched). The
aggregator stacks these into the per-item contamination score c_i.

No single metric is trusted (design section 1.4): the ensemble is the product.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

from ..benchmark import BenchmarkItem
from ..corpus import Corpus


@dataclass
class Evidence:
    """One matcher's finding for one benchmark item."""

    matcher: str
    score: float                       # normalized signal in [0, 1]
    matched_doc_id: str | None = None  # which corpus doc fired
    detail: str = ""                   # human-readable "why"
    extra: dict = field(default_factory=dict)

    def fired(self, threshold: float) -> bool:
        return self.score >= threshold


@runtime_checkable
class Matcher(Protocol):
    """Interface every matcher implements.

    fit() is given the (possibly contaminated) corpus once; match() is called
    per benchmark item and returns this matcher's Evidence.
    """

    name: str

    def fit(self, corpus: Corpus) -> "Matcher":
        ...

    def match(self, item: BenchmarkItem) -> Evidence:
        ...


from .ngram import NGramMatcher            # noqa: E402
from .embedding import EmbeddingMatcher    # noqa: E402
from .paraphrase import ParaphraseMatcher  # noqa: E402
from .answer import AnswerMatcher          # noqa: E402

__all__ = [
    "Evidence",
    "Matcher",
    "NGramMatcher",
    "EmbeddingMatcher",
    "ParaphraseMatcher",
    "AnswerMatcher",
]
