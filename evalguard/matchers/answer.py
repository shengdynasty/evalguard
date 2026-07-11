"""Answer-leak matcher (open-data; targets the `answer_only` contamination form).

The `answer_only` leak puts ONLY the item's answer into D (no question). The
Q/A-oriented matchers barely see it because they compare the whole item text
against a doc that contains just the answer. This matcher looks specifically for
the item's ANSWER tokens co-occurring in a single corpus doc, weighted by IDF so
that a distinctive answer ("Thexesca-160") fires while a generic one ("441")
does not (keeping precision).

This is honest per design section 4.1: a generic answer that also appears in
clean text SHOULD get low evidence - we cannot claim membership from a common
string. Detectability of answer-only leakage is therefore bounded by how
distinctive the answer is, and the matcher reflects that rather than hiding it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from ..benchmark import BenchmarkItem
from ..corpus import Corpus
from . import Evidence

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "of", "a", "an", "is", "to", "in", "on", "at", "and", "or",
         "captain", "note", "answer"}


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


class AnswerMatcher:
    name = "answer"

    def __init__(self, threshold: float | None = None):
        # score-shaping floor (calibrated via evalguard.config), not hardcoded.
        if threshold is None:
            from ..config import load_shaping
            threshold = load_shaping()["answer_floor"]
        self.threshold = threshold
        self._doc_bags: List[Counter] = []
        self._doc_ids: List[str] = []
        self._idf: Dict[str, float] = {}
        self._n = 0

    def fit(self, corpus: Corpus) -> "AnswerMatcher":
        self._doc_ids = list(corpus.doc_ids)
        self._doc_bags = [Counter(_tokens(d)) for d in corpus.docs]
        self._n = max(1, len(self._doc_bags))
        df: Counter = Counter()
        for bag in self._doc_bags:
            for tok in bag:
                df[tok] += 1
        self._idf = {t: math.log((1 + self._n) / (1 + d)) + 1.0 for t, d in df.items()}
        return self

    def _idf_of(self, tok: str) -> float:
        return self._idf.get(tok, math.log(1 + self._n) + 1.0)

    def match(self, item: BenchmarkItem) -> Evidence:
        ans_tokens = set(_tokens(item.answer))
        if not ans_tokens or not self._doc_bags:
            return Evidence(self.name, 0.0, detail="no answer tokens")
        total = sum(self._idf_of(t) for t in ans_tokens)
        best_conf = 0.0
        best_doc = None
        for di, bag in enumerate(self._doc_bags):
            if not bag:
                continue
            matched = sum(self._idf_of(t) for t in ans_tokens if t in bag)
            conf = matched / total if total else 0.0
            if conf > best_conf:
                best_conf = conf
                best_doc = self._doc_ids[di]
        score = max(0.0, (best_conf - self.threshold) / (1.0 - self.threshold))
        score = min(1.0, score)
        detail = (
            f"answer tokens {best_conf:.0%} present (IDF-wt) in {best_doc}"
            if best_doc and score > 0
            else "answer not found in corpus"
        )
        return Evidence(
            self.name,
            score=float(score),
            matched_doc_id=best_doc if score > 0 else None,
            detail=detail,
            extra={"raw_conf": best_conf},
        )
