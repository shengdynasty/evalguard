"""Rule-based paraphrase judge (design section 5.2).

For borderline retrieval hits, decide "is this corpus doc the same benchmark
item, just reworded?" This is the matcher that catches the rephrasing attack
(Yang et al.'s "rephrased samples" problem) that defeats pure n-gram overlap.

Key idea for keeping precision up: a real "same item?" judge does NOT key on
boilerplate ("which delegation proposed the ... clause") shared by many items;
it keys on the SPECIFIC, rare content (the named entities, the exact answer).
We approximate that with IDF-weighted content overlap over the corpus, so
distinctive tokens dominate the judgment and template siblings do not trigger a
false positive.

# SWAP: real LLM judge.
# Replace `_judge()` with an LLM call:
#   "Here is a benchmark item and a candidate training passage. Are they the
#    same question/answer, merely reworded? Answer yes/no + confidence 0-1."
# behind the same interface. The matcher only needs a score in [0,1].
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Set

from ..benchmark import BenchmarkItem
from ..corpus import Corpus
from . import Evidence

_WORD_RE = re.compile(r"[a-z0-9]+")

# Normalization map so the injector's synonym reword ("seat of government" <->
# "capital") is seen through. A real LLM judge would not need this.
_NORMALIZE: Dict[str, str] = {
    "seat": "capital", "government": "capital",
    "biggest": "large", "largest": "large",
    "tiniest": "small", "smallest": "small",
    "calculate": "compute", "quantity": "value", "figure": "number",
    "writer": "author", "penned": "wrote", "started": "began",
    "finished": "ended", "annum": "year", "nation": "country",
    "town": "city", "waterway": "river", "world": "planet",
    "substance": "element", "which": "what", "person": "who",
    "question": "q", "prompt": "q",
    "correct": "a", "response": "a", "answer": "a", "expected": "a",
    "output": "a", "note": "", "the": "",
}

_STOP: Set[str] = {"the", "of", "a", "an", "is", "to", "in", "on", "at", "and",
                   "or", "", "which", "who", "what", "did", "before", "first"}


def _content_tokens(text: str) -> List[str]:
    out = []
    for w in _WORD_RE.findall(text.lower()):
        w = _NORMALIZE.get(w, w)
        if w and w not in _STOP:
            out.append(w)
    return out


class ParaphraseMatcher:
    name = "paraphrase"

    def __init__(self, shortlist: int = 25, min_overlap: float = 0.2, threshold: float | None = None):
        # shortlist: judge only the top-N lexically closest docs (an LLM judge
        # is expensive; run it only on borderline retrieval hits).
        self.shortlist = shortlist
        self.min_overlap = min_overlap
        # score-shaping floor (calibrated, not a magic number): confidence below
        # this template-sibling level maps toward 0 evidence.
        if threshold is None:
            from ..config import load_shaping
            threshold = load_shaping()["paraphrase_floor"]
        self.threshold = threshold
        self._docs: List[str] = []
        self._doc_ids: List[str] = []
        self._doc_bags: List[Counter] = []
        self._idf: Dict[str, float] = {}

    def fit(self, corpus: Corpus) -> "ParaphraseMatcher":
        self._docs = list(corpus.docs)
        self._doc_ids = list(corpus.doc_ids)
        self._doc_bags = [Counter(_content_tokens(d)) for d in self._docs]
        # IDF over the corpus: rare tokens (unique entities, answers) get high
        # weight; boilerplate template words get near-zero weight.
        n = max(1, len(self._docs))
        df: Counter = Counter()
        for bag in self._doc_bags:
            for tok in bag:
                df[tok] += 1
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        return self

    def _idf_of(self, tok: str) -> float:
        # unseen-in-corpus tokens are maximally distinctive
        return self._idf.get(tok, math.log(1 + max(1, len(self._docs))) + 1.0)

    def _judge(self, item_tokens: List[str], doc_bag: Counter) -> float:
        """IDF-weighted containment of the item's content in the doc, in [0,1]."""
        item_set = set(item_tokens)
        if not item_set:
            return 0.0
        total = sum(self._idf_of(t) for t in item_set)
        matched = sum(self._idf_of(t) for t in item_set if t in doc_bag)
        return matched / total if total else 0.0

    def match(self, item: BenchmarkItem) -> Evidence:
        if not self._docs:
            return Evidence(self.name, 0.0, detail="empty corpus")
        item_tokens = _content_tokens(item.text())
        item_set = set(item_tokens)
        if not item_set:
            return Evidence(self.name, 0.0, detail="empty item")

        # cheap unweighted prefilter to build the judge shortlist
        scored = []
        for di, bag in enumerate(self._doc_bags):
            if not bag:
                continue
            overlap = len(item_set & set(bag)) / len(item_set)
            if overlap > self.min_overlap:
                scored.append((overlap, di))
        scored.sort(reverse=True)
        shortlist = scored[: self.shortlist]

        best_conf = 0.0
        best_doc = None
        for _, di in shortlist:
            conf = self._judge(item_tokens, self._doc_bags[di])
            if conf > best_conf:
                best_conf = conf
                best_doc = self._doc_ids[di]

        # rescale so that confidence below `threshold` (template-sibling level)
        # maps toward 0 and a near-perfect specific match maps toward 1.
        score = max(0.0, (best_conf - self.threshold) / (1.0 - self.threshold))
        score = min(1.0, score)
        detail = (
            f"judge: {best_conf:.0%} IDF-weighted content match (reworded) in {best_doc}"
            if best_doc and score > 0
            else "no paraphrase match"
        )
        return Evidence(
            self.name,
            score=float(score),
            matched_doc_id=best_doc if score > 0 else None,
            detail=detail,
            extra={"raw_conf": best_conf},
        )
