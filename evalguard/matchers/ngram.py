"""Exact / n-gram overlap matcher via MinHash-LSH over D (design section 5.2).

This is the GPT-3/PaLM-era baseline: cheap, high precision, low recall. It
catches copy-paste (verbatim) contamination and, thanks to the containment
score, format-shifts that preserve token n-grams. It is defeated by
paraphrasing - which is exactly why the ensemble also has an embedding and a
paraphrase matcher.

MinHash-LSH is implemented from scratch (no datasketch dependency) so the
package stays dependency-light: numpy only. The LSH banding gives an
approximate-nearest-neighbour shortlist; we then compute an exact containment
score on the shortlist.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import numpy as np

from ..benchmark import BenchmarkItem
from ..corpus import Corpus
from . import Evidence

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _shingles(text: str, k: int) -> Set[str]:
    toks = _tokens(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


class _MinHasher:
    """Deterministic MinHash over string shingles using random hash coeffs."""

    def __init__(self, num_perm: int = 64, seed: int = 1):
        self.num_perm = num_perm
        rng = np.random.RandomState(seed)
        self._mersenne = (1 << 61) - 1
        self._a = rng.randint(1, self._mersenne, size=num_perm, dtype=np.int64)
        self._b = rng.randint(0, self._mersenne, size=num_perm, dtype=np.int64)

    def signature(self, shingles: Set[str]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, self._mersenne, dtype=np.int64)
        # hash each shingle to a 64-bit int deterministically
        base = np.array(
            [hash_str(s) for s in shingles], dtype=np.int64
        ).reshape(-1, 1)
        # (num_shingles, num_perm) permuted hashes; take column-wise min
        permuted = (self._a * base + self._b) % self._mersenne
        return permuted.min(axis=0)


def hash_str(s: str) -> int:
    """Stable 63-bit hash of a string (independent of PYTHONHASHSEED)."""
    import hashlib

    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


class NGramMatcher:
    name = "ngram"

    def __init__(self, k: int = 5, num_perm: int = 64, bands: int = 16, seed: int = 1):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.k = k
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self._hasher = _MinHasher(num_perm=num_perm, seed=seed)
        self._doc_shingles: List[Set[str]] = []
        self._doc_ids: List[str] = []
        self._buckets: List[Dict[Tuple, List[int]]] = [dict() for _ in range(bands)]

    def fit(self, corpus: Corpus) -> "NGramMatcher":
        self._doc_shingles = []
        self._doc_ids = list(corpus.doc_ids)
        self._buckets = [dict() for _ in range(self.bands)]
        for di, doc in enumerate(corpus.docs):
            sh = _shingles(doc, self.k)
            self._doc_shingles.append(sh)
            sig = self._hasher.signature(sh)
            for b in range(self.bands):
                band = tuple(sig[b * self.rows : (b + 1) * self.rows].tolist())
                self._buckets[b].setdefault(band, []).append(di)
        return self

    def _candidates(self, sig: np.ndarray) -> Set[int]:
        cand: Set[int] = set()
        for b in range(self.bands):
            band = tuple(sig[b * self.rows : (b + 1) * self.rows].tolist())
            cand.update(self._buckets[b].get(band, ()))
        return cand

    def match(self, item: BenchmarkItem) -> Evidence:
        q_shingles = _shingles(item.text(), self.k)
        if not q_shingles:
            return Evidence(self.name, 0.0, detail="empty item")
        sig = self._hasher.signature(q_shingles)
        cands = self._candidates(sig)

        best_score = 0.0
        best_doc = None
        for di in cands:
            doc_sh = self._doc_shingles[di]
            if not doc_sh:
                continue
            inter = len(q_shingles & doc_sh)
            # containment: fraction of the item's shingles present in the doc.
            # Robust when the item is embedded inside a larger training doc.
            containment = inter / len(q_shingles)
            if containment > best_score:
                best_score = containment
                best_doc = self._doc_ids[di]

        detail = (
            f"{best_score:.0%} of {self.k}-gram shingles found in {best_doc}"
            if best_doc
            else "no n-gram overlap"
        )
        return Evidence(
            self.name,
            score=float(best_score),
            matched_doc_id=best_doc,
            detail=detail,
            extra={"k": self.k},
        )
