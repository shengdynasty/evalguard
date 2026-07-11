"""Embedding retrieval matcher (design section 5.2).

Encode each benchmark item and each corpus doc, then nearest-neighbour search
D for near-duplicates above a cosine threshold. Catches lightly-edited copies
(paraphrase, format-shift) that break exact n-gram matching.

Real embeddings:
`try_sentence_transformer()` attempts to load sentence-transformers
'all-MiniLM-L6-v2'. When it is installed and the model is available it is wired
in behind the same .encode() interface used by the TF-IDF stand-in. If the
package is missing or the model cannot be loaded (offline / too heavy for a
CPU-only box), we fall back to the TF-IDF character/word n-gram encoder, which
is a lightweight, dependency-light proxy for a sentence-embedding model. The
choice is reported so the run is honest about which encoder produced the numbers.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..benchmark import BenchmarkItem
from ..corpus import Corpus
from . import Evidence


class _Embedder:
    """TF-IDF stand-in for a real embedding model. L2-normalized rows so that
    a dot product == cosine similarity."""

    def __init__(self):
        # word 1-2 grams + char 3-5 grams => robust to light rewording/typos.
        self._word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "_Embedder":
        self._word.fit(texts)
        self._char.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        w = self._word.transform(texts)
        c = self._char.transform(texts)
        from scipy.sparse import hstack

        m = hstack([w, c]).tocsr()
        # L2 normalize
        norms = np.sqrt(m.multiply(m).sum(axis=1))
        norms = np.asarray(norms).ravel()
        norms[norms == 0] = 1.0
        from scipy.sparse import diags

        return diags(1.0 / norms) @ m


class _SentenceTransformerEmbedder:
    """Real sentence-embedding encoder (all-MiniLM-L6-v2) behind the same
    .fit()/.encode() interface as the TF-IDF stand-in. Rows are L2-normalized so
    a dot product == cosine similarity."""

    def __init__(self, model):
        self._model = model
        self.backend = "sentence-transformers/all-MiniLM-L6-v2"

    def fit(self, texts):  # nothing to fit for a pretrained encoder
        return self

    def encode(self, texts):
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs)


def try_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """One honest attempt to load real embeddings. Returns an embedder or None.

    Never raises: a missing package or a failed/too-heavy model download simply
    yields None and the caller falls back to the TF-IDF stand-in.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        model = SentenceTransformer(model_name, device="cpu")
        return _SentenceTransformerEmbedder(model)
    except Exception:
        return None


class EmbeddingMatcher:
    name = "embedding"

    def __init__(self, sim_floor: float | None = None, embedder=None, use_real: bool = False):
        # sim_floor scales raw cosine to a [0,1] evidence score: cosine at or
        # below sim_floor -> ~0 evidence; cosine of 1.0 -> 1.0 evidence.
        # Default comes from the calibrated shaping constants (evalguard.config),
        # not a hardcoded magic number.
        if sim_floor is None:
            from ..config import load_shaping
            sim_floor = load_shaping()["embedding_sim_floor"]
        self.sim_floor = sim_floor
        # Encoder selection: an explicit embedder wins; else optionally try real
        # sentence-transformers; else the TF-IDF stand-in.
        if embedder is not None:
            self._embedder = embedder
        elif use_real:
            self._embedder = try_sentence_transformer() or _Embedder()
        else:
            self._embedder = _Embedder()
        self.backend = getattr(self._embedder, "backend", "tfidf-stand-in")
        self._doc_mat = None
        self._doc_ids: List[str] = []

    def fit(self, corpus: Corpus) -> "EmbeddingMatcher":
        self._doc_ids = list(corpus.doc_ids)
        self._embedder.fit(list(corpus.docs))
        self._doc_mat = self._embedder.encode(list(corpus.docs))
        return self

    def match(self, item: BenchmarkItem) -> Evidence:
        if self._doc_mat is None or self._doc_mat.shape[0] == 0:
            return Evidence(self.name, 0.0, detail="empty corpus")
        q = self._embedder.encode([item.text()])
        sims = linear_kernel(q, self._doc_mat).ravel()
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        # rescale cosine -> evidence in [0,1]
        score = max(0.0, (best_sim - self.sim_floor) / (1.0 - self.sim_floor))
        score = min(1.0, score)
        best_doc = self._doc_ids[best_i] if best_sim > 0 else None
        detail = (
            f"cosine {best_sim:.2f} to {best_doc}"
            if best_doc
            else "no similar doc"
        )
        return Evidence(
            self.name,
            score=float(score),
            matched_doc_id=best_doc,
            detail=detail,
            extra={"cosine": best_sim},
        )
