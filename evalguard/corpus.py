"""The training corpus D: a list of documents we can search over.

In the open-data regime the buyer (fine-tuning team) literally has this corpus,
so membership of a benchmark item is a search problem rather than a black-box
inference problem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class Corpus:
    """A collection of training documents.

    docs[i] is the raw text of document i. doc_ids[i] is a stable identifier.
    """

    docs: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.doc_ids:
            self.doc_ids = [f"doc-{i}" for i in range(len(self.docs))]
        if len(self.doc_ids) != len(self.docs):
            raise ValueError("doc_ids and docs must be the same length")

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self) -> Iterable[str]:
        return iter(self.docs)

    def add(self, text: str, doc_id: str | None = None) -> None:
        self.docs.append(text)
        self.doc_ids.append(doc_id if doc_id is not None else f"doc-{len(self.docs) - 1}")

    def copy(self) -> "Corpus":
        return Corpus(list(self.docs), list(self.doc_ids))


def load_corpus(path: str | Path) -> Corpus:
    """Load a corpus from a .jsonl (one {"id","text"} per line) or .txt file.

    .txt files are split into documents on blank lines.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        docs, ids = [], []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs.append(obj.get("text", ""))
                ids.append(str(obj.get("id", f"doc-{i}")))
        return Corpus(docs, ids)
    text = path.read_text(encoding="utf-8")
    docs = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return Corpus(docs)


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for doc_id, text in zip(corpus.doc_ids, corpus.docs):
            fh.write(json.dumps({"id": doc_id, "text": text}) + "\n")
