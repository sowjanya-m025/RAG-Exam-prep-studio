"""Tiny in-memory vector store (cosine similarity via dot product on normalised vectors)."""
import json
from pathlib import Path

import numpy as np

from .models import Chunk, Hit


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0))

    def add(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if len(chunks) != len(matrix):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks, self.matrix = list(chunks), np.asarray(matrix)

    def search(self, qvec: np.ndarray, k: int = 6) -> list[Hit]:
        scores = self.matrix @ qvec
        idx = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in idx]

    def search_mmr(self, qvec: np.ndarray, k: int = 6, fetch_k: int = 20, lambda_: float = 0.6) -> list[Hit]:
        """Maximal Marginal Relevance: relevant AND diverse passages (less duplicate context)."""
        scores = self.matrix @ qvec
        candidates = [int(i) for i in np.argsort(-scores)[:fetch_k]]
        chosen: list[int] = []
        while candidates and len(chosen) < k:
            if not chosen:
                best = candidates[0]
            else:
                selected = self.matrix[chosen]
                best = max(
                    candidates,
                    key=lambda i: lambda_ * scores[i] - (1 - lambda_) * float(np.max(selected @ self.matrix[i])),
                )
            chosen.append(best)
            candidates.remove(best)
        return [Hit(self.chunks[i], float(scores[i])) for i in chosen]

    def save(self, folder: str) -> None:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "matrix.npy", self.matrix)
        (path / "chunks.json").write_text(json.dumps([c.__dict__ for c in self.chunks]), encoding="utf-8")

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        path = Path(folder)
        store = cls()
        chunks = [Chunk(**d) for d in json.loads((path / "chunks.json").read_text(encoding="utf-8"))]
        store.add(chunks, np.load(path / "matrix.npy"))
        return store
