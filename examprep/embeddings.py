"""Embedding backends. Both expose embed_docs(list[str]) and embed_query(str) -> L2-normalised vectors."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import EMBEDDING_BACKEND, EMBEDDING_MODEL


class TfidfEmbedder:
    """Lightweight lexical fallback. Needs no downloads; used for tests and offline demos."""

    name = "tfidf"

    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return normalize(self.vec.fit_transform(texts)).toarray()

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(self.vec.transform([text])).toarray()[0]


class SbertEmbedder:
    """Semantic embeddings via sentence-transformers (downloads the model on first use)."""

    name = "sbert"

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0])


def get_embedder(backend: str | None = None):
    backend = (backend or EMBEDDING_BACKEND).lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "sbert":
        return SbertEmbedder()
    try:  # auto
        return SbertEmbedder()
    except Exception:
        return TfidfEmbedder()
