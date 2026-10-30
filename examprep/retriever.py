"""Retriever: the one object the LLM side talks to.  retriever.retrieve(query, k) -> list[Hit]"""
from .chunking import chunk_pages
from .config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from .embeddings import get_embedder
from .models import Hit, Page
from .vectorstore import VectorStore


class Retriever:
    def __init__(self, embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    @classmethod
    def from_pages(cls, pages: list[Page], backend: str | None = None,
                   size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> "Retriever":
        chunks = chunk_pages(pages, size, overlap)
        if not chunks:
            raise ValueError("No text to index")
        embedder = get_embedder(backend)
        store = VectorStore()
        store.add(chunks, embedder.embed_docs([c.text for c in chunks]))
        return cls(embedder, store)

    def retrieve(self, query: str, k: int = TOP_K, diverse: bool = True) -> list[Hit]:
        qvec = self.embedder.embed_query(query)
        k = min(k, len(self.store.chunks))
        return self.store.search_mmr(qvec, k) if diverse else self.store.search(qvec, k)

    @property
    def n_chunks(self) -> int:
        return len(self.store.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.store.chunks})
