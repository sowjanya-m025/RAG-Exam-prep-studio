"""Shared data models. This is the contract between the retrieval side and the LLM side."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    source: str   # file name
    number: int   # 1-based page number (virtual pages for .txt/.md)
    text: str


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}, p.{self.page}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
