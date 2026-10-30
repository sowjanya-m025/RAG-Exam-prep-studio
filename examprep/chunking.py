"""Sentence-aware chunking. Chunks never cross page boundaries, so citations stay exact."""
import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .models import Chunk, Page

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def _tail(sentences: list[str], overlap: int) -> list[str]:
    keep, total = [], 0
    for s in reversed(sentences):
        if total + len(s) > overlap:
            break
        keep.insert(0, s)
        total += len(s) + 1
    return keep


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    out, cur, cur_len = [], [], 0
    for sentence in split_sentences(text):
        pieces = [sentence[i:i + size] for i in range(0, len(sentence), size)] or [sentence]
        for p in pieces:
            if cur and cur_len + len(p) + 1 > size:
                out.append(" ".join(cur))
                cur = _tail(cur, overlap)
                cur_len = sum(len(x) + 1 for x in cur)
                if cur_len + len(p) + 1 > size:
                    cur, cur_len = [], 0
            cur.append(p)
            cur_len += len(p) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_pages(pages: list[Page], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for text in _chunk_text(page.text, size, overlap):
            chunks.append(Chunk(len(chunks), page.source, page.number, text))
    return chunks
