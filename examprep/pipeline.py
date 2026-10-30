"""Glue: uploaded files -> ready-to-use Retriever."""
from .ingestion import load_file
from .retriever import Retriever


def build_retriever(files: list[tuple[str, bytes]], backend: str | None = None) -> Retriever:
    pages = []
    for name, data in files:
        pages.extend(load_file(name, data))
    return Retriever.from_pages(pages, backend)
