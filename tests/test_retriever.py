from pathlib import Path

import pytest

from examprep.ingestion import load_file
from examprep.retriever import Retriever
from examprep.vectorstore import VectorStore

NOTES = Path(__file__).resolve().parent.parent / "sample_data" / "sample_notes.txt"


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_pages(load_file(NOTES.name, NOTES.read_bytes()), backend="tfidf", size=500, overlap=80)


def test_top_hit_is_relevant(retriever):
    hits = retriever.retrieve("what is overfitting", k=3, diverse=False)
    assert "overfitting" in hits[0].chunk.text.lower()
    assert hits[0].score >= hits[-1].score


def test_mmr_returns_k_unique_chunks(retriever):
    hits = retriever.retrieve("gradient descent learning rate", k=4, diverse=True)
    ids = [h.chunk.id for h in hits]
    assert len(ids) == 4 and len(set(ids)) == 4


def test_k_larger_than_corpus_is_safe(retriever):
    assert len(retriever.retrieve("anything", k=999)) == retriever.n_chunks


def test_unknown_query_does_not_crash(retriever):
    assert retriever.retrieve("zzzzqqq", k=3)


def test_store_save_load_roundtrip(retriever, tmp_path):
    retriever.store.save(str(tmp_path))
    loaded = VectorStore.load(str(tmp_path))
    assert [c.text for c in loaded.chunks] == [c.text for c in retriever.store.chunks]
    assert loaded.matrix.shape == retriever.store.matrix.shape


def test_empty_pages_rejected():
    with pytest.raises(ValueError):
        Retriever.from_pages([], backend="tfidf")
