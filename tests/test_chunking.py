from examprep.chunking import chunk_pages, split_sentences
from examprep.models import Page


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_chunks_respect_size_and_page():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    pages = [Page("a.txt", 1, text), Page("a.txt", 2, "Short page two.")]
    chunks = chunk_pages(pages, size=200, overlap=50)
    assert all(len(c.text) <= 200 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2 and chunks[-1].text == "Short page two."


def test_overlap_repeats_context():
    text = " ".join(f"S{i} is a sentence." for i in range(40))
    chunks = chunk_pages([Page("a.txt", 1, text)], size=150, overlap=60)
    assert len(chunks) > 2
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text


def test_very_long_sentence_is_hard_split():
    chunks = chunk_pages([Page("a.txt", 1, "x" * 1000)], size=300, overlap=50)
    assert len(chunks) == 4 and all(len(c.text) <= 300 for c in chunks)
