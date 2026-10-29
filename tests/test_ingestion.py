import pytest

from examprep.ingestion import clean_text, load_file
from tests.helpers import make_pdf


def test_pdf_keeps_page_numbers():
    pages = load_file("notes.pdf", make_pdf(["Alpha page one", "Beta page two"]))
    assert [p.number for p in pages] == [1, 2]
    assert "Alpha" in pages[0].text and "Beta" in pages[1].text
    assert all(p.source == "notes.pdf" for p in pages)


def test_text_file_gets_virtual_pages():
    big = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(12))
    pages = load_file("long.txt", big.encode())
    assert len(pages) > 1
    assert [p.number for p in pages] == list(range(1, len(pages) + 1))


def test_unsupported_extension():
    with pytest.raises(ValueError):
        load_file("slides.pptx", b"data")


def test_empty_file_rejected():
    with pytest.raises(ValueError):
        load_file("empty.txt", b"   \n\n  ")


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\n\n\nc\x00") == "a b\n\nc"
