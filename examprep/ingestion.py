"""Read PDF / TXT / MD files into Page objects that keep page numbers."""
import io
import re

from pypdf import PdfReader

from .models import Page

VIRTUAL_PAGE_CHARS = 2500  # .txt/.md have no pages, so we create "virtual pages"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(name: str, data: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(Page(name, i, text))
    return pages


def load_text(name: str, data: bytes) -> list[Page]:
    text = clean_text(data.decode("utf-8", errors="ignore"))
    pages, buf, number = [], "", 1
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) > VIRTUAL_PAGE_CHARS:
            pages.append(Page(name, number, buf.strip()))
            number += 1
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        pages.append(Page(name, number, buf.strip()))
    return pages


def load_file(name: str, data: bytes) -> list[Page]:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf":
        pages = load_pdf(name, data)
    elif ext in {"txt", "md"}:
        pages = load_text(name, data)
    else:
        raise ValueError(f"Unsupported file type: .{ext} (use PDF, TXT or MD)")
    if not pages:
        raise ValueError(f"No extractable text in {name} (scanned PDF? try OCR first)")
    return pages
