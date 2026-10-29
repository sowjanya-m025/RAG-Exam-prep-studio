"""Test helper: build a minimal valid PDF with one text line per page (no extra dependencies)."""


def make_pdf(page_texts: list[str]) -> bytes:
    n = len(page_texts)
    font_id = 3 + 2 * n
    objs = {1: b"<< /Type /Catalog /Pages 2 0 R >>"}
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    for i, text in enumerate(page_texts):
        pid, cid = 3 + 2 * i, 4 + 2 * i
        objs[pid] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {cid} 0 R "
                     f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>").encode()
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs[cid] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    objs[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out, offsets = b"%PDF-1.4\n", {}
    for i in sorted(objs):
        offsets[i] = len(out)
        out += f"{i} 0 obj\n".encode() + objs[i] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for i in sorted(objs):
        out += f"{offsets[i]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out
