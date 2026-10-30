from examprep.exporters import flashcards_to_csv, study_pack_to_markdown
from examprep.generators import MCQ, Flashcard, ShortQ, Summary


def test_csv_has_header_and_rows():
    csv_text = flashcards_to_csv([Flashcard("Term", "Def, with comma", ["n.pdf, p.2"])])
    lines = csv_text.strip().splitlines()
    assert lines[0] == "Front,Back,Source"
    assert '"Def, with comma"' in lines[1]


def test_markdown_contains_all_sections():
    md = study_pack_to_markdown(
        "ML", Summary("sum", ["kp"], ["n.pdf, p.1"]),
        [MCQ("q?", ["a", "b", "c", "d"], 1, "because", ["n.pdf, p.1"])],
        [ShortQ("sq?", "ans", ["k1"], ["n.pdf, p.2"])], [Flashcard("f", "b", ["n.pdf, p.3"])])
    for heading in ("# Study pack: ML", "## Summary", "## Multiple-choice", "## Short-answer", "## Flashcards"):
        assert heading in md
    assert "Answer: B" in md
