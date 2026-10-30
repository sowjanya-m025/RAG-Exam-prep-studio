"""Export study material: Anki-importable CSV and a Markdown study pack."""
import csv
import io

from .generators import MCQ, Flashcard, ShortQ, Summary


def flashcards_to_csv(cards: list[Flashcard]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Front", "Back", "Source"])
    for c in cards:
        writer.writerow([c.front, c.back, "; ".join(c.citations)])
    return buf.getvalue()


def study_pack_to_markdown(topic: str, summary: Summary | None, mcqs: list[MCQ],
                           shorts: list[ShortQ], cards: list[Flashcard]) -> str:
    lines = [f"# Study pack: {topic}", ""]
    if summary:
        lines += ["## Summary", summary.text, ""] + [f"- {p}" for p in summary.key_points]
        lines += [f"\n*Sources: {', '.join(summary.citations) or 'n/a'}*", ""]
    if mcqs:
        lines += ["## Multiple-choice questions", ""]
        for i, q in enumerate(mcqs, 1):
            lines.append(f"**{i}. {q.question}**")
            lines += [f"   {chr(65 + j)}. {o}" for j, o in enumerate(q.options)]
            lines += [f"   - Answer: {chr(65 + q.answer_index)} — {q.explanation} *({', '.join(q.citations)})*", ""]
    if shorts:
        lines += ["## Short-answer questions", ""]
        for i, q in enumerate(shorts, 1):
            lines += [f"**{i}. {q.question}**", f"   - Model answer: {q.model_answer}",
                      f"   - Key points: {'; '.join(q.key_points)}", f"   - *({', '.join(q.citations)})*", ""]
    if cards:
        lines += ["## Flashcards", ""] + [f"- **{c.front}** → {c.back} *({', '.join(c.citations)})*" for c in cards]
    return "\n".join(lines) + "\n"
