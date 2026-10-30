"""Grade a student's written answer against freshly retrieved source passages."""
from dataclasses import dataclass, field

from . import prompts
from .generators import MCQ, format_context, resolve_citations
from .llm import LLM, complete_json
from .models import Hit


@dataclass
class Grade:
    score: int                     # 0-10
    verdict: str                   # correct | partial | incorrect
    covered: list[str]
    missed: list[str]
    feedback: str
    citations: list[str] = field(default_factory=list)
    evidence: list[Hit] = field(default_factory=list)


def verdict_for(score: int) -> str:
    return "correct" if score >= 8 else "partial" if score >= 4 else "incorrect"


def grade_mcq(mcq: MCQ, chosen_index: int | None) -> bool:
    return chosen_index == mcq.answer_index


def grade_answer(llm: LLM, retriever, question: str, student_answer: str, k: int = 4,
                 reference_answer: str | None = None) -> Grade:
    if not student_answer.strip():
        return Grade(0, "incorrect", [], ["No answer was written."], "Write an answer first, then grade it.")
    hits = retriever.retrieve(f"{question} {reference_answer or ''}".strip(), k=k)
    data = complete_json(llm, prompts.SYSTEM,
                         prompts.grade_prompt(format_context(hits), question, student_answer, reference_answer))
    try:
        score = max(0, min(10, int(round(float(data.get("score", 0))))))
    except (TypeError, ValueError):
        score = 0
    return Grade(score, verdict_for(score), [str(x) for x in data.get("covered", [])],
                 [str(x) for x in data.get("missed", [])], str(data.get("feedback", "")),
                 resolve_citations(hits, data.get("sources")), hits)
