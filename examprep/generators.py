"""Turn retrieved passages into study material: summary, MCQs, short answers, flashcards.

Citations are resolved from source NUMBERS returned by the LLM, so page references always come
from real retrieved chunks and cannot be hallucinated.
"""
import random
from dataclasses import dataclass, field

from . import prompts
from .llm import LLM, complete_json
from .models import Hit

# ===================== Part 1: models, context, summary, MCQs =====================


@dataclass
class Summary:
    text: str
    key_points: list[str]
    citations: list[str]


@dataclass
class MCQ:
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ShortQ:
    question: str
    model_answer: str
    key_points: list[str]
    citations: list[str] = field(default_factory=list)


@dataclass
class Flashcard:
    front: str
    back: str
    citations: list[str] = field(default_factory=list)


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[S{i}] ({h.chunk.citation})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def resolve_citations(hits: list[Hit], ids) -> list[str]:
    out: list[str] = []
    for raw in ids or []:
        try:
            idx = int(raw) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(hits) and hits[idx].chunk.citation not in out:
            out.append(hits[idx].chunk.citation)
    return out


def _items(data, key: str) -> list:
    if isinstance(data, list):
        return data
    return data.get(key, []) if isinstance(data, dict) else []


def generate_summary(llm: LLM, hits: list[Hit], topic: str) -> Summary:
    data = complete_json(llm, prompts.SYSTEM, prompts.summary_prompt(format_context(hits), topic))
    return Summary(str(data.get("summary", "")), [str(p) for p in data.get("key_points", [])],
                   resolve_citations(hits, data.get("sources")))


def generate_mcqs(llm: LLM, hits: list[Hit], n: int = 5, difficulty: str = "medium",
                  seed: int | None = None) -> list[MCQ]:
    data = complete_json(llm, prompts.SYSTEM, prompts.mcq_prompt(format_context(hits), n, difficulty))
    rng = random.Random(seed)
    out = []
    for q in _items(data, "questions"):
        options, idx = q.get("options", []), q.get("answer_index")
        if len(options) != 4 or not isinstance(idx, int) or not 0 <= idx < 4:
            continue  # drop malformed items instead of crashing
        order = list(range(4))
        rng.shuffle(order)  # LLMs love putting the answer first; shuffle so it's not predictable
        out.append(MCQ(str(q.get("question", "")), [str(options[i]) for i in order], order.index(idx),
                       str(q.get("explanation", "")), resolve_citations(hits, q.get("sources"))))
    return out

# ===================== Part 2: short answers and flashcards =====================


def generate_short_questions(llm: LLM, hits: list[Hit], n: int = 3, difficulty: str = "medium") -> list[ShortQ]:
    data = complete_json(llm, prompts.SYSTEM, prompts.short_prompt(format_context(hits), n, difficulty))
    return [ShortQ(str(q.get("question", "")), str(q.get("model_answer", "")),
                   [str(k) for k in q.get("key_points", [])], resolve_citations(hits, q.get("sources")))
            for q in _items(data, "questions") if q.get("question")]


def generate_flashcards(llm: LLM, hits: list[Hit], n: int = 8) -> list[Flashcard]:
    data = complete_json(llm, prompts.SYSTEM, prompts.flashcards_prompt(format_context(hits), n))
    return [Flashcard(str(c.get("front", "")), str(c.get("back", "")), resolve_citations(hits, c.get("sources")))
            for c in _items(data, "cards") if c.get("front") and c.get("back")]
