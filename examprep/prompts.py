"""All prompt templates in one place (easy to tune). Every task asks for JSON + source numbers."""

SYSTEM = """You are a careful exam-prep tutor.
Rules:
1. Use ONLY the numbered SOURCES in the user message. Never add outside facts.
2. If the sources are too thin for the request, return fewer items rather than inventing content.
3. For every item, list the source numbers you used in "sources" (e.g. [1, 3]).
4. Reply with valid JSON only. No markdown fences, no commentary."""


def summary_prompt(context: str, topic: str) -> str:
    return f"""TASK: summary
TOPIC: {topic}

SOURCES:
{context}

Write a concise revision summary of the topic (120-180 words) plus 4-6 key points.
JSON schema: {{"summary": str, "key_points": [str], "sources": [int]}}"""


def mcq_prompt(context: str, n: int, difficulty: str) -> str:
    return f"""TASK: mcq
COUNT: {n}
DIFFICULTY: {difficulty}

SOURCES:
{context}

Write {n} {difficulty} multiple-choice questions. Exactly 4 options each, one correct answer,
plausible distractors, and a one-sentence explanation of why the answer is correct.
JSON schema: {{"questions": [{{"question": str, "options": [str, str, str, str], "answer_index": int, "explanation": str, "sources": [int]}}]}}"""


def short_prompt(context: str, n: int, difficulty: str) -> str:
    return f"""TASK: short
COUNT: {n}
DIFFICULTY: {difficulty}

SOURCES:
{context}

Write {n} {difficulty} short-answer questions (answerable in 2-4 sentences). For each give a model answer
and 2-4 key points a correct answer must contain.
JSON schema: {{"questions": [{{"question": str, "model_answer": str, "key_points": [str], "sources": [int]}}]}}"""


def flashcards_prompt(context: str, n: int) -> str:
    return f"""TASK: flashcards
COUNT: {n}

SOURCES:
{context}

Create {n} flashcards (term/concept on the front, short precise answer on the back).
JSON schema: {{"cards": [{{"front": str, "back": str, "sources": [int]}}]}}"""


def grade_prompt(context: str, question: str, student_answer: str, reference: str | None) -> str:
    ref = f"\nREFERENCE ANSWER (from generation step): {reference}\n" if reference else ""
    return f"""TASK: grade

SOURCES:
{context}
{ref}
QUESTION: {question}

STUDENT ANSWER: {student_answer}

Grade the student answer ONLY against the sources. Score 0-10 (10 = fully correct and complete).
List what the student covered correctly and what they missed or got wrong, then give 2-3 sentences of
constructive feedback that tell them exactly what to revise.
JSON schema: {{"score": int, "covered": [str], "missed": [str], "feedback": str, "sources": [int]}}"""
