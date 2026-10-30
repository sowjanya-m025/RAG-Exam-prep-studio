from examprep.generators import (format_context, generate_flashcards, generate_mcqs,
                                 generate_short_questions, generate_summary, resolve_citations)
from examprep.llm import MockLLM
from examprep.models import Chunk, Hit

HITS = [Hit(Chunk(0, "notes.pdf", 3, "Overfitting is when..."), 0.9),
        Hit(Chunk(1, "notes.pdf", 7, "Gradient descent is..."), 0.8)]


def test_format_context_numbers_sources():
    ctx = format_context(HITS)
    assert "[S1] (notes.pdf, p.3)" in ctx and "[S2] (notes.pdf, p.7)" in ctx


def test_resolve_citations_ignores_bad_ids():
    assert resolve_citations(HITS, [1, 2, 2, 99, "x", None]) == ["notes.pdf, p.3", "notes.pdf, p.7"]


def test_mcqs_valid_and_cited():
    qs = generate_mcqs(MockLLM(), HITS, n=3, seed=1)
    assert len(qs) == 3
    for q in qs:
        assert len(q.options) == 4 and 0 <= q.answer_index < 4
        assert q.options[q.answer_index] == "A"          # shuffle must keep the right answer tracked
        assert q.citations == ["notes.pdf, p.3"]


def test_malformed_mcq_is_dropped():
    class Bad:
        def complete(self, s, u, max_tokens=2000):
            return '{"questions": [{"question": "q", "options": ["a","b"], "answer_index": 5}]}'

    assert generate_mcqs(Bad(), HITS, n=1) == []


def test_summary_short_and_flashcards():
    s = generate_summary(MockLLM(), HITS, "ML")
    assert s.text and s.key_points and s.citations
    assert len(generate_short_questions(MockLLM(), HITS, n=2)) == 2
    cards = generate_flashcards(MockLLM(), HITS, n=4)
    assert len(cards) == 4 and cards[0].citations
