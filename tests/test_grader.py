from examprep.grader import grade_answer, grade_mcq, verdict_for
from examprep.generators import MCQ
from examprep.llm import MockLLM
from examprep.models import Chunk, Hit


class FakeRetriever:
    def retrieve(self, query, k=4, diverse=True):
        return [Hit(Chunk(0, "n.pdf", 5, "Overfitting means..."), 0.9)]


def test_grade_with_mock():
    g = grade_answer(MockLLM(), FakeRetriever(), "What is overfitting?", "It is when a model memorises.")
    assert g.score == 6 and g.verdict == "partial"
    assert g.covered and g.missed and g.citations == ["n.pdf, p.5"] and g.evidence


def test_empty_answer_scores_zero_without_llm_call():
    class Boom:
        def complete(self, *a, **k):
            raise AssertionError("LLM should not be called")

    g = grade_answer(Boom(), FakeRetriever(), "Q?", "   ")
    assert g.score == 0 and g.verdict == "incorrect"


def test_score_is_clamped():
    class Wild:
        def complete(self, *a, **k):
            return '{"score": 42, "covered": [], "missed": [], "feedback": "x", "sources": [1]}'

    assert grade_answer(Wild(), FakeRetriever(), "Q?", "answer").score == 10


def test_verdict_thresholds():
    assert (verdict_for(9), verdict_for(5), verdict_for(1)) == ("correct", "partial", "incorrect")


def test_grade_mcq():
    q = MCQ("q", ["a", "b", "c", "d"], 2, "why")
    assert grade_mcq(q, 2) and not grade_mcq(q, 0) and not grade_mcq(q, None)
