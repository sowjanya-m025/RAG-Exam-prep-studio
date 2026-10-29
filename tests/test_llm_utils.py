import pytest

from examprep.llm import MockLLM, complete_json, extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_json_with_chatter():
    assert extract_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}


def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_json_retries_once():
    class Flaky:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, max_tokens=2000):
            self.calls += 1
            return "garbage" if self.calls == 1 else '{"ok": true}'

    llm = Flaky()
    assert complete_json(llm, "s", "u") == {"ok": True}
    assert llm.calls == 2


def test_mock_llm_returns_json():
    response = MockLLM().complete("s", "TASK: flashcards\nCOUNT: 2")
    assert "cards" in extract_json(response)