import pytest

from examprep import config, llm
from examprep.llm import MockLLM, OpenAICompatLLM, get_llm


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, str(body)

    def json(self):
        return self._body


OK = {"choices": [{"message": {"content": '{"ok": true}'}}]}


def test_provider_selection(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "mock")
    assert isinstance(get_llm(), MockLLM)
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_llm()
    monkeypatch.setattr(config, "GEMINI_API_KEY", "abc")
    assert isinstance(get_llm(), OpenAICompatLLM)
    monkeypatch.setattr(config, "LLM_PROVIDER", "nonsense")
    with pytest.raises(RuntimeError, match="Unknown"):
        get_llm()


def test_request_payload_and_token_floor(monkeypatch):
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return FakeResp(200, OK)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = OpenAICompatLLM("http://x/v1/", "KEY", "m", min_tokens=4096).complete("sys", "usr", max_tokens=1000)
    assert out == '{"ok": true}'
    assert seen["url"] == "http://x/v1/chat/completions"
    assert seen["json"]["max_tokens"] == 4096
    assert seen["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["headers"]["Authorization"] == "Bearer KEY"


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    replies = [FakeResp(429), FakeResp(429, headers={"Retry-After": "2"}), FakeResp(200, OK)]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: replies.pop(0))
    waits = []
    client = OpenAICompatLLM("http://x/v1", "k", "m")
    client._sleep = waits.append
    assert client.complete("s", "u") == '{"ok": true}'
    assert waits == [6, 2.0]


def test_client_error_raises(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(400, {"error": "bad key"}))
    with pytest.raises(RuntimeError, match="400"):
        OpenAICompatLLM("http://x/v1", "k", "m").complete("s", "u")
