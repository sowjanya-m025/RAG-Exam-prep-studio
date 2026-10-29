"""LLM access: Gemini (free tier), Ollama (free, local), any OpenAI-compatible API, Anthropic, or Mock.
Also contains robust JSON parsing."""
import json
import re
import time
from typing import Protocol

import requests

from . import config


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OpenAICompatLLM:
    """Any server that speaks the OpenAI chat-completions format: Gemini (free tier), Ollama (free,
    runs on your laptop), Groq, OpenRouter... Retries with back-off on rate limits (HTTP 429)."""

    def __init__(self, base_url: str, api_key: str, model: str, min_tokens: int = 0,
                 max_retries: int = 4, timeout: int = 120):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.min_tokens, self.max_retries, self.timeout = min_tokens, max_retries, timeout
        self._sleep = time.sleep

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        payload = {"model": self.model, "max_tokens": max(max_tokens, self.min_tokens),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key or 'none'}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers=headers, timeout=self.timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                try:
                    wait = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 6 * 2 ** attempt          # 6s, 12s, 24s, 48s: free tiers allow only a few requests/minute
                self._sleep(min(wait, 60))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text[:300]}")
            return resp.json()["choices"][0]["message"]["content"] or ""
        raise RuntimeError("LLM request failed after retries")


class MockLLM:
    """Deterministic fake so the whole app can be demoed/tested without an API key."""

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        task = re.search(r"TASK:\s*(\w+)", user).group(1)
        m = re.search(r"COUNT:\s*(\d+)", user)
        n = int(m.group(1)) if m else 1
        if task == "summary":
            return json.dumps({"summary": "Mock summary of the retrieved material [S1].",
                               "key_points": ["Mock key point A", "Mock key point B"], "sources": [1]})
        if task == "mcq":
            return json.dumps({"questions": [
                {"question": f"Mock MCQ {i + 1}?", "options": ["A", "B", "C", "D"], "answer_index": 0,
                 "explanation": "Mock explanation.", "sources": [1]} for i in range(n)]})
        if task == "short":
            return json.dumps({"questions": [
                {"question": f"Mock short question {i + 1}?", "model_answer": "Mock model answer.",
                 "key_points": ["point one", "point two"], "sources": [1]} for i in range(n)]})
        if task == "flashcards":
            return json.dumps({"cards": [
                {"front": f"Mock term {i + 1}", "back": "Mock definition.", "sources": [1]} for i in range(n)]})
        if task == "grade":
            return json.dumps({"score": 6, "covered": ["point one"], "missed": ["point two"],
                               "feedback": "Mock feedback: you covered part of the idea.", "sources": [1]})
        return "{}"


def get_llm() -> LLM:
    provider = config.LLM_PROVIDER
    if provider == "mock":
        return MockLLM()
    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Get a free key at aistudio.google.com, "
                               "or set LLM_PROVIDER=mock for an offline demo.")
        # floor of 4096 output tokens: Gemini 'thinking' models can use part of the budget before the JSON
        return OpenAICompatLLM(GEMINI_BASE_URL, config.GEMINI_API_KEY, config.GEMINI_MODEL, min_tokens=4096)
    if provider == "ollama":
        if not config.LLM_MODEL:
            raise RuntimeError("Set LLM_MODEL to a model you pulled, e.g. llama3.1:8b (run: ollama pull llama3.1:8b)")
        return OpenAICompatLLM(config.LLM_BASE_URL or OLLAMA_BASE_URL, "ollama", config.LLM_MODEL)
    if provider == "openai_compat":
        if not (config.LLM_BASE_URL and config.LLM_MODEL):
            raise RuntimeError("openai_compat needs LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY if required).")
        return OpenAICompatLLM(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env, or use LLM_PROVIDER=gemini (free).")
        return AnthropicLLM(config.ANTHROPIC_API_KEY, config.LLM_MODEL or "claude-sonnet-5-5")
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'. Use gemini, ollama, openai_compat, anthropic or mock.")


def extract_json(text: str):
    """Parse JSON even if the model wrapped it in ``` fences or added chatter around it."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
    if not m:
        raise ValueError("No JSON found in LLM output")
    return json.loads(m.group(1))


def complete_json(llm: LLM, system: str, user: str, max_tokens: int = 2500, retries: int = 1):
    last = None
    for _ in range(retries + 1):
        try:
            return extract_json(llm.complete(system, user, max_tokens))
        except ValueError as e:  # JSONDecodeError is a ValueError
            last = e
            user += "\n\nYour previous reply was not valid JSON. Reply with valid JSON only."
    raise ValueError(f"LLM did not return valid JSON: {last}")
