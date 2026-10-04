"""Thin LLM client layer so the pipeline does not depend on Ollama directly (and can be unit-tested offline)."""
import requests

from src.config import OLLAMA_URL, OLLAMA_TAGS_URL, MODEL


class LLMUnavailable(RuntimeError):
    """Raised when the local model server cannot be reached or the model is missing."""


class OllamaLLM:
    def __init__(self, url=OLLAMA_URL, tags_url=OLLAMA_TAGS_URL, retries=1):
        self.url, self.tags_url, self.retries = url, tags_url, retries

    def available(self, model=MODEL):
        try:
            r = requests.get(self.tags_url, timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
        except Exception:
            return False

    def generate(self, prompt, model=MODEL, temperature=0.0, timeout=120):
        last = None
        for _ in range(self.retries + 1):
            try:
                r = requests.post(self.url, json={"model": model, "prompt": prompt, "stream": False,
                                                  "options": {"temperature": temperature}}, timeout=timeout)
                r.raise_for_status()
                return r.json()["response"].strip()
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
        raise LLMUnavailable(f"Could not reach Ollama at {self.url}. Run `ollama serve` and `ollama pull {model}`. ({last})")


class FakeLLM:
    """Deterministic stand-in for tests: returns `reply` (a string or a callable(prompt) -> str)."""
    def __init__(self, reply="OK"):
        self.reply, self.calls = reply, []

    def available(self, model=MODEL):
        return True

    def generate(self, prompt, model=MODEL, temperature=0.0, timeout=120):
        self.calls.append(prompt)
        return self.reply(prompt) if callable(self.reply) else self.reply
