"""Thin, dependency-free client for the Ollama REST API.

Configurable purely through environment variables (OLLAMA_BASE_URL,
OLLAMA_MODEL). No API keys, no cloud LLM calls - this is the only place the
system talks to a language model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama cannot be reached or the configured model is missing."""


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout_seconds

    def check_availability(self) -> tuple[bool, str]:
        """Returns (is_available, message). Never raises."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = {m.get("name", "").split(":")[0] for m in tags}
            model_base = self.model.split(":")[0]
            if names and model_base not in names:
                available = ", ".join(sorted(names)) or "none"
                return False, (
                    f"Ollama is running at {self.base_url} but model '{self.model}' is not pulled. "
                    f"Available models: {available}. Run: ollama pull {self.model}"
                )
            return True, "Ollama is reachable."
        except httpx.ConnectError:
            return False, (
                f"Cannot reach Ollama at {self.base_url}. Start it with 'ollama serve' "
                f"(or check OLLAMA_BASE_URL in your .env)."
            )
        except httpx.TimeoutException:
            return False, f"Ollama at {self.base_url} timed out while checking availability."
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Unexpected error contacting Ollama: {exc}"

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.1, format_json: bool = False) -> LLMResponse:
        """Call Ollama's /api/generate endpoint (non-streaming) and return the text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        try:
            resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is 'ollama serve' running? ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError(
                f"Ollama request timed out after {self.timeout}s. The model may be too large "
                f"for this machine, or still loading."
            ) from exc

        if resp.status_code == 404:
            raise OllamaUnavailableError(
                f"Model '{self.model}' was not found on the Ollama server at {self.base_url}. "
                f"Run: ollama pull {self.model}"
            )
        if resp.status_code != 200:
            raise OllamaUnavailableError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise OllamaUnavailableError(f"Ollama returned a non-JSON response: {resp.text[:200]}") from exc

        text = data.get("response", "")
        logger.info(
            "llm_generate",
            extra={"extra_fields": {"model": self.model, "prompt_chars": len(prompt), "response_chars": len(text)}},
        )
        return LLMResponse(text=text, model=self.model, raw=data)


def get_llm_client() -> OllamaClient:
    return OllamaClient()
