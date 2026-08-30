"""
core/llm/ollama.py
Ollama local model implementation of BaseLLMClient.
Communicates directly with the Ollama REST API on http://localhost:11434.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error

from core.llm.base import BaseLLMClient
from core.helpers import log_it

_ENTITY = "llm_ollama"


class OllamaClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
            },
        }
        if system_instruction:
            payload["system"] = system_instruction
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result.get("response", "")
                if json_mode:
                    return self.clean_json_response(content)
                return content.strip()
        except Exception as e:
            log_it(f"Ollama generate failed: {e}", _ENTITY)
            raise

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text[:8_000],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["embedding"]
        except Exception as e:
            log_it(f"Ollama embed failed: {e}", _ENTITY)
            raise RuntimeError(f"Ollama embedding call failed: {e}") from e
