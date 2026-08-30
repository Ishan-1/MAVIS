"""
core/llm/openai_compat.py
OpenAI-compatible client (works with OpenAI, Groq, DeepSeek, vLLM, local endpoints).
Uses standard urllib to avoid extra hard dependencies.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from core.llm.base import BaseLLMClient
from core.helpers import log_it

_ENTITY = "llm_openai_compat"


class OpenAICompatClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.2,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                if json_mode:
                    return self.clean_json_response(content)
                return content.strip()
        except Exception as e:
            log_it(f"OpenAICompat generate failed: {e}", _ENTITY)
            raise

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": text[:8_000],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["data"][0]["embedding"]
        except Exception as e:
            log_it(f"OpenAICompat embed failed: {e}", _ENTITY)
            raise RuntimeError(f"OpenAICompat embedding call failed: {e}") from e
