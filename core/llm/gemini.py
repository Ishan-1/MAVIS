"""
core/llm/gemini.py
Google Gemini implementation of BaseLLMClient using google-genai SDK.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

from google import genai
from core.llm.base import BaseLLMClient
from core.helpers import log_it

_ENTITY = "llm_gemini"


class GeminiClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        embedding_model: str = "text-embedding-004",
        api_key: str | None = None,
        vertexai: bool = True,
        temperature: float = 0.2,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature

        api_key = api_key or os.getenv("VERTEX_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(vertexai=vertexai, api_key=api_key)

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        config = {}
        if temp is not None:
            config["temperature"] = temp
        if system_instruction:
            config["system_instruction"] = system_instruction

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config if config else None,
            )
            raw = response.text or ""
            if json_mode:
                return self.clean_json_response(raw)
            return raw.strip()
        except Exception as e:
            log_it(f"Gemini generate_content failed: {e}", _ENTITY)
            raise

    def embed(self, text: str) -> list[float]:
        # Defensive truncation — embedding model has a token limit
        truncated = text[:8_000] if len(text) > 8_000 else text
        try:
            result = self._client.models.embed_content(
                model=self.embedding_model,
                contents=truncated,
            )
            vector = result.embeddings[0].values
            return list(vector)
        except Exception as e:
            log_it(f"Gemini embedding failed: {e}", _ENTITY)
            raise RuntimeError(f"Gemini embedding call failed: {e}") from e
