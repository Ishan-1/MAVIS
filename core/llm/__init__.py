"""
core/llm package
Provides model-agnostic LLM client abstractions for MAVIS.
"""
from __future__ import annotations

import os
from typing import Any

from core.config import cfg
from core.llm.base import BaseLLMClient
from core.llm.gemini import GeminiClient
from core.llm.openai_compat import OpenAICompatClient
from core.llm.ollama import OllamaClient

_CLIENT_CACHE: dict[str, BaseLLMClient] = {}


def get_llm_client(provider: str | None = None, **kwargs: Any) -> BaseLLMClient:
    """
    Factory function returning the configured BaseLLMClient singleton.
    Reads defaults from `cfg.get("llm", ...)` in mavis_config.json.
    """
    llm_cfg = cfg.llm
    target_provider = provider or llm_cfg.get("provider", "gemini").lower()

    cache_key = f"{target_provider}_{kwargs}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    if target_provider == "gemini":
        model = kwargs.get("model") or llm_cfg.get("model", "gemini-2.5-flash")
        embed_model = kwargs.get("embedding_model") or llm_cfg.get("embedding_model", "text-embedding-004")
        vertexai = kwargs.get("vertexai", llm_cfg.get("vertexai", True))
        client = GeminiClient(model=model, embedding_model=embed_model, vertexai=vertexai)

    elif target_provider in ("openai", "openai_compat", "groq", "deepseek", "vllm"):
        model = kwargs.get("model") or llm_cfg.get("model", "gpt-4o-mini")
        embed_model = kwargs.get("embedding_model") or llm_cfg.get("embedding_model", "text-embedding-3-small")
        base_url = kwargs.get("base_url") or llm_cfg.get("base_url", "https://api.openai.com/v1")
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = OpenAICompatClient(model=model, embedding_model=embed_model, base_url=base_url, api_key=api_key)

    elif target_provider == "ollama":
        model = kwargs.get("model") or llm_cfg.get("model", "llama3.2")
        embed_model = kwargs.get("embedding_model") or llm_cfg.get("embedding_model", "nomic-embed-text")
        base_url = kwargs.get("base_url") or llm_cfg.get("base_url", "http://localhost:11434")
        client = OllamaClient(model=model, embedding_model=embed_model, base_url=base_url)

    else:
        raise ValueError(f"Unknown LLM provider '{target_provider}'. Supported: 'gemini', 'openai', 'ollama'.")

    _CLIENT_CACHE[cache_key] = client
    return client


__all__ = [
    "BaseLLMClient",
    "GeminiClient",
    "OpenAICompatClient",
    "OllamaClient",
    "get_llm_client",
]
