"""
core/llm/base.py
Abstract base class defining the uniform LLM provider interface for MAVIS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import re
from typing import Any

from core.helpers import MAV_ROOT


def is_prompt_logging_enabled() -> bool:
    """
    Check if prompt debugging/logging is enabled.
    Enabled if:
    - Environment variable MAVIS_LOG_PROMPTS or MAVIS_DEBUG_PROMPTS is truthy ('1', 'true', 'yes', 'on')
    - cfg.llm.get('log_prompts') is True
    - cfg.output.get('verbosity') == 'debug'
    """
    env_val = os.getenv("MAVIS_LOG_PROMPTS", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    debug_val = os.getenv("MAVIS_DEBUG_PROMPTS", "").strip().lower()
    if debug_val in ("1", "true", "yes", "on"):
        return True
    try:
        from core.config import cfg
        if cfg.llm.get("log_prompts", False):
            return True
        if str(cfg.output.get("verbosity", "")).lower() == "debug":
            return True
    except Exception:
        pass
    return False


def log_llm_prompt(
    prompt: str,
    system_instruction: str | None = None,
    model: str = "",
    provider: str = "",
    extra: dict | None = None,
    entity: str = "llm",
) -> None:
    """
    Log the final prompt dispatched to an LLM provider.
    Outputs to:
    1. logs/llm_prompts.log (centralized prompt log)
    2. logs/{entity}.log (provider-specific log, e.g. llm_gemini.log)
    3. Terminal via mavis_debug() if debug output is active
    """
    if not is_prompt_logging_enabled():
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"=== LLM PROMPT [{ts}] | Provider: {provider or 'unknown'} | Model: {model or 'default'} ==="
    lines = [header]
    if extra:
        extra_str = ", ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        if extra_str:
            lines.append(f"Config: {extra_str}")
    if system_instruction:
        lines.append("--- SYSTEM INSTRUCTION ---")
        lines.append(system_instruction.strip())
    lines.append("--- USER PROMPT ---")
    lines.append(prompt.strip())
    lines.append("=" * len(header))
    entry = "\n".join(lines) + "\n\n"

    # Terminal output via mavis_debug
    try:
        from core.output import mavis_debug
        mavis_debug(f"\n{entry.strip()}", entity=entity)
    except Exception:
        pass

    # File logging
    try:
        log_dir = os.path.join(MAV_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "llm_prompts.log"), "a", encoding="utf-8") as f:
            f.write(entry)
        if entity:
            with open(os.path.join(log_dir, f"{entity}.log"), "a", encoding="utf-8") as f:
                f.write(entry)
    except Exception:
        pass


class BaseLLMClient(ABC):
    """
    Abstract LLM client interface. All model providers (Gemini, OpenAI,
    Anthropic, Ollama, etc.) implement this contract.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate text response from a prompt.

        Args:
            prompt: User prompt text.
            json_mode: If True, guarantees markdown code blocks (```json ... ```)
                       are stripped from the returned text.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature override.

        Returns:
            The raw text response (stripped).
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input string.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    def log_prompt(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model: str = "",
        provider: str = "",
        extra: dict | None = None,
        entity: str = "llm",
    ) -> None:
        """Helper to invoke prompt logging if enabled."""
        log_llm_prompt(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model or getattr(self, "model", ""),
            provider=provider or getattr(self, "provider", ""),
            extra=extra,
            entity=entity,
        )

    @staticmethod
    def clean_json_response(raw: str) -> str:
        """
        Helper to strip markdown formatting like ```json ... ``` or ``` ... ```
        from LLM output.
        """
        text = raw.strip()
        if text.startswith("```"):
            # Strip opening ```json or ```
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

