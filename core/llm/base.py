"""
core/llm/base.py
Abstract base class defining the uniform LLM provider interface for MAVIS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re


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
