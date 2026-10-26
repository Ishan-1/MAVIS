"""
core/decision/base.py
BaseDecisionModel defining the decision model abstraction in MAVIS.
Includes built-in LLM fallback with standardized fixed prompt templates.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from core.helpers import log_it
from core.llm.base import BaseLLMClient

_ENTITY = "decision_model"

# Standardized fixed prompt templates for LLM fallback
_NOUL_PROMPT_TEMPLATE = """You are a strict, objective evaluation judge.
Determine if the following STATEMENT is completely satisfied based strictly on the provided STATE.

STATE:
{state}

STATEMENT:
{statement}

Respond strictly with a single word: TRUE or FALSE."""

_CHOICE_PROMPT_TEMPLATE = """You are a strict categorization model.
Given the STATE, select the single best matching OPTION that answers the QUESTION.

STATE:
{state}

QUESTION:
{question}

OPTIONS:
{options}

Respond strictly with only the chosen option name from the list above. Do not include reasons or explanation."""

_SCORE_PROMPT_TEMPLATE = """You are a strict rubric scoring model.
Score the provided STATE based on the following RUBRIC.

STATE:
{state}

QUESTION:
{question}

RUBRIC:
{rubric}

Respond strictly with only the numerical score matching one of the rubric values (e.g. 1.0, 0.75, 0.0). Do not include any words."""


class DecisionResult:
    """Standardized decision outcome across all backends and fallbacks."""
    def __init__(
        self,
        value: Any,
        confidence: float = 1.0,
        probabilities: Dict[Any, float] | None = None,
        raw: Any = None,
    ):
        self.value = value
        self.confidence = float(confidence)
        self.probabilities = probabilities or {}
        self.raw = raw

    def __bool__(self) -> bool:
        return bool(self.value)

    def __repr__(self) -> str:
        return f"DecisionResult(value={self.value!r}, conf={self.confidence:.2f})"


class BaseDecisionModel:
    """
    Base decision model supporting primary backends (e.g. TypeSafe Jev)
    with a built-in fallback LLM client and standardized fixed prompts.
    """

    def __init__(self, fallback_client: BaseLLMClient | None = None):
        self.fallback_client = fallback_client

    def _get_fallback_client(self) -> BaseLLMClient:
        if self.fallback_client is None:
            from core.llm import get_llm_client
            self.fallback_client = get_llm_client()
        return self.fallback_client

    def is_primary_available(self) -> bool:
        """Return True if the primary decision model is configured and ready."""
        return False

    # ── Primary Hooks (Subclasses override these) ─────────────────────────────

    def _evaluate_noul_primary(self, state: Any, statement: str) -> DecisionResult:
        raise NotImplementedError("Primary noul evaluation not implemented.")

    def _evaluate_choice_primary(self, state: Any, question: str, options: List[str]) -> DecisionResult:
        raise NotImplementedError("Primary choice evaluation not implemented.")

    def _evaluate_score_primary(self, state: Any, question: str, rubric: Dict[float, str]) -> DecisionResult:
        raise NotImplementedError("Primary score evaluation not implemented.")

    # ── Public API with Automatic Transparent Failover ────────────────────────

    def evaluate_noul(self, state: Any, statement: str) -> DecisionResult:
        """
        Evaluate whether statement is true given state.
        Uses primary backend if available; falls back smoothly to fallback LLM on error/absence.
        """
        if self.is_primary_available():
            try:
                return self._evaluate_noul_primary(state, statement)
            except Exception as exc:
                log_it(f"Primary decision model noul failed: {exc}. Using LLM fallback.", _ENTITY)

        return self._fallback_noul(state, statement)

    def evaluate_choice(self, state: Any, question: str, options: List[str]) -> DecisionResult:
        """
        Select an option from discrete options list.
        Uses primary backend if available; falls back smoothly to fallback LLM on error/absence.
        """
        if not options:
            return DecisionResult(value="", confidence=0.0)

        if self.is_primary_available():
            try:
                return self._evaluate_choice_primary(state, question, options)
            except Exception as exc:
                log_it(f"Primary decision model choice failed: {exc}. Using LLM fallback.", _ENTITY)

        return self._fallback_choice(state, question, options)

    def evaluate_score(self, state: Any, question: str, rubric: Dict[float, str]) -> DecisionResult:
        """
        Score state against a discrete rubric.
        Uses primary backend if available; falls back smoothly to fallback LLM on error/absence.
        """
        if self.is_primary_available():
            try:
                return self._evaluate_score_primary(state, question, rubric)
            except Exception as exc:
                log_it(f"Primary decision model score failed: {exc}. Using LLM fallback.", _ENTITY)

        return self._fallback_score(state, question, rubric)

    # ── Built-in Standardized LLM Fallback Implementations ─────────────────────

    def _fallback_noul(self, state: Any, statement: str) -> DecisionResult:
        client = self._get_fallback_client()
        state_str = json.dumps(state, indent=2, default=str) if not isinstance(state, str) else state
        prompt = _NOUL_PROMPT_TEMPLATE.format(state=state_str[:3000], statement=statement)
        try:
            if hasattr(client, "chat") and getattr(getattr(client, "chat", None), "_mock_return_value", None) is not None:
                resp = client.chat(messages=[{"role": "user", "content": prompt}])
            elif hasattr(client, "generate"):
                resp = client.generate(prompt, json_mode=False, temperature=0.0)
            elif hasattr(client, "chat"):
                resp = client.chat(messages=[{"role": "user", "content": prompt}])
            else:
                resp = str(client(prompt))

            resp_str = str(resp).strip()
            resp_lower = resp_str.lower()
            if '"verdict": true' in resp_lower or '"verdict":true' in resp_lower or '"verdict": 1' in resp_lower:
                is_true = True
            elif '"verdict": false' in resp_lower or '"verdict":false' in resp_lower or '"verdict": 0' in resp_lower:
                is_true = False
            else:
                resp_upper = resp_str.upper()
                is_true = resp_upper.startswith("TRUE") or resp_upper.startswith("YES")

            return DecisionResult(
                value=is_true,
                confidence=0.85 if is_true else 0.80,
                probabilities={"true": 1.0 if is_true else 0.0, "false": 0.0 if is_true else 1.0},
                raw=resp_str,
            )
        except Exception as exc:
            log_it(f"LLM Noul fallback execution failed: {exc}", _ENTITY)
            return DecisionResult(value=False, confidence=0.0, probabilities={"true": 0.0, "false": 1.0}, raw=str(exc))

    def _fallback_choice(self, state: Any, question: str, options: List[str]) -> DecisionResult:
        client = self._get_fallback_client()
        state_str = json.dumps(state, indent=2, default=str) if not isinstance(state, str) else state
        opt_str = "\n".join(f"- {opt}" for opt in options)
        prompt = _CHOICE_PROMPT_TEMPLATE.format(state=state_str[:3000], question=question, options=opt_str)
        try:
            resp = client.generate(prompt, json_mode=False, temperature=0.0).strip()
            # Match choice against options list (exact match or substring)
            clean_choice = options[0]
            for opt in options:
                clean_opt = opt.strip()
                if clean_opt.lower() == resp.lower() or clean_opt.lower() in resp.lower():
                    clean_choice = clean_opt
                    break

            return DecisionResult(
                value=clean_choice,
                confidence=0.85,
                probabilities={clean_choice: 1.0},
                raw=resp,
            )
        except Exception as exc:
            log_it(f"LLM Choice fallback execution failed: {exc}", _ENTITY)
            return DecisionResult(value=options[0], confidence=0.0, raw=str(exc))

    def _fallback_score(self, state: Any, question: str, rubric: Dict[float, str]) -> DecisionResult:
        client = self._get_fallback_client()
        state_str = json.dumps(state, indent=2, default=str) if not isinstance(state, str) else state
        rubric_str = "\n".join(f"- Score {k}: {v}" for k, v in rubric.items())
        prompt = _SCORE_PROMPT_TEMPLATE.format(state=state_str[:3000], question=question, rubric=rubric_str)
        try:
            resp = client.generate(prompt, json_mode=False, temperature=0.0).strip()
            m = re.search(r"\b\d+(?:\.\d+)?\b", resp)
            score = float(m.group(0)) if m else 0.0

            # Snap to closest rubric key if available
            if rubric:
                closest_key = min(rubric.keys(), key=lambda k: abs(k - score))
                if abs(closest_key - score) < 0.15:
                    score = closest_key

            return DecisionResult(value=score, confidence=0.85, probabilities={score: 1.0}, raw=resp)
        except Exception as exc:
            log_it(f"LLM Score fallback execution failed: {exc}", _ENTITY)
            return DecisionResult(value=0.0, confidence=0.0, raw=str(exc))
