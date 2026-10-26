"""
tests/test_decision_model.py
Unit tests for the MAVIS Decision Model subsystem (core/decision),
covering TypeSafe primary evaluations, standardized LLM fallback,
and transparent failover routing under API errors, timeouts, and edge cases.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
import requests

from core.decision.base import BaseDecisionModel, DecisionResult
from core.decision.typesafe import TypeSafeDecisionModel
from core.decision import get_decision_model
from core.llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """Simple mock LLM client for testing fallback prompt responses."""
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "temp": temperature})
        if isinstance(self.response_text, Exception):
            raise self.response_text
        return self.response_text

    def embed(self, text: str) -> list[float]:
        return [0.0] * 10


# ── 1. BaseDecisionModel Standardized Fallback Tests ─────────────────────────

def test_base_fallback_noul():
    """Verify BaseDecisionModel evaluates Noul using standardized fixed prompt."""
    mock_llm = MockLLMClient("TRUE")
    model = BaseDecisionModel(fallback_client=mock_llm)

    res = model.evaluate_noul({"status": 0}, "Did the command succeed?")
    assert res.value is True
    assert res.confidence == 0.85
    assert len(mock_llm.calls) == 1
    assert "Did the command succeed?" in mock_llm.calls[0]["prompt"]
    assert "TRUE or FALSE" in mock_llm.calls[0]["prompt"]

    # Test FALSE response
    mock_llm.response_text = "FALSE"
    res_false = model.evaluate_noul({"status": 1}, "Did the command succeed?")
    assert res_false.value is False
    assert res_false.confidence == 0.80


def test_base_fallback_choice():
    """Verify BaseDecisionModel evaluates Choice with option matching."""
    mock_llm = MockLLMClient("Option B")
    model = BaseDecisionModel(fallback_client=mock_llm)

    options = ["Option A", "Option B", "Option C"]
    res = model.evaluate_choice({"metric": 42}, "Which category matches best?", options)
    assert res.value == "Option B"
    assert res.confidence == 0.85
    assert "Option A" in mock_llm.calls[0]["prompt"]

    # Substring / case-insensitive match
    mock_llm.response_text = "The answer is option a."
    res2 = model.evaluate_choice({}, "Choose", options)
    assert res2.value == "Option A"


def test_base_fallback_score():
    """Verify BaseDecisionModel evaluates Score against rubric."""
    mock_llm = MockLLMClient("0.75")
    model = BaseDecisionModel(fallback_client=mock_llm)

    rubric = {1.0: "High", 0.75: "Medium", 0.0: "None"}
    res = model.evaluate_score({"content": "good"}, "Rate quality", rubric)
    assert res.value == 0.75
    assert res.confidence == 0.85


def test_base_fallback_llm_exception():
    """Verify BaseDecisionModel safely returns degraded result when LLM raises exception."""
    mock_llm = MockLLMClient()
    mock_llm.response_text = RuntimeError("LLM rate limit exceeded")
    model = BaseDecisionModel(fallback_client=mock_llm)

    res_noul = model.evaluate_noul({}, "Is valid?")
    assert res_noul.value is False
    assert res_noul.confidence == 0.0

    res_choice = model.evaluate_choice({}, "Choose", ["default_opt", "other_opt"])
    assert res_choice.value == "default_opt"
    assert res_choice.confidence == 0.0

    res_score = model.evaluate_score({}, "Rate", {1.0: "High", 0.0: "Low"})
    assert res_score.value == 0.0
    assert res_score.confidence == 0.0


# ── 2. TypeSafe Primary Evaluation Tests ──────────────────────────────────────

@patch("requests.post")
def test_typesafe_primary_success_noul(mock_post):
    """Verify TypeSafe primary Noul evaluation parses structured response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": [{"noul": 0.95, "confidence": 0.98}]
    }
    mock_post.return_value = mock_resp

    model = TypeSafeDecisionModel(api_key="ts_test_key_123")
    assert model.is_primary_available() is True

    res = model.evaluate_noul({"query": "tech news"}, "Is this a general knowledge request?")
    assert res.value is True
    assert res.confidence == 0.98
    assert res.probabilities["true"] == 0.95
    assert mock_post.called


@patch("requests.post")
def test_typesafe_primary_success_choice(mock_post):
    """Verify TypeSafe primary Choice evaluation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": [{"choice": "specialized", "confidence": 0.92, "probabilities": {"specialized": 0.85, "generalized": 0.15}}]
    }
    mock_post.return_value = mock_resp

    model = TypeSafeDecisionModel(api_key="ts_test_key_123")
    res = model.evaluate_choice("query text", "Classify query", ["generalized", "specialized"])
    assert res.value == "specialized"
    assert res.confidence == 0.92
    assert res.probabilities["specialized"] == 0.85


@patch("requests.post")
def test_typesafe_primary_success_score(mock_post):
    """Verify TypeSafe primary Score evaluation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": [{"score": 1.0, "confidence": 0.94}]
    }
    mock_post.return_value = mock_resp

    model = TypeSafeDecisionModel(api_key="ts_test_key_123")
    rubric = {1.0: "Actionable", 0.0: "Trivia"}
    res = model.evaluate_score("Topic X", "Rate topic", rubric)
    assert res.value == 1.0
    assert res.confidence == 0.94


# ── 3. Routing & Failover Hierarchy Tests ─────────────────────────────────────

@patch("requests.post")
def test_failover_routing_on_http_error(mock_post):
    """Verify transparent failover to LLM when TypeSafe raises an HTTP error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_post.return_value = mock_resp

    mock_llm = MockLLMClient("TRUE")
    model = TypeSafeDecisionModel(api_key="ts_test_key_123", fallback_client=mock_llm)

    # Calling evaluate_noul should NOT crash; it should seamlessly failover to mock_llm
    res = model.evaluate_noul({"step": "done"}, "Did step complete?")
    assert res.value is True
    assert res.confidence == 0.85
    assert len(mock_llm.calls) == 1


@patch("requests.post")
def test_failover_routing_on_timeout(mock_post):
    """Verify transparent failover to LLM when TypeSafe times out."""
    mock_post.side_effect = requests.Timeout("Connection timed out after 5.0s")

    mock_llm = MockLLMClient("choice_b")
    model = TypeSafeDecisionModel(api_key="ts_test_key_123", fallback_client=mock_llm)

    res = model.evaluate_choice("state", "Pick", ["choice_a", "choice_b"])
    assert res.value == "choice_b"
    assert len(mock_llm.calls) == 1


def test_failover_when_api_key_absent():
    """Verify model directly uses LLM fallback when TypeSafe API key is empty."""
    mock_llm = MockLLMClient("0.75")
    model = TypeSafeDecisionModel(api_key="", fallback_client=mock_llm)
    assert model.is_primary_available() is False

    with patch("requests.post") as mock_post:
        res = model.evaluate_score("content", "Rate", {1.0: "A", 0.75: "B"})
        assert not mock_post.called
        assert res.value == 0.75
        assert len(mock_llm.calls) == 1


@patch("requests.post")
def test_dual_failure_safe_degradation(mock_post):
    """Verify safe degradation when BOTH TypeSafe and Fallback LLM fail."""
    mock_post.side_effect = requests.ConnectionError("Network down")
    mock_llm = MockLLMClient()
    mock_llm.response_text = RuntimeError("LLM service unavailable")

    model = TypeSafeDecisionModel(api_key="ts_key", fallback_client=mock_llm)

    # Must return safe default without raising unhandled exceptions
    res = model.evaluate_noul("state", "Condition")
    assert res.value is False
    assert res.confidence == 0.0


def test_factory_get_decision_model_singleton():
    """Verify get_decision_model factory returns singleton and respects force_refresh."""
    model1 = get_decision_model(force_refresh=True)
    model2 = get_decision_model(force_refresh=False)
    assert model1 is model2
