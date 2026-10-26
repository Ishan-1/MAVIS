"""
core/decision/typesafe.py
TypeSafe Jev System One implementation of BaseDecisionModel.
Inherits built-in LLM fallback from BaseDecisionModel.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
import requests

from core.decision.base import BaseDecisionModel, DecisionResult
from core.llm.base import BaseLLMClient

_ENTITY = "typesafe_decision_model"


class TypeSafeDecisionModel(BaseDecisionModel):
    """
    Client for TypeSafe AI's Jev model.
    Evaluates questions in parallel and returns structured typed answers,
    probabilities, and confidence directly.
    Inherits automatic LLM fallback from BaseDecisionModel.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.typesafe.ai/v1",
        timeout: float = 5.0,
        fallback_client: BaseLLMClient | None = None,
    ):
        super().__init__(fallback_client=fallback_client)
        self.api_key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_primary_available(self) -> bool:
        """TypeSafe is considered available if an API key is provided."""
        return bool(self.api_key and self.api_key.strip())

    def _post(self, questions: List[Dict[str, Any]], state: Any) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"state": state, "questions": questions}
        resp = requests.post(
            f"{self.base_url}/evaluate",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answers", [])

    def _evaluate_noul_primary(self, state: Any, statement: str) -> DecisionResult:
        answers = self._post([{"type": "noul", "statement": statement}], state)
        if not answers:
            raise ValueError("TypeSafe returned empty answers list")
        ans = answers[0]
        noul_score = float(ans.get("noul", 0.0))
        conf = float(ans.get("confidence", 1.0))
        return DecisionResult(
            value=(noul_score >= 0.5),
            confidence=conf,
            probabilities={"true": noul_score, "false": max(0.0, 1.0 - noul_score)},
            raw=ans,
        )

    def _evaluate_choice_primary(self, state: Any, question: str, options: List[str]) -> DecisionResult:
        answers = self._post([{"type": "choice", "question": question, "options": options}], state)
        if not answers:
            raise ValueError("TypeSafe returned empty answers list")
        ans = answers[0]
        return DecisionResult(
            value=str(ans.get("choice")),
            confidence=float(ans.get("confidence", 1.0)),
            probabilities=ans.get("probabilities", {}),
            raw=ans,
        )

    def _evaluate_score_primary(self, state: Any, question: str, rubric: Dict[float, str]) -> DecisionResult:
        rubric_payload = [{"score": k, "criteria": v} for k, v in rubric.items()]
        answers = self._post([{"type": "score", "question": question, "rubric": rubric_payload}], state)
        if not answers:
            raise ValueError("TypeSafe returned empty answers list")
        ans = answers[0]
        return DecisionResult(
            value=float(ans.get("score", 0.0)),
            confidence=float(ans.get("confidence", 1.0)),
            probabilities=ans.get("probabilities", {}),
            raw=ans,
        )
