"""
benchmarks/evaluators/proactivity.py
Proactivity (PROC) evaluation engine.
Scores early autonomous discovery and satisfaction of hidden intents across turns/waves.
"""
from __future__ import annotations

import re
from typing import Any

from benchmarks.harness.models import HiddenIntent, TaskSpec


class ProactivityEvaluator:
    """
    Computes PROC score based on wave/turn decay:
    PROC = (1 / |H|) * sum(gamma^(turn(h) - 1)) for satisfied intents h in H.
    """

    def __init__(self, gamma: float = 0.85, llm_client: Any | None = None) -> None:
        self.gamma = gamma
        self.llm_client = llm_client

    def evaluate(
        self,
        task: TaskSpec,
        turns: list[dict[str, Any]],
    ) -> tuple[float, list[str]]:
        """
        Evaluate proactivity given a task spec and ordered list of turns.
        Each turn is a dict containing {"turn": int, "role": str, "content": str, "metadata": dict}.

        Returns (proc_score, satisfied_intent_ids).
        """
        if not task.hidden_intents:
            return 1.0, []

        satisfied_intents: dict[str, int] = {}

        # Scan turns in chronological order
        for t_idx, turn in enumerate(turns):
            if turn.get("role") != "assistant":
                continue
            turn_number = turn.get("turn", t_idx + 1)
            content = str(turn.get("content", ""))

            for intent in task.hidden_intents:
                if intent.id in satisfied_intents:
                    continue  # Already recorded at earlier turn

                if self._matches_intent(intent, content):
                    satisfied_intents[intent.id] = turn_number

        # Calculate score with decay factor
        total_weight = sum(h.weight for h in task.hidden_intents)
        weighted_score = 0.0

        for intent in task.hidden_intents:
            if intent.id in satisfied_intents:
                turn_satisfied = satisfied_intents[intent.id]
                # Turn 1 -> gamma^0 = 1.0. Later turns decay by gamma^(turn - 1)
                decay = self.gamma ** max(0, turn_satisfied - 1)
                weighted_score += intent.weight * decay

        proc_score = round(weighted_score / total_weight, 4) if total_weight > 0 else 1.0
        return proc_score, list(satisfied_intents.keys())

    def _matches_intent(self, intent: HiddenIntent, content: str) -> bool:
        """
        Check whether the assistant's content satisfies the given hidden intent.
        Uses key phrase matching with semantic fallback if LLM client is present.
        """
        intent_lower = intent.content.lower()
        content_lower = content.lower()

        # Extract keywords (words with length >= 4, ignoring common stop words)
        stopwords = {"that", "this", "with", "from", "make", "sure", "also", "into", "their", "will", "have"}
        words = [w for w in re.findall(r"\b[a-z]{4,}\b", intent_lower) if w not in stopwords]

        if not words:
            return False

        # If at least 60% of core intent keywords appear in the response turn
        matched_words = sum(1 for w in words if w in content_lower)
        keyword_match = (matched_words / len(words)) >= 0.6

        if keyword_match:
            return True

        # Optional LLM semantic check if available
        if self.llm_client is not None and len(content) > 50:
            try:
                check_prompt = (
                    f"Evaluate if the following assistant response satisfies this specific requirement:\n"
                    f"Requirement: {intent.content}\n\n"
                    f"Assistant Response: {content[:1500]}\n\n"
                    f"Answer YES or NO."
                )
                res = self.llm_client.generate(check_prompt)
                return "yes" in res.strip().lower()
            except Exception:
                pass

        return False
