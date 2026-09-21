"""
benchmarks/evaluators/proactivity.py
Proactivity (PROC) evaluation engine.
Scores early autonomous discovery and satisfaction of hidden intents across turns/waves.
"""
from __future__ import annotations

import json
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
        self._batch_cache: dict[tuple, list[str]] = {}

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

            remaining_intents = [h for h in task.hidden_intents if h.id not in satisfied_intents]
            if not remaining_intents:
                break

            still_unmatched: list[HiddenIntent] = []
            for intent in remaining_intents:
                if self._matches_intent_fast(intent, content):
                    satisfied_intents[intent.id] = turn_number
                else:
                    still_unmatched.append(intent)

            # If semantic LLM check is available, batch check still-unmatched intents for this turn
            if still_unmatched and self.llm_client is not None and len(content) > 30:
                batch_satisfied = self._batch_llm_check(still_unmatched, content)
                for intent_id in batch_satisfied:
                    satisfied_intents[intent_id] = turn_number

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

    def _matches_intent_fast(self, intent: HiddenIntent, content: str) -> bool:
        """
        Fast heuristic check whether assistant content satisfies the given hidden intent.
        Examines file basenames, phrases, and keyword density.
        """
        intent_lower = intent.content.lower()
        content_lower = content.lower()

        # Check for explicit file references (e.g. "executive_summary.md", "reply_sarah.md")
        file_matches = re.findall(r"[\w\-_]+\.(?:md|json|py|txt|csv|yaml|yml)", intent_lower)
        for fm in file_matches:
            if fm in content_lower:
                return True
            # Also check without extension if length >= 5
            base = fm.split(".")[0]
            if len(base) >= 5 and base in content_lower:
                return True

        # Extract keywords (length >= 4, ignoring common stop words)
        stopwords = {
            "that", "this", "with", "from", "make", "sure", "also", "into", "their",
            "will", "have", "been", "then", "when", "what", "which", "some", "only",
        }
        words = [w for w in re.findall(r"\b[a-z]{4,}\b", intent_lower) if w not in stopwords]

        if not words:
            return False

        matched_words = 0
        for w in words:
            stem = w[:5] if len(w) >= 6 else (w[:4] if len(w) >= 5 else w)
            if w in content_lower or stem in content_lower:
                matched_words += 1

        # Match if at least 35% of keywords present, or at least 2 keywords match
        if (matched_words / len(words) >= 0.35) or matched_words >= 2:
            return True

        return False

    def _batch_llm_check(self, intents: list[HiddenIntent], content: str) -> list[str]:
        """
        Evaluate multiple un-matched intents in a single prompt to minimize API calls and latency.
        """
        cache_key = tuple(sorted(i.id for i in intents)) + (hash(content[:1500]),)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        intents_bullets = "\n".join(f"- [{i.id}]: {i.content}" for i in intents)
        check_prompt = (
            f"You are an evaluator assessing whether an AI assistant's reply satisfied specific requirements.\n\n"
            f"Assistant Response:\n\"\"\"{content[:2000]}\"\"\"\n\n"
            f"Requirements to check:\n{intents_bullets}\n\n"
            f"Output a valid JSON list of IDs for requirements that were satisfied or addressed. "
            f"Example: [\"{intents[0].id}\"]. If none were satisfied, output []."
        )

        try:
            raw = self.llm_client.generate(check_prompt, json_mode=True)
            clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            data = json.loads(clean)
            if isinstance(data, list):
                valid_ids = {i.id for i in intents}
                result = [str(x) for x in data if str(x) in valid_ids]
                self._batch_cache[cache_key] = result
                return result
        except Exception:
            pass

        self._batch_cache[cache_key] = []
        return []
