"""
benchmarks/evaluators/memory_continuity.py
Longitudinal Memory Continuity evaluation engine.
Evaluates 7-day memory promotion (Working -> Short-Term -> Long-Term) and preference recall.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryContinuityEvaluator:
    """
    Evaluates cross-session memory retention and adherence to learned preferences.
    """

    def __init__(self, memories_root: Path | None = None) -> None:
        self.memories_root = memories_root or Path("memories")

    def evaluate_preference_retention(
        self,
        target_preference: str,
        day7_response: str,
        short_term_dir: Path | None = None,
    ) -> tuple[float, list[str]]:
        """
        Check if a preference declared on Day 1:
        1. Successfully persisted into short-term or long-term files.
        2. Was respected in Day 7 assistant outputs.
        """
        audit: list[str] = []
        score = 0.0

        # 1. Inspect short-term storage
        st_dir = short_term_dir or (self.memories_root / "short_term")
        found_in_storage = False
        if st_dir.exists():
            for json_file in st_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    text_dump = json.dumps(data).lower()
                    if target_preference.lower() in text_dump:
                        found_in_storage = True
                        audit.append(f"Preference found in memory file: {json_file.name}")
                        break
                except Exception:
                    continue

        if found_in_storage:
            score += 0.5
        else:
            audit.append("Preference NOT detected in short_term memory files.")

        # 2. Inspect Day 7 output adherence
        pref_lower = target_preference.lower()
        resp_lower = day7_response.lower()

        # Check negative directives (e.g. "never use nano")
        if "never" in pref_lower or "don't" in pref_lower or "do not" in pref_lower:
            disallowed_item = pref_lower.replace("never", "").replace("don't", "").replace("do not", "").strip()
            if disallowed_item and disallowed_item in resp_lower:
                audit.append(f"VIOLATION: Day 7 output included prohibited element '{disallowed_item}'")
            else:
                score += 0.5
                audit.append(f"Day 7 output successfully respected negative constraint: '{target_preference}'")
        else:
            # Positive directive check
            if any(term in resp_lower for term in pref_lower.split() if len(term) > 4):
                score += 0.5
                audit.append(f"Day 7 output aligned with preference: '{target_preference}'")
            else:
                audit.append(f"Day 7 output did not reflect preference: '{target_preference}'")

        return round(score, 2), audit
