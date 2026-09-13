"""
memories/knowledge_extractor.py
Decoupled background extractor for forming Knowledge Graph triples
from conversation turns and directives.
"""
from __future__ import annotations

import json
from typing import Any
from core.helpers import log_it
from core.llm.base import BaseLLMClient

_ENTITY = "knowledge_extractor"

_EXTRACTION_PROMPT = """
You are an ontology and knowledge graph extraction assistant for MAVIS.
Analyze the provided conversation text and extract concrete, long-term facts,
user preferences, environment attributes, or system rules.

For each fact, extract:
- "subject": canonical entity name (e.g. "User", "Environment", "Python", tool name)
- "predicate": relation verb in SCREAMING_SNAKE_CASE (e.g. "PREFERS_EDITOR", "HAS_OS", "USES_SHELL", "DEFAULT_TONE", "FAVORITE_LANGUAGE")
- "object": target entity value (e.g. "Neovim", "Linux", "zsh", "concise")
- "topic": dot-notated topic:
    * "user.profile" (personal facts, tone, habits, identity)
    * "user.rules" (explicit directives: always/never instructions)
    * "env.system" (OS, shell, paths, hardware, python runtime)
    * "env.binaries" (installed CLI tools)
    * "tooling.tools" (tool capabilities)
- "is_functional": boolean (true if 1-to-1 attribute like OS, editor, primary shell that supersedes previous values; false for multi-valued relations)

Return ONLY a valid JSON array of objects:
[
  {
    "subject": "User",
    "predicate": "PREFERS_EDITOR",
    "object": "Neovim",
    "topic": "user.profile",
    "is_functional": true
  }
]

If no concrete long-term facts or directives are declared, return an empty array [].

Text to analyze:
\"\"\"{content}\"\"\"
""".strip()


def extract_facts_from_turn(content: str, client: BaseLLMClient) -> list[dict]:
    """
    Extract structured knowledge triples from conversational content.
    Executes in background worker or post-turn queue, off the main conversational path.
    """
    if not content or not content.strip():
        return []

    prompt = _EXTRACTION_PROMPT.replace("{content}", content.strip())
    try:
        raw = client.generate(prompt, json_mode=True)
        # Parse JSON
        data = json.loads(raw)
        if isinstance(data, list):
            valid_facts = []
            for item in data:
                if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                    item.setdefault("topic", "user.profile")
                    item.setdefault("is_functional", True)
                    valid_facts.append(item)
            log_it(f"Extracted {len(valid_facts)} facts from text (len={len(content)}).", _ENTITY)
            return valid_facts
        return []
    except Exception as exc:
        log_it(f"Fact extraction failed: {exc}", _ENTITY)
        return []
