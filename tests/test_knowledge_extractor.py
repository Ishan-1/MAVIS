"""
tests/test_knowledge_extractor.py
Unit tests for decoupled background knowledge extractor.
"""
from unittest.mock import MagicMock
import pytest

from memories.knowledge_extractor import extract_facts_from_turn


def test_extract_facts_from_turn_empty():
    client = MagicMock()
    assert extract_facts_from_turn("", client) == []
    assert extract_facts_from_turn("   ", client) == []
    client.generate.assert_not_called()


def test_extract_facts_from_turn_success():
    client = MagicMock()
    client.generate.return_value = """
    [
      {
        "subject": "User",
        "predicate": "PREFERS_EDITOR",
        "object": "Neovim",
        "topic": "user.profile",
        "is_functional": true
      },
      {
        "subject": "Environment",
        "predicate": "HAS_OS",
        "object": "Ubuntu",
        "topic": "env.system",
        "is_functional": true
      }
    ]
    """

    facts = extract_facts_from_turn("From now on I use Neovim on Ubuntu.", client)
    assert len(facts) == 2
    assert facts[0]["subject"] == "User"
    assert facts[0]["predicate"] == "PREFERS_EDITOR"
    assert facts[0]["object"] == "Neovim"
    assert facts[0]["topic"] == "user.profile"
    assert facts[0]["is_functional"] is True

    assert facts[1]["subject"] == "Environment"
    assert facts[1]["predicate"] == "HAS_OS"
    assert facts[1]["object"] == "Ubuntu"
    assert facts[1]["topic"] == "env.system"


def test_extract_facts_from_turn_invalid_json_fallback():
    client = MagicMock()
    client.generate.return_value = "Sorry, cannot extract facts."
    facts = extract_facts_from_turn("Hello world", client)
    assert facts == []
