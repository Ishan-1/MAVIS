"""
tests/test_neo4j_knowledge_graph.py
Unit tests for Neo4jKnowledgeGraph with mock driver and verification of
Cypher query construction, topic filtering, and deterministic helpers.
"""
from unittest.mock import MagicMock, patch
import pytest

from memories.neo4j_graph import (
    Neo4jKnowledgeGraph,
    _slugify,
    _normalize_predicate,
)


def test_slugify_and_normalize_predicate():
    assert _slugify("Neovim Editor!") == "neovim_editor"
    assert _slugify("  User:Primary-123  ") == "user:primary-123"
    assert _normalize_predicate("prefers editor") == "PREFERS_EDITOR"
    assert _normalize_predicate("has-os") == "HAS_OS"
    assert _normalize_predicate("") == "RELATED_TO"


def test_neo4j_graceful_offline_fallback():
    with patch("memories.neo4j_graph.GraphDatabase.driver", side_effect=Exception("Connection refused")):
        kg = Neo4jKnowledgeGraph(uri="bolt://localhost:9999")
        assert kg.is_available() is False
        assert kg.upsert_entity("e1", "Test", "user.profile") == "e1"
        assert kg.query_active_facts("query", ["user.*"]) == []


def test_neo4j_upsert_entity_with_mock():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value.single.return_value = {"id": "entity:user"}

    with patch("memories.neo4j_graph.GraphDatabase.driver", return_value=mock_driver):
        kg = Neo4jKnowledgeGraph()
        res = kg.upsert_entity("entity:user", "User", "user.profile", "User", embedding=[0.1, 0.2])
        assert res == "entity:user"
        # Verify MERGE query was executed among session runs
        found_merge = False
        for call in mock_session.run.call_args_list:
            cypher_str = call[0][0]
            if "MERGE (e:Entity {id: $id})" in cypher_str:
                found_merge = True
                assert call[1]["id"] == "entity:user"
                assert call[1]["topic"] == "user.profile"
        assert found_merge is True


def test_neo4j_add_fact_functional_invalidation():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("memories.neo4j_graph.GraphDatabase.driver", return_value=mock_driver):
        with patch.object(Neo4jKnowledgeGraph, "_get_embedding", return_value=[0.1, 0.2]):
            kg = Neo4jKnowledgeGraph()
            kg.add_fact("User", "PREFERS_EDITOR", "Neovim", "user.profile", is_functional=True)

            # Check that Cypher contains functional superseding logic
            found_functional_cypher = False
            for call in mock_session.run.call_args_list:
                cypher_str = call[0][0]
                if "old_r.is_active = false" in cypher_str and "PREFERS_EDITOR" in cypher_str:
                    found_functional_cypher = True
                    assert call[1]["topic"] == "user.profile"
                    assert call[1]["is_functional"] is True
            assert found_functional_cypher is True


def test_neo4j_deterministic_subagent_helpers():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("memories.neo4j_graph.GraphDatabase.driver", return_value=mock_driver):
        with patch.object(Neo4jKnowledgeGraph, "_get_embedding", return_value=[0.1, 0.2]):
            kg = Neo4jKnowledgeGraph()

            # 1. Tool definition
            kg.add_tool_definition(
                tool_name="git_diff_tool",
                signature="git_diff_tool() -> str",
                summary="Generates git diffs",
                required_binaries=["git"],
                required_libs=["rich"],
            )

            # 2. Tool fix
            kg.add_tool_fix(
                tool_name="git_diff_tool",
                error_snippet="Banned import subprocess",
                fix_summary="Use oni.call_process instead",
            )

            # 3. Agent fix
            kg.add_agent_fix(
                agent_name="news_agent",
                failure_reason="Output was not valid JSON",
                prompt_remedy="Added few-shot JSON example in system prompt",
            )

            calls = mock_session.run.call_args_list
            assert len(calls) > 0


def test_neo4j_query_active_facts_formatting():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock query result
    mock_session.run.return_value = [
        {"subject": "User", "predicate": "PREFERS_EDITOR", "object": "Neovim", "topic": "user.profile"},
        {"subject": "Environment", "predicate": "HAS_OS", "object": "Linux", "topic": "env.system"},
    ]

    with patch("memories.neo4j_graph.GraphDatabase.driver", return_value=mock_driver):
        with patch.object(Neo4jKnowledgeGraph, "_get_embedding", return_value=[0.1, 0.2]):
            kg = Neo4jKnowledgeGraph()
            kg._vector_index_ready = True
            facts = kg.query_active_facts("what editor do I use?", allowed_topics=["user.*", "env.*"])
            assert "User prefers editor Neovim" in facts
            assert "Environment has os Linux" in facts

            # Verify that query filtered with prefixes
            args, kwargs = mock_session.run.call_args
            assert kwargs["prefixes"] == ["user.", "env."]
