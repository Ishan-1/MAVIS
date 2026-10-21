"""
tests/test_memory_store_neo4j.py
Unit tests verifying MemoryStore topic subscription boundaries and
integration with Neo4j Knowledge Graph without ChromaDB.
"""
from unittest.mock import MagicMock, patch
import pytest

from memories.memory_store import MemoryStore


def test_memory_store_topic_boundaries():
    with patch("memories.memory_store.Neo4jKnowledgeGraph"):
        store_interp = MemoryStore(namespace="interpreter")
        store_tool = MemoryStore(namespace="toolbuilder")
        store_debug = MemoryStore(namespace="debugger")
        store_agent = MemoryStore(namespace="agent_debugger")

        # Interpreter has access to user and environment
        assert any(t.startswith("user") for t in store_interp.read_topics)
        assert any(t.startswith("env") for t in store_interp.read_topics)

        # ToolBuilder MUST NOT have access to user data
        assert not any(t.startswith("user") for t in store_tool.read_topics)
        assert any(t.startswith("tooling") for t in store_tool.read_topics)
        assert any(t.startswith("env") for t in store_tool.read_topics)

        # Debugger MUST NOT have access to user data
        assert not any(t.startswith("user") for t in store_debug.read_topics)
        assert any(t.startswith("debugging") for t in store_debug.read_topics)

        # AgentDebugger MUST NOT have access to user data
        assert not any(t.startswith("user") for t in store_agent.read_topics)
        assert any(t.startswith("agents") for t in store_agent.read_topics)


def test_retrieve_context_active_knowledge_injection():
    with patch("memories.memory_store.embed", return_value=[0.1] * 768):
        mock_kg = MagicMock()
        mock_kg.is_available.return_value = True
        mock_kg.query_active_facts.return_value = [
            "User prefers editor Neovim",
            "Environment has os Linux",
        ]

        with patch("memories.memory_store.Neo4jKnowledgeGraph", return_value=mock_kg):
            store = MemoryStore(namespace="interpreter")
            context = store.retrieve_context("what editor?")

            assert "### Active Knowledge (interpreter subscribed)" in context
            assert "User prefers editor Neovim" in context
            assert "Environment has os Linux" in context

            # Verify kg was queried with interpreter's read_topics
            mock_kg.query_active_facts.assert_called_once()
            args, kwargs = mock_kg.query_active_facts.call_args
            assert kwargs["allowed_topics"] == store.read_topics


def test_toolbuilder_retrieval_does_not_call_user_topics():
    with patch("memories.memory_store.embed", return_value=[0.1] * 768):
        mock_kg = MagicMock()
        mock_kg.is_available.return_value = True
        mock_kg.query_active_facts.return_value = ["Tool curl requires binary curl"]

        with patch("memories.memory_store.Neo4jKnowledgeGraph", return_value=mock_kg):
            store = MemoryStore(namespace="toolbuilder")
            context = store.retrieve_context("build curl tool")

            assert "### Active Knowledge (toolbuilder subscribed)" in context
            assert "Tool curl requires binary curl" in context

            # Verify toolbuilder read_topics contains no user topic
            args, kwargs = mock_kg.query_active_facts.call_args
            for topic in kwargs["allowed_topics"]:
                assert not topic.startswith("user")


def test_memory_store_helpers_call_neo4j():
    mock_kg = MagicMock()
    mock_kg.is_available.return_value = True

    with patch("memories.memory_store.Neo4jKnowledgeGraph", return_value=mock_kg):
        store = MemoryStore(namespace="toolbuilder")

        # write_pattern calls kg.add_tool_definition
        store.write_pattern("my_tool", "my_tool() -> int", "summarized")
        mock_kg.add_tool_definition.assert_called_once_with(
            "my_tool", "my_tool() -> int", "summarized", client=store._client
        )

        # write_fix calls kg.add_tool_fix
        store.write_fix("my_tool", "ImportError", "Use oni")
        mock_kg.add_tool_fix.assert_called_once_with(
            "my_tool", "ImportError", "Use oni", client=store._client
        )

        # write_agent_fix calls kg.add_agent_fix
        store.write_agent_fix("agent_a", "Failed JSON", "Add few-shot")
        mock_kg.add_agent_fix.assert_called_once_with(
            "agent_a", "Failed JSON", "Add few-shot", client=store._client
        )

        # add_fact calls kg.add_fact
        store.add_fact("Subject", "PREDICATE", "Object")
        mock_kg.add_fact.assert_called_once()
        _, kwargs = mock_kg.add_fact.call_args
        assert kwargs["subject"] == "Subject"
        assert kwargs["predicate"] == "PREDICATE"
        assert kwargs["obj"] == "Object"
        assert kwargs["topic"] == store.write_topic
