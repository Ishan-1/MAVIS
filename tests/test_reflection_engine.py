"""
tests/test_reflection_engine.py
End-to-end unit tests for ReflectionEngine using mocked BaseLLMClient.
"""
import json
import pytest
from unittest.mock import MagicMock
from core.llm.base import BaseLLMClient
from core.reflection.engine import ReflectionEngine, ReflectionResult


class MockLLM(BaseLLMClient):
    def __init__(self, responses: list[str] | None = None, embed_dim: int = 4):
        self.responses = list(responses or [])
        self.call_count = 0
        self.embed_dim = embed_dim

    def generate(self, prompt: str, json_mode: bool = False, system_instruction: str | None = None, temperature: float | None = None) -> str:
        if self.responses:
            return self.responses.pop(0)
        return "{}"

    def embed(self, text: str) -> list[float]:
        idx = abs(hash(text)) % self.embed_dim
        vec = [0.0] * self.embed_dim
        vec[idx] = 1.0
        return vec


def test_reflection_triage_abort(tmp_path):
    # Setup mock LLM that generates low triage score
    triage_reject = json.dumps({
        "utility_score": 0.45,
        "category": "trivia",
        "rationale": "General fact with no operational impact.",
        "pass": False,
    })
    mock_llm = MockLLM(responses=[triage_reject])

    engine = ReflectionEngine(client=mock_llm)
    result = engine.run_cycle(seed="trivial trivia fact")

    assert result.passed_triage is False
    assert result.status == "triaged_out"
    assert result.iterations_used == 0
    assert result.committed is False


def test_reflection_full_accepted_cycle(tmp_path, monkeypatch):
    # 1. Triage pass (score = 0.88)
    triage_resp = json.dumps({
        "utility_score": 0.88,
        "category": "rule_formation",
        "rationale": "High-value CLI flag rule for ripgrep.",
        "pass": True,
    })
    # 2. Generator turn 1: web search
    gen_resp_1 = json.dumps({
        "hypothesis": "Ripgrep requires --pcre2 for complex lookaround assertions.",
        "action_type": "web_search",
        "action_payload": "ripgrep lookaround pcre2",
        "reasoning": "Verify lookaround support",
    })
    # 3. Critic turn 1: ACCEPT
    critic_resp_1 = json.dumps({
        "status": "ACCEPT",
        "critique": "Solid empirical evidence from ripgrep documentation.",
        "synthesis": "When executing lookaround regex in ripgrep, pass --pcre2 to enable PCRE2 engine and avoid silent syntax errors.",
    })

    mock_llm = MockLLM(responses=[triage_resp, gen_resp_1, critic_resp_1])
    engine = ReflectionEngine(client=mock_llm, reflections_dir=tmp_path)

    # Mock web search to return dummy docs
    monkeypatch.setattr(
        "core.reflection.engine.search_web_sandboxed",
        lambda query, max_results=3: (0, "ripgrep manpage: --pcre2 enables PCRE2 engine supporting lookahead/lookbehind assertions.")
    )

    result = engine.run_cycle(seed="ripgrep regex assertions")

    assert result.passed_triage is True
    assert result.status == "accepted"
    assert result.iterations_used == 1
    assert result.web_searches_count == 1
    assert result.committed is True
    assert result.storage_path is not None
    assert "pcre2" in result.synthesis.lower()


def test_reflection_low_delta_rejection(tmp_path, monkeypatch):
    triage_resp = json.dumps({
        "utility_score": 0.90,
        "category": "rule_formation",
        "rationale": "Appears useful initially.",
        "pass": True,
    })
    gen_resp = json.dumps({
        "hypothesis": "Test hypothesis",
        "action_type": "conclude",
        "action_payload": "Redundant text",
        "reasoning": "done",
    })
    critic_resp = json.dumps({
        "status": "ACCEPT",
        "critique": "Accepted but redundant",
        "synthesis": "Identical synthesis",
    })

    # Embed method returns identical vector for all texts -> similarity = 1.0 -> delta = 0.0 < 0.25
    class FlatLLM(MockLLM):
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

    mock_llm = FlatLLM(responses=[triage_resp, gen_resp, critic_resp])
    engine = ReflectionEngine(client=mock_llm)

    # Provide existing reference vector identical to synthesis
    monkeypatch.setattr(
        engine,
        "_get_reference_embeddings",
        lambda query_vec, top_k=10: [[1.0, 0.0, 0.0, 0.0]]
    )

    result = engine.run_cycle(seed="some seed")
    assert result.passed_triage is True
    assert result.status == "low_delta"
    assert result.committed is False
    assert result.semantic_delta == pytest.approx(0.0)


def test_reflection_turn_ceiling_consolidation(tmp_path, monkeypatch):
    triage_resp = json.dumps({
        "utility_score": 0.85,
        "category": "rule_formation",
        "rationale": "High-value operational topic",
        "pass": True,
    })
    # Turn 1
    gen_1 = json.dumps({"hypothesis": "Hyp 1", "action_type": "web_search", "action_payload": "search 1", "reasoning": "r1"})
    critic_1 = json.dumps({"status": "REFINE", "critique": "Needs more proof", "synthesis": None})
    # Turn 2
    gen_2 = json.dumps({"hypothesis": "Hyp 2", "action_type": "web_search", "action_payload": "search 2", "reasoning": "r2"})
    critic_2 = json.dumps({"status": "REFINE", "critique": "Almost there", "synthesis": None})
    # Turn 3
    gen_3 = json.dumps({"hypothesis": "Hyp 3", "action_type": "conclude", "action_payload": "conclusion 3", "reasoning": "r3"})
    critic_3 = json.dumps({"status": "REFINE", "critique": "Still fine-tuning", "synthesis": None})
    # Consolidation
    cons_resp = json.dumps({
        "synthesis": "Verified consolidated insight extracted from the 3 turns of empirical evidence.",
        "summary_hypothesis": "Final validated hypothesis",
    })

    mock_llm = MockLLM(
        responses=[triage_resp, gen_1, critic_1, gen_2, critic_2, gen_3, critic_3, cons_resp]
    )
    engine = ReflectionEngine(client=mock_llm, reflections_dir=tmp_path)

    monkeypatch.setattr(
        "core.reflection.engine.search_web_sandboxed",
        lambda query, max_results=3: (0, "Sample search snippet for " + query)
    )

    result = engine.run_cycle(seed="test ceiling topic")

    assert result.passed_triage is True
    assert result.status == "max_turns_consolidated"
    assert result.iterations_used == 3
    assert result.committed is True
    assert "consolidated insight" in result.synthesis.lower()


def test_reflection_rejected_by_critic_preserves_reasoning(tmp_path, monkeypatch):
    triage_resp = json.dumps({
        "utility_score": 0.85,
        "category": "rule_formation",
        "rationale": "Appears to be a valid decision framework",
        "pass": True,
    })
    gen_resp = json.dumps({
        "hypothesis": "Jev Decision Model is an established framework.",
        "action_type": "web_search",
        "action_payload": "Jev Decision Model",
        "reasoning": "Verify public documentation exists.",
    })
    rejection_critique = "No evidence found across searches. Concept has no grounding in technical literature."
    critic_resp = json.dumps({
        "status": "REJECT",
        "critique": rejection_critique,
        "synthesis": None,
    })

    mock_llm = MockLLM(responses=[triage_resp, gen_resp, critic_resp])
    engine = ReflectionEngine(client=mock_llm, reflections_dir=tmp_path)

    monkeypatch.setattr(
        "core.reflection.engine.search_web_sandboxed",
        lambda query, max_results=3: (0, "No results found")
    )

    result = engine.run_cycle(seed="Jev Decision Model")

    assert result.passed_triage is True
    assert result.status == "rejected"
    assert result.committed is False
    assert result.rejection_reason == rejection_critique
    assert len(result.history) == 1
    assert result.history[0]["generator_reasoning"] == "Verify public documentation exists."
    assert result.history[0]["critique"] == rejection_critique


def test_seed_pool_management(tmp_path):
    from core.reflection.engine import load_seed_pool, add_seed_to_pool, remove_seed_from_pool

    assert load_seed_pool(tmp_path) == []

    add_seed_to_pool("Subprocess non-blocking stream multiplexing", reflections_dir=tmp_path)
    add_seed_to_pool("DAG idempotent execution recovery", reflections_dir=tmp_path)

    pool = load_seed_pool(tmp_path)
    assert len(pool) == 2
    assert "Subprocess non-blocking stream multiplexing" in pool

    remove_seed_from_pool("Subprocess non-blocking stream multiplexing", reflections_dir=tmp_path)
    pool_after = load_seed_pool(tmp_path)
    assert len(pool_after) == 1
    assert "Subprocess non-blocking stream multiplexing" not in pool_after
    assert "DAG idempotent execution recovery" in pool_after


def test_concept_extraction_from_memory(tmp_path):
    concept_resp = json.dumps({
        "concept": "Large payload offloading in subprocess execution"
    })
    mock_llm = MockLLM(responses=[concept_resp])
    engine = ReflectionEngine(client=mock_llm, reflections_dir=tmp_path)

    memory_text = "Tool outputs exceeding 4000 bytes offloaded to data/scratch/ with compact head/tail digest."
    concept = engine._extract_concept_from_memory(memory_text)
    assert concept == "Large payload offloading in subprocess execution"


def test_empty_seed_pool_aborts_cleanly(tmp_path):
    mock_llm = MockLLM()
    # Empty reflections directory with no seeds and empty store
    engine = ReflectionEngine(client=mock_llm, reflections_dir=tmp_path)

    result = engine.run_cycle(seed=None)
    assert result.status == "no_seeds_available"
    assert result.passed_triage is False
    assert result.iterations_used == 0
    assert result.committed is False


def test_journal_verify_and_delete_lifecycle(tmp_path, monkeypatch):
    from core.reflection.engine import (
        load_reflection_journal,
        verify_and_promote_reflection,
        delete_reflection_entry,
    )

    triage_resp = json.dumps({
        "utility_score": 0.88,
        "category": "rule_formation",
        "rationale": "High-value CLI flag rule for ripgrep.",
        "pass": True,
    })
    gen_resp_1 = json.dumps({
        "hypothesis": "Ripgrep requires --pcre2 for complex lookaround assertions.",
        "action_type": "web_search",
        "action_payload": "ripgrep lookaround pcre2",
        "reasoning": "Verify lookaround support",
    })
    critic_resp_1 = json.dumps({
        "status": "ACCEPT",
        "critique": "Solid empirical evidence from ripgrep documentation.",
        "synthesis": "When executing lookaround regex in ripgrep, pass --pcre2 to enable PCRE2 engine and avoid silent syntax errors.",
    })

    mock_llm = MockLLM(responses=[triage_resp, gen_resp_1, critic_resp_1])
    engine = ReflectionEngine(client=mock_llm, reflections_dir=tmp_path)

    monkeypatch.setattr(
        "core.reflection.engine.search_web_sandboxed",
        lambda query, max_results=3: (0, "ripgrep documentation: --pcre2 enables PCRE2 engine supporting lookarounds.")
    )

    result = engine.run_cycle(seed="ripgrep regex assertions")
    assert result.committed is True

    # 1. Verify entry is in journal as pending_review
    journal = load_reflection_journal(tmp_path)
    assert len(journal) == 1
    entry = journal[0]
    assert entry["id"] == result.id
    assert entry["verified"] is False
    assert entry["status"] == "pending_review"

    # 2. Promote reflection
    promoted = verify_and_promote_reflection(result.id, client=mock_llm, reflections_dir=tmp_path)
    assert promoted is True

    journal_after_prom = load_reflection_journal(tmp_path)
    assert journal_after_prom[0]["verified"] is True
    assert journal_after_prom[0]["status"] == "verified"

    # 3. Delete reflection
    deleted = delete_reflection_entry(result.id, reflections_dir=tmp_path)
    assert deleted is True

    journal_after_del = load_reflection_journal(tmp_path)
    assert len(journal_after_del) == 0

