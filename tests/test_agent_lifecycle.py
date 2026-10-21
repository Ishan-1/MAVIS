"""
tests/test_agent_lifecycle.py
Comprehensive test suite for Sub-Agent architecture, AgentBuilder, AgentTester (LLM-as-a-Judge),
AgentDebugger, BaseAgent contracts, and Answerer synthesis.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.base import BaseAgent
from core.agents import load_agent
from agents.semantic_transform import SemanticTransformAgent
from agent_builder.tester import AgentTester
from agent_builder.debugger import AgentDebugger
from core.answerer import Answerer
from core.llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for deterministic unit testing without hitting external network."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.last_prompt = None
        self.last_system_instruction = None

    def generate(self, prompt: str, json_mode: bool = False, system_instruction: str = None) -> str:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction

        # Check for specific prompt keywords
        if "expert LLM-as-a-Judge" in (system_instruction or "") or "verdict" in prompt:
            return json.dumps({
                "verdict": "passed",
                "reason": "Test output matches all criteria faithfully."
            })
        elif "A candidate sub-agent failed its LLM-as-a-Judge" in prompt or "DEBUGGING INSTRUCTIONS" in prompt:
            return json.dumps({
                "code": "# fixed code",
                "fix_summary": "Added negative constraint banning pleasantries."
            })
        elif "generate 2–3 realistic, diverse synthetic test inputs" in prompt:
            return json.dumps({
                "test_cases": [
                    {"id": "tc1", "description": "Happy path", "inputs": {"content": "Sample news text", "instruction": "Extract headline"}},
                    {"id": "tc2", "description": "Edge case", "inputs": {"content": "", "instruction": "Extract headline"}}
                ]
            })
        elif "You are an internal semantic processing agent" in (system_instruction or ""):
            return "Extracted Headline: Mars Rover Finds Ancient Riverbed"
        elif "You are MAVIS" in (system_instruction or ""):
            return "Here are the top headlines: Mars rover discovered ancient riverbeds on Mars."

        return self.responses.get("default", "mocked response")

    def embed(self, text: str) -> list[float]:
        return [0.0] * 128


def test_base_agent_contract_and_payload_guard():
    """Verify BaseAgent handles payload size guards and execution correctly."""
    mock_llm = MockLLMClient()
    agent = SemanticTransformAgent(mock_llm)

    # 1. Normal execution
    status, result = agent.run(content="Some article content", instruction="Summarize it")
    assert status == 0
    assert "Extracted Headline" in result

    # 2. Payload guard test (large payload)
    large_payload = "A" * 40000
    status, result = agent.run(content=large_payload, instruction="Summarize")
    assert status == 0
    assert "[TRUNCATED DUE TO SIZE LIMIT]" in mock_llm.last_prompt


def test_base_agent_output_schema_validation():
    """Verify BaseAgent validates output against output_schema."""
    class StructuredAgent(BaseAgent):
        name = "structured_agent"
        output_schema = {"required": ["summary", "score"]}

    # Case A: Valid JSON output
    mock_llm_valid = MockLLMClient({"default": json.dumps({"summary": "Great", "score": 10})})
    agent_valid = StructuredAgent(mock_llm_valid)
    status, res = agent_valid.run(text="input")
    assert status == 0
    assert res == {"summary": "Great", "score": 10}

    # Case B: Missing required key
    mock_llm_invalid = MockLLMClient({"default": json.dumps({"summary": "Great"})})
    agent_invalid = StructuredAgent(mock_llm_invalid)
    status, res = agent_invalid.run(text="input")
    assert status == -1
    assert "Missing required key 'score'" in res


def test_agent_tester_discrete_judge():
    """Verify AgentTester returns strictly binary 'passed' / 'failed' verdicts."""
    mock_llm = MockLLMClient()
    tester = AgentTester(mock_llm)
    agent = SemanticTransformAgent(mock_llm)

    # Run test suite
    status, res = tester.test_agent(agent)
    assert status == 0
    assert "passed" in res["summary"]

    # Test judge evaluation directly
    verdict, reason = tester.evaluate_output(
        agent_name="test_agent",
        agent_description="Extracts data",
        system_instruction="Role instruction",
        output_schema=None,
        test_input={"text": "hello"},
        actual_output="hello extracted",
    )
    assert verdict in ("passed", "failed")
    assert isinstance(reason, str)


def test_agent_debugger():
    """Verify AgentDebugger analyzes Judge failures and produces prompt fixes."""
    mock_llm = MockLLMClient()
    debugger = AgentDebugger(mock_llm)

    fixed_code, fix_summary = debugger.debug_agent(
        agent_name="summarizer",
        agent_description="Summarizes articles",
        broken_code="class SummarizerAgent(BaseAgent): pass",
        failed_case={"inputs": {"text": "hello"}},
        actual_output="Sure, here is the summary: hello",
        failure_reason="Included conversational pleasantry 'Sure, here is'",
    )

    assert fixed_code == "# fixed code"
    assert "negative constraint" in fix_summary


def test_answerer_synthesis():
    """Verify Answerer synthesizes terminal responses with <tool_data> quarantine."""
    mock_llm = MockLLMClient()
    answerer = Answerer(mock_llm)

    results = {
        "n1": [{"title": "Mars Discovery", "desc": "Water found"}]
    }

    answer = answerer.synthesize(
        query="What is the latest Mars news?",
        pipeline_results=results,
        memory_context="User prefers concise answers",
    )

    assert "Mars rover discovered" in answer
    assert "<tool_data>" in mock_llm.last_prompt
    assert "Mars Discovery" in mock_llm.last_prompt


def test_load_agent_dynamic():
    """Verify dynamic loading of agents from agents/ directory."""
    mock_llm = MockLLMClient()
    agent = load_agent("semantic_transform", mock_llm)
    assert agent is not None
    assert agent.name == "semantic_transform"


def test_pipeline_heterogeneous_dag():
    """Verify execute_pipeline can run a heterogeneous DAG (tool + subagent) and synthesize with Answerer."""
    import main as mavis_main

    # Setup pipeline with a deterministic tool and a subagent
    pipeline = [
        {
            "id": "n1",
            "type": "tool",
            "function_name": "extract_date_from_datetime",
            "params": {"datetime_string": "2026-08-31 11:30:00"}
        },
        {
            "id": "n2",
            "type": "cognitive",
            "function_name": "semantic_transform",
            "params": {
                "content": "$n1.output",
                "instruction": "Format the date as human readable text"
            }
        }
    ]

    # Save original answerer and inject mock
    orig_answerer = mavis_main.answerer
    orig_notify = mavis_main._notify
    mavis_main.answerer = Answerer(MockLLMClient())
    mavis_main._notify = lambda *args, **kwargs: None

    try:
        # execute_pipeline runs without error
        mavis_main.execute_pipeline(pipeline, query="What is the date in 2026-08-31 11:30:00?")
    finally:
        mavis_main.answerer = orig_answerer
        mavis_main._notify = orig_notify


if __name__ == "__main__":
    print("Running test_base_agent_contract_and_payload_guard...")
    test_base_agent_contract_and_payload_guard()
    print("OK")

    print("Running test_base_agent_output_schema_validation...")
    test_base_agent_output_schema_validation()
    print("OK")

    print("Running test_agent_tester_discrete_judge...")
    test_agent_tester_discrete_judge()
    print("OK")

    print("Running test_agent_debugger...")
    test_agent_debugger()
    print("OK")

    print("Running test_answerer_synthesis...")
    test_answerer_synthesis()
    print("OK")

    print("Running test_load_agent_dynamic...")
    test_load_agent_dynamic()
    print("OK")

    print("Running test_pipeline_heterogeneous_dag...")
    test_pipeline_heterogeneous_dag()
    print("OK")

    print("\nAll Sub-Agent & Lifecycle tests passed successfully!")
