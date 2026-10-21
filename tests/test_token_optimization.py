"""
tests/test_token_optimization.py
Unit tests for Prefix Ordering, Prompt Compression (Strategy 1),
App-Layer History Control & Compaction (Strategy 2),
Dynamic Tool Retrieval (All General + Top-K Specific) (Strategy 3),
and Emotion / Directive Modernization.
"""
import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.prompt_templates import (
    interpreter_system_prompt,
    format_interpreter_user_prompt,
    interpreter_prompt,
)
from core.config import cfg
from core.llm import get_llm_client
from memories.memory_store import MemoryStore
from memories.emotion_classifier import (
    parse_classifier_fields,
    should_promote_short_term,
    should_promote_long_term,
)
from core.tool_retriever import ToolRetriever


def test_static_prompt_compression():
    """Verify Strategy 1: interpreter_system_prompt is static and contains rules & few-shot."""
    assert "direct_response" in interpreter_system_prompt
    assert "pipeline" in interpreter_system_prompt
    assert "missing_commands" in interpreter_system_prompt
    assert "Stanford prison experiment" in interpreter_system_prompt, "Few-shot example should be retained"
    assert "directive" in interpreter_system_prompt, "Directive boolean should be specified"
    assert '"low"' in interpreter_system_prompt, "Emotion strength should reference discrete levels"
    # Ensure static system prompt has no unformatted format variables
    assert "{commands_list}" not in interpreter_system_prompt
    assert "{user_input}" not in interpreter_system_prompt


def test_prefix_ordering_formatter():
    """Verify prefix ordering: Tools -> Memory Context -> User Input."""
    tools_str = '{"test_tool": "does test"}'
    context_str = "User likes concise answers"
    user_input = "Run the system test"

    formatted = format_interpreter_user_prompt(tools_str, context_str, user_input)

    pos_tools = formatted.find("### COMMANDS LIST:")
    pos_context = formatted.find("### MEMORY CONTEXT:")
    pos_input = formatted.find("### USER INPUT:")

    assert pos_tools != -1, "Tools section missing"
    assert pos_context != -1, "Context section missing"
    assert pos_input != -1, "User input missing"
    assert pos_tools < pos_context < pos_input, "Prefix ordering violated!"


def test_working_memory_capping_and_truncation():
    """Verify Strategy 2: Working memory is capped to active turns and entries truncated."""
    llm = get_llm_client()
    ms = MemoryStore(llm, namespace="interpreter")

    # Add 15 turns
    with ms._lock:
        ms._working.clear()
        for i in range(15):
            ms._working.append({
                "id": f"turn-{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Turn message {i}: " + ("x" * 800),  # Long content > max_memory_entry_chars (600)
                "timestamp": 1000.0 + i,
                "emotion": "neutral",
                "emotion_strength": "low",
                "directive": False,
            })

    ctx = ms.retrieve_context("test query")

    # Check active turns cap (default 8)
    # The oldest turns (0 to 6) should not appear in working memory section
    assert "Turn message 14:" in ctx
    assert "Turn message 13:" in ctx
    assert "Turn message 0:" not in ctx
    assert "Turn message 1:" not in ctx

    # Check defensive truncation
    assert "... [truncated]" in ctx

    # Check compaction budget is 0.5 * max_token
    assert ms._token_budget() == int(0.5 * cfg.memory.get("max_token", 12000))


def test_classifier_modernization():
    """Verify emotion_strength (low/medium/high) and directive boolean parsing and promotions."""
    # Test parser
    emo, emo_s, directive = parse_classifier_fields({
        "emotion": "FRUSTRATION",
        "emotion_strength": "HIGH",
        "directive": True,
    })
    assert emo == "frustration"
    assert emo_s == "high"
    assert directive is True

    # Test short-term promotions
    assert should_promote_short_term("high", False) is True
    assert should_promote_short_term("low", True) is True
    assert should_promote_short_term("low", False) is False

    # Test long-term promotions
    promote, ltype = should_promote_long_term(True, "Always prefer dark mode")
    assert promote is True
    assert ltype == "behaviour"

    promote_no_dir, _ = should_promote_long_term(False, "Always prefer dark mode")
    assert promote_no_dir is False


def test_all_general_plus_topk_specific_tools():
    """Verify Strategy 3: All 'generalizable' tools are supplied + Top-K domain tools."""
    temp_dir = tempfile.mkdtemp()
    try:
        llm = get_llm_client()
        tr = ToolRetriever(client=llm, chroma_path=temp_dir)

        # Register 4 general tools
        general_commands = {
            "get_current_datetime() -> tuple[int, str]": "Retrieves current date and time.",
            "extract_date_from_datetime(datetime_string: str) -> tuple[int, str]": "Extracts date part.",
            "parse_natural_date_to_yyyymmdd(date_string: str) -> tuple[int, str]": "Parses natural dates.",
            "read_file_contents(filename: str) -> tuple[int, str]": "Reads file contents.",
        }
        for k, v in general_commands.items():
            tr.index_tool(k, v, generalizability="generalizable")
            assert tr.get_tool_generalizability(k) == "generalizable"

        # Register 4 repurposable domain tools and 4 specialized tools (total 12 tools)
        domain_commands = {}
        for i in range(4):
            sig = f"repurposable_tool_{i}(query: str) -> tuple[int, str]"
            desc = f"Searches domain source {i}."
            domain_commands[sig] = desc
            tr.index_tool(sig, desc, generalizability="repurposable")
            assert tr.get_tool_generalizability(sig) == "repurposable"

        for i in range(4):
            sig = f"specialized_tool_{i}(query: str) -> tuple[int, str]"
            desc = f"Executes specialized routine {i}."
            domain_commands[sig] = desc
            tr.index_tool(sig, desc, generalizability="specialized")
            assert tr.get_tool_generalizability(sig) == "specialized"

        all_commands = {**general_commands, **domain_commands}

        # Query for tools
        selected = tr.get_relevant_tools(
            "What is today's date?",
            all_commands,
            specific_top_k=2,
        )

        # ALL 4 general tools MUST be present
        for gen_k in general_commands:
            assert gen_k in selected, f"General tool '{gen_k}' was omitted!"

        # Domain tools should be bounded by specific_top_k (2)
        domain_selected = [k for k in selected if k in domain_commands]
        assert len(domain_selected) <= 2

        # Total selected tools = 4 general + at most 2 domain = at most 6 tools (out of 12 total)
        assert len(selected) <= 6
        assert len(selected) < len(all_commands)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_commands_list_json_format():
    """Verify data/commands_list.json matches the structured generalizability schema."""
    import json
    with open("data/commands_list.json", "r") as f:
        cmds = json.load(f)

    for sig, val in cmds.items():
        assert isinstance(val, dict), f"Command '{sig}' is not a dict"
        assert "description" in val, f"Missing description in '{sig}'"
        assert "generalizability" in val, f"Missing generalizability in '{sig}'"
        assert val["generalizability"] in ("specialized", "repurposable", "generalizable")


if __name__ == "__main__":
    print("Running test_static_prompt_compression...")
    test_static_prompt_compression()
    print("OK")

    print("Running test_prefix_ordering_formatter...")
    test_prefix_ordering_formatter()
    print("OK")

    print("Running test_working_memory_capping_and_truncation...")
    test_working_memory_capping_and_truncation()
    print("OK")

    print("Running test_classifier_modernization...")
    test_classifier_modernization()
    print("OK")

    print("Running test_all_general_plus_topk_specific_tools...")
    test_all_general_plus_topk_specific_tools()
    print("OK")

    print("Running test_commands_list_json_format...")
    test_commands_list_json_format()
    print("OK")

    print("All token optimization & classifier tests passed successfully!")
