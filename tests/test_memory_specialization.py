"""
Test memory specialization across namespaces and cross-namespace retrieval.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import get_llm_client
from memories.memory_store import MemoryStore


def test_namespaces_initialization():
    llm = get_llm_client()
    im = MemoryStore(llm, namespace="interpreter")
    tb = MemoryStore(llm, namespace="toolbuilder")
    dbg = MemoryStore(llm, namespace="debugger")

    assert im.namespace == "interpreter"
    assert tb.namespace == "toolbuilder"
    assert dbg.namespace == "debugger"


def test_cross_namespace_retrieval():
    llm = get_llm_client()
    tb = MemoryStore(llm, namespace="toolbuilder")
    dbg = MemoryStore(llm, namespace="debugger")

    # Record pattern in toolbuilder
    tb.write_pattern("calendar_query", "get_calendar_events()", "Opens data/calendar.json")

    # Record fix in debugger
    dbg.write_fix("calendar_query", "JSONDecodeError", "Ensure file exists before parsing")

    # Retrieve context from toolbuilder with debugger peer namespace
    ctx = tb.retrieve_context("calendar", extra_namespaces=["debugger"], top_k=2)

    assert "calendar_query" in ctx
    assert "Reference memories (debugger)" in ctx


if __name__ == "__main__":
    print("Running test_namespaces_initialization...")
    test_namespaces_initialization()
    print("test_namespaces_initialization: OK")

    print("Running test_cross_namespace_retrieval...")
    test_cross_namespace_retrieval()
    print("test_cross_namespace_retrieval: OK")
    print("All memory specialization tests passed!")
