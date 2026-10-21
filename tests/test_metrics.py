"""
tests/test_metrics.py
Tests for MAVIS Telemetry & Metrics System (MetricEmitter, aggregation, trace, CLI tables, and component integration).
"""
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.metrics import (
    MetricEmitter,
    get_metrics_summary,
    get_turn_trace,
    format_metrics_tables,
    CSV_SCHEMAS,
)
from core.answerer import Answerer
from core.caching import CacheManager
from core.agents.base import BaseAgent


@pytest.fixture
def temp_metrics_dir(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("core.metrics.METRICS_DIR", str(metrics_path))
    return metrics_path


def test_metric_emitter_creates_header_and_logs(temp_metrics_dir):
    emitter = MetricEmitter("caching")
    file_path = Path(temp_metrics_dir) / "caching.csv"

    # Log record
    emitter.log({
        "turn_id": "t100",
        "cache_status": "hit",
        "hit_tier": "instant",
        "similarity_score": 0.95,
        "ttl_valid": True,
        "latency_ms": 1.2,
        "estimated_tokens_saved": 500,
    })

    assert file_path.exists()
    content = file_path.read_text()
    lines = content.strip().splitlines()
    assert len(lines) == 2  # Header + 1 record
    assert "turn_id" in lines[0]
    assert "t100" in lines[1]
    assert "instant" in lines[1]


def test_get_metrics_summary_aggregations(temp_metrics_dir):
    interp_emitter = MetricEmitter("interpreter")
    caching_emitter = MetricEmitter("caching")

    # Add 3 interpreter records
    interp_emitter.log({
        "turn_id": "t1",
        "latency_ms": 100.0,
        "status": "pipeline",
        "input_tokens": 1000,
        "output_tokens": 200,
        "tools_retrieved_count": 5,
    })
    interp_emitter.log({
        "turn_id": "t2",
        "latency_ms": 200.0,
        "status": "pipeline",
        "input_tokens": 1200,
        "output_tokens": 300,
        "tools_retrieved_count": 5,
    })
    interp_emitter.log({
        "turn_id": "t3",
        "latency_ms": 300.0,
        "status": "direct_response",
        "input_tokens": 800,
        "output_tokens": 100,
        "tools_retrieved_count": 5,
    })

    # Add caching records (2 hits, 1 miss)
    caching_emitter.log({
        "turn_id": "t1",
        "cache_status": "hit",
        "hit_tier": "instant",
        "similarity_score": 0.95,
        "ttl_valid": True,
        "latency_ms": 2.0,
        "estimated_tokens_saved": 500,
    })
    caching_emitter.log({
        "turn_id": "t2",
        "cache_status": "hit",
        "hit_tier": "llm_verified",
        "similarity_score": 0.88,
        "ttl_valid": True,
        "latency_ms": 150.0,
        "estimated_tokens_saved": 400,
    })
    caching_emitter.log({
        "turn_id": "t3",
        "cache_status": "miss",
        "hit_tier": "miss",
        "similarity_score": 0.40,
        "ttl_valid": False,
        "latency_ms": 1.0,
        "estimated_tokens_saved": 0,
    })

    summary = get_metrics_summary()

    # Verify interpreter stats: avg=200, median=200, max=300
    assert summary["interpreter"]["total_queries"] == 3
    assert summary["interpreter"]["latency_ms"]["avg"] == 200.0
    assert summary["interpreter"]["latency_ms"]["median"] == 200.0
    assert summary["interpreter"]["latency_ms"]["max"] == 300.0
    assert summary["tokens_total"]["input"] == 3000
    assert summary["tokens_total"]["output"] == 600

    # Verify caching stats: 2 hits / 3 checks = 66.7%
    assert summary["caching"]["checks"] == 3
    assert summary["caching"]["hits"] == 2
    assert summary["caching"]["misses"] == 1
    assert summary["caching"]["tokens_saved"] == 900
    assert 66.0 <= summary["caching"]["hit_rate_pct"] <= 67.0


def test_get_turn_trace_lazy_join(temp_metrics_dir):
    MetricEmitter("interpreter").log({
        "turn_id": "turn_abc",
        "latency_ms": 120.0,
        "status": "pipeline",
        "input_tokens": 500,
        "output_tokens": 100,
        "tools_retrieved_count": 4,
    })
    MetricEmitter("caching").log({
        "turn_id": "turn_abc",
        "cache_status": "miss",
        "hit_tier": "miss",
        "similarity_score": 0.2,
        "ttl_valid": False,
        "latency_ms": 1.5,
        "estimated_tokens_saved": 0,
    })
    MetricEmitter("dag_execution").log({
        "turn_id": "turn_abc",
        "start_time": "2026-09-01T12:00:00",
        "end_time": "2026-09-01T12:00:01",
        "latency_ms": 850.0,
        "status": "success",
        "dag_size": 2,
        "dag_depth": 2,
        "tool_nodes_count": 2,
        "subagent_nodes_count": 0,
        "failed_node_id": "",
    })

    # Unrelated turn
    MetricEmitter("interpreter").log({
        "turn_id": "turn_xyz",
        "latency_ms": 90.0,
        "status": "pipeline",
        "input_tokens": 400,
        "output_tokens": 80,
        "tools_retrieved_count": 2,
    })

    trace = get_turn_trace("turn_abc")
    assert len(trace) == 3
    components_in_trace = [entry["component"] for entry in trace]
    assert "interpreter" in components_in_trace
    assert "caching" in components_in_trace
    assert "dag_execution" in components_in_trace
    assert all(entry["turn_id"] == "turn_abc" for entry in trace)


def test_format_metrics_tables_returns_rich_tables(temp_metrics_dir):
    MetricEmitter("interpreter").log({
        "turn_id": "t1",
        "latency_ms": 150.0,
        "status": "pipeline",
        "input_tokens": 800,
        "output_tokens": 150,
        "tools_retrieved_count": 3,
    })

    tables = format_metrics_tables()
    assert len(tables) >= 3  # Summary KPI, Latency, Token tables


def test_answerer_metric_emission(temp_metrics_dir):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Synthesized answer from LLM."

    answerer = Answerer(mock_llm)
    res = answerer.synthesize(
        query="What is the weather?",
        pipeline_results={"step1": {"temp": 72}},
        turn_id="turn_answer_test",
    )
    assert res == "Synthesized answer from LLM."

    ans_file = Path(temp_metrics_dir) / "answerer.csv"
    assert ans_file.exists()
    content = ans_file.read_text()
    assert "turn_answer_test" in content
    assert "success" in content


def test_subagent_metric_emission(temp_metrics_dir):
    class DummyAgent(BaseAgent):
        name = "dummy_agent"
        description = "A test agent"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Agent processed response."

    agent = DummyAgent(mock_llm)
    status, result = agent.run(turn_id="turn_agent_test", data="test data")
    assert status == 0
    assert result == "Agent processed response."

    sub_file = Path(temp_metrics_dir) / "cognitive.csv"
    assert sub_file.exists()
    content = sub_file.read_text()
    assert "turn_agent_test" in content
    assert "dummy_agent" in content
