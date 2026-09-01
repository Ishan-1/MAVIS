"""
core/metrics.py

Lightweight, append-only CSV metrics emitter and aggregator for MAVIS.
Provides:
  - MetricEmitter class initialized per component with .log(data)
  - Thread-safe metric emission to data/metrics/{component_name}.csv
  - Independent component aggregation (Average, Median, Max, Totals)
  - Rich table formatting for the CLI /metrics command
  - On-demand single-turn trace inspection across CSVs
"""
from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
from typing import Any
from rich.table import Table

_MAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METRICS_DIR = os.path.join(_MAV_ROOT, "data", "metrics")
METRICS_DIR = _METRICS_DIR
_LOCK = threading.Lock()

# Standard field schemas per component
CSV_SCHEMAS: dict[str, list[str]] = {
    "interpreter": [
        "timestamp", "turn_id", "latency_ms", "status",
        "input_tokens", "output_tokens", "tools_retrieved_count"
    ],
    "answerer": [
        "timestamp", "turn_id", "latency_ms", "status",
        "input_tokens", "output_tokens"
    ],
    "dag_execution": [
        "timestamp", "turn_id", "start_time", "end_time", "latency_ms",
        "status", "dag_size", "dag_depth", "tool_nodes_count",
        "subagent_nodes_count", "failed_node_id"
    ],
    "caching": [
        "timestamp", "turn_id", "cache_status", "hit_tier",
        "similarity_score", "llm_verify_result", "ttl_valid",
        "latency_ms", "tokens_saved_estimate", "estimated_tokens_saved"
    ],
    "builders": [
        "timestamp", "target_name", "builder_type", "latency_ms",
        "status", "attempt_count", "failure_reason",
        "debugger_prior_used", "input_tokens", "output_tokens"
    ],
    "subagents": [
        "timestamp", "turn_id", "agent_name", "latency_ms",
        "status", "input_tokens", "output_tokens", "payload_truncated"
    ],
    "oni": [
        "timestamp", "turn_id", "target_command_or_path", "phase",
        "trust_level", "oni_decision", "user_decision", "dwell_time_ms"
    ],
    "memory": [
        "timestamp", "event_type", "working_tokens_count",
        "compaction_triggered", "tokens_freed", "turns_evaluated",
        "turns_promoted", "facts_consolidated"
    ],
}


class MetricEmitter:
    """
    Component-specific metrics emitter.

    Usage:
        emitter = MetricEmitter("caching")
        emitter.log({"turn_id": "8a7f1e92", "cache_status": "hit", "latency_ms": 42.5})
    """
    def __init__(self, component_name: str, metrics_dir: str | None = None):
        self.component_name = component_name
        self._custom_dir = metrics_dir
        self.schema = CSV_SCHEMAS.get(component_name)

    @property
    def metrics_dir(self) -> str:
        return self._custom_dir or METRICS_DIR

    @property
    def filepath(self) -> str:
        return os.path.join(self.metrics_dir, f"{self.component_name}.csv")

    def log(self, data: dict[str, Any]) -> None:
        """
        Append a metrics record to data/metrics/{component_name}.csv.
        Guaranteed not to raise exceptions to caller.
        """
        try:
            os.makedirs(self.metrics_dir, exist_ok=True)
            row = dict(data)
            if "timestamp" not in row:
                row["timestamp"] = datetime.now().isoformat()

            # Normalize tokens_saved key if provided under alternate name
            if "estimated_tokens_saved" in row and "tokens_saved_estimate" not in row:
                row["tokens_saved_estimate"] = row["estimated_tokens_saved"]
            elif "tokens_saved_estimate" in row and "estimated_tokens_saved" not in row:
                row["estimated_tokens_saved"] = row["tokens_saved_estimate"]

            schema = self.schema or list(row.keys())

            with _LOCK:
                file_exists = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0
                with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=schema, extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception:
            # Observability must never crash application execution
            pass


# Alias for convenience
Emitter = MetricEmitter


def emit_metric(category: str, data: dict[str, Any]) -> None:
    """Global convenience helper to log a metric for a category."""
    MetricEmitter(category).log(data)


def _read_csv_rows(category: str, since_timestamp: float | None = None) -> list[dict[str, Any]]:
    """Read rows from category CSV, optionally filtering by cutoff timestamp."""
    filepath = os.path.join(METRICS_DIR, f"{category}.csv")
    if not os.path.exists(filepath):
        return []

    rows: list[dict[str, Any]] = []
    try:
        with _LOCK:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if since_timestamp is not None:
                        ts_str = row.get("timestamp", "")
                        try:
                            if "T" in ts_str or "-" in ts_str:
                                dt = datetime.fromisoformat(ts_str)
                                row_ts = dt.timestamp()
                            else:
                                row_ts = float(ts_str)
                            if row_ts < since_timestamp:
                                continue
                        except Exception:
                            pass
                    rows.append(row)
    except Exception:
        pass
    return rows


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute average, median, and max for a list of numbers."""
    if not values:
        return {"avg": 0.0, "median": 0.0, "max": 0.0, "count": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    avg = sum(sorted_vals) / n
    median = (sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0)
    return {
        "avg": round(avg, 2),
        "median": round(median, 2),
        "max": round(max(sorted_vals), 2),
        "count": n,
    }


def get_metrics_summary(since_timestamp: float | None = None) -> dict[str, Any]:
    """
    Compute standalone metrics summaries across main components.
    Processes each CSV independently without expensive multi-file joins.
    """
    summary: dict[str, Any] = {}

    # 1. Interpreter & Routing
    interp_rows = _read_csv_rows("interpreter", since_timestamp)
    interp_latencies = [float(r["latency_ms"]) for r in interp_rows if r.get("latency_ms")]
    interp_in_tokens = sum(int(r.get("input_tokens") or 0) for r in interp_rows)
    interp_out_tokens = sum(int(r.get("output_tokens") or 0) for r in interp_rows)
    direct_count = sum(1 for r in interp_rows if r.get("status") == "direct_response")
    pipeline_count = sum(1 for r in interp_rows if r.get("status") == "pipeline")
    interp_stats = _compute_stats(interp_latencies)

    summary["interpreter"] = {
        "total_queries": len(interp_rows),
        "direct_responses": direct_count,
        "pipelines": pipeline_count,
        "latency": interp_stats,
        "latency_ms": interp_stats,
        "input_tokens": {"sum": interp_in_tokens},
        "output_tokens": {"sum": interp_out_tokens},
        "_raw_in": interp_in_tokens,
        "_raw_out": interp_out_tokens,
    }

    # 2. DAG Execution Engine
    dag_rows = _read_csv_rows("dag_execution", since_timestamp)
    dag_latencies = [float(r["latency_ms"]) for r in dag_rows if r.get("latency_ms")]
    dag_sizes = [float(r["dag_size"]) for r in dag_rows if r.get("dag_size")]
    dag_depths = [float(r["dag_depth"]) for r in dag_rows if r.get("dag_depth")]
    success_count = sum(1 for r in dag_rows if r.get("status") == "success")
    dag_stats = _compute_stats(dag_latencies)

    summary["dag"] = {
        "total_executions": len(dag_rows),
        "success_rate": round(success_count / len(dag_rows) * 100, 1) if dag_rows else 0.0,
        "latency": dag_stats,
        "latency_ms": dag_stats,
        "avg_steps": round(sum(dag_sizes) / len(dag_sizes), 1) if dag_sizes else 0.0,
        "avg_depth": round(sum(dag_depths) / len(dag_depths), 1) if dag_depths else 0.0,
    }
    summary["dag_execution"] = summary["dag"]

    # 3. Answerer
    answerer_rows = _read_csv_rows("answerer", since_timestamp)
    answerer_latencies = [float(r["latency_ms"]) for r in answerer_rows if r.get("latency_ms")]
    answerer_in = sum(int(r.get("input_tokens") or 0) for r in answerer_rows)
    answerer_out = sum(int(r.get("output_tokens") or 0) for r in answerer_rows)
    ans_stats = _compute_stats(answerer_latencies)

    summary["answerer"] = {
        "total_syntheses": len(answerer_rows),
        "latency": ans_stats,
        "latency_ms": ans_stats,
        "input_tokens": {"sum": answerer_in},
        "output_tokens": {"sum": answerer_out},
        "_raw_in": answerer_in,
        "_raw_out": answerer_out,
    }

    # 4. Caching
    cache_rows = _read_csv_rows("caching", since_timestamp)
    cache_hits = sum(1 for r in cache_rows if r.get("cache_status") == "hit")
    instant_hits = sum(1 for r in cache_rows if r.get("hit_tier") in ("instant", "tier_instant_hit"))
    verified_hits = sum(1 for r in cache_rows if r.get("hit_tier") in ("llm_verified", "tier_llm_verified"))
    tokens_saved = sum(int(r.get("tokens_saved_estimate") or r.get("estimated_tokens_saved") or 0) for r in cache_rows)
    cache_latencies = [float(r["latency_ms"]) for r in cache_rows if r.get("latency_ms")]
    cache_stats = _compute_stats(cache_latencies)
    hit_rate = round(cache_hits / len(cache_rows) * 100, 1) if cache_rows else 0.0

    summary["caching"] = {
        "total_checks": len(cache_rows),
        "checks": len(cache_rows),
        "hits": cache_hits,
        "misses": len(cache_rows) - cache_hits,
        "hit_rate": hit_rate,
        "hit_rate_pct": hit_rate,
        "instant_hits": instant_hits,
        "verified_hits": verified_hits,
        "tokens_saved": tokens_saved,
        "latency": cache_stats,
        "latency_ms": cache_stats,
    }

    # 5. Builders (Tools & Agents)
    builder_rows = _read_csv_rows("builders", since_timestamp)
    builder_latencies = [float(r["latency_ms"]) for r in builder_rows if r.get("latency_ms")]
    builder_in = sum(int(r.get("input_tokens") or 0) for r in builder_rows)
    builder_out = sum(int(r.get("output_tokens") or 0) for r in builder_rows)
    tool_builds = sum(1 for r in builder_rows if r.get("builder_type") == "tool")
    agent_builds = sum(1 for r in builder_rows if r.get("builder_type") == "agent")
    passes = sum(1 for r in builder_rows if r.get("status") == "passed")
    retries = sum(int(r.get("attempt_count") or 0) for r in builder_rows)
    builder_stats = _compute_stats(builder_latencies)

    summary["builders"] = {
        "total_builds": len(builder_rows),
        "tool_builds": tool_builds,
        "agent_builds": agent_builds,
        "pass_rate": round(passes / len(builder_rows) * 100, 1) if builder_rows else 0.0,
        "total_retries": retries,
        "latency": builder_stats,
        "latency_ms": builder_stats,
        "input_tokens": {"sum": builder_in},
        "output_tokens": {"sum": builder_out},
        "_raw_in": builder_in,
        "_raw_out": builder_out,
    }

    # 6. Subagents
    subagent_rows = _read_csv_rows("subagents", since_timestamp)
    subagent_in = sum(int(r.get("input_tokens") or 0) for r in subagent_rows)
    subagent_out = sum(int(r.get("output_tokens") or 0) for r in subagent_rows)
    subagent_latencies = [float(r["latency_ms"]) for r in subagent_rows if r.get("latency_ms")]
    subagent_stats = _compute_stats(subagent_latencies)

    summary["subagents"] = {
        "total_calls": len(subagent_rows),
        "latency": subagent_stats,
        "latency_ms": subagent_stats,
        "input_tokens": {"sum": subagent_in},
        "output_tokens": {"sum": subagent_out},
        "_raw_in": subagent_in,
        "_raw_out": subagent_out,
    }

    # 7. ONI Security
    oni_rows = _read_csv_rows("oni", since_timestamp)
    prompts = sum(1 for r in oni_rows if r.get("oni_decision") == "greylist_prompted")
    allowed = sum(1 for r in oni_rows if r.get("user_decision") == "allowed")
    denied = sum(1 for r in oni_rows if r.get("user_decision") == "denied")

    summary["oni"] = {
        "total_events": len(oni_rows),
        "greylist_prompts": prompts,
        "user_allowed": allowed,
        "user_denied": denied,
    }

    # 8. Memory
    mem_rows = _read_csv_rows("memory", since_timestamp)
    compactions = sum(1 for r in mem_rows if str(r.get("compaction_triggered")).lower() in ("true", "1"))
    tokens_freed = sum(int(r.get("tokens_freed") or 0) for r in mem_rows)

    summary["memory"] = {
        "total_events": len(mem_rows),
        "compactions": compactions,
        "tokens_freed": tokens_freed,
    }

    # Overall Turn Latency
    all_turn_latencies = interp_latencies + dag_latencies
    summary["overall_turn_latency_ms"] = _compute_stats(all_turn_latencies)

    # Global Tokens
    total_in = interp_in_tokens + answerer_in + subagent_in + builder_in
    total_out = interp_out_tokens + answerer_out + subagent_out + builder_out
    summary["tokens_total"] = {
        "input": total_in,
        "output": total_out,
        "grand_total": total_in + total_out,
    }

    return summary


def format_metrics_tables(since_timestamp: float | None = None) -> list[Table]:
    """Build formatted Rich tables for display in the MAVIS CLI."""
    summary = get_metrics_summary(since_timestamp)
    tables: list[Table] = []

    # Table 1: Core Performance & Latency
    t1 = Table(title="MAVIS Performance & Latency (ms)", title_style="bold cyan", expand=True)
    t1.add_column("Component", style="cyan", no_wrap=True)
    t1.add_column("Count", justify="right")
    t1.add_column("Average (ms)", justify="right")
    t1.add_column("Median (ms)", justify="right")
    t1.add_column("Max (ms)", justify="right")

    interp_lat = summary["interpreter"]["latency"]
    t1.add_row("Interpreter (Planning)", str(interp_lat["count"]), f"{interp_lat['avg']:.0f}", f"{interp_lat['median']:.0f}", f"{interp_lat['max']:.0f}")

    dag_lat = summary["dag"]["latency"]
    t1.add_row("DAG Execution", str(dag_lat["count"]), f"{dag_lat['avg']:.0f}", f"{dag_lat['median']:.0f}", f"{dag_lat['max']:.0f}")

    ans_lat = summary["answerer"]["latency"]
    t1.add_row("Answerer (Synthesis)", str(ans_lat["count"]), f"{ans_lat['avg']:.0f}", f"{ans_lat['median']:.0f}", f"{ans_lat['max']:.0f}")

    cache_lat = summary["caching"]["latency"]
    t1.add_row("Cache Check", str(cache_lat["count"]), f"{cache_lat['avg']:.0f}", f"{cache_lat['median']:.0f}", f"{cache_lat['max']:.0f}")

    sub_lat = summary["subagents"]["latency"]
    t1.add_row("Subagents (Cognitive)", str(sub_lat["count"]), f"{sub_lat['avg']:.0f}", f"{sub_lat['median']:.0f}", f"{sub_lat['max']:.0f}")
    tables.append(t1)

    # Table 2: Token Economics
    t2 = Table(title="Token Economics & Components", title_style="bold green", expand=True)
    t2.add_column("Component", style="green", no_wrap=True)
    t2.add_column("Input Tokens", justify="right")
    t2.add_column("Output Tokens", justify="right")
    t2.add_column("Total Tokens", justify="right")

    i_in, i_out = summary['interpreter']['_raw_in'], summary['interpreter']['_raw_out']
    s_in, s_out = summary['subagents']['_raw_in'], summary['subagents']['_raw_out']
    a_in, a_out = summary['answerer']['_raw_in'], summary['answerer']['_raw_out']

    t2.add_row("Interpreter", f"{i_in:,}", f"{i_out:,}", f"{i_in + i_out:,}")
    t2.add_row("Subagents", f"{s_in:,}", f"{s_out:,}", f"{s_in + s_out:,}")
    t2.add_row("Answerer", f"{a_in:,}", f"{a_out:,}", f"{a_in + a_out:,}")
    t2.add_row("[bold]Grand Total[/bold]", f"[bold]{summary['tokens_total']['input']:,}[/bold]", f"[bold]{summary['tokens_total']['output']:,}[/bold]", f"[bold]{summary['tokens_total']['grand_total']:,}[/bold]")
    tables.append(t2)

    # Table 3: System, Caching & Security Health
    t3 = Table(title="Caching, Reliability & Security", title_style="bold yellow", expand=True)
    t3.add_column("Metric", style="yellow", no_wrap=True)
    t3.add_column("Value", justify="right")
    t3.add_column("Details", style="dim")

    c = summary["caching"]
    t3.add_row("Cache Hit Rate", f"{c['hit_rate']}%", f"{c['hits']}/{c['total_checks']} hits ({c['instant_hits']} instant, {c['verified_hits']} verified)")
    t3.add_row("Tokens Saved (Cache)", f"{c['tokens_saved']:,}", "Estimated tokens avoided")

    d = summary["dag"]
    t3.add_row("Pipeline Success Rate", f"{d['success_rate']}%", f"{d['total_executions']} pipelines (avg depth {d['avg_depth']})")

    b = summary["builders"]
    t3.add_row("Tool/Agent Synthesis", f"{b['pass_rate']}% Pass", f"{b['total_builds']} builds ({b['total_retries']} debug retries)")

    o = summary["oni"]
    t3.add_row("ONI Security Gates", f"{o['greylist_prompts']} Prompts", f"{o['user_allowed']} allowed, {o['user_denied']} denied")

    m = summary["memory"]
    t3.add_row("Memory Compactions", f"{m['compactions']}", f"{m['tokens_freed']:,} tokens freed")
    tables.append(t3)

    return tables


def get_turn_trace(turn_id: str) -> list[dict[str, Any]]:
    """
    On-demand lazy cross-file join for a single turn_id.
    Searches all CSVs to build a full trace timeline for that query.
    """
    trace: list[dict[str, Any]] = []
    categories = ["interpreter", "caching", "dag_execution", "subagents", "answerer", "oni", "memory"]

    for cat in categories:
        rows = _read_csv_rows(cat)
        matched = [r for r in rows if r.get("turn_id") == turn_id]
        for m in matched:
            event = dict(m)
            event["component"] = cat
            trace.append(event)

    trace.sort(key=lambda x: x.get("timestamp", ""))
    return trace
