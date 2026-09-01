"""
scripts/dashboard.py
MAVIS Performance & Observability Dashboard (Streamlit + Pandas).

Architecture:
- High-speed independent single-CSV reads per tab/panel (no joins by default).
- On-Demand Lazy Cross-CSV Joins for Single-Turn Trace Inspector.
- Track Average, Median, Max aggregations.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.metrics import (
    METRICS_DIR,
    get_metrics_summary,
    get_turn_trace,
)

# ── Page Configuration & Theming ─────────────────────────────────────────────
st.set_page_config(
    page_title="MAVIS Observability",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8892b0;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #111927;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data Loading Helpers (Isolated & Fast) ───────────────────────────────────

def load_component_csv(component: str) -> pd.DataFrame:
    """Read an isolated component CSV without joining any other tables."""
    csv_path = Path(METRICS_DIR) / f"{component}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error reading {csv_path.name}: {e}")
        return pd.DataFrame()


def filter_by_time(df: pd.DataFrame, cutoff: datetime | None) -> pd.DataFrame:
    """Filter DataFrame by timestamp_dt >= cutoff if applicable."""
    if df.empty or cutoff is None or "timestamp_dt" not in df.columns:
        return df
    return df[df["timestamp_dt"] >= cutoff]


# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚡ **MAVIS Telemetry**")
st.sidebar.caption("Personal AI Performance & Observability")

time_filter_option = st.sidebar.selectbox(
    "Time Horizon",
    options=["All Time", "Last 15 Minutes", "Last 1 Hour", "Last 24 Hours", "Last 7 Days"],
    index=0,
)

now_utc = datetime.now(timezone.utc)
time_cutoff: datetime | None = None

if time_filter_option == "Last 15 Minutes":
    time_cutoff = now_utc - timedelta(minutes=15)
elif time_filter_option == "Last 1 Hour":
    time_cutoff = now_utc - timedelta(hours=1)
elif time_filter_option == "Last 24 Hours":
    time_cutoff = now_utc - timedelta(days=1)
elif time_filter_option == "Last 7 Days":
    time_cutoff = now_utc - timedelta(days=7)

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

st.sidebar.divider()
st.sidebar.info(
    "💡 **Architecture**: Independent CSV streams in `data/metrics/`. "
    "Tabular views perform zero cross-joins by default for maximum speed."
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">MAVIS Observability Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Showing metrics for: <b>{time_filter_option}</b> • Storage: <code>{METRICS_DIR}</code></div>',
    unsafe_allow_html=True,
)

# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab_overview, tab_latency, tab_caching, tab_dag, tab_builders, tab_memory, tab_trace = st.tabs([
    "📊 Overview",
    "⏱ Latency Breakdown",
    "⚡ Caching",
    "🔄 DAG Execution",
    "🛠 Builders & Agents",
    "🧠 Memory & Safety",
    "🔍 Turn Trace Inspector",
])

# ── Tab 1: Overview ──────────────────────────────────────────────────────────
with tab_overview:
    cutoff_ts = time_cutoff.timestamp() if time_cutoff else None
    summary = get_metrics_summary(cutoff_ts)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Queries", f"{summary['interpreter']['total_queries']:,}")
    with col2:
        st.metric("Avg Turn Latency", f"{summary['overall_turn_latency_ms']['avg']:.1f} ms")
    with col3:
        st.metric("Cache Hit Rate", f"{summary['caching']['hit_rate_pct']:.1f}%")
    with col4:
        st.metric("Tokens In / Out", f"{summary['tokens_total']['input']:,} / {summary['tokens_total']['output']:,}")
    with col5:
        st.metric("Est. Tokens Saved", f"{summary['caching']['tokens_saved']:,}")

    st.divider()

    st.subheader("High-Level Activity Snapshot")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Component Latencies Summary (ms)**")
        latency_rows = []
        for comp in ["interpreter", "answerer", "dag_execution", "subagents", "builders"]:
            stats = summary.get(comp, {})
            if "latency_ms" in stats:
                latency_rows.append({
                    "Component": comp.capitalize(),
                    "Average (ms)": stats["latency_ms"]["avg"],
                    "Median (ms)": stats["latency_ms"]["median"],
                    "Max (ms)": stats["latency_ms"]["max"],
                })
        if latency_rows:
            st.dataframe(pd.DataFrame(latency_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No latency records found.")

    with c2:
        st.markdown("**Token Consumption by Component**")
        token_rows = []
        for comp in ["interpreter", "answerer", "subagents", "builders"]:
            stats = summary.get(comp, {})
            if "input_tokens" in stats and "output_tokens" in stats:
                token_rows.append({
                    "Component": comp.capitalize(),
                    "Input Tokens": stats["input_tokens"]["sum"],
                    "Output Tokens": stats["output_tokens"]["sum"],
                    "Total": stats["input_tokens"]["sum"] + stats["output_tokens"]["sum"],
                })
        if token_rows:
            st.dataframe(pd.DataFrame(token_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No token records found.")

# ── Tab 2: Latency Breakdown ─────────────────────────────────────────────────
with tab_latency:
    st.subheader("Component Latency Distribution")
    st.caption("Tracking Average, Median, and Max execution times across core modules.")

    comp_select = st.selectbox(
        "Select Component Stream",
        options=["interpreter", "answerer", "dag_execution", "subagents", "builders"],
        format_func=lambda x: x.capitalize(),
    )

    df_comp = load_component_csv(comp_select)
    df_comp = filter_by_time(df_comp, time_cutoff)

    if df_comp.empty or "latency_ms" not in df_comp.columns:
        st.info(f"No latency telemetry available for {comp_select}.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Record Count", f"{len(df_comp):,}")
        m2.metric("Average Latency", f"{df_comp['latency_ms'].mean():.1f} ms")
        m3.metric("Median Latency", f"{df_comp['latency_ms'].median():.1f} ms")
        m4.metric("Max Latency", f"{df_comp['latency_ms'].max():.1f} ms")

        st.line_chart(df_comp, y="latency_ms")

        with st.expander("View Raw Component Records"):
            st.dataframe(df_comp.tail(50), use_container_width=True)

# ── Tab 3: Caching Performance ───────────────────────────────────────────────
with tab_caching:
    st.subheader("Semantic Pipeline Cache Performance")
    df_cache = load_component_csv("caching")
    df_cache = filter_by_time(df_cache, time_cutoff)

    if df_cache.empty:
        st.info("No caching telemetry recorded yet.")
    else:
        hits = (df_cache["cache_status"] == "hit").sum()
        total = len(df_cache)
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        tokens_saved = df_cache["estimated_tokens_saved"].sum() if "estimated_tokens_saved" in df_cache.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cache Checks", f"{total:,}")
        c2.metric("Cache Hits", f"{hits:,}")
        c3.metric("Hit Rate", f"{hit_rate:.1f}%")
        c4.metric("Tokens Saved", f"{tokens_saved:,}")

        col_tier, col_sim = st.columns(2)
        with col_tier:
            st.markdown("**Hit Tier Distribution**")
            if "hit_tier" in df_cache.columns:
                tier_counts = df_cache["hit_tier"].value_counts().reset_index()
                tier_counts.columns = ["Tier", "Count"]
                st.dataframe(tier_counts, use_container_width=True, hide_index=True)

        with col_sim:
            st.markdown("**Similarity Score Stats (Hits)**")
            if "similarity_score" in df_cache.columns:
                sim_hits = df_cache[df_cache["cache_status"] == "hit"]["similarity_score"]
                if not sim_hits.empty:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Avg Similarity", f"{sim_hits.mean():.3f}")
                    s2.metric("Median Similarity", f"{sim_hits.median():.3f}")
                    s3.metric("Max Similarity", f"{sim_hits.max():.3f}")

        st.dataframe(df_cache.tail(50), use_container_width=True)

# ── Tab 4: DAG Execution ─────────────────────────────────────────────────────
with tab_dag:
    st.subheader("Topological Pipeline & DAG Telemetry")
    df_dag = load_component_csv("dag_execution")
    df_dag = filter_by_time(df_dag, time_cutoff)

    if df_dag.empty:
        st.info("No DAG execution records found.")
    else:
        success_count = (df_dag["status"] == "success").sum()
        total_dag = len(df_dag)
        success_rate = (success_count / total_dag * 100) if total_dag > 0 else 0.0

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total DAGs Executed", f"{total_dag:,}")
        d2.metric("Success Rate", f"{success_rate:.1f}%")
        d3.metric("Avg DAG Depth", f"{df_dag['dag_depth'].mean():.2f}" if "dag_depth" in df_dag.columns else "N/A")
        d4.metric("Avg DAG Size (Nodes)", f"{df_dag['dag_size'].mean():.2f}" if "dag_size" in df_dag.columns else "N/A")

        st.markdown("**DAG Executions Log**")
        st.dataframe(df_dag.tail(50), use_container_width=True)

# ── Tab 5: Builders & Agents ─────────────────────────────────────────────────
with tab_builders:
    st.subheader("Synthesis & Synthesis Debugging Telemetry")
    df_builders = load_component_csv("builders")
    df_builders = filter_by_time(df_builders, time_cutoff)

    if df_builders.empty:
        st.info("No tool/agent builder telemetry recorded yet.")
    else:
        total_builds = len(df_builders)
        passed_builds = (df_builders["status"] == "passed").sum()
        pass_rate = (passed_builds / total_builds * 100) if total_builds > 0 else 0.0
        avg_attempts = df_builders["attempt_count"].mean() if "attempt_count" in df_builders.columns else 0.0

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Builds", f"{total_builds:,}")
        b2.metric("Passed Verification", f"{passed_builds:,}")
        b3.metric("Pass Rate", f"{pass_rate:.1f}%")
        b4.metric("Avg Attempt Retries", f"{avg_attempts:.2f}")

        st.dataframe(df_builders.tail(50), use_container_width=True)

# ── Tab 6: Memory & Safety ───────────────────────────────────────────────────
with tab_memory:
    st.subheader("Memory Pressure, Compaction & ONI Gate Operations")
    
    col_m, col_o = st.columns(2)
    with col_m:
        st.markdown("#### 🧠 Memory Store Operations")
        df_mem = load_component_csv("memory")
        df_mem = filter_by_time(df_mem, time_cutoff)
        if df_mem.empty:
            st.info("No memory telemetry records.")
        else:
            compactions = (df_mem["compaction_triggered"] == True).sum() if "compaction_triggered" in df_mem.columns else 0
            freed = df_mem["tokens_freed"].sum() if "tokens_freed" in df_mem.columns else 0
            
            mc1, mc2 = st.columns(2)
            mc1.metric("Compaction Events", f"{compactions}")
            mc2.metric("Tokens Freed", f"{freed:,}")

            if "working_tokens_count" in df_mem.columns:
                st.line_chart(df_mem, y="working_tokens_count")

    with col_o:
        st.markdown("#### 🛡 ONI Safety Gate")
        df_oni = load_component_csv("oni")
        df_oni = filter_by_time(df_oni, time_cutoff)
        if df_oni.empty:
            st.info("No ONI safety gate interventions recorded.")
        else:
            allowed = (df_oni["user_decision"] == "allowed").sum() if "user_decision" in df_oni.columns else 0
            denied = (df_oni["user_decision"] == "denied").sum() if "user_decision" in df_oni.columns else 0
            avg_dwell = df_oni["dwell_time_ms"].mean() if "dwell_time_ms" in df_oni.columns else 0.0

            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Greylist Prompts", f"{len(df_oni)}")
            oc2.metric("Allowed / Denied", f"{allowed} / {denied}")
            oc3.metric("Avg Dwell Time", f"{avg_dwell:.0f} ms")

            st.dataframe(df_oni.tail(20), use_container_width=True)

# ── Tab 7: Turn Trace Inspector (On-Demand Lazy Join) ─────────────────────────
with tab_trace:
    st.subheader("🔍 Turn Trace Inspector (On-Demand Lazy Join)")
    st.caption("Drill down into a single turn to reconstruct the full end-to-end execution path.")

    # Collect available turn_ids from interpreter table
    df_interp = load_component_csv("interpreter")
    recent_turns = []
    if not df_interp.empty and "turn_id" in df_interp.columns:
        recent_turns = [t for t in df_interp["turn_id"].dropna().unique().tolist() if t]

    target_turn_id = st.selectbox(
        "Select Turn ID to Inspect",
        options=[""] + list(reversed(recent_turns)),
        help="Select a turn_id to lazily scan and join records across all component CSVs.",
    )

    custom_turn_id = st.text_input("Or enter custom Turn ID", value="")
    inspect_id = custom_turn_id.strip() or target_turn_id

    if inspect_id:
        with st.spinner(f"Joining telemetry records for turn {inspect_id}..."):
            trace = get_turn_trace(inspect_id)

        if not trace:
            st.warning(f"No records found across CSV files for turn_id: `{inspect_id}`")
        else:
            st.success(f"Reconstructed trace for turn `{inspect_id}` ({len(trace)} component events)")
            
            trace_df = pd.DataFrame(trace)
            st.dataframe(trace_df[["component", "timestamp", "status", "latency_ms"]], use_container_width=True)

            for idx, event in enumerate(trace):
                comp = event.get("component", "unknown").upper()
                status = event.get("status", "info")
                latency = event.get("latency_ms", "N/A")
                with st.expander(f"Step {idx+1}: [{comp}] Status: {status} • Latency: {latency} ms"):
                    st.json(event)
    else:
        st.info("Select or enter a Turn ID to inspect.")
