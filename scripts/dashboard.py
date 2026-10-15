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
from core.reflection import (
    load_seed_pool,
    add_seed_to_pool,
    remove_seed_from_pool,
    load_reflection_journal,
    verify_and_promote_reflection,
    delete_reflection_entry,
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
            # utc=True normalizes mixed naive/aware values into one tz-aware UTC dtype
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df
    except Exception as e:
        st.error(f"Error reading {csv_path.name}: {e}")
        return pd.DataFrame()


def filter_by_time(df: pd.DataFrame, cutoff: datetime | None) -> pd.DataFrame:
    """Filter DataFrame by timestamp_dt >= cutoff if applicable."""
    if df.empty or cutoff is None or "timestamp_dt" not in df.columns:
        return df
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize("UTC")
    else:
        cutoff_ts = cutoff_ts.tz_convert("UTC")
    ts = pd.to_datetime(df["timestamp_dt"], errors="coerce", utc=True)
    return df[ts >= cutoff_ts]


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
tab_overview, tab_latency, tab_caching, tab_dag, tab_builders, tab_memory, tab_reflection, tab_trace = st.tabs([
    "📊 Overview",
    "⏱ Latency Breakdown",
    "⚡ Caching",
    "🔄 DAG Execution",
    "🛠 Builders & Agents",
    "🧠 Memory & Safety",
    "💭 Reflection",
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
        for comp in ["interpreter", "answerer", "dag_execution", "cognitive", "subagents", "builders"]:
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
        for comp in ["interpreter", "answerer", "cognitive", "subagents", "builders"]:
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
        options=["interpreter", "answerer", "dag_execution", "cognitive", "subagents", "builders"],
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
        total = len(df_cache)
        hits = (df_cache["cache_status"] == "hit").sum()
        bypassed = (df_cache["cache_status"].isin(["bypassed", "prefiltered"])).sum()
        eligible = total - bypassed
        hit_rate = (hits / eligible * 100) if eligible > 0 else 0.0
        tokens_saved = df_cache["estimated_tokens_saved"].sum() if "estimated_tokens_saved" in df_cache.columns else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cache Checks", f"{total:,}")
        c2.metric("Eligible Checks", f"{eligible:,}")
        c3.metric("Pre-filtered (Context)", f"{bypassed:,}")
        c4.metric("Hit Rate (Eligible)", f"{hit_rate:.1f}%")
        c5.metric("Tokens Saved", f"{tokens_saved:,}")

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

        df_goal = load_component_csv("goal_runner")
        df_goal = filter_by_time(df_goal, time_cutoff)
        if not df_goal.empty:
            st.divider()
            st.markdown("#### 🎯 Autonomous Goal Runs & Wave Progressions")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Goals Initiated", f"{len(df_goal):,}")
            completed_goals = (df_goal["status"] == "completed").sum() if "status" in df_goal.columns else 0
            g2.metric("Goals Completed", f"{completed_goals:,}")
            avg_waves = df_goal["waves_executed"].mean() if "waves_executed" in df_goal.columns else 0.0
            g3.metric("Avg Waves / Goal", f"{avg_waves:.1f}")
            avg_dur = df_goal["duration_s"].mean() if "duration_s" in df_goal.columns else 0.0
            g4.metric("Avg Duration", f"{avg_dur:.1f}s")
            st.dataframe(df_goal.tail(25), use_container_width=True)

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

# ── Tab 7: Autonomous Reflection ─────────────────────────────────────────────
with tab_reflection:
    st.subheader("Autonomous Reflection & Conceptual Growth")

    # 1. Telemetry Metrics
    df_refl = load_component_csv("reflection")
    df_refl = filter_by_time(df_refl, time_cutoff)

    if not df_refl.empty:
        committed_n = int((df_refl["committed"].astype(str).str.lower() == "true").sum()) if "committed" in df_refl.columns else 0
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Cycles", f"{len(df_refl):,}")
        rc2.metric("Committed Insights", f"{committed_n}")
        if "semantic_delta" in df_refl.columns:
            rc3.metric("Avg Semantic Delta", f"{pd.to_numeric(df_refl['semantic_delta'], errors='coerce').mean():.3f}")
        if "latency_ms" in df_refl.columns:
            rc4.metric("Avg Latency", f"{df_refl['latency_ms'].mean():.0f} ms")

    st.markdown("---")

    # 2. Conceptual Seed Pool Management
    st.subheader("🌱 Conceptual Seed Pool")
    st.caption("Manage conceptual seeds that MAVIS randomly samples from to initiate reflection cycles. (No fallback is assumed if empty).")

    with st.form("seed_pool_form", clear_on_submit=True):
        sc_col1, sc_col2 = st.columns([5, 1])
        new_seed_val = sc_col1.text_input("New Conceptual Seed", placeholder="e.g. Subprocess non-blocking stream multiplexing with selectors", label_visibility="collapsed")
        submit_seed = sc_col2.form_submit_button("➕ Add Seed", use_container_width=True)
        if submit_seed and new_seed_val.strip():
            if add_seed_to_pool(new_seed_val.strip()):
                st.success(f"Added '{new_seed_val.strip()}' to Seed Pool.")
                st.rerun()

    current_seeds = load_seed_pool()
    if not current_seeds:
        st.info("The Seed Pool is currently empty. Add conceptual seeds above to enable autonomous reflection.")
    else:
        st.markdown(f"**Current Seed Pool ({len(current_seeds)}):**")
        for idx, s in enumerate(current_seeds):
            c_txt, c_btn = st.columns([6, 1])
            c_txt.markdown(f"• `{s}`")
            if c_btn.button("🗑️ Remove", key=f"rm_seed_{idx}", help="Remove seed from pool"):
                remove_seed_from_pool(s)
                st.rerun()

    st.markdown("---")

    # 3. Self-Reflection Journal Review
    st.subheader("📔 Self-Reflection Journal")
    st.caption("Review completed exploration cycles. Insights remain sandboxed from the DAG planner until you verify & promote them into Long-Term Memory.")

    journal_records = load_reflection_journal()
    if not journal_records:
        st.info("No reflection journal records recorded yet.")
    else:
        filter_opt = st.radio("Filter", ["All", "Pending Review", "Verified"], horizontal=True, key="journal_filter")
        filtered_entries = list(reversed(journal_records))
        if filter_opt == "Pending Review":
            filtered_entries = [e for e in filtered_entries if not e.get("verified")]
        elif filter_opt == "Verified":
            filtered_entries = [e for e in filtered_entries if e.get("verified")]

        if not filtered_entries:
            st.info(f"No records matching filter '{filter_opt}'.")
        else:
            for entry in filtered_entries:
                is_ver = entry.get("verified", False)
                status_badge = "🟢 Verified" if is_ver else "🟡 Pending Review"
                card_title = f"{entry.get('date', 'Unknown Date')} • {entry.get('explored_topic', 'Unnamed Topic')} ({status_badge})"

                with st.expander(card_title, expanded=(not is_ver)):
                    st.markdown(f"**Seed Concept:** `{entry.get('seed_concept', 'N/A')}`")
                    if entry.get("hypothesis"):
                        st.markdown(f"**Hypothesis:** {entry.get('hypothesis')}")
                    st.markdown("**Distilled Insight:**")
                    st.info(entry.get("content", ""))
                    st.caption(f"Semantic Delta: {entry.get('semantic_delta', 0.0):.3f} | Turns Used: {entry.get('iterations_count', 0)} | ID: `{entry.get('id', '')}`")

                    act_col1, act_col2, _ = st.columns([1.5, 1.5, 4])
                    if not is_ver:
                        if act_col1.button("✔️ Verify & Promote", key=f"vp_{entry['id']}", type="primary"):
                            if verify_and_promote_reflection(entry["id"]):
                                st.success("Verified and promoted to Long-Term Memory & Seed Pool!")
                                st.rerun()

                    if act_col2.button("🗑️ Delete", key=f"del_{entry['id']}", type="secondary"):
                        if delete_reflection_entry(entry["id"]):
                            st.warning("Deleted reflection entry.")
                            st.rerun()

    # 4. Raw Metrics Table
    if not df_refl.empty:
        with st.expander("📊 Raw Reflection Telemetry Logs"):
            if "status" in df_refl.columns:
                st.bar_chart(df_refl["status"].value_counts())
            st.dataframe(df_refl.tail(50), use_container_width=True)

# ── Tab 8: Turn & Goal Wave Trace Inspector (On-Demand Lazy Join) ─────────────
with tab_trace:
    st.subheader("🔍 Turn & Goal Wave Trace Inspector (On-Demand Lazy Join)")
    st.caption("Drill down into a single turn or an autonomous multi-wave goal to reconstruct the full end-to-end execution path.")

    # Collect available turn_ids and goal_ids
    df_interp = load_component_csv("interpreter")
    df_goal = load_component_csv("goal_runner")

    recent_options: list[tuple[str, str]] = []

    if not df_goal.empty and "goal_id" in df_goal.columns:
        for _, row in df_goal.tail(25).iterrows():
            gid = str(row.get("goal_id", ""))
            goal_text = str(row.get("goal", ""))[:45]
            status = str(row.get("status", ""))
            recent_options.append((f"🎯 [GOAL] {gid} ({status}): {goal_text}", gid))

    if not df_interp.empty and "turn_id" in df_interp.columns:
        for t in df_interp["turn_id"].dropna().unique().tolist()[-35:]:
            recent_options.append((f"💬 [TURN] {t}", str(t)))

    seen_ids = set()
    unique_choices: list[tuple[str, str]] = []
    for label, tid in reversed(recent_options):
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            unique_choices.append((label, tid))

    label_to_id = {label: tid for label, tid in unique_choices}

    selected_label = st.selectbox(
        "Select Turn or Goal to Inspect",
        options=[""] + [lbl for lbl, _ in unique_choices],
        help="Select a Turn ID or Goal ID to lazily join records across all component CSVs.",
    )

    custom_turn_id = st.text_input("Or enter custom Turn ID / Goal ID", value="")
    inspect_id = custom_turn_id.strip() or label_to_id.get(selected_label, "")

    if inspect_id:
        with st.spinner(f"Joining telemetry records for {inspect_id}..."):
            trace = get_turn_trace(inspect_id)

        if not trace:
            st.warning(f"No records found across CSV files for: `{inspect_id}`")
        else:
            st.success(f"Reconstructed trace for `{inspect_id}` ({len(trace)} component events)")

            # Check if this trace represents a multi-wave goal or contains DAG execution waves
            wave_events = [e for e in trace if e.get("component") == "dag_execution"]
            gate_events = [e for e in trace if e.get("component") == "gate_evaluator"]
            debug_events = [e for e in trace if e.get("component") == "pipeline_debugger"]
            goal_events = [e for e in trace if e.get("component") == "goal_runner"]

            if goal_events or len(wave_events) > 1:
                st.markdown("### 🌊 Multi-Wave DAG Progression")
                w_cols = st.columns(4)
                w_cols[0].metric("Waves Executed", f"{len(wave_events)}")
                w_cols[1].metric("Gate Decisions", f"{len(gate_events)}")
                w_cols[2].metric("Debugger Repairs", f"{len(debug_events)}")
                total_steps = sum(int(e.get("dag_size", 0)) for e in wave_events if str(e.get("dag_size", "")).isdigit())
                w_cols[3].metric("Total Nodes Dispatched", f"{total_steps}")

                for w_idx, w_event in enumerate(wave_events):
                    w_turn = w_event.get("turn_id", f"Wave {w_idx+1}")
                    w_status = w_event.get("status", "unknown")
                    w_size = w_event.get("dag_size", "N/A")
                    w_depth = w_event.get("dag_depth", "N/A")
                    w_lat = w_event.get("latency_ms", "N/A")

                    w_gates = [g for g in gate_events if g.get("turn_id") == w_turn]
                    w_debugs = [d for d in debug_events if d.get("turn_id") == w_turn]

                    with st.container():
                        st.markdown(
                            f"**Wave {w_idx+1}** (`{w_turn}`): Status: `{w_status}` • Nodes: {w_size} • Depth: {w_depth} • Latency: {w_lat} ms"
                        )
                        if w_gates:
                            for g in w_gates:
                                st.caption(f"  ↳ 🚦 Gate `{g.get('gate_id')}` evaluated `{g.get('condition')}` ➔ **Verdict: {g.get('verdict')}**")
                        if w_debugs:
                            for d in w_debugs:
                                st.caption(f"  ↳ 🩺 Debugger action: `{d.get('action')}` on node `{d.get('node_id')}`")
                    st.divider()

            trace_df = pd.DataFrame(trace)
            cols_to_show = [c for c in ["component", "timestamp", "status", "latency_ms"] if c in trace_df.columns]
            st.dataframe(trace_df[cols_to_show], use_container_width=True)

            for idx, event in enumerate(trace):
                comp = event.get("component", "unknown").upper()
                status = event.get("status", "info")
                latency = event.get("latency_ms", "N/A")
                with st.expander(f"Step {idx+1}: [{comp}] Status: {status} • Latency: {latency} ms"):
                    st.json(event)
    else:
        st.info("Select or enter a Turn ID or Goal ID to inspect.")
