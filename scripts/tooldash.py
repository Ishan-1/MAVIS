"""
scripts/tooldash.py
MAVIS Tool, Subagent & MCP Management Dashboard (Streamlit).
Runs by default on port 8502.

Capabilities:
- Manage Tools: view, inspect code, deregister (archive), restore, or permanently delete.
- Manage Subagents & Cognitive Nodes: view, inspect, deregister, restore, or delete.
- Manage MCP: inspect server status, toggle, add, test handshake, delete, view discovered tools.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.manager import (
    PROTECTED_BASELINE_TOOLS,
    list_all_tools,
    deregister_tool,
    restore_tool,
    delete_tool_permanently,
    list_all_agents,
    deregister_agent,
    restore_agent,
    delete_agent_permanently,
    get_mcp_overview,
    save_mcp_server,
    toggle_mcp_server,
    delete_mcp_server,
    test_mcp_server,
    reload_mcp_system,
)

# ── Page Configuration & Theming ─────────────────────────────────────────────
st.set_page_config(
    page_title="MAVIS Tool & Agent Studio",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #00d2ff 0%, #9d4edd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .item-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    .item-card:hover {
        border-color: #475569;
    }

    .badge-generalizable {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-repurposable {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-specialized {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-protected {
        background-color: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(236, 72, 153, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-archived {
        background-color: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-mcp {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load Live State ────────────────────────────────────────────────────────────
all_tools = list_all_tools()
all_agents = list_all_agents()
mcp_data = get_mcp_overview()

active_tools = [t for t in all_tools if t["status"] == "active"]
archived_tools = [t for t in all_tools if t["status"] == "archived"]
mcp_tools = [t for t in all_tools if t["status"] == "mcp"]

active_agents = [a for a in all_agents if a["status"] == "active"]
archived_agents = [a for a in all_agents if a["status"] == "archived"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ **MAVIS Studio**")
    st.caption("Lifecycle Management & Control Console")
    st.markdown("---")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Active Tools", len(active_tools))
        st.metric("Subagents", len(active_agents))
    with col_s2:
        st.metric("Archived Tools", len(archived_tools))
        st.metric("MCP Tools", len(mcp_tools))

    st.markdown("---")
    st.markdown("#### **System Info**")
    st.markdown("**Baseline Primitive (Tool):** `run_shell_command`")
    st.markdown("**Baseline Primitive (Agent):** `semantic_transform`")
    st.markdown(f"**Archive Storage:** `data/archived_tools/`")
    st.markdown(f"**Active MCP Servers:** `{len([s for s in mcp_data['servers'].values() if s.get('enabled')])}`")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">MAVIS Tool & Agent Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Inspect, configure, archive, restore, and permanently manage tools, subagents, and MCP integrations.</div>', unsafe_allow_html=True)

# ── Primary Navigation Tabs ───────────────────────────────────────────────────
tab_tools, tab_agents, tab_mcp = st.tabs([
    f"🛠️ Tools Manager ({len(all_tools)})",
    f"🤖 Subagents & Cognitive Nodes ({len(all_agents)})",
    f"🔌 MCP Servers & Tools ({len(mcp_data['servers'])})",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: TOOLS MANAGER
# ═════════════════════════════════════════════════════════════════════════════
with tab_tools:
    view_mode = st.radio(
        "Tool Category",
        options=["Active Tools", "Archived Tools", "MCP Discovered Tools", "All"],
        horizontal=True,
    )

    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        search_tool = st.text_input("🔍 Search Tools by name or description", key="search_tool").strip().lower()
    with col_f2:
        gen_filter = st.selectbox(
            "Generalizability Class",
            options=["All Classes", "generalizable", "repurposable", "specialized"],
            index=0,
        )

    # Filter tools list
    display_tools = all_tools
    if view_mode == "Active Tools":
        display_tools = [t for t in display_tools if t["status"] == "active"]
    elif view_mode == "Archived Tools":
        display_tools = [t for t in display_tools if t["status"] == "archived"]
    elif view_mode == "MCP Discovered Tools":
        display_tools = [t for t in display_tools if t["status"] == "mcp"]

    if search_tool:
        display_tools = [
            t for t in display_tools
            if search_tool in t["name"].lower() or search_tool in t["description"].lower()
        ]

    if gen_filter != "All Classes":
        display_tools = [t for t in display_tools if t["generalizability"] == gen_filter]

    st.markdown(f"Showing **{len(display_tools)}** tool(s)")

    if not display_tools:
        st.info("No tools found matching the selected filters.")

    for tool in display_tools:
        name = tool["name"]
        sig = tool["signature"]
        status = tool["status"]
        is_prot = tool["is_protected"]
        gen = tool["generalizability"]

        with st.container():
            col_t1, col_t2 = st.columns([7, 3])
            with col_t1:
                st.markdown(f"#### `{name}`")
                st.markdown(f"**Signature:** `{sig}`")
                st.markdown(f"_{tool['description']}_")

                badges = []
                if is_prot:
                    badges.append('<span class="badge-protected">🔒 Baseline Primitive</span>')
                if status == "archived":
                    badges.append('<span class="badge-archived">📦 Archived</span>')
                elif status == "mcp":
                    badges.append(f'<span class="badge-mcp">🔌 MCP ({tool.get("mcp_server", "")})</span>')
                else:
                    badges.append('<span class="badge-generalizable">🟢 Active</span>')

                if gen == "generalizable":
                    badges.append('<span class="badge-generalizable">Generalizable</span>')
                elif gen == "repurposable":
                    badges.append('<span class="badge-repurposable">Repurposable</span>')
                elif gen == "specialized":
                    badges.append('<span class="badge-specialized">Specialized</span>')

                st.markdown(" ".join(badges), unsafe_allow_html=True)

            with col_t2:
                if is_prot:
                    st.caption("⚠️ Baseline primitive is protected from deletion or archiving.")
                elif status == "active":
                    # Active tool actions: Deregister (Archive) or Delete Permanently
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("📦 Deregister", key=f"dreg_{name}", help="Archive tool. Can be restored anytime."):
                            ok, msg = deregister_tool(name)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col_b2:
                        with st.popover("🗑️ Delete"):
                            st.warning(f"Permanently wipe `{name}`?")
                            st.caption("Destroys file, memory, SQLite index, and Neo4j node.")
                            if st.button("Confirm Delete", key=f"del_act_{name}", type="primary"):
                                ok, msg = delete_tool_permanently(name)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)

                elif status == "archived":
                    # Archived tool actions: Restore or Permanent Delete
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("🔄 Restore", key=f"res_{name}", help="Re-enable tool and return to active pool."):
                            ok, msg = restore_tool(name)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col_b2:
                        with st.popover("🗑️ Delete"):
                            st.warning(f"Permanently destroy archived `{name}`?")
                            if st.button("Confirm Delete", key=f"del_arch_{name}", type="primary"):
                                ok, msg = delete_tool_permanently(name)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                elif status == "mcp":
                    st.caption("Managed via MCP Server configuration.")

            if tool.get("source_code"):
                with st.expander(f"Inspect Code: `{name}.py`"):
                    st.code(tool["source_code"], language="python")

            st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: SUBAGENTS & COGNITIVE NODES
# ═════════════════════════════════════════════════════════════════════════════
with tab_agents:
    agent_view = st.radio(
        "Agent State",
        options=["Active Subagents", "Archived Subagents", "All"],
        horizontal=True,
    )

    col_a1, col_a2 = st.columns([3, 2])
    with col_a1:
        search_agent = st.text_input("🔍 Search Subagents", key="search_agent").strip().lower()
    with col_a2:
        type_filter = st.selectbox("Agent Architecture", options=["All Architectures", "cognitive", "subagent"])

    display_agents = all_agents
    if agent_view == "Active Subagents":
        display_agents = [a for a in display_agents if a["status"] == "active"]
    elif agent_view == "Archived Subagents":
        display_agents = [a for a in display_agents if a["status"] == "archived"]

    if search_agent:
        display_agents = [
            a for a in display_agents
            if search_agent in a["name"].lower() or search_agent in a["description"].lower()
        ]

    if type_filter != "All Architectures":
        display_agents = [a for a in display_agents if a["type"] == type_filter]

    st.markdown(f"Showing **{len(display_agents)}** subagent(s)")

    if not display_agents:
        st.info("No subagents found matching the selected filters.")

    for agent in display_agents:
        name = agent["name"]
        sig = agent["signature"]
        status = agent["status"]
        atype = agent["type"]
        is_prot = agent.get("is_protected", False)

        with st.container():
            col_ag1, col_ag2 = st.columns([7, 3])
            with col_ag1:
                st.markdown(f"#### `{name}`")
                st.markdown(f"**Signature:** `{sig}`")
                st.markdown(f"_{agent['description']}_")

                badges = []
                if is_prot:
                    badges.append('<span class="badge-protected">🔒 Baseline Primitive</span>')

                if status == "active":
                    badges.append('<span class="badge-generalizable">🟢 Active</span>')
                else:
                    badges.append('<span class="badge-archived">📦 Archived</span>')

                if atype == "subagent":
                    badges.append('<span class="badge-mcp">🤖 ReAct Subagent</span>')
                else:
                    badges.append('<span class="badge-repurposable">🧠 Cognitive Node</span>')

                st.markdown(" ".join(badges), unsafe_allow_html=True)

                if atype == "subagent":
                    st.caption(f"Allowed Tools: {agent.get('allowed_tools') or 'All'} | Max Turns: {agent.get('default_max_turns', 4)}")

            with col_ag2:
                if is_prot:
                    st.caption("⚠️ Baseline cognitive primitive is protected from deletion or archiving.")
                elif status == "active":
                    col_ab1, col_ab2 = st.columns(2)
                    with col_ab1:
                        if st.button("📦 Deregister", key=f"ag_dreg_{name}", help="Archive agent."):
                            ok, msg = deregister_agent(name)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col_ab2:
                        with st.popover("🗑️ Delete"):
                            st.warning(f"Permanently destroy agent `{name}`?")
                            if st.button("Confirm Delete", key=f"ag_del_{name}", type="primary"):
                                ok, msg = delete_agent_permanently(name)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                else:
                    col_ab1, col_ab2 = st.columns(2)
                    with col_ab1:
                        if st.button("🔄 Restore", key=f"ag_res_{name}", help="Restore agent to active pool."):
                            ok, msg = restore_agent(name)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col_ab2:
                        with st.popover("🗑️ Delete"):
                            st.warning(f"Permanently delete archived agent `{name}`?")
                            if st.button("Confirm Delete", key=f"ag_del_arch_{name}", type="primary"):
                                ok, msg = delete_agent_permanently(name)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)

            if agent.get("source_code"):
                with st.expander(f"Inspect Agent Code: `{name}.py`"):
                    st.code(agent["source_code"], language="python")

            st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: MCP SERVERS & TOOLS
# ═════════════════════════════════════════════════════════════════════════════
with tab_mcp:
    col_m_top1, col_m_top2 = st.columns([8, 2])
    with col_m_top1:
        st.markdown("### Configured Model Context Protocol (MCP) Servers")
        st.caption("Manage external tool servers via standard stdio JSON-RPC transport.")
    with col_m_top2:
        if st.button("⚡ Reload MCP System", type="primary", use_container_width=True):
            with st.spinner("Reconnecting servers and discovering tools..."):
                ok, msg = reload_mcp_system()
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    servers = mcp_data.get("servers", {})

    if not servers:
        st.info("No MCP servers currently configured.")

    for s_name, s_info in servers.items():
        with st.container():
            col_ms1, col_ms2, col_ms3 = st.columns([5, 3, 2])
            with col_ms1:
                is_run = s_info.get("is_running", False)
                enabled = s_info.get("enabled", True)

                status_pill = "🟢 Online" if is_run else ("🟡 Enabled (Idle)" if enabled else "⚪ Disabled")
                st.markdown(f"#### `{s_name}` &nbsp; <small>{status_pill}</small>", unsafe_allow_html=True)
                st.markdown(f"**Command:** `{s_info.get('command')}` `{' '.join(s_info.get('args', []))}`")
                if s_info.get("env"):
                    st.caption(f"Environment: {', '.join(s_info['env'].keys())}")

            with col_ms2:
                new_state = st.toggle("Enabled", value=enabled, key=f"mcp_tog_{s_name}")
                if new_state != enabled:
                    ok, msg = toggle_mcp_server(s_name, new_state)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

                st.caption(f"Discovered Tools: **{s_info.get('tool_count', 0)}**")

            with col_ms3:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🧪 Test", key=f"test_{s_name}", help="Verify server handshake and list tools"):
                        with st.spinner("Testing connection..."):
                            ok, tools_found, test_msg = test_mcp_server(s_name, s_info)
                            if ok:
                                st.success(test_msg)
                                if tools_found:
                                    st.write("Tools found:", tools_found)
                            else:
                                st.error(test_msg)
                with col_btn2:
                    with st.popover("🗑️ Remove"):
                        st.warning(f"Delete MCP server `{s_name}`?")
                        if st.button("Confirm", key=f"del_mcp_{s_name}", type="primary"):
                            ok, msg = delete_mcp_server(s_name)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

            # Server Discovered Tools Expander
            server_tools = s_info.get("tools", [])
            if server_tools:
                with st.expander(f"View {len(server_tools)} Tool(s) from `{s_name}`"):
                    for st_item in server_tools:
                        st.markdown(f"**`{st_item['name']}`** — {st_item.get('description', '')}")
                        st.code(st_item.get("signature", ""), language="python")

            st.markdown("---")

    # ── Add New MCP Server Form ────────────────────────────────────────────────
    with st.expander("➕ Add New MCP Server"):
        with st.form("add_mcp_form"):
            new_name = st.text_input("Server Name (e.g. 'github', 'brave-search', 'postgres')")
            new_command = st.text_input("Command", value="npx", help="Executable name or absolute path (e.g. npx, uvx, python, node)")
            new_args_str = st.text_input("Arguments (space-separated)", value="-y @modelcontextprotocol/server-example")
            new_env_str = st.text_area("Environment Variables (JSON object)", value='{\n  "API_KEY": "${API_KEY}"\n}')
            new_cwd = st.text_input("Working Directory (Optional)", value="")

            submitted = st.form_submit_button("Save Server Configuration", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Server name is required.")
                elif not new_command.strip():
                    st.error("Command is required.")
                else:
                    args_list = [a.strip() for a in new_args_str.split(" ") if a.strip()]
                    env_dict = {}
                    if new_env_str.strip():
                        try:
                            env_dict = json.loads(new_env_str)
                        except Exception as e:
                            st.error(f"Invalid Environment JSON: {e}")
                            st.stop()

                    conf = {
                        "enabled": True,
                        "command": new_command.strip(),
                        "args": args_list,
                        "env": env_dict,
                    }
                    if new_cwd.strip():
                        conf["cwd"] = new_cwd.strip()

                    ok, msg = save_mcp_server(new_name.strip(), conf)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
