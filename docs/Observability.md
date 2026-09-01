# MAVIS Observability & Performance Analytics

## 1. Objectives & Overview

The MAVIS Observability system tracks the performance, efficiency, and reliability of MAVIS's core execution path as a personal assistant:
- Measure end-to-end response latency and core phase execution times.
- Track token consumption across the main cognitive components.
- Evaluate semantic cache hit rates and compute savings.
- Track DAG pipeline execution success and structure.
- Monitor tool and sub-agent generation, usage, and failure rates.
- Observe working memory context pressure and security gate decisions.

---

## 2. Core Metrics by Component

### A. Latency & Execution Time (Main Phases)
Tracked as **Average**, **Median**, and **Max** (in ms / seconds):
- **`turn_latency_e2e`**: Total wall-clock time from user query submission to final displayed answer.
- **`latency_cache_check`**: Semantic cache lookup and LLM verification time.
- **`latency_dag_planning`**: Interpreter LLM time to plan the execution DAG.
- **`latency_pipeline_execution`**: Total execution time for all DAG steps (tools & subagents).
- **`latency_tool_build`**: Time taken when a missing tool must be dynamically generated and tested.
- **`latency_answerer`**: Presentation module time synthesizing the final user response.

---

### B. Token Usage & Cost (Main Cognitive Components)
Tracked by total counts and per-turn breakdown:
- **Total Input / Output Tokens**: Overall prompt and completion tokens.
- **Tokens by Component**:
  - `interpreter`: DAG pipeline planning and tool selection.
  - `subagents`: In-memory cognitive transformations (e.g. `semantic_transform`).
  - `answerer`: Final response formatting and presentation.
  - `tool_builder` / `agent_builder`: Dynamic code/prompt synthesis and debugger retries.
- **Tokens Saved via Cache**: Estimated tokens avoided when serving results directly from cache.

---

### C. Execution Routing & DAG Dynamics
- **Routing Ratio**: Percentage of queries handled via direct response vs multi-step DAG pipeline.
- **DAG Complexity**: Average DAG step count and critical path depth for successful runs.
- **Pipeline Success Rate**: Percentage of pipelines that run all nodes to completion.

---

### D. Semantic Caching Efficiency
- **Cache Hit Rate**:
  - `tier_instant_hit` (>0.95 similarity, instant return)
  - `tier_llm_verified` (0.85–0.95 similarity, verified by LLM)
  - `tier_miss` (<0.85 similarity or LLM rejected)
- **Result vs Pipeline Reuse**: Percentage of cache hits served instantly (valid TTL) vs re-executing cached pipelines with new extracted parameters.

---

### E. Tool & Sub-Agent Lifecycle
- **Tool Creation vs Reuse**: Ratio of queries using existing tools vs triggering new tool synthesis.
- **Build Success Rate**: Percentage of generated tools/agents passing on first attempt vs after debugger retry vs failed.
- **Tool Execution Failure Rate**: Frequency of tool execution errors or timeouts (30s limit).

---

### F. Memory & Security Health
- **Working Memory Pressure**: Active tokens in working memory vs 12,000 token cap.
- **Memory Compactions**: Frequency of compaction triggers.
- **Memory Promotion**: % of working memory promoted to short memory and % of short memory promoted to long term memory.
- **ONI Gate Decisions**: Ratio of user approvals vs denials on interactive confirmation prompts.

---

## 3. How to Actually Track Metrics
The main idea is to build a lightweight **Metric Emitter** module that will be integrated into each core component. Each component records and appends its relevant metrics directly to CSV files in `data/metrics/`. 

Storing metrics in CSV format allows:
- Zero-overhead, append-only metric writes during runtime.
- Easy manual viewing and simple programmatic aggregation (Average, Median, Max, Totals) across any time window.
- Seamless consumption by both the CLI summary command and the local web dashboard.

Every emitted metric row includes an ISO-8601 `timestamp` and a correlating `turn_id` for cross-component joins and timeline analysis.

### Metrics Emitted by Component

1. **Interpreter** (`data/metrics/interpreter.csv`)
   - `timestamp`, `turn_id`, `latency_ms` (time from entry to output), `status` (`direct_response` | `pipeline` | `error`), `input_tokens`, `output_tokens`, `tools_retrieved_count`

2. **Answerer** (`data/metrics/answerer.csv`)
   - `timestamp`, `turn_id`, `latency_ms` (time from entry to output), `status` (`success` | `error`), `input_tokens`, `output_tokens`

3. **DAG Execution Engine** (`data/metrics/dag_execution.csv`)
   - `timestamp`, `turn_id`, `start_time`, `end_time`, `latency_ms`, `status` (`success` | `node_failed` | `aborted`), `dag_size` (step count), `dag_depth`, `tool_nodes_count`, `subagent_nodes_count`, `failed_node_id`

4. **CachingManager** (`data/metrics/caching.csv`)
   - `timestamp`, `turn_id`, `cache_status` (`hit` | `miss`), `hit_tier` (`instant` | `llm_verified` | `miss`), `similarity_score`, `llm_verify_result` (`verified` | `rejected` | `n/a`), `ttl_valid` (`true` | `false`), `latency_ms`, `tokens_saved_estimate`

5. **ToolBuilder & AgentBuilder** (`data/metrics/builders.csv`)
   - `timestamp`, `target_name`, `builder_type` (`tool` | `agent`), `latency_ms`, `status` (`passed` | `failed`), `attempt_count` (0 = first pass, 1–3 = debugger retry), `failure_reason` (`ast_violation` | `syntax` | `pytest_fail` | `judge_fail` | `none`), `debugger_prior_used` (`true` | `false`), `input_tokens`, `output_tokens`

6. **Subagents** (`data/metrics/subagents.csv`)
   - `timestamp`, `turn_id`, `agent_name` (e.g. `semantic_transform`), `latency_ms`, `status` (`success` | `error`), `input_tokens`, `output_tokens`, `payload_truncated` (`true` | `false`)

7. **ONI Security Harness** (`data/metrics/oni.csv`)
   - `timestamp`, `turn_id`, `target_command_or_path`, `phase` (`pre_flight` | `runtime`), `trust_level`, `oni_decision` (`whitelist_allowed` | `greylist_prompted` | `blacklist_blocked`), `user_decision` (`allowed` | `denied` | `n/a`), `dwell_time_ms` (time waiting for user approval)

8. **Memory Manager & Workers** (`data/metrics/memory.csv`)
   - `timestamp`, `event_type` (`working_turn` | `compaction` | `short_term_worker` | `long_term_worker`), `working_tokens_count`, `compaction_triggered` (`true` | `false`), `tokens_freed`, `turns_evaluated`, `turns_promoted`, `facts_consolidated`

## 4. How to Display Metrics

MAVIS presents metrics in two dedicated interfaces designed for distinct usage contexts: a streamlined **CLI Interface** for real-time awareness during interactive sessions, and a comprehensive **Locally Hosted Web Dashboard** for in-depth historical analysis.

---

### A. CLI Display (In-Terminal Observability)

The CLI provides lightweight, immediate feedback without distracting from conversation flow:

1. **Persistent Bottom Toolbar**:
   - Integrated into the interactive prompt (`prompt_toolkit`).
   - Displays real-time session counters: `Session Time` | `Tokens: in/out` | `Cache Hits` | `Memory Pressure (tokens / 12k cap)`.

2. **`/metrics` Slash Command**:
   - Renders a rich formatted terminal table summarizing recent performance across:
     - **Session Summary**: Total queries handled, direct response vs pipeline ratio, total session duration.
     - **Latency**: Average, Median, and Max for end-to-end turns, planning, and execution.
     - **Token Usage**: Total input/output tokens and component breakdown (`interpreter`, `subagents`, `answerer`, `builders`).
     - **Semantic Cache**: Hit rate percentage, instant vs LLM-verified hits, and estimated tokens saved.
     - **Tool & Agent Lifecycle**: Tools reused vs created, build pass rates, and failure counts.
     - **Security & Memory**: ONI gate approvals/denials, working memory token count, and compaction count.

3. **Session Exit Summary**:
   - Printed automatically when exiting MAVIS (`exit`, `quit`, or `Ctrl+C`).
   - Displays a compact summary card showing session length, total turns, tokens consumed, and cache hit efficiency.

---

### B. Locally Hosted Web Dashboard

A dedicated, lightweight web dashboard for visual and historical analytics:

1. **Architecture & Data Access Pattern**:
   - **Local Delivery**: Served locally on `http://localhost:8000` (or configurable port), launched on-demand via the `/dashboard` slash command or standalone script (`python -m core.dashboard`).
   - **Isolated Component Reads by Default**: To guarantee fast rendering without CPU/IO bottlenecks, all dashboard panels read **only their corresponding CSV file** independently (e.g. Caching panel reads `caching.csv`, Execution panel reads `dag_execution.csv`). No multi-file joins are performed by default.
   - **Lazy Cross-File Join (On-Demand Turn Inspector)**: Cross-file joining on `turn_id` is performed *only* when the user explicitly clicks a specific turn to inspect its full end-to-end trace and component waterfall.

2. **Dashboard Views & Panels**:
   - **Component KPI Cards**: Standalone summaries for Interpreter, DAG Engine, Caching, and Builders computed directly from their respective CSVs.
   - **Latency & Performance Views**: Independent latency timelines per component (Average, Median, Max).
   - **Token Economics & Cost**: Component-level token consumption curves over time.
   - **Semantic Caching Performance**: Visual breakdown of similarity score distribution, cache hit tiers (`instant` vs `llm_verified` vs `miss`), and TTL validity.
   - **DAG & Tool Reliability**: Distribution of DAG step counts/depths, tool synthesis pass vs retry rates, and failure taxonomy breakdown.
   - **Security & Memory Health**: ONI gate decision history (Allowed vs Denied) and working memory token growth/compaction curves.
   - **Turn Inspector (Trace View)**: On-demand drill-down that joins rows by `turn_id` across CSVs to display the full lifecycle of a single query.

3. **Interactivity & Controls**:
   - **Time Filters**: Filter data by `Current Session`, `Last 24 Hours`, `Last 7 Days`, or `All-Time`.
   - **Live Auto-Refresh**: Configurable poll interval (e.g. 5s) to monitor metrics in real-time while using MAVIS.