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
   - `timestamp`, `turn_id`, `latency_ms`, `status` (`direct_response` | `pipeline` | `error`), `input_tokens`, `output_tokens`, `tools_retrieved_count`

2. **Answerer** (`data/metrics/answerer.csv`)
   - `timestamp`, `turn_id`, `latency_ms`, `status` (`success` | `error`), `input_tokens`, `output_tokens`

3. **DAG Execution Engine** (`data/metrics/dag_execution.csv`)
   - `timestamp`, `turn_id`, `start_time`, `end_time`, `latency_ms`, `status` (`success` | `node_failed` | `aborted`), `dag_size` (step count), `dag_depth`, `tool_nodes_count`, `cognitive_nodes_count`, `subagent_nodes_count`, `failed_node_id`

4. **CachingManager** (`data/metrics/caching.csv`)
   - `timestamp`, `turn_id`, `cache_status` (`hit` | `miss`), `hit_tier` (`instant` | `llm_verified` | `miss`), `similarity_score`, `llm_verify_result` (`verified` | `rejected` | `n/a`), `ttl_valid` (`true` | `false`), `latency_ms`, `tokens_saved_estimate`, `estimated_tokens_saved`

5. **ToolBuilder & AgentBuilder** (`data/metrics/builders.csv`)
   - `timestamp`, `target_name`, `builder_type` (`tool` | `agent`), `latency_ms`, `status` (`passed` | `failed`), `attempt_count` (0 = first pass, 1–3 = debugger retry), `failure_reason` (`ast_violation` | `syntax` | `pytest_fail` | `judge_fail` | `none`), `debugger_prior_used` (`true` | `false`), `input_tokens`, `output_tokens`

6. **Cognitive Nodes** (`data/metrics/cognitive.csv`)
   - `timestamp`, `turn_id`, `agent_name`, `latency_ms`, `status` (`success` | `error`), `input_tokens`, `output_tokens`, `payload_truncated` (`true` | `false`)

7. **ReAct Subagents** (`data/metrics/subagents.csv`)
   - `timestamp`, `turn_id`, `agent_name`, `latency_ms`, `status` (`success` | `error`), `input_tokens`, `output_tokens`, `turns_count`, `tools_called_count`, `hit_turn_cap` (`true` | `false`)

8. **Tool Usage & Subprocess Execution** (`data/metrics/tool_usage.csv`)
   - `timestamp`, `turn_id`, `tool_name`, `status` (`0` | `-1`), `latency_ms`, `payload_truncated`, `is_mcp`, `cached`

9. **Gate & Control Node Evaluator** (`data/metrics/gate_evaluator.csv`)
   - `timestamp`, `turn_id`, `node_id`, `gate_type`, `mode` (`ast` | `nlp`), `condition`, `result` (`true` | `false`), `latency_ms`, `input_tokens`, `output_tokens`, `pruned_nodes_count`

10. **Autonomous Goal Runner** (`data/metrics/goal_runner.csv`)
    - `timestamp`, `goal_id`, `wave_index`, `dag_size`, `nodes_succeeded`, `nodes_failed`, `latency_ms`, `status`, `input_tokens`, `output_tokens`

11. **Memory Manager & Workers** (`data/metrics/memory.csv`)
    - `timestamp`, `event_type` (`working_turn` | `compaction` | `short_term_worker` | `long_term_worker`), `working_tokens_count`, `compaction_triggered` (`true` | `false`), `tokens_freed`, `turns_evaluated`, `turns_promoted`, `facts_consolidated`

## 4. How to Display Metrics

MAVIS presents metrics in dedicated interfaces designed for distinct usage contexts: a streamlined **CLI Interface** for real-time awareness during interactive sessions, and two **Locally Hosted Web Dashboards** for visual analytics and tool management.

---

### A. CLI Display (In-Terminal Observability)

The CLI provides lightweight, immediate feedback without distracting from conversation flow:

1. **Persistent Bottom Toolbar**:
   - Integrated into the interactive prompt (`prompt_toolkit`).
   - Displays real-time session counters via `core.metrics.get_metrics_summary`:
     `[MAVIS v1.0] Session: Xm • Tokens: X in / Y out • Cache Hits: Z • WM: X/12k tokens • Type / for commands`

2. **`/metrics` Slash Command**:
   - Renders rich formatted terminal tables (`core.metrics.format_metrics_tables`) summarizing:
     - **Session Overview**: Total queries, direct responses, pipeline runs, and runtime duration.
     - **Token Economics & Components**: Input, output, and grand total tokens across Interpreter, Answerer, Cognitive nodes, Subagents, and Builders.
     - **Latency & Performance**: Average, Median, and Max execution times per subsystem.
     - **Semantic Caching**: Cache hits, hit rates, and tokens avoided.
     - **Memory & Security**: Working memory pressure, compaction counts, and active tokens.

3. **Session Exit Summary**:
   - Printed automatically upon clean exit (`exit`, `quit`, or `Ctrl+C`).
   - Displays a concise summary of tokens used, duration, and caching efficiency.

---

### B. Locally Hosted Streamlit Dashboards

1. **MAVIS Local Observability Dashboard (`scripts/dashboard.py`)**:
   - **Access**: Served at `http://localhost:8501`, launched via `/dashboard` or standalone:
     ```bash
     streamlit run scripts/dashboard.py --server.port 8501
     ```
   - **Isolated Component Reads by Default**: Panels independently read their isolated CSV files (`caching.csv`, `dag_execution.csv`, `gate_evaluator.csv`, `goal_runner.csv`) to prevent CPU/IO bottlenecks.
   - **Turn & Goal Wave Inspector**: Dynamically correlates `turn_id` or `goal_id` to render the complete multi-wave DAG waterfall, gate branching decisions, and runtime debugger patches.

2. **MAVIS Tool Studio (`scripts/tooldash.py`)**:
   - **Access**: Served at `http://localhost:8502`, launched via `/tooldash` or standalone:
     ```bash
     streamlit run scripts/tooldash.py --server.port 8502
     ```
   - Provides live visual management, syntax inspection, and status monitoring across registered tools, cognitive subagents, and connected MCP servers.