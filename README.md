# MAVIS — My Awesome Virtual Intelligence Suite

MAVIS is a self-modifying, local-first personal AI assistant that dynamically adapts to user requirements. It decomposes natural language instructions into validated Directed Acyclic Graph (DAG) pipelines, synthesizes and self-debugs missing tools and sub-agents on-the-fly, enforces OS-level security through the **ONI** harness, and maintains long-term memory via a **Topic-Subscribed Neo4j Knowledge Graph**.

MAVIS is **LLM provider-agnostic**, supporting cloud models (Google Gemini, OpenAI, Claude) and 100% offline local inference (via Ollama or OpenAI-compatible endpoints like vLLM).

---

## System Architecture

```
User Input (text)
      │
      ▼
handle_slash_command()       ← /goal, /config, /status, /trust, /metrics, /mcp, /save
      │  (if standard prompt)
      ▼
cache_manager.check_cache()  ──► [Cache Hit >0.95] ─────────► Answerer.synthesize() (sub-ms instant reply)
      │                      └─► [Pipeline Hit 0.85-0.95] ─► Fast LLM verify ──► execute_pipeline()
      ▼  (if miss)
interpret_command() / /goal   ← Plans Heterogeneous DAG with Tool, Subagent, & Gate nodes
      │                          ├─ Discovers & syncs external tools via Model Context Protocol (MCP)
      │                          └─ /goal orchestrates iterative wave DAGs with ONI task-scoped lease
      ├─ missing tools?  ──► ToolBuilder.build_tool()
      │                          ├─ Queries past tooling patterns & debugger fixes
      │                          ├─ Generates Python module & scans AST (blocks dangerous imports)
      │                          ├─ Runs automated pytest validation in isolated process
      │                          └─ FAIL? → debug_tool() retry loop → writes fix to Neo4j
      │                             PASS? → registers in commands_list.json & Neo4j
      │
      ├─ missing agents? ──► AgentBuilder.build_agent()
      │                          ├─ Queries past prompt failure priors
      │                          ├─ Synthesizes BaseAgent with structured I/O schemas
      │                          ├─ Runs AgentTester (LLM-as-a-Judge discrete evaluation)
      │                          └─ FAIL? → AgentDebugger refines prompts & constraints
      │                             PASS? → registers in agents_list.json & Neo4j
      ▼
execute_pipeline()
      ├─ ONI pre-flight scan   ← Security gate: blacklist aborts, greylist requests confirmation
      ├─ Live output streaming ← Streams subprocess stderr/stdout live to terminal (no frozen screens)
      ├─ Gate evaluation       ← Evaluates condition ($node.status == 0) & prunes untaken branches
      ├─ Dispatch nodes        ← Executes "tools" (sandboxed subprocess), "MCP tools", & "subagents"
      ├─ Scratchpad offload    ← Large outputs (>4KB) offloaded to data/scratch/ with pointer digest
      ├─ node failure?         ──► PipelineDebugger.diagnose_and_repair()
      │                                ├─ Diagnoses root cause from error message & prior outputs
      │                                ├─ Patches parameters (e.g. path prefixes) or replaces node
      │                                └─ Retries execution closed-loop (up to 2 attempts)
      ├─ cache_manager.save()  ← Caches executed pipeline & results to SQLite (on success)
      └─ Answerer.synthesize() ← Terminal presentation layer: synthesizes final output (or explains failures gracefully)
```

---

## Main Components & Unique Design Points

### 1. Heterogeneous DAG, Gate Nodes, & Self-Healing Execution (`core/dag.py`, `core/gate_evaluator.py`, `core/pipeline_debugger.py`, `core/scratchpad.py`, `core/goal_runner.py`)
- **Tool Nodes vs. Sub-Agent Nodes vs. Gate Nodes**: Treats deterministic code execution, cognitive reasoning, and conditional control flow as first-class primitives.
- **Gate Nodes (`type: "gate"`) & Branch Pruning**: Supports deterministic evaluation (0 LLM tokens, $<1\text{ms}$ AST evaluator for conditions like `$n1.status == 0`, `len($n1.output) > 0`, `'failed' not in $n1.output.lower()`) and semantic NLP evaluations via LLM-as-a-Judge. Automatically sorts gates before dependent steps and prunes inactive branch subtrees.
- **Autonomous Goal Runner (`/goal`)**: Executes high-level objectives across dynamically planned wave DAGs up to `max_iterations = 8`. Preserves strict mathematical acyclicity (preventing cycle errors) while enabling autonomous self-correction and looping. Runs under a single-prompt ONI Task-Scoped Lease.
- **Live Output Streaming**: Subprocess execution prints live to the terminal line-by-line, providing immediate visual feedback during long compilation, test, or git tasks.
- **Scratchpad Pointer Pattern (`core/scratchpad.py`)**: Automatically offloads tool outputs $>4\text{ KB}$ into `data/scratch/`, injecting compact Head/Tail summaries and file pointers into prompts to prevent token bloat while retaining 100% raw data fidelity.
- **Closed-Loop Runtime Self-Healing (`PipelineDebugger`)**: Resolves live execution failures. When a tool or sub-agent fails, the debugger inspects the failure trace and prior node outputs, patches arguments (e.g. relative path resolution) or substitutes alternative nodes, and retries execution.
- **Graceful Failure Handoff & Truthful Memory**: Unrecoverable failures pass partial outputs to the Answerer to converse naturally with the user about what went wrong, recording failures accurately with `tool_failure=True` so follow-up prompts retain complete context.
- **Terminal Presentation Plane**: DAG execution is strictly for data acquisition. Raw outputs are quarantined within `<tool_data>` blocks and passed to the Answerer, eliminating prompt injection and infinite re-planning loops.

### 2. Tiered Memory & Topic-Subscribed Knowledge Graph (`memories/`)
- **Three-Tiered Memory Hierarchy**:
  - **Working Memory** (`memories/<ns>/working_memory.json`): Real-time conversational buffer with turn embeddings. When it crosses the token budget (1,500 tokens), compaction promotes salient turns (emotion, intent, directives, failures) to short-term storage and condenses routine chatter into an LLM session summary.
  - **Short-Term Memory** (`memories/<ns>/short_term/json/`): Rolling 7-day episodic logs stored as daily JSON files (`YYYY-MM-DD.json`), queried via in-memory cosine similarity.
  - **Long-Term Memory**: Structured Neo4j Property Graph combined with domain-specific JSON catalogs (`facts.json`, `behaviours.json`, `patterns.json`, `fixes.json`).
- **Decoupled Promotion & Async Extraction**: Background daemons (`tasks/long_term_worker.py`) scan episodic turn deltas and extract structured entity triples asynchronously (`memories/knowledge_extractor.py`), keeping live conversational turns fast.
- **Hybrid Dense-Graph Retrieval**: Uses Neo4j 5+ native vector indexing (`entity_embeddings`) to find seed entities, then traverses active 1-hop relationships within subscribed topics in a single Cypher query.
- **Strict Topic Boundaries**:
  - `interpreter`: Reads `["user.*", "env.*", "tooling.*", "agents.*", "debugging.*"]`, writes `user.profile`.
  - `toolbuilder`, `agent_debugger`, & `pipeline_debugger`: Read `["env.*", "tooling.*", "agents.*", "debugging.*"]`. **Hard-blocked from `user.*`**, protecting user privacy and eliminating context pollution in code/pipeline generation.
- **Deterministic Temporal Deduplication**: Functional 1-to-1 predicates (`PREFERS_EDITOR`, `HAS_OS`, `USES_SHELL`) automatically supersede prior active edges (`is_active = false, superseded_at = datetime()`).
- **Zero-LLM Subagent & Debugger Writes**: ToolBuilder, ToolDebugger, AgentDebugger, and PipelineDebugger register tool capabilities, AST rules, prompt remedies, and runtime DAG parameter repairs directly into Neo4j without extra LLM overhead.

### 3. Model Context Protocol (MCP) Integration (`core/mcp_client.py`)
- **Stdio JSON-RPC Client**: Connects to external MCP servers defined in `data/mcp_servers.json` (or central config) over standard input/output.
- **Dynamic Schema Translation**: Discovers tools via `tools/list` and translates JSON Schema definitions into standard MAVIS Python signatures (`format_mcp_tool_signature`), seamlessly syncing with `commands_list` and the SQLite tool retriever.
- **First-Class Pipeline Execution**: Dispatches external MCP tools directly within DAG pipelines alongside native tools.
- **Live Lifecycle Management**: Interactive inspection and hot-reloading via `/mcp [status|list|reload]`.

### 4. ONI Security Harness (`oni/`)
- **Process Isolation**: All dynamic tools run in independent sandboxed subprocesses (`core/run_tool.py`).
- **AST Code Guard**: Rejects scripts attempting forbidden imports (`subprocess`, `socket`, `pty`, `eval`) before execution.
- **Safe Read Whitelist**: Read-only inspection commands (`ls`, `find`, `read_file_contents`, `read_and_concatenate_files`, `git status`) execute without interactive prompt interruption.
- **Task-Scoped Execution Leases**: Long-running autonomous goals (`/goal`) acquire a temporary single-prompt task lease (`oni.acquire_task_lease()`), allowing iterative wave pipelines to complete without repetitive permission dialogs.
- **Tiered Trust Levels**:
  - `ask`: Interactive user confirmation for write/network actions.
  - `yolo`: Automated execution for trusted sessions.
  - `whitelist_only`: Strict deny-by-default execution.
- **Audit Logging**: Append-only security decisions recorded to `logs/oni_audit.jsonl`.

### 5. Autonomous Tool & Agent Builders (`tool_builder/`, `agent_builder/`)
- **Self-Modifying Assistant**: Automatically synthesizes missing capabilities when encountering unknown commands.
- **Closed-Loop Verification**:
  - **Tools**: Validated with auto-generated `pytest` suites and module eviction before registry insertion.
  - **Agents**: Evaluated by an LLM-as-a-Judge against a 4-part rubric (Schema, Fidelity, Negative Constraints, Containment).
- **Automated Self-Debugging**: Failed components enter an automated debug loop (up to 3 retries) and record verified fixes into Neo4j.

### 6. SQLite Semantic Caching (`core/caching.py`)
- **Zero External Vector Overhead**: A fast SQLite store with in-memory cosine similarity.
- **Tiered Retrieval**:
  - `> 0.95` similarity: Instant cache hit.
  - `0.85 - 0.95` similarity: Fast LLM verification checks if the cached pipeline satisfies the query.
- **Dynamic Parameter Extraction**: Adapts generalized pipelines to new inputs without replanning.
- **Lifecycle Management**: TTL invalidation for time-sensitive results and LRU eviction for storage caps.

### 7. Categorical Tool Retrieval (`core/tool_retriever.py`)
- **Prefix Caching Optimization**: Tools are classified into:
  - `generalizable`: Essential primitives (file reading, datetime). **Always included** in the prompt to prevent planning hallucinations.
  - `repurposable` & `specialized`: Domain tools retrieved dynamically via cosine ranking down to top-K.
- Backed by an embedded SQLite registry.

### 8. Telemetry & Observability (`core/metrics.py`, `scripts/dashboard.py`)
- **Append-Only CSV Streams**: High-throughput metric emissions in `data/metrics/` across all components (including `gate_evaluator`, `goal_runner`, and `pipeline_debugger`).
- **Zero-Join Analytics**: Fast aggregations (Average, Median, Max) per component.
- **Turn & Goal Wave Trace Inspector**: Reconstructs end-to-end execution lifecycles by lazily correlating `turn_id` or `goal_id`, visualizing multi-wave DAG progressions, gate decisions, and debugger repairs.
- **Local Streamlit Dashboard**: Web UI accessible via `/dashboard` at `http://localhost:8501`.

### 9. Provider-Agnostic LLM Layer (`core/llm/`)
- Unified `BaseLLMClient` supporting:
  - **Google Gemini**: Vertex AI & Gemini Studio (`gemini-2.5-flash`, `text-embedding-004`).
  - **OpenAI Compatible**: Native support for OpenAI, Groq, DeepSeek, and vLLM.
  - **Ollama**: Local offline inference (`llama3.2`, `nomic-embed-text`).

---

## Setup & Quickstart

### Prerequisites
- Linux / macOS with Python 3.11+
- Neo4j 5.0+ (Local Docker or Neo4j Aura cloud instance)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ishan-1/MAVIS.git ~/PerTools/MAV
cd ~/PerTools/MAV

# Create virtual environment
python3 -m venv .
source bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration (`.env`)

Create `.env` in the project root:

```env
# Primary LLM Provider
VERTEX_API_KEY="your-gemini-or-vertex-key"
# OPENAI_API_KEY="your-openai-key"

# Neo4j Knowledge Graph
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-password"
```

### Running MAVIS

```bash
source bin/activate
python main.py
```

---

## Runtime Slash Commands

Control MAVIS dynamically inside the interactive shell:

```bash
/help                           # View help and command assistance
/goal <description>             # Run autonomous multi-wave goal execution until completed
/status                         # View heartbeats, worker processes, and memory state
/metrics [session]              # Print terminal performance and latency tables
/dashboard                      # Launch local Streamlit observability dashboard
/config                         # Display active configuration table
/config set llm.provider ollama # Switch active LLM provider (gemini | openai | ollama)
/config set llm.model llama3.2  # Change active model
/config save                    # Persist runtime settings to data/mavis_config.json
/mcp status|list|reload         # Inspect, list, or hot-reload external MCP servers & tools
/trust ask|yolo|whitelist       # Change ONI security trust level
/allow <tool_name>              # Whitelist tool
/block <tool_name>              # Blacklist tool
/save [filename.md]             # Export conversation session to Markdown
exit | quit                     # Clean shutdown
```

---

## Project Directory Layout

```
MAV/
├── main.py                     # Interactive shell & orchestrator
├── core/                       # Core runtime package
│   ├── config.py               # Central configuration manager (MAVISConfig)
│   ├── caching.py              # SQLite semantic pipeline cache
│   ├── tool_retriever.py       # SQLite categorical tool retriever
│   ├── mcp_client.py           # Model Context Protocol (MCP) stdio client & schema mapper
│   ├── goal_runner.py          # Autonomous multi-wave goal execution engine
│   ├── gate_evaluator.py       # Deterministic AST and NLP gate node evaluator
│   ├── pipeline_debugger.py    # Closed-loop live DAG execution debugger & repair
│   ├── scratchpad.py           # Scratchpad disk offload (>4KB) with head/tail digests
│   ├── answerer.py             # Presentation layer synthesizing final responses
│   ├── metrics.py              # CSV telemetry emitter & aggregator
│   ├── dag.py                  # DAG parsing, topological sorting & branch pruning
│   ├── run_tool.py             # Subprocess sandbox for tool execution
│   └── llm/                    # Provider-agnostic LLM interface (Gemini, OpenAI, Ollama)
├── tool_builder/               # Autonomous tool builder, tester, & debug loop
├── agent_builder/              # Cognitive sub-agent builder, Judge tester, & debugger
├── agents/                     # Built-in and dynamically synthesized sub-agents
├── memories/                   # Topic-subscribed memory subsystem
│   ├── memory_store.py         # MemoryStore manager & topic subscriptions
│   ├── neo4j_graph.py          # Neo4j property graph & native vector index
│   ├── knowledge_extractor.py  # Decoupled background knowledge extraction
│   └── embedding.py            # Vector embedding wrapper & cosine similarity
├── oni/                        # ONI security harness (AST guard, permissions, task leases)
├── tasks/                      # Background daemons (episodic promotion, consolidation worker)
├── data/                       # Configs, mcp_servers.json, registries, SQLite databases
│   └── scratch/                # Automatic disk scratchpad for large step outputs
├── docs/                       # Architecture specifications (Memory, Caching, Subagents, ONI)
├── scripts/                    # Web dashboard (dashboard.py with Turn & Wave Inspector)
└── tests/                      # Automated test suite (60 unit and integration tests)
```
