# MAVIS — My Awesome Virtual Intelligence Suite

MAVIS is an extensible, local-first personal AI assistant built for resilient automation. It decomposes natural language instructions into validated DAG execution pipelines, synthesizes and tests missing tools on-the-fly, maintains specialized long-term memory across isolated domain topics, and enforces strict security through the **ONI** harness.

MAVIS is **LLM provider-agnostic**, supporting cloud backbones (Gemini, OpenAI, Claude) and 100% offline local models (via Ollama or OpenAI-compatible local endpoints like vLLM).

---

## Architecture

```
User Input (text)
      │
      ▼
handle_slash_command()       ← intercepts /config, /trust, /allow, /block, etc.
      │  (if standard prompt)
      ▼
interpret_command()          ← LLM Client (Gemini / OpenAI / Ollama): decomposes into heterogeneous DAG
      │
      ├─ missing tools? ──► ToolBuilder.build_tool()
      │                          ├─ Cross-read memories: loads conventions & debugger fixes
      │                          ├─ LLM generates Python tool module
      │                          ├─ AST scan (ONI: blocks subprocess, socket, os.system, …)
      │                          ├─ ToolTester runs auto-generated pytest validation (with sys.modules eviction)
      │                          ├─ FAIL? → debug_tool() retry loop (max_retries from config)
      │                          │           └─ Fixed? → writes fix to memories/debugger/
      │                          └─ PASS (attempt 0) → writes pattern to memories/toolbuilder/
      │                                                & registers in commands_list.json
      │
      ├─ missing agents? ─► AgentBuilder.build_agent()
      │                          ├─ Cross-read memories: loads agent_debugger failure priors
      │                          ├─ Synthesizes BaseAgent configuration with input/output schemas
      │                          ├─ AgentTester runs LLM-as-a-Judge discrete evaluation (passed/failed)
      │                          ├─ FAIL? → AgentDebugger refines prompt & negative constraints
      │                          │           └─ Fixed? → writes fix to memories/agent_debugger/
      │                          └─ PASS → registers in data/agents_list.json
      ▼
execute_pipeline()
      ├─ ONI pre-flight scan   ← blacklist → abort; greylist → batch interactive confirmation
      │
      ├─ Dispatch nodes        ← runs "tool" (sandboxed subprocess) & "subagent" (in-memory LLM)
      │                          Strict execution contract: (status_code: int, output: Any)
      │
      └─ Answerer.synthesize() ← Terminal presentation layer quarantines tool data and synthesizes final answer

Background Daemons (Out-of-process workers)
─────────────────────────────────────────────────────────────────────────────
worker_process (whitelist_only trust)
  ├─ short_term_worker   — every 15 min: promote high-signal dialogue turns to ChromaDB
  └─ long_term_worker    — every 8 h:   consolidate facts & behaviours into long-term JSON
```

---

## Core Capabilities

### 1. Provider-Agnostic LLM Layer (`core/llm/`)
MAVIS abstracts all model generation and vector embeddings behind a uniform `BaseLLMClient` contract. You can switch models live without changing any application code:
- **Google Gemini**: Vertex AI & Gemini Studio API keys (`gemini-2.5-flash`, `text-embedding-004`).
- **OpenAI Compatible**: Native support for OpenAI, Groq, DeepSeek, Together AI, and vLLM via standard HTTP.
- **Local Ollama**: Native local inference targeting `http://localhost:11434` (`llama3.2`, `nomic-embed-text`).

### 2. Topic-Specialized Memory Namespaces (`memories/`)
Rather than maintaining a single monolithic memory pool, MAVIS separates memories into isolated disk and vector collections:

| Namespace | Focus Area | Promotion Strategy | Storage Path |
| :--- | :--- | :--- | :--- |
| **`interpreter`** | User dialogue turns, emotion, intent | Progressive (15-min & 8-hr workers) | `memories/interpreter/` |
| **`toolbuilder`** | Proven code conventions, successful builds | Event-Driven (Immediate on clean build) | `memories/toolbuilder/` |
| **`debugger`** | Tool failure fixes, syntax/import pitfalls | Event-Driven (Immediate on debug fix) | `memories/debugger/` |
| **`agent_debugger`** | Sub-agent prompt remedies & negative constraints | Event-Driven (Immediate on agent fix) | `memories/agent_debugger/` |
| **`tasks`** | Scheduled job runs, background health | Event-Driven (Immediate on job run) | `memories/tasks/` |

**Cross-Namespace Learning**: ToolBuilder subscribes to `debugger` memories at generation time to pre-emptively avoid past failure modes without polluting the conversational dialogue context.

### 3. ONI — Operating System & Network Interface
A security boundary that gates all filesystem, system, and network access:
- **Trust Levels**: `ask` (default prompt on sensitive actions), `yolo` (unrestricted), `whitelist_only` (strict deny-by-default).
- **Pre-Flight Scans**: Entire DAG pipeline inspected before any node executes.
- **AST Code Guard**: Dynamically generated tools scanned for forbidden imports (`subprocess`, `socket`, `pty`, `eval`, etc.).
- **Process Isolation**: All tools run inside independent Python subprocesses managed by `core/run_tool.py`.
- **Append-Only Audit Log**: Every security decision logged to `logs/oni_audit.jsonl`.

### 4. Token Optimization & Tool Categorization (`core/tool_retriever.py`)
MAVIS is architected to minimize token overhead and maximize prompt caching:
- **Stable Prefix Ordering**: Static system instructions (`interpreter_system_prompt`) are passed via native `system_instruction` headers. Dynamic user turns follow a deterministic order (`COMMANDS LIST` $\to$ `MEMORY CONTEXT` $\to$ `USER INPUT`), unlocking provider-level prompt caching (Gemini Context Caching, OpenAI Prompt Caching).
- **Categorical Tool Generalizability**: Tools in `data/commands_list.json` are classified into three distinct categories:
  - **`generalizable`**: Universal primitives and glue tools (datetime, parsing, file reading, state). **Always passed in full** to eliminate missing-primitive hallucinations during multi-step DAG planning.
  - **`repurposable`**: Reusable domain utilities (news search, web scrapers, email).
  - **`specialized`**: Bespoke, single-purpose tools for narrow tasks.
  - Domain tools (`repurposable` + `specialized`) are dynamically retrieved via ChromaDB cosine similarity down to a configurable top-K (default: 5).
- **Proactive Memory Compaction**: Working memory injection is capped to the last 8 turns with automatic compaction triggered at `0.5 * max_token` to maintain a lean context window.
- **Discrete Emotion & Directive Classifiers**: Replaced noisy continuous floats with discrete levels (`"low" | "medium" | "high"`) and a clean boolean `directive: bool` for persistent instructions.

### 5. Cognitive Sub-Agent Architecture (`core/agents/`, `agent_builder/`)
MAVIS provides full architectural symmetry between deterministic Python tools and cognitive sub-agents:
- **Heterogeneous DAG Execution**: The Interpreter plans execution graphs containing both deterministic environment tools (`"type": "tool"`) and cognitive LLM sub-agents (`"type": "subagent"`).
- **`BaseAgent` Abstraction**: All sub-agents enforce:
  - Input payload guards (>32k characters automatically truncated to safeguard the context window).
  - `<tool_input>` tag isolation for prompt injection defense.
  - Strict JSON/primitive output schema validation to prevent downstream hallucination cascades.
- **Cognitive Triad (Builder, Tester, Debugger)**:
  - **`AgentBuilder`**: Dynamically writes sub-agents to `agents/` and registers them in `data/agents_list.json`.
  - **`AgentTester` (LLM-as-a-Judge)**: Runs synthetic test cases (happy path, edge-cases, prompt injections) and evaluates them against a 4-part rubric (Schema, Fidelity, Negative Constraints, Containment), emitting strictly discrete `"passed"` or `"failed"` verdicts.
  - **`AgentDebugger`**: Refines agent system prompts and enforces negative constraints (e.g. banning pleasantries or verbose prose).
- **`semantic_transform` Primitive**: A built-in universal sub-agent for ad-hoc unstructured extractions, filtering, and multi-document summarization.
- **Terminal Answerer (`core/answerer.py`)**: A dedicated presentation module that synthesizes final pipeline answers for the user while quarantining raw tool returns inside `<tool_data>` blocks.

### 6. Strict Internal Execution Contract & Error Handling
MAVIS enforces a clean separation of concerns between high-level planning and low-level runtime execution:
- **Internal Execution Contract**: All tools and agents adhere to a mandatory 2-element runtime return convention: `(status_code: int, output: Any)` (`0` for success, `-1` for failure).
- **Deterministic Error Handling**: In `execute_pipeline()`, non-zero status codes immediately halt the pipeline and display formatted error diagnostics, preventing silent failures or cascade errors. Successful outputs (`status == 0`) are automatically unpacked into `$node_id.output`.
- **Clean Functional Planning**: The Interpreter is relieved of execution plumbing and plans in natural functional return signatures (`-> str`, `-> list`), while `core/run_tool.py`, `ToolTester`, and `AgentTester` strictly enforce the internal tuple contract at runtime.
- **Automated Test Cache Eviction**: `ToolTester` explicitly evicts `sys.modules` entries between test retries to ensure that newly debugged code is reloaded fresh from disk.

### 7. Multi-File Summarization & Memory Resilience
- **Multi-File Processing Flow**: Tools like `read_and_concatenate_files` aggregate raw multi-file contents and direct their combined text (`$read_node.output`) into `semantic_transform` for semantic summarization before terminal Answerer presentation.
- **ChromaDB Date Filtering**: Short-term memory retrieval computes rolling TTL cutoffs across metadata dates directly in Python, bypassing ChromaDB operator restrictions on string values.
- **REPL Crash Guard**: The interactive prompt loop in `main.py` is protected by defensive exception handling, ensuring that runtime tool errors or rejected security prompts never crash the active session.

---

## Setup & Quickstart

### Prerequisites
- Linux / macOS with Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Ishan-1/MAVIS.git ~/PerTools/MAV
cd ~/PerTools/MAV

# Set up dedicated virtual environment
python3 -m venv .
source bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Important:** Always activate the local venv (`source bin/activate`) before running MAVIS or running tests to ensure the pinned ChromaDB and SDK libraries are used.

### Configuration (`.env`)

Create or update `.env` in the project root:

```env
# For Gemini provider (default)
VERTEX_API_KEY="your-vertex-or-gemini-key"

# For OpenAI / Groq / DeepSeek (if using openai provider)
OPENAI_API_KEY="your-openai-key"

# Optional tool API keys
OPENWEATHER_API_KEY=""
NEWS_API_KEY=""
```

### Starting MAVIS

```bash
source bin/activate
python main.py
```

To exit cleanly at any time, type `exit`, `quit`, or press `Ctrl+C`.

---

## Configuration & Runtime Commands

All application configuration is managed centrally in **[`data/mavis_config.json`](file:///home/ishan07/PerTools/MAV/data/mavis_config.json)**. Settings can be updated dynamically at runtime via slash commands:

```bash
/help                           # View command assistance
/config                         # Print active configuration table
/config set llm.provider ollama # Switch active LLM provider (gemini | openai | ollama)
/config set llm.model llama3.2  # Change target model name
/config set memory.top_k 8      # Change retrieved context depth
/config save                    # Persist runtime updates to disk
/config reload                  # Reload settings from disk
/trust ask|yolo|whitelist       # Change ONI security trust level
/allow <tool_name>              # Whitelist tool
/block <tool_name>              # Blacklist tool
/audit [N]                      # Display last N lines of security audit log
```

### Example `mavis_config.json`

```json
{
    "llm": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "embedding_model": "text-embedding-004",
        "temperature": 0.2,
        "vertexai": true,
        "base_url": null
    },
    "oni": {
        "trust_level": "ask",
        "whitelist": ["get_current_datetime", "search_news"],
        "greylist": ["restart_process", "pip"],
        "blacklist": [],
        "tool_execution_timeout_seconds": 30
    },
    "memory": {
        "max_token": 12000,
        "top_k": 5,
        "short_term_ttl_days": 7,
        "session_timeout_minutes": 30,
        "working_memory_active_turns": 8,
        "compact_token_threshold": 1500,
        "max_memory_entry_chars": 600,
        "tool_retrieval_threshold": 8,
        "tool_retrieval_top_k": 6,
        "general_tool_threshold": 0.75,
        "specific_tools_top_k": 5
    },
    "toolbuilder": {
        "max_retries": 3,
        "forbidden_imports": ["subprocess", "socket", "fork", "pty"]
    },
    "scheduler": {
        "tick_seconds": 30,
        "short_term_worker_interval_minutes": 15,
        "long_term_worker_interval_minutes": 480
    }
}
```

---

## Project Structure

```
MAV/
├── main.py                   # Application entry point & interactive shell
├── core/                     # Foundational runtime package
│   ├── __init__.py           # Re-exports cfg, log_it, TaskRunner, get_llm_client
│   ├── config.py             # Central config manager (MAVISConfig)
│   ├── answerer.py           # Presentation layer synthesizing final responses
│   ├── helpers.py            # Structured logging (log_it)
│   ├── output.py             # Rich console formatting & UI theme
│   ├── scheduler.py          # Background task scheduler
│   ├── run_tool.py           # Subprocess harness for sandboxed tool execution
│   ├── tool_retriever.py     # Categorical & semantic tool retrieval engine
│   ├── agents/               # Dynamic agent loading & BaseAgent abstraction
│   │   ├── __init__.py       # load_agent dynamic loader
│   │   └── base.py           # BaseAgent class (payload guard, tag isolation)
│   └── llm/                  # Provider-agnostic LLM subsystem
│       ├── __init__.py       # get_llm_client() factory
│       ├── base.py           # BaseLLMClient interface
│       ├── gemini.py         # Google Gemini provider adapter
│       ├── openai_compat.py  # OpenAI / Groq / vLLM provider adapter
│       └── ollama.py         # Local Ollama provider adapter
├── tool_builder/             # Autonomous tool synthesis package
│   ├── __init__.py           # Exports ToolBuilder, ToolTester, ToolBuildError
│   ├── toolbuilder.py        # Code synthesis, self-reflection & debug loop
│   └── tester.py             # ONI AST scanner & pytest test executor
├── agent_builder/            # Cognitive sub-agent synthesis package
│   ├── __init__.py           # Exports AgentBuilder, AgentTester, AgentDebugger
│   ├── agent_builder.py      # Sub-agent synthesis & lifecycle orchestrator
│   ├── tester.py             # LLM-as-a-Judge discrete evaluation harness
│   └── debugger.py           # Prompt constraint refiner & anti-pleasantry debugger
├── agents/                   # Built-in and dynamically synthesized sub-agents
│   ├── __init__.py
│   └── semantic_transform.py # Built-in semantic transformer primitive
├── memories/                 # Multi-namespace memory subsystem
│   ├── memory_store.py       # Namespaced MemoryStore manager (working/ST/LT)
│   ├── embedding.py          # Vector embedding wrapper & cosine similarity
│   ├── emotion_classifier.py # Dialogue turn classification & promotion rules
│   ├── interpreter/          # Conversational memory (working_memory.json, ST/LT Chroma)
│   ├── toolbuilder/          # Successful tool patterns (patterns.json, Chroma)
│   ├── debugger/             # Tool failure fixes (fixes.json, Chroma)
│   ├── agent_debugger/       # Agent prompt remedies & negative constraints
│   └── tasks/                # Background job histories (events.json, Chroma)
├── data/
│   ├── mavis_config.json     # Master configuration file
│   ├── commands_list.json    # Tool registry with generalizability classes
│   ├── agents_list.json      # Sub-agent registry mirroring commands list
│   └── user_profile.json     # User preferences & profile information
├── docs/                     # Specifications and architectural documentation
│   ├── bugs.md               # Tracked issues & resolution history
│   ├── Subagents.md          # Cognitive sub-agent architecture spec
│   ├── Harness.md            # ONI security harness architecture
│   ├── Memory.md             # Multi-namespace memory design & schemas
│   └── UX.md                 # UI/UX interaction standards
├── scripts/
│   └── reset_chroma.sh       # Maintenance tool to rebuild ChromaDB schemas
├── oni/                      # ONI security harness
│   ├── __init__.py           # ONI singleton exports
│   ├── oni.py                # Preflight scanner, call_system_command, call_fs
│   ├── config.py             # Central config adapter for ONI
│   ├── permissions.py        # Command and path classification rules
│   ├── gate.py               # User confirmation gate & prompt
│   └── audit.py              # Append-only security audit log writer
├── tasks/                    # Background worker daemons
│   ├── short_term_worker.py  # Working memory → short-term promoter
│   ├── long_term_worker.py   # Short-term → long-term archiver & pruner
│   └── worker_process.py     # Isolated worker daemon subprocess
├── tools/                    # Dynamic and manual Python tool files (gitignored)
├── tests/                    # Unit tests and generated tool tests (gitignored)
├── prompts/                  # LLM prompt templates
│   ├── prompt_templates.py   # Interpreter, tool builder & tester prompts
│   └── agent_prompt_templates.py # Agent builder, tester & debugger prompts
└── logs/                     # Component logs and audit records (gitignored)
```

---

## Troubleshooting

### ChromaDB `KeyError: '_type'`
If the ChromaDB on-disk schema was created with an incompatible version, run:
```bash
./scripts/reset_chroma.sh
```
This cleanly wipes stale vector collections across all namespaces and regenerates them fresh.

