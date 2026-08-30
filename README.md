# MAVIS — Modular Autonomous Virtual Intelligence Suite

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
interpret_command()          ← LLM Client (Gemini / OpenAI / Ollama): decomposes into DAG JSON
      │
      ├─ missing tools? ──► ToolBuilder.build_tool()
      │                          ├─ Cross-read memories: loads conventions & debugger fixes
      │                          ├─ LLM generates Python tool module
      │                          ├─ AST scan (ONI: blocks subprocess, socket, os.system, …)
      │                          ├─ ToolTester runs auto-generated pytest validation
      │                          ├─ FAIL? → debug_tool() retry loop (max_retries from config)
      │                          │           └─ Fixed? → writes fix to memories/debugger/
      │                          └─ PASS (attempt 0) → writes pattern to memories/toolbuilder/
      │                                                & registers in commands_list.json
      ▼
execute_pipeline()
      ├─ ONI pre-flight scan   ← blacklist → abort; greylist → batch interactive confirmation
      │
      └─ call_command()        ← isolated subprocess execution via core/run_tool.py
            │
            └─ Final Result Display (Rich formatting)

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
| **`debugger`** | Failure-to-fix pairs, syntax/import pitfalls | Event-Driven (Immediate on debug fix) | `memories/debugger/` |
| **`tasks`** | Scheduled job runs, background health | Event-Driven (Immediate on job run) | `memories/tasks/` |

**Cross-Namespace Learning**: ToolBuilder subscribes to `debugger` memories at generation time to pre-emptively avoid past failure modes without polluting the conversational dialogue context.

### 3. ONI — Operating System & Network Interface
A security boundary that gates all filesystem, system, and network access:
- **Trust Levels**: `ask` (default prompt on sensitive actions), `yolo` (unrestricted), `whitelist_only` (strict deny-by-default).
- **Pre-Flight Scans**: Entire DAG pipeline inspected before any node executes.
- **AST Code Guard**: Dynamically generated tools scanned for forbidden imports (`subprocess`, `socket`, `pty`, `eval`, etc.).
- **Process Isolation**: All tools run inside independent Python subprocesses managed by `core/run_tool.py`.
- **Append-Only Audit Log**: Every security decision logged to `logs/oni_audit.jsonl`.

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
        "session_timeout_minutes": 30
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
│   ├── helpers.py            # Structured logging (log_it)
│   ├── output.py             # Rich console formatting & UI theme
│   ├── scheduler.py          # Background task scheduler
│   ├── run_tool.py           # Subprocess harness for sandboxed tool execution
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
├── memories/                 # Multi-namespace memory subsystem
│   ├── memory_store.py       # Namespaced MemoryStore manager (working/ST/LT)
│   ├── embedding.py          # Vector embedding wrapper & cosine similarity
│   ├── emotion_classifier.py # Dialogue turn classification & promotion rules
│   ├── interpreter/          # Conversational memory (working_memory.json, ST/LT Chroma)
│   ├── toolbuilder/          # Successful tool patterns (patterns.json, Chroma)
│   ├── debugger/             # Tool failure fixes (fixes.json, Chroma)
│   └── tasks/                # Background job histories (events.json, Chroma)
├── data/
│   ├── mavis_config.json     # Master configuration file
│   ├── commands_list.json    # Tool signature registry
│   └── user_profile.json     # User preferences & profile information
├── docs/                     # Specifications and architectural documentation
│   ├── bugs.md               # Tracked issues & resolution history
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
├── tests/                    # Unit tests and generated tool tests
├── prompts/                  # LLM prompt templates
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

### Running Automated Tests
To run memory and unit tests:
```bash
source bin/activate
python3 tests/test_memory_specialization.py
```
