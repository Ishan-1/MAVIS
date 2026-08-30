# MAVIS Memory

### How Human Memory Works
Humans form new memories in response to the following:
- Strong emotions
- Something very new/interesting
- Repetition

Memory consolidation happens usually during REM sleep.

### Memory Architecture

Three tiers of memory, mirroring human cognition:

| Tier | Scope | Storage | TTL |
|---|---|---|---|
| Working memory | Current session | In-memory (Python list/dict) | Session lifetime |
| Short-term memory | Recent sessions | `memories/short_term/YYYY-MM-DD.json` (one file per day) | 7 days (files deleted after expiry) |
| Long-term memory | Permanent | `memories/long_term/` (structured files or vector DB) | Indefinite |

---

### Working Memory

- **Token cap**: `min(MAX_TOKEN, 0.4 * context_window)` — adapts to different models while providing a user-configurable hard ceiling to control cost.
- **Overflow handling (compaction)**: When the cap is exceeded, the oldest entries are summarized via LLM. The summary is written to today's short-term file and the in-memory buffer is trimmed to make room.
- **Structure**: Ordered list of turns `{role, content, timestamp, emotion, emotion_strength}`. Emotion fields are populated by the classifier (see below) at write time.

---

### Promotion: Working → Short-term

Handled by the **short-term background worker** (runs every 15 minutes when in active session).

**Active session detection**: The worker checks the timestamp of the last user input. If it is within the last 30 minutes, the session is considered active and the worker runs. Otherwise it skips its cycle.

**Delta processing**: The worker tracks a `last_processed` cursor (timestamp). On each run it only inspects working memory entries added after that cursor, then advances the cursor.

**Promotion triggers** — any one of the following qualifies an entry for promotion:

1. **Emotion strength** — the entry's `emotion_strength` score (from the classifier) exceeds a threshold (e.g. > 0.75). Captures strong frustration, excitement, urgency, etc.
2. **Explicit intent** — the entry contains a directive to remember something or permanently change behaviour (detected via intent classifier or keyword heuristic).
3. **Repetition** — cosine similarity between the entry's embedding and embeddings of previous working memory entries exceeds threshold (e.g. ≥ 0.85), and this is the 3rd or more such occurrence. Embedding model: Gemini `text-embedding-004` (or `sentence-transformers` locally). Search scope: rolling window of last 50 turns.
4. **Tool failure** — MAVIS failed to build or call a tool after `MAX_RETRIES` attempts. The failure is stored as a known limitation.

Promoted entries are appended to today's short-term file (`memories/short_term/YYYY-MM-DD.json`).

---

### Promotion: Short-term → Long-term

Handled by the **long-term background worker** (runs every 8 hours regardless of session state).

**Delta processing**: Tracks its own `last_processed` cursor across short-term files. On each run it scans only entries added since the last run.

**Promotion triggers** — any one of the following:

1. **Permanent behaviour change** — user explicitly asks to always or never do something (e.g. "always reply concisely", "never open attachments").
2. **Explicit long-term request** — user explicitly asks MAVIS to remember something for a long time or indefinitely.

Promoted entries are written to the appropriate long-term memory file.

---

### Emotion & Intent Classifier

Both short-term promotion triggers (emotion strength, explicit intent) are detected by a **dual-output classifier** embedded in the main interpreter prompt — no separate API call. The existing Gemini call is extended to return two additional fields:

```json
{
  "pipeline": [...],
  "missing_commands": [...],
  "emotion": "frustration",
  "emotion_strength": 0.82,
  "intent_strength": 0.6
}
```

- `emotion`: categorical label (frustration, excitement, urgency, neutral, etc.)
- `emotion_strength`: float 0–1, strength of that emotion
- `intent_strength`: float 0–1, strength of the directive regardless of emotional tone — captures neutral but high-intent commands like "never show me ads again"

Promotion fires if `emotion_strength > 0.75` **or** `intent_strength > 0.85`.

---

### Retrieval

On every command, before calling the LLM, context is assembled as:

```
[Long-term memories (top-K relevant)]
[Short-term memories — last 7 days, top-K relevant]
[Working memory — full, up to token cap]
[Current user input]
```

Retrieval for both short-term and long-term uses **ChromaDB** (vector similarity search). The current user input is embedded at query time and the top-K most similar memories are retrieved from each tier's collection. Top-K = 5 per tier (tunable).

Short-term retrieval is scoped to the last 7 days using ChromaDB metadata filtering on the `date` field. Entries older than 7 days are deleted from the collection and their JSON files are archived/removed.

---

### File Layout

```
memories/
├── short_term/
│   ├── chroma/              ← ChromaDB collection (embeddings + metadata)
│   └── json/
│       ├── 2026-08-20.json  ← deleted after 7 days (human-readable reference)
│       ├── 2026-08-25.json
│       └── 2026-08-26.json  ← today
└── long_term/
    ├── chroma/              ← ChromaDB collection (embeddings + metadata)
    └── json/
        ├── behaviours.json  ← permanent behaviour rules
        └── facts.json       ← things explicitly remembered long-term
```

Every write goes to **both** the ChromaDB collection (for retrieval) and the corresponding JSON file (human-readable reference and recovery). ChromaDB is the source of truth for retrieval; JSON is the source of truth for inspection and debugging.

---

- ~~Vector DB vs. flat JSON~~ → **Resolved**: ChromaDB for both tiers; JSON written in parallel as human-readable reference and recovery backup.
- ~~Compaction latency~~ → **Resolved**: inline compaction is an acceptable tradeoff. Compaction should be rare in practice given the large token cap.
- ~~Cross-session working memory persistence~~ → **Resolved**: not needed. Working memory is session-only; short-term memory captures what matters.

---

## Scoped Subagent Memory

### Motivation

The three core subagents — **Interpreter**, **ToolBuilder**, and **Debugger** — have fundamentally different memory needs. A single global `MemoryStore` forces all retrieval context through one index, meaning ToolBuilder queries pollute conversation history and vice versa. Each subagent should retrieve only what is relevant to its job.

---

### Subagent Memory Profiles

| Subagent | Working Memory | Short-term | Long-term |
|---|---|---|---|
| **Interpreter** | Yes — full conversation turns, emotion-tagged, token-capped | Yes — high-signal turns promoted by background worker | Yes — behaviours & facts |
| **ToolBuilder** | No — stateless per invocation | No | Yes — successful build patterns, file/import conventions |
| **Debugger** | No — stateless per invocation | No | Yes — failure→fix pairs learned from debug loops |

ToolBuilder and Debugger are stateless within a pipeline step; they do not benefit from session-scoped working memory. They only need long-term retrieval at call time to improve generation quality.

---

### Namespace Architecture

`MemoryStore` gains a `namespace` parameter. Each namespace is fully isolated on disk — its own ChromaDB collection and JSON directory. The existing flat `memories/short_term/` and `memories/long_term/` paths become `memories/interpreter/short_term/` and `memories/interpreter/long_term/`.

```
memories/
├── interpreter/
│   ├── short_term/
│   │   ├── chroma/          ← ChromaDB collection
│   │   └── json/
│   │       ├── 2026-08-26.json
│   │       └── ...
│   └── long_term/
│       ├── chroma/
│       └── json/
│           ├── behaviours.json
│           └── facts.json
├── toolbuilder/
│   └── long_term/
│       ├── chroma/
│       └── json/
│           └── patterns.json   ← successful build conventions
└── debugger/
    └── long_term/
        ├── chroma/
        └── json/
            └── fixes.json      ← failure→fix pairs
```

---

### Cross-Read, Own-Write Policy

ToolBuilder and Debugger are peers — each owns its own long-term store but may retrieve from the other's at query time. The Interpreter remains fully isolated.

| Subagent | Writes to | Reads from (at query time) |
|---|---|---|
| Interpreter | `interpreter/` | `interpreter/long_term` + `interpreter/short_term` |
| ToolBuilder | `toolbuilder/long_term` | `toolbuilder/long_term` + `debugger/long_term` |
| Debugger | `debugger/long_term` | `debugger/long_term` + `toolbuilder/long_term` |

**Rationale for cross-reading:**

- *ToolBuilder reads Debugger long-term*: Debugger's `fixes.json` records failure→fix pairs (e.g. "replaced `import os` with `from oni import call_fs`"). At generation time ToolBuilder retrieves relevant past failures and pre-emptively avoids those patterns, reducing the number of tools that enter the debug loop. This creates a compounding feedback loop — the longer MAVIS runs, the fewer initial build failures occur.

- *Debugger reads ToolBuilder long-term*: ToolBuilder's `patterns.json` records established conventions (e.g. "tools that read `user_profile.json` open it relative to `_MAV_ROOT`"). When the Debugger writes a fix, retrieving these patterns produces fixes that are consistent with the broader codebase, not just technically correct.

---

### Long-term Promotion Logic (ToolBuilder & Debugger)

Neither ToolBuilder nor Debugger has emotion signals. Promotion uses operational triggers instead:

**ToolBuilder → `patterns.json`**

Promoted when a tool passes its test on the **first attempt** (no debug loop needed). These are clean, idiomatic builds worth learning from.

Entry format:
```json
{
  "id": "...",
  "date": "2026-08-30",
  "signature": "get_user_name() -> tuple[int, str]",
  "summary": "Reads from user_profile.json relative to _MAV_ROOT; returns (0, name) or (-1, error).",
  "conventions": ["_MAV_ROOT-relative paths", "ONI call_fs for file read"]
}
```

**Debugger → `fixes.json`**

Promoted when a debug loop **succeeds** (broken → fixed after ≥1 retry). The failure+fix pair is stored so the same class of error can be avoided or fixed faster in future.

Entry format:
```json
{
  "id": "...",
  "date": "2026-08-30",
  "error_signature": "AST scan: forbidden import 'os'",
  "fix_summary": "Replaced os.path calls with oni.call_fs('read', path) pattern.",
  "attempts": 1
}
```

Entries are embedded and upserted into the namespace's ChromaDB collection so retrieval is semantic (not exact-match on error string).

---

### `MemoryStore` API Changes

```python
# Construction — namespace selects the storage root
interpreter_mem = MemoryStore(client, namespace="interpreter")
toolbuilder_mem = MemoryStore(client, namespace="toolbuilder")
debugger_mem    = MemoryStore(client, namespace="debugger")

# Retrieval — cross-namespace peers passed as extra read sources
context = toolbuilder_mem.retrieve_context(
    query,
    extra_read_namespaces=["debugger"],   # read debugger LT at query time
)

# Write — always own namespace only (no extra_write_namespaces)
toolbuilder_mem.write_long_term(pattern_entry, ltype="pattern")
debugger_mem.write_long_term(fix_entry, ltype="fix")
```

`extra_read_namespaces` causes `retrieve_context` to query those namespaces' long-term ChromaDB collections and merge results (top-K per source, then union). Short-term is never cross-queried.

---

### Migration Plan

1. **Rename existing paths** on disk:
   - `memories/short_term/` → `memories/interpreter/short_term/`
   - `memories/long_term/` → `memories/interpreter/long_term/`
   - `memories/working_memory.json` → `memories/interpreter/working_memory.json`
2. **Update path constants** in `memory_store.py` to derive from `namespace`.
3. **Update `main.py`** — pass `namespace="interpreter"` when constructing `MemoryStore`.
4. **Instantiate ToolBuilder and Debugger stores** in `main.py` and pass them into `ToolBuilder.__init__`.
5. **Wire retrieval** in `ToolBuilder.build_tool()` — call `toolbuilder_mem.retrieve_context()` with `extra_read_namespaces=["debugger"]` before the build prompt.
6. **Wire promotion** — after a clean first-pass build, call `toolbuilder_mem.write_long_term(pattern_entry)`. After a successful debug loop, call `debugger_mem.write_long_term(fix_entry)`.
7. **Update `reset_chroma.sh`** to recreate all three namespace collections.

---

- ~~Scoped subagent memory~~ → **Resolved**: namespaced `MemoryStore` instances; cross-read own-write between ToolBuilder and Debugger; Interpreter fully isolated.
