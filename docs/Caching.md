# MAVIS Caching Strategies

## Objectives
- Cache reusable DAG pipelines across similar user tasks.
- Cache results of previous tool executions to avoid redundant computations.
- Cache direct conversational and conceptual answers (Direct Response Caching) to bypass DAG planning and Answerer synthesis.
- Intercept and pre-filter context-dependent queries to prevent false-positive semantic collisions.

---

## Basic Structure
- Caching is managed by `core/caching.py` (`CacheManager`), backed by a lightweight, fast **SQLite** store (`data/pipeline_cache.db`).
- Embeddings are computed via `memories.embedding.embed()` and queried using in-memory cosine similarity for sub-millisecond retrieval.
- Database Schema (`pipeline_cache`):
  - `id` (TEXT PRIMARY KEY): Unique hash identifier.
  - `query` (TEXT): Normalized original user query text.
  - `pipeline` (TEXT): Serialized JSON array of DAG steps (or empty string for direct responses).
  - `result` (TEXT): Serialized JSON execution output or direct answer.
  - `embedding` (TEXT): JSON vector embedding.
  - `ttl_timestamp` (INTEGER): Expiration epoch in seconds.
  - `generalizability` (TEXT): `"generalized"` or `"specialized"`.
  - `inserted_at` (INTEGER): Inserter epoch in seconds.

---

## Context-Dependent Query Pre-Filtering (`core/caching_filter.py`)

Semantic vector similarity cannot reliably distinguish whether a short, deictic query (such as *"Do that"*, *"Repeat this"*, *"Why?"*, or *"Run the second one"*) refers to the current turn or an earlier session. Treating such queries as isolated cache keys causes catastrophic false-positive cache hits.

To prevent this:
1. **Heuristic & Regex Pre-Filter**: Before computing embeddings or querying SQLite, `check_cache()` invokes `is_context_dependent_query(query)`:
   - Intercepts anaphora and deictic references (*"it"*, *"that"*, *"this"*, *"previous"*, *"the above"*).
   - Intercepts follow-up question starters (*"why"*, *"how come"*, *"what about"*).
   - Intercepts ordinal and relative indicators (*"first one"*, *"second one"*, *"other one"*).
   - Intercepts repetitive instructions (*"repeat"*, *"retry"*, *"try again"*).
2. **Immediate Bypass**: If flagged as context-dependent:
   - The query bypasses the cache entirely (`cache_status: "bypassed"`, `hit_tier: "prefiltered_context"`).
   - Execution passes directly to the Interpreter with full working memory context.
3. **Database Purging**: `CacheManager.purge_context_dependent_entries()` runs automatically on startup if `cfg.caching.get("filter_context_dependent", True)` is active, deleting any legacy ambiguous entries from SQLite.

---

## Direct Response Caching Strategy

Not all user queries require planning a tool execution pipeline. Greetings, definitions, conceptual explanations, or factual summaries synthesized directly by MAVIS can be cached as direct responses:
- **Identification**: When `interpret_command()` produces a `direct_response` and an empty `pipeline` (`is_direct_response = True`), the response is cached with a sensible TTL.
- **Hit Condition**: During cache lookup, if the candidate has an empty pipeline:
  - Similarity must be $\ge 0.95$.
  - The TTL must be strictly valid (`time.time() < ttl_timestamp`).
  - LLM verification tier (0.85–0.95) is bypassed for direct responses to prevent hallucinated answers.
- **Efficiency**: Direct cache hits return instantly with $0$ LLM tokens and sub-millisecond latency, bypassing DAG generation, subprocess execution, and Answerer synthesis.

---

## Pipeline Caching Strategy
- **Generalizability Classification**:
  - The Interpreter classifies pipelines as `"generalized"` (parameterizable templates like *"search news about X"*) or `"specialized"` (specific actions like *"git status in ~/repo"*).
  - Generalized pipelines take priority over specialized ones when evaluating candidates.
- **Parameter Extraction**:
  - When a generalized pipeline matches with similarity $< 1.0$, a lightweight parameter extraction prompt extracts dynamic parameters from the new query and updates the pipeline's `params` fields before execution.

---

## Tiered Retrieval & Decision Model Verification Flow

```
Incoming User Query
         │
         ▼
[core/caching_filter.py] ──(Context-Dependent / Deictic?)──► Bypass Cache (Pass to Planner)
         │ (Clean Query)
         ▼
[Vector Embedding & Cosine Ranking]
         │
         ├─► [Similarity ≥ 0.95] ───────────────────────► Automatic Instant Hit
         │                                                ├─ Direct Response (if valid TTL)
         │                                                └─ Pipeline Execution / Extraction
         │
         ├─► [0.85 ≤ Similarity < 0.95] (Pipeline Only)
         │         │
         │         ▼
         │   [Decision Model (Noul)] ──────────────────► Verified?
         │   - dm.evaluate_noul(state, statement)         ├─ Yes ──► Pipeline Execution / Extraction
         │                                                └─ No  ──► Cache Miss
         │
         └─► [Similarity < 0.85] ───────────────────────► Automatic Cache Miss
```

1. **Similarity $\ge 0.95$ (Instant Hit)**:
   - If direct response: returns answer immediately if TTL is valid.
   - If pipeline: executes cached pipeline (or extracts parameters if generalized). If the execution output TTL is still valid, serves the cached result without re-executing.
2. **$0.85 \le \text{Similarity} < 0.95$ (Decision Model Verification)**:
   - Instead of noisy free-form prompts, MAVIS routes verification through the typesafe **Decision Model** (`core/decision/`, `get_decision_model()`).
   - `dm.evaluate_noul(state, statement)` evaluates whether the candidate pipeline fundamentally answers and satisfies the user query using discrete truth semantics.
   - If verified, the pipeline is accepted; if rejected, hands off to the Interpreter planner.
3. **Similarity $< 0.85$ (Cache Miss)**:
   - Emits a miss metric and proceeds to normal DAG planning.

---

## Cache Eviction & Lifecycle Management
- **TTL Enforcement**: Time-sensitive queries (e.g. weather, news, git status) receive short TTLs (hours/days), while invariant concepts receive indefinite or multi-week TTLs.
- **LRU Eviction**: When cache storage limits are approached, least-recently-used entries are pruned.
- **Telemetry**: Every cache check logs structured performance metrics to `data/metrics/caching.csv`:
  `timestamp, turn_id, cache_status, hit_tier, similarity_score, llm_verify_result, ttl_valid, latency_ms, tokens_saved_estimate, estimated_tokens_saved`.