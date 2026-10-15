# MAVIS Autonomous Reflection Architecture

## 1. Executive Summary & Rationale

MAVIS is predominantly a reactive system: it builds Directed Acyclic Graphs (DAGs) and executes tool pipelines in response to explicit user input. 

The **Autonomous Reflection Architecture** introduces a self-directed, compute-efficient learning cycle during idle periods—analogous to the human brain's REM sleep and hippocampal-neocortical memory consolidation. It extracts recent memory seeds, mathematically determines whether the agent is in an informational rut, explores adjacent concepts through an adversarial hypothesis-and-critic loop, and stores only actionable, high-delta insights into dedicated reflection memory.

### Core Design Constraints & Guarantees
- **Hallucination / Pollution Mitigation** (not elimination): Gated by a low-temperature triage filter and a strict mathematical semantic delta threshold.
- **Compute Ceiling**: Hard-capped turns, early-exit triage gates, and temperature-controlled exploration to eliminate infinite token burn.
- **Strict ONI Sandboxing**: Background execution runs isolated at `whitelist_only`. It cannot mutate the filesystem outside `memories/reflections/` or execute unapproved shell commands.
- **Immutable Baseline Tools**: Invokes existing search and inspection primitives via `core/run_tool.py` without modifying files in `tools/`.
- **Telemetry Integrity**: Emits structured cycle metrics directly to `data/metrics/reflection.csv`.

---

## 2. Four-Phase Lifecycle

```
                                [ Daily Worker / Idle Trigger / /reflect ]
                                                    │
                                                    ▼
                     ┌────────────────────────────────────────────────────────┐
                     │          Phase 1: Initiation & Topic Selection         │
                     │  - Sample from Seed Pool / Conceptual Long-Term Memory │
                     │  - Extract 3-8 Word Technical Concept                  │
                     │  - Vectorize & Compute Density (D) & Variance (V)      │
                     │  - Rut Check: Direct Focus vs. 5-Candidate Pivot       │
                     │  - Novelty Weighting: W_i = 1 - max(S_i)               │
                     │  - Temperature-Controlled Softmax Sampling P(c_i)      │
                     └──────────────────────┬─────────────────────────────────┘
                                            │ Selected Topic
                                            ▼
                     ┌────────────────────────────────────────────────────────┐
                     │              Phase 2: The Triage Gate                  │
                     │  - Low-temp (0.0) Router Prompt                        │
                     │  - Score utility: Rule, Contradiction, Compression     │
                     │  - Score < 0.70 ──► [ INSTANT ABORT (0 wasted tokens) ]│
                     └──────────────────────┬─────────────────────────────────┘
                                            │ Score ≥ 0.70
                                            ▼
                     ┌────────────────────────────────────────────────────────┐
                     │         Phase 3: Adversarial Exploration Loop          │
                     │  - Generator: Formulate Testable Falsifiable Hypothesis│
                     │  - Dual-Source Verification:                           │
                     │      * Web Search (tools/search_web via run_tool)      │
                     │      * Local Codebase & Logs (read_file_contents)      │
                     │  - Critic: Logic attack, redundancy & coherence check  │
                     │  - State Machine: ACCEPT / REFINE / REJECT (Max 3 turns│
                     └──────────────────────┬─────────────────────────────────┘
                                            │ Validated Insight
                                            ▼
                     ┌────────────────────────────────────────────────────────┐
                     │           Phase 4: Evaluation & Storage                │
                     │  - Compute Semantic Delta: Δ = 1 - max(Similarity)     │
                     │  - Discard if Δ < 0.25 (paraphrase redundancy)         │
                     │  - Append to memories/reflections/journal.json         │
                     │  - Status: "pending_review", verified: false           │
                     │  - Write Telemetry to data/metrics/reflection.csv      │
                     │  - Dashboard Review: Verify & Promote vs. Delete       │
                     └────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Initiation and Topic Selection

### 1. Seed Shard Extraction & Concept Distillation
The cycle begins by drawing a candidate from:
1. **The Dedicated Seed Pool** (`memories/reflections/seeds.json`): Conceptual topics injected by the user via the Dashboard.
2. **Conceptual Long-Term Memory**: Structured `patterns.json`, `facts.json`, and prior verified reflections. Tactical debug traces (`fixes.json`) and personal habits (`behaviours.json`) are strictly excluded.

If no seeds or conceptual memories exist, the cycle halts cleanly with zero token waste (`status: "no_seeds_available"`).

When a candidate is selected, Phase 1 invokes an LLM Concept Extractor (`CONCEPT_EXTRACTION_PROMPT`) to distill the underlying technical, algorithmic, or architectural concept into a sharp 3 to 8 word phrase (e.g., *"Subprocess output offloading in execution"*). This clean concept serves as $X$.

### 2. Quantifying Existing Knowledge
The agent vectorizes $X$ using `memories.embedding.embed(X)` and queries the vector store for the top-$k$ nearest memories $\{m_1, m_2, \dots, m_k\}$ (with $k=10$).

* **Vector Density ($D$)**: Measures how saturated the agent's memory already is around concept $X$:
  $$D = \sum_{i=1}^{k} \mathbf{1}_{\left[\text{cosine\_similarity}(V_X, V_{m_i}) \ge \theta_{\text{dense}}\right]} \quad (\text{default } \theta_{\text{dense}} = 0.75)$$

* **Semantic Variance ($V$)**: Measures the internal diversity or distance among the returned shards:
  $$V = \frac{2}{k(k - 1)} \sum_{1 \le i < j \le k} \left(1 - \text{cosine\_similarity}(V_{m_i}, V_{m_j})\right)$$

### 3. Determine Pivot
* **High Density, Low Variance ($D \ge 5$ and $V < 0.25$)**: The agent is stuck in an informational or behavioral rut regarding $X$. It already possesses homogeneous knowledge on this topic. **Action: Pivot.**
* **Low Density ($D < 3$)**: The agent's knowledge on $X$ is sparse or fragmentary. **Action: Direct Exploration of $X$.**
* **Balanced ($3 \le D < 5$ or $V \ge 0.25$)**: The agent possesses foundational knowledge with healthy variance; it pivots to structurally adjacent frontiers.

### 4. Candidate Generation (When Pivoting)
When pivoting, the core LLM receives $X$ and generates 5 structurally adjacent or analogical concepts $\{c_1, c_2, c_3, c_4, c_5\}$.

### 5. Inverse Weighting & Softmax Sampling
For each candidate $c_i$:
1. Compute the maximum cosine similarity against all existing memories:
   $$S_i = \max_{m \in \text{Memories}} \text{cosine\_similarity}\left(V_{c_i}, V_m\right)$$
2. Convert into a novelty weight:
   $$W_i = 1 - S_i$$
3. Convert weights into a normalized probability distribution using a temperature parameter $T \in [0.2, 1.0]$:
   $$P(c_i) = \frac{\exp(W_i / T)}{\sum_{j=1}^{5} \exp(W_j / T)}$$
4. Sample from $P(c_i)$ to select the winning concept $c^*$.
   - **Low $T$ (e.g. 0.3)**: Exploits strict novelty maximization (chooses the most unexplored adjacent territory).
   - **High $T$ (e.g. 0.8)**: Flattens the distribution for wider, serendipitous discovery.

---

## 4. Phase 2: The Triage Gate

Before executing expensive web searches, code reading, or multi-turn reasoning loops, $c^*$ must clear a strict utility gate.

### Evaluation Criteria
A single low-temperature prompt (`temperature=0.0`) evaluates $(X, c^*)$ against three actionable dimensions:
1. **Rule Formation**: Does exploring this concept likely yield an explicit operational heuristic, rule, or architectural improvement for MAVIS?
2. **Contradiction Resolution**: Does it resolve an ambiguity, contradiction, or known tool/pipeline failure recorded in recent logs?
3. **Conceptual Compression**: Does it unify disparate mechanisms into a simpler, higher-level mental model?

### Hard Kill Gate
The router outputs a JSON payload with an LLM-estimated (uncalibrated) utility score $\in [0.0, 1.0]$:
```json
{
  "utility_score": 0.82,
  "category": "rule_formation",
  "rationale": "Clarifies boundary conditions between ripgrep multiline flags and memory-safe streaming for pipeline log parsers.",
  "pass": true
}
```
* **Threshold**: If `utility_score < 0.70`, the cycle terminates immediately. Zero web searches and zero downstream inference calls are made.

---

## 5. Phase 3: The Active Exploration Loop (Adversarial Generator vs. Critic)

If the topic clears triage, an adversarial state machine runs up to a hard ceiling of `max_iterations = 3`.

```
                    ┌───────────────────────────────┐
                    │  Generator: State Hypothesis  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Targeted Verification Action  │
                    │ - Web Search (tools/search_web│
                    │ - Local Inspection (read_file)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Critic: Attack & Test     │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
             Status == "REFINE"             Status == "ACCEPT"
                    │                       or Status == "REJECT"
                    ▼                               │
         [ Next Iteration Loop ]                    ▼
           (Up to 3 turns max)            [ Exit to Phase 4 ]
```

### 1. The Generator (Hypothesis-Driven Search)
* Formulates a specific, testable hypothesis about $c^*$.
* Can issue targeted verification requests:
  - **External Web Verification**: Calls `tools/search_web.py` via `core/run_tool.py` subprocess execution.
  - **Internal Workspace Verification**: Reads existing logs, metrics, or implementations (e.g., inspecting `logs/pipeline_debugger.log`, `core/dag.py`, or `data/metrics/`).
* Open-ended, conversational browsing is strictly forbidden: queries must be explicitly designed to confirm, falsify, or fill an exact variable in the hypothesis.

### 2. The Critic (Adversarial Review)
* Reviews the generator's hypothesis and returned evidence.
* Aggressively looks for:
  - Logical leaps or confirmation bias.
  - Redundancy with existing MAVIS architecture or existing knowledge.
  - Hallucinated imports or ungrounded assumptions.
* Outputs:
  - `status: "ACCEPT"` — Hypothesis is verified, coherent, and actionable.
  - `status: "REFINE"` — Flaws found; generator must revise hypothesis or retrieve targeted counter-evidence.
  - `status: "REJECT"` — Dead end or fundamentally flawed premise; abort loop.

### 3. Stop Conditions & Ceiling Consolidation
The loop terminates as soon as:
1. **Critic returns `"ACCEPT"`**: Direct success with synthesized insight.
2. **Critic returns `"REJECT"`**: Fundamental dead end or false premise; cycle terminates with zero writes.
3. **`iteration_count >= 3` (Compute Ceiling Reached)**:
   - Rather than dropping the empirical evidence gathered across 3 turns, the engine triggers the **Insight Consolidator** (`CONSOLIDATION_PROMPT`).
   - It extracts verified facts from the turn history into a best-effort synthesis.
   - This consolidated synthesis then proceeds directly to **Phase 4 (Semantic Delta & Rubric Evaluation)**. If it passes the novelty barrier ($\Delta \ge 0.25$), it is committed to long-term memory.

---

## 6. Phase 4: Evaluation and Storage

Validated thoughts that exit the exploration loop undergo a mathematical delta audit before any persistent write occurs.

### 1. Semantic Delta Rubric
Let $V_{\text{insight}}$ be the vector embedding of the synthesized thought:
$$\Delta = 1 - \max_{m \in \text{RefVectors}} \text{cosine\_similarity}(V_{\text{insight}}, V_m)$$
*(where $\text{RefVectors}$ contains up to 10 vector embeddings from the active working memory and recent reflection files closest to the seed concept, rather than the entire historical memory database)*

* **Paraphrase Rejection ($\Delta < 0.25$)**: If the synthesis has high similarity to pre-existing memories, it is discarded as a redundant restatement.
* **Grounding Sanity Check ($\Delta > 0.85$)**: *Not implemented.* (A planned filter to flag insights that completely drift from grounded context).

### 2. Dedicated Storage: `memories/reflections/`
All validated reflection memories are quarantined in their own domain to prevent polluting standard conversational working memory:
- **File System Tier**: `memories/reflections/long_term/json/YYYY-MM-DD.json`
  ```json
  {
    "id": "refl_6f8b2d1c",
    "timestamp": 1728912345.0,
    "seed_concept": "ripgrep regex assertions",
    "explored_topic": "ripgrep_pcre2_lookaround_limits",
    "triage_score": 0.88,
    "semantic_delta": 0.41,
    "content": "When using ripgrep with complex lookaround expressions, pass --pcre2 explicitly to avoid silent fallback failures on large multi-line buffers.",
    "iterations_count": 2,
    "verified": false,
    "status": "pending_review"
  }
  ```
- **Reflection Journal & Consent Sandbox**: Output is appended to `memories/reflections/journal.json` with `verified: false, status: "pending_review"`. Invariants guarantee the DAG planner and interpreter never read unverified entries.
- **Dashboard Review Studio (`scripts/dashboard.py`)**: Users inspect journal entries under the Reflection tab:
  * **Delete**: Permanently expunges unaligned reflections.
  * **Verify & Promote**: Marks the reflection verified and commits it to permanent Long-Term Memory (`ltype="fact"`). This immediately makes the insight available to the active DAG planner and re-enrolls it into the Phase 1 conceptual seed pool.

### 3. Telemetry Logging
Every reflection execution appends a row to `data/metrics/reflection.csv`:
```csv
timestamp,seed_topic,chosen_concept,pivoted,triage_score,turns_used,web_searches,local_reads,semantic_delta,outcome,duration_ms
```

---

## 7. Execution Lifecycle & Deployment

### 1. Daily Background Worker (`tasks/reflection_worker.py`)
- Configured in `core/config.py` under `cfg.scheduler`:
  ```json
  "reflection_worker_interval_minutes": 1440
  ```
- Spawned as an independent subprocess by `main.py`'s `_start_workers()`:
  - Inherits heartbeat monitoring via `data/.mavis_heartbeat`.
  - Automatically terminates if MAVIS shuts down.
  - Isolates execution trust via `set_context_trust("whitelist_only")`.

### 2. Interactive CLI Trigger (`/reflect`)
- Added to `main.py` prompt session:
  - Users can invoke `/reflect` or `/reflect --concept <seed>` directly in the CLI.
  - Streams generator and critic turns in real time with styled terminal spinners, allowing inspection and tuning.

---

## 8. Implementation Plan

| Step | Scope | Target Files |
|---|---|---|
| **Step 1: Core Engine & Math** | Mathematical utilities (density, variance, softmax), prompts (triage, generator, critic), and tool adapter (sandboxed web search & local inspection). | `core/reflection/math_utils.py`<br>`core/reflection/prompts.py`<br>`core/reflection/tools_adapter.py`<br>`core/reflection/engine.py` |
| **Step 2: Storage & Background Worker** | Worker script, scheduler interval wiring, memory namespace setup (`memories/reflections/`), and telemetry writer. | `tasks/reflection_worker.py`<br>`tasks/worker_process.py`<br>`core/config.py`<br>`data/metrics/reflection.csv` |
| **Step 3: CLI Integration & Observability** | Slash command `/reflect` in `main.py` with live progress visualization, and metrics panel integration in `scripts/dashboard.py`. | `main.py`<br>`scripts/dashboard.py` |
