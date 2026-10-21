"""
core/reflection/prompts.py
Prompt templates for the MAVIS Autonomous Reflection Architecture:
1. Candidate Pivot Generator
2. Triage Router Gate (low-temperature)
3. Generator (Hypothesis formulation + targeted verification action)
4. Critic (Adversarial challenge + synthesis gate)
5. Insight Consolidator & Concept Extractor
"""
from __future__ import annotations

PIVOT_CANDIDATES_PROMPT = """
You are an ontology and concept-expansion engine for MAVIS, an autonomous personal AI assistant.
The agent has detected an informational saturation or rut around the following seed concept:

SEED CONCEPT:
\"\"\"{seed_concept}\"\"\"

Generate exactly 5 distinct, structurally adjacent concepts, interaction patterns, or operational frontiers.
Explore both:
1. Systems & Tooling: Unix internals, memory optimization, error recovery, API resilience, algorithmic heuristics, execution safety.
2. Assistant Intelligence & Human-AI Interaction: Cognitive ergonomics, progressive disclosure, workflow pacing, task decomposition frameworks (e.g. GTD, Eisenhower), personal knowledge management (PKM), context retention, and proactive vs. reactive intervention boundaries.

Return ONLY a valid JSON array of 5 concept strings:
[
  "Concept 1",
  "Concept 2",
  "Concept 3",
  "Concept 4",
  "Concept 5"
]
""".strip()


TRIAGE_ROUTER_PROMPT = """
You are the strict Utility Triage Gate for MAVIS Autonomous Reflection.
Your goal is to prevent compute waste, trivia chasing, and database pollution while enabling both technical and personal assistant operational growth.

SEED:
\"\"\"{seed_concept}\"\"\"

PROPOSED EXPLORATION TOPIC:
\"\"\"{candidate_topic}\"\"\"

EVALUATION RUBRIC:
Evaluate the proposed topic against the following discrete utility tiers:
- 1.00 (High Utility): Directly yields concrete operational rules, CLI flags, DAG execution heuristics, error recovery procedures, or explicit assistant interaction policies (e.g., communication pacing, memory retention thresholds, confirmation vs. execution boundaries).
- 0.75 (Actionable Pattern): Specific API convention, systems heuristic, or validated human-assistant cognitive pattern (e.g., progressive disclosure, task decomposition model, cognitive load reduction) with clear practical utility.
- 0.40 (Marginal / General Theory): High-level abstract theory, general philosophy, or generic advice with no concrete operational translation.
- 0.00 (Reject / Trivia): Generic trivia, empty self-help platitudes, surface-level definitions, subjective opinion, or common knowledge.

DECISION RULE:
- Select exactly one of the rubric scores: 1.00, 0.75, 0.40, or 0.00.
- Set "pass": true if utility_score >= 0.75, otherwise "pass": false.

Return ONLY a valid JSON object:
{{
  "utility_score": 1.00,
  "category": "rule_formation" | "assistant_heuristic" | "systems_pattern",
  "rationale": "One-line justification citing the specific rubric criteria.",
  "pass": true
}}
""".strip()


GENERATOR_PROMPT = """
You are the Hypothesis Generator in the MAVIS Adversarial Reflection loop.
You explore technical topics and personal assistant behavioral patterns to produce verified, high-utility operational insights.

EXPLORATION TOPIC:
\"\"\"{topic}\"\"\"

ITERATION: {iteration} of {max_iterations}

PREVIOUS CRITIQUE / HISTORY:
{history}

INSTRUCTIONS:
1. Formulate a specific, testable, and falsifiable hypothesis about this topic:
   - For Technical Topics: State a precise technical assertion (e.g., library flag, memory limit, failure recovery condition).
   - For Assistant / Interaction Patterns: State an Operational Interaction Hypothesis linking [Trigger Context] -> [Assistant Heuristic / Policy] -> [Cognitive/Utility Benefit], including its [Boundary Limit or Counter-Condition]. Avoid vague motivational or conversational fluff.
2. Decide whether you need targeted empirical evidence or are ready to conclude:
   - "web_search": if external docs, RFCs, API signatures, empirical HCI research, or validated workflow models are needed.
     query must be laser-focused to verify or falsify a specific variable.
   - "local_read": if internal MAVIS patterns, logs, prompts, or metrics need verification.
     path must be a clean relative path (e.g. "core/answerer.py", "core/dag.py", "logs/pipeline_debugger.log").
   - "conclude": if you have sufficient evidence and are ready for the Critic to make a final ruling.

Return ONLY a valid JSON object:
{{
  "hypothesis": "Concrete, falsifiable assertion with operational impact and boundary conditions",
  "action_type": "web_search" | "local_read" | "conclude",
  "action_payload": "search query string OR relative file path OR proposed final synthesis",
  "reasoning": "Why this specific test or conclusion was chosen"
}}
""".strip()


CRITIC_PROMPT = """
You are the Adversarial Critic in the MAVIS Reflection loop.
Your duty is to relentlessly challenge hypotheses, expose logical fallacies, prevent hallucinations, and reject superficial platitudes.

TOPIC:
\"\"\"{topic}\"\"\"

CURRENT HYPOTHESIS:
\"\"\"{hypothesis}\"\"\"

EVIDENCE RETRIEVED:
\"\"\"{evidence}\"\"\"

ITERATION: {iteration} of {max_iterations}

CRITIQUE CRITERIA:
1. Empirical Support: Is the hypothesis strongly supported by the evidence, or is there confirmation bias?
2. The Anti-Platitude Test: Is this a generic truism/pop-psychology platitude (e.g., "be polite", "prioritize well")? If it cannot be encoded into an explicit prompt rule, DAG gate, or memory policy, REJECT immediately.
3. Boundary Conditions: Does the hypothesis define when this heuristic FAILS or should NOT be applied? (If boundary conditions are missing, demand REFINE).
4. Grounded Actionability: Can MAVIS directly apply this insight to improve tool execution, user interaction pacing, or task planning?

DECISION RULES:
- "ACCEPT": Evidence is solid, boundary conditions are explicit, and insight is actionable. Provide a clear, dense synthesis summary.
- "REFINE": Hypothesis has merit but contains unaddressed edge cases, missing boundaries, or weak grounding. State what must be fixed.
- "REJECT": The hypothesis is fundamentally vague, circular, trivial, or a dead end. Abort exploration.

Return ONLY a valid JSON object:
{{
  "status": "ACCEPT" | "REFINE" | "REJECT",
  "critique": "Sharp adversarial analysis pointing out gaps, boundary omissions, or strengths.",
  "synthesis": "If ACCEPT: 2-3 sentence canonical insight ready for long-term memory. If REFINE or REJECT: null."
}}
""".strip()


CONSOLIDATION_PROMPT = """
You are the Insight Consolidator for MAVIS Autonomous Reflection.
The exploration loop for topic "{topic}" has reached its turn ceiling without an explicit rejection. Substantive empirical evidence was gathered across {iterations} iterations.

EXPLORATION HISTORY AND GATHERED EVIDENCE:
\"\"\"{history}\"\"\"

TASK:
Synthesize the verified, grounded facts gathered across all iterations into a concise, actionable 2-3 sentence insight suitable for long-term memory.
Discard any unverified assertions, hallucinations, platitudes, or disputed claims.

Return ONLY a valid JSON object:
{{
  "synthesis": "2-3 sentence actionable insight based strictly on the gathered evidence.",
  "summary_hypothesis": "The refined hypothesis backed by the evidence."
}}
""".strip()


CONCEPT_EXTRACTION_PROMPT = """
You are a concept extractor for MAVIS Autonomous Reflection.
Given the following excerpt from long-term memory or a seed entry, extract the core technical, interaction, or architectural concept into a concise phrase (3 to 8 words).

INPUT EXCERPT:
\"\"\"{memory_text}\"\"\"

Extract the underlying concept, interaction heuristic, or design principle. Do not include specific variable names, IDs, dates, or error traces.
Examples:
- Input: "Tool Pattern [fetch_web_page]: Uses urllib3 PoolManager with ExponentialBackoffRetry on status codes 429 and 503." -> "HTTP backoff retry strategies with urllib3"
- Input: "Subprocess output offloaded to scratchpad when exceeding 4000 bytes." -> "Large payload offloading in subprocess execution"
- Input: "Multi-step plan presentation collapses intermediate completed nodes unless verbose mode is requested." -> "Progressive disclosure in multi-step task execution"

Return ONLY a valid JSON object:
{{
  "concept": "Extracted Concept Name"
}}
""".strip()
