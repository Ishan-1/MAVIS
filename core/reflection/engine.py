"""
core/reflection/engine.py
Orchestrator for the MAVIS Autonomous Reflection Architecture.
Coordinates:
  Phase 1: Initiation & Topic Selection (math-gated exploration vs pivot)
  Phase 2: Triage Gate (low-temperature utility filter)
  Phase 3: Adversarial Exploration Loop (Generator vs Critic + sandboxed tools)
  Phase 4: Semantic Delta Evaluation & Dedicated Reflection Storage
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from core.config import cfg
from core.helpers import log_it
from core.llm.base import BaseLLMClient
from core.metrics import MetricEmitter
from memories.embedding import cosine_similarity, embed
from memories.memory_store import MemoryStore
from core.reflection.math_utils import (
    compute_novelty_weights,
    compute_semantic_variance,
    compute_vector_density,
    evaluate_pivot,
    sample_concept,
    temperature_softmax,
)
from core.reflection.prompts import (
    CRITIC_PROMPT,
    GENERATOR_PROMPT,
    PIVOT_CANDIDATES_PROMPT,
    TRIAGE_ROUTER_PROMPT,
    CONSOLIDATION_PROMPT,
)
from core.reflection.tools_adapter import read_local_context, search_web_sandboxed

_ENTITY = "reflection_engine"
_METRICS = MetricEmitter("reflection")
_MAV_ROOT = Path(__file__).resolve().parent.parent.parent
_REFLECTIONS_DIR = _MAV_ROOT / "memories" / "reflections"


@dataclass
class ReflectionResult:
    id: str
    seed_concept: str
    selected_topic: str
    pivoted: bool
    pivot_reason: str
    triage_score: float
    triage_category: str
    passed_triage: bool
    iterations_used: int
    web_searches_count: int
    local_reads_count: int
    status: str  # "accepted", "triaged_out", "rejected", "max_turns_exceeded", "low_delta", "error"
    hypothesis: Optional[str] = None
    synthesis: Optional[str] = None
    semantic_delta: float = 0.0
    committed: bool = False
    storage_path: Optional[str] = None
    duration_seconds: float = 0.0
    triage_rationale: Optional[str] = None
    rejection_reason: Optional[str] = None
    history: list[dict] = field(default_factory=list)


class ReflectionEngine:
    """
    Autonomous Reflection Engine executing compute-bounded learning cycles
    during idle intervals or upon explicit user invocation.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        store: Optional[MemoryStore] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        reflections_dir: Optional[Path] = None,
    ):
        self.client = client
        self.store = store or MemoryStore(client, namespace="reflections")
        self.progress_callback = progress_callback
        self.reflections_dir = Path(reflections_dir) if reflections_dir else _REFLECTIONS_DIR

    def _notify(self, phase: str, message: str) -> None:
        log_it(f"[{phase}] {message}", _ENTITY)
        if self.progress_callback:
            try:
                self.progress_callback(phase, message)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Initiation and Topic Selection
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_seed(self) -> str:
        """Sample a recent substantive memory turn or fallback to system seed."""
        try:
            turns = self.store.get_working_memory()
            # Prioritize turns with directives, tool failures, or user input
            candidates = [
                t["content"] for t in turns
                if t.get("role") in ("user", "assistant") and len(t.get("content", "")) > 20
            ]
            if candidates:
                # Return most recent substantive turn
                return candidates[-1][:400].strip()
        except Exception as exc:
            log_it(f"Failed to sample turn from working memory: {exc}", _ENTITY)

        # Fallback to recent log or default frontier
        return "MAVIS tool orchestration and DAG execution resilience"

    def _get_reference_embeddings(self, query_vec: list[float], top_k: int = 10) -> list[list[float]]:
        """Retrieve vector embeddings of closest existing memories."""
        vecs: list[list[float]] = []

        # 1. From active working memory
        try:
            for turn in self.store.get_working_memory():
                v = turn.get("embedding")
                if v and isinstance(v, list) and len(v) == len(query_vec):
                    vecs.append(v)
        except Exception:
            pass

        # 2. From short-term/long-term reflection files
        try:
            refl_json_dir = self.reflections_dir / "long_term" / "json"
            if refl_json_dir.exists():
                for fname in sorted(os.listdir(refl_json_dir), reverse=True)[:5]:
                    if not fname.endswith(".json"):
                        continue
                    with open(refl_json_dir / fname, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            v = item.get("embedding")
                            if v and isinstance(v, list) and len(v) == len(query_vec):
                                vecs.append(v)
        except Exception:
            pass

        # Sort by similarity and keep top-K
        if vecs:
            vecs.sort(key=lambda m: cosine_similarity(query_vec, m), reverse=True)
            return vecs[:top_k]

        return []

    def _phase1_select_topic(self, seed: str) -> tuple[str, bool, str, list[list[float]]]:
        """
        Evaluate vector density and semantic variance to decide whether to pivot.
        Returns: (selected_topic, pivoted, reason, reference_memory_vectors)
        """
        self._notify("Phase 1", f"Initiating reflection on seed: '{seed[:60]}...'")
        seed_vec = self.client.embed(seed)
        ref_vecs = self._get_reference_embeddings(seed_vec, top_k=10)

        density = compute_vector_density(seed_vec, ref_vecs, threshold=0.75)
        variance = compute_semantic_variance(ref_vecs)
        pivoted, reason = evaluate_pivot(density, variance)

        self._notify("Phase 1", f"Density={density}, Variance={variance:.3f} -> {reason}")

        if not pivoted:
            return seed, False, reason, ref_vecs

        # Pivot: Generate 5 adjacent candidates
        self._notify("Phase 1", "Generating adjacent candidate concepts...")
        pivot_prompt = PIVOT_CANDIDATES_PROMPT.format(seed_concept=seed)
        raw_candidates = self.client.generate(pivot_prompt, json_mode=True, temperature=0.7)
        clean_json = self.client.clean_json_response(raw_candidates)

        candidates: list[str] = []
        try:
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                candidates = [str(c).strip() for c in parsed if str(c).strip()]
        except Exception as exc:
            log_it(f"Failed to parse pivot candidates: {exc}. Raw: {raw_candidates}", _ENTITY)

        if not candidates:
            # Fallback if generation failed
            return seed, False, "Pivot candidate generation empty; defaulting to seed", ref_vecs

        # Vectorize candidates and compute inverse novelty weights
        cand_vecs = [self.client.embed(c) for c in candidates]
        weights = compute_novelty_weights(cand_vecs, ref_vecs)

        temp = cfg.get("reflection", "softmax_temperature", default=0.7)
        probabilities = temperature_softmax(weights, temperature=temp)
        chosen_topic = sample_concept(candidates, probabilities)

        self._notify("Phase 1", f"Selected exploration topic: '{chosen_topic}'")
        return chosen_topic, True, reason, ref_vecs

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: The Triage Gate
    # ──────────────────────────────────────────────────────────────────────────

    def _phase2_triage(self, seed: str, topic: str) -> tuple[bool, float, str, str]:
        """
        Run low-temperature triage prompt to prevent compute waste.
        Returns: (pass_gate, score, category, rationale)
        """
        self._notify("Phase 2", f"Evaluating utility of topic '{topic}' at Triage Gate...")
        prompt = TRIAGE_ROUTER_PROMPT.format(seed_concept=seed, candidate_topic=topic)
        raw = self.client.generate(prompt, json_mode=True, temperature=0.0)
        clean = self.client.clean_json_response(raw)

        score = 0.0
        category = "unknown"
        rationale = ""
        passed = False

        try:
            res = json.loads(clean)
            score = float(res.get("utility_score", 0.0))
            category = str(res.get("category", "general"))
            rationale = str(res.get("rationale", ""))
            threshold = float(cfg.get("reflection", "triage_threshold", default=0.70))
            passed = score >= threshold
        except Exception as exc:
            log_it(f"Triage JSON parse failed: {exc}. Defaulting to fail.", _ENTITY)
            passed = False

        status_str = "PASSED" if passed else "REJECTED (Circuit breaker triggered)"
        self._notify("Phase 2", f"Triage score: {score:.2f} ({category}) -> {status_str}")
        if rationale:
            log_it(f"[Phase 2] Triage rationale: {rationale}", _ENTITY)
        return passed, score, category, rationale

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: Adversarial Exploration Loop
    # ──────────────────────────────────────────────────────────────────────────

    def _phase3_adversarial_loop(
        self,
        topic: str,
        max_iterations: int = 3,
    ) -> tuple[str, Optional[str], Optional[str], list[dict], int, int]:
        """
        Run hypothesis-driven generator vs critic adversarial loop.
        Returns: (status, hypothesis, synthesis, history, web_count, local_count)
        """
        history: list[dict] = []
        web_count = 0
        local_count = 0
        current_hypothesis: Optional[str] = None
        synthesis: Optional[str] = None
        status = "max_turns_exceeded"

        for it in range(1, max_iterations + 1):
            self._notify("Phase 3", f"Iteration {it}/{max_iterations}: Generator formulating hypothesis...")

            hist_text = json.dumps(history, indent=2) if history else "None (first turn)"
            gen_prompt = GENERATOR_PROMPT.format(
                topic=topic,
                iteration=it,
                max_iterations=max_iterations,
                history=hist_text,
            )
            raw_gen = self.client.generate(gen_prompt, json_mode=True, temperature=0.5)
            gen_data = {}
            try:
                gen_data = json.loads(self.client.clean_json_response(raw_gen))
            except Exception:
                gen_data = {
                    "hypothesis": f"{topic} optimizes system workflows.",
                    "action_type": "conclude",
                    "action_payload": f"{topic} has direct operational utility.",
                }

            current_hypothesis = gen_data.get("hypothesis", "")
            action_type = gen_data.get("action_type", "conclude")
            action_payload = gen_data.get("action_payload", "")
            gen_reasoning = gen_data.get("reasoning", "")

            log_it(f"[Phase 3] [Iteration {it}] Hypothesis: '{current_hypothesis}'", _ENTITY)
            if gen_reasoning:
                log_it(f"[Phase 3] [Iteration {it}] Generator reasoning: {gen_reasoning}", _ENTITY)
            log_it(f"[Phase 3] [Iteration {it}] Action: {action_type} -> '{action_payload}'", _ENTITY)

            # Execute targeted verification action
            evidence = ""
            if action_type == "web_search":
                self._notify("Phase 3", f"Targeted web search: '{action_payload}'")
                s_code, evidence = search_web_sandboxed(action_payload, max_results=3)
                web_count += 1
                log_it(f"[Phase 3] [Iteration {it}] Web search status={s_code}, returned {len(evidence)} chars", _ENTITY)
            elif action_type == "local_read":
                self._notify("Phase 3", f"Local inspection: '{action_payload}'")
                s_code, evidence = read_local_context(action_payload)
                local_count += 1
                log_it(f"[Phase 3] [Iteration {it}] Local read status={s_code}, returned {len(evidence)} chars", _ENTITY)
            else:
                evidence = f"Generator submitted conclusion: {action_payload}"
                log_it(f"[Phase 3] [Iteration {it}] Conclusion payload: {action_payload}", _ENTITY)

            # Adversarial Critic challenge
            self._notify("Phase 3", f"Critic reviewing hypothesis and evidence...")
            critic_prompt = CRITIC_PROMPT.format(
                topic=topic,
                hypothesis=current_hypothesis,
                evidence=evidence[:3000],
                iteration=it,
                max_iterations=max_iterations,
            )
            raw_critic = self.client.generate(critic_prompt, json_mode=True, temperature=0.0)
            critic_data = {}
            try:
                critic_data = json.loads(self.client.clean_json_response(raw_critic))
            except Exception:
                critic_data = {"status": "REFINE", "critique": "Incomplete response", "synthesis": None}

            critic_status = str(critic_data.get("status", "REFINE")).upper()
            critique = str(critic_data.get("critique", ""))
            critic_synthesis = critic_data.get("synthesis")
            if critic_synthesis and str(critic_synthesis).strip():
                synthesis = str(critic_synthesis).strip()

            log_it(f"[Phase 3] [Iteration {it}] Critic verdict: {critic_status}. Full critique: {critique}", _ENTITY)

            history.append({
                "iteration": it,
                "hypothesis": current_hypothesis,
                "action": action_type,
                "payload": action_payload,
                "generator_reasoning": gen_reasoning,
                "evidence_summary": evidence[:500] + ("..." if len(evidence) > 500 else ""),
                "critic_status": critic_status,
                "critique": critique,
            })

            if critic_status == "ACCEPT":
                self._notify("Phase 3", f"Hypothesis accepted by Critic in turn {it}!")
                status = "accepted"
                break
            elif critic_status == "REJECT":
                reject_msg = f"Hypothesis rejected as dead end by Critic: {critique}" if critique else "Hypothesis rejected as dead end by Critic."
                self._notify("Phase 3", reject_msg)
                status = "rejected"
                break
            else:
                self._notify("Phase 3", f"Critic requested refinement: {critique}")

        # If turn limit reached without explicit rejection, consolidate gathered evidence into synthesis
        if status == "max_turns_exceeded" and history:
            self._notify("Phase 3", f"Turn ceiling reached ({max_iterations}). Consolidating gathered evidence into synthesis...")
            try:
                hist_text = json.dumps(history, indent=2)
                cons_prompt = CONSOLIDATION_PROMPT.format(
                    topic=topic,
                    iterations=len(history),
                    history=hist_text[:4000],
                )
                raw_cons = self.client.generate(cons_prompt, json_mode=True, temperature=0.2)
                cons_data = json.loads(self.client.clean_json_response(raw_cons))
                cand_syn = cons_data.get("synthesis")
                if cand_syn and len(str(cand_syn).strip()) > 20:
                    synthesis = str(cand_syn).strip()
                    current_hypothesis = cons_data.get("summary_hypothesis", current_hypothesis)
                    status = "max_turns_consolidated"
                    self._notify("Phase 3", "Successfully consolidated learning at turn ceiling into synthesis.")
            except Exception as exc:
                log_it(f"Turn ceiling consolidation failed: {exc}", _ENTITY)

        return status, current_hypothesis, synthesis, history, web_count, local_count

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Semantic Delta Evaluation and Dedicated Storage
    # ──────────────────────────────────────────────────────────────────────────

    def _phase4_evaluate_and_store(
        self,
        seed: str,
        topic: str,
        hypothesis: Optional[str],
        synthesis: Optional[str],
        ref_vecs: list[list[float]],
        history: list[dict],
    ) -> tuple[float, bool, Optional[str]]:
        """
        Measure mathematical Semantic Delta: Δ = 1 - max(Similarity).
        If Δ >= threshold, persists insight to memories/reflections/long_term/json/
        Returns: (delta, committed, file_path)
        """
        if not synthesis:
            return 0.0, False, None

        self._notify("Phase 4", "Computing Semantic Delta against prior memory...")
        syn_vec = self.client.embed(synthesis)

        max_sim = 0.0
        if ref_vecs:
            for m in ref_vecs:
                if len(m) == len(syn_vec):
                    sim = cosine_similarity(syn_vec, m)
                    if sim > max_sim:
                        max_sim = sim

        delta = max(0.0, 1.0 - max_sim)
        delta_thresh = float(cfg.get("reflection", "novelty_delta_threshold", default=0.25))

        self._notify("Phase 4", f"Semantic Delta = {delta:.3f} (Threshold = {delta_thresh})")
        if delta < delta_thresh:
            self._notify("Phase 4", "Delta below threshold: insight is redundant with prior knowledge. Discarded.")
            return delta, False, None

        # Persist to memories/reflections/long_term/json/
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        refl_dir = self.reflections_dir / "long_term" / "json"
        refl_dir.mkdir(parents=True, exist_ok=True)
        file_path = refl_dir / f"{today}.json"

        record_id = f"refl_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": record_id,
            "timestamp": time.time(),
            "date": today,
            "seed_concept": seed,
            "explored_topic": topic,
            "hypothesis": hypothesis,
            "content": synthesis,
            "semantic_delta": delta,
            "embedding": syn_vec,
            "iterations_count": len(history),
            "history": history,
        }

        existing_entries = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_entries = json.load(f)
                    if not isinstance(existing_entries, list):
                        existing_entries = []
            except Exception:
                existing_entries = []

        existing_entries.append(entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_entries, f, indent=2)

        # Optional Neo4j extraction
        if self.store.kg and self.store.kg.is_available():
            try:
                from memories.knowledge_extractor import extract_facts_from_turn
                facts = extract_facts_from_turn(synthesis, self.client)
                for fact in facts:
                    self.store.add_fact(
                        subject=fact["subject"],
                        predicate=fact["predicate"],
                        obj=fact["object"],
                        topic="reflection.insights",
                        is_functional=bool(fact.get("is_functional", False)),
                    )
            except Exception as exc:
                log_it(f"Neo4j reflection facts extraction failed: {exc}", _ENTITY)

        self._notify("Phase 4", f"Successfully committed reflection insight to {file_path.name}")
        return delta, True, str(file_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Public Execution
    # ──────────────────────────────────────────────────────────────────────────

    def run_cycle(self, seed: Optional[str] = None) -> ReflectionResult:
        """
        Execute one full Autonomous Reflection cycle (Phases 1 through 4).
        """
        start_time = time.time()
        refl_id = f"refl_{uuid.uuid4().hex[:8]}"

        seed_concept = (seed or self._extract_seed()).strip()
        topic = seed_concept
        pivoted = False
        pivot_reason = "Initial"
        ref_vecs: list[list[float]] = []

        try:
            # Phase 1: Initiation & Topic Selection
            topic, pivoted, pivot_reason, ref_vecs = self._phase1_select_topic(seed_concept)

            # Phase 2: Triage Gate
            passed_triage, triage_score, category, rationale = self._phase2_triage(seed_concept, topic)
            if not passed_triage:
                dur = time.time() - start_time
                rejection_msg = f"Triage score ({triage_score:.2f}) below threshold ({cfg.get('reflection', 'triage_threshold', default=0.70)}): {rationale}"
                log_it(f"Reflection cycle triaged out: {rejection_msg}", _ENTITY)
                res = ReflectionResult(
                    id=refl_id,
                    seed_concept=seed_concept,
                    selected_topic=topic,
                    pivoted=pivoted,
                    pivot_reason=pivot_reason,
                    triage_score=triage_score,
                    triage_category=category,
                    passed_triage=False,
                    iterations_used=0,
                    web_searches_count=0,
                    local_reads_count=0,
                    status="triaged_out",
                    duration_seconds=dur,
                    triage_rationale=rationale,
                    rejection_reason=rejection_msg,
                )
                self._emit_metrics(res)
                return res

            # Phase 3: Adversarial Exploration Loop
            max_turns = int(cfg.get("reflection", "max_exploration_turns", default=3))
            status, hypothesis, synthesis, history, web_c, local_c = self._phase3_adversarial_loop(
                topic, max_iterations=max_turns
            )

            if status not in ("accepted", "max_turns_consolidated") or not synthesis:
                dur = time.time() - start_time
                final_critique = history[-1].get("critique", "") if history else ""
                rejection_msg = final_critique if status == "rejected" else f"Exploration ended with status: {status}"
                log_it(f"Reflection cycle finished without commitment (status={status}). Rejection reason: {rejection_msg}", _ENTITY)
                res = ReflectionResult(
                    id=refl_id,
                    seed_concept=seed_concept,
                    selected_topic=topic,
                    pivoted=pivoted,
                    pivot_reason=pivot_reason,
                    triage_score=triage_score,
                    triage_category=category,
                    passed_triage=True,
                    iterations_used=len(history),
                    web_searches_count=web_c,
                    local_reads_count=local_c,
                    status=status,
                    hypothesis=hypothesis,
                    duration_seconds=dur,
                    triage_rationale=rationale,
                    rejection_reason=rejection_msg,
                    history=history,
                )
                self._emit_metrics(res)
                return res

            # Phase 4: Delta Evaluation & Storage
            delta, committed, path = self._phase4_evaluate_and_store(
                seed_concept, topic, hypothesis, synthesis, ref_vecs, history
            )
            final_status = status if committed else "low_delta"
            rejection_reason = None
            if not committed:
                rejection_reason = f"Semantic delta ({delta:.3f}) fell below novelty threshold (redundant with prior memory)."
                log_it(f"Reflection cycle rejected at Phase 4: {rejection_reason}", _ENTITY)
            else:
                log_it(f"Reflection cycle completed successfully and committed to {path} (delta={delta:.3f})", _ENTITY)

            dur = time.time() - start_time
            res = ReflectionResult(
                id=refl_id,
                seed_concept=seed_concept,
                selected_topic=topic,
                pivoted=pivoted,
                pivot_reason=pivot_reason,
                triage_score=triage_score,
                triage_category=category,
                passed_triage=True,
                iterations_used=len(history),
                web_searches_count=web_c,
                local_reads_count=local_c,
                status=final_status,
                hypothesis=hypothesis,
                synthesis=synthesis,
                semantic_delta=delta,
                committed=committed,
                storage_path=path,
                duration_seconds=dur,
                triage_rationale=rationale,
                rejection_reason=rejection_reason,
                history=history,
            )
            self._emit_metrics(res)
            return res

        except Exception as exc:
            log_it(f"Reflection cycle encountered unhandled error: {exc}", _ENTITY)
            dur = time.time() - start_time
            res = ReflectionResult(
                id=refl_id,
                seed_concept=seed_concept,
                selected_topic=topic,
                pivoted=pivoted,
                pivot_reason=pivot_reason,
                triage_score=0.0,
                triage_category="error",
                passed_triage=False,
                iterations_used=0,
                web_searches_count=0,
                local_reads_count=0,
                status="error",
                duration_seconds=dur,
                rejection_reason=str(exc),
            )
            self._emit_metrics(res)
            return res

    def _emit_metrics(self, res: ReflectionResult) -> None:
        try:
            _METRICS.log({
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "reflection_id": res.id,
                "seed_concept": res.seed_concept[:50].replace(",", " "),
                "selected_topic": res.selected_topic[:50].replace(",", " "),
                "pivoted": res.pivoted,
                "triage_score": f"{res.triage_score:.2f}",
                "iterations_used": res.iterations_used,
                "web_searches_count": res.web_searches_count,
                "local_reads_count": res.local_reads_count,
                "semantic_delta": f"{res.semantic_delta:.3f}",
                "status": res.status,
                "committed": res.committed,
                "latency_ms": int(res.duration_seconds * 1000),
            })
        except Exception as exc:
            log_it(f"Failed to emit reflection metrics: {exc}", _ENTITY)
