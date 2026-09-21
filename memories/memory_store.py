"""
memories/memory_store.py
Central memory manager for MAVIS with Neo4j Knowledge Graph,
topic subscriptions, and file-backed episodic short-term storage.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from google import genai

from core.config import cfg
from core.helpers import log_it
from core.llm import get_llm_client, BaseLLMClient
from memories.embedding import embed, cosine_similarity
from core.metrics import MetricEmitter

try:
    from memories.neo4j_graph import Neo4jKnowledgeGraph
except ImportError:
    Neo4jKnowledgeGraph = None

_ENTITY = "memory_store"
_METRICS_EMITTER = MetricEmitter("memory")

_DEFAULT_TOPICS = {
    "interpreter": {
        "read": ["user.*", "env.*", "tooling.*", "agents.*", "debugging.*"],
        "write": "user.profile",
    },
    "toolbuilder": {
        "read": ["env.*", "tooling.*", "debugging.*"],
        "write": "tooling.tools",
    },
    "debugger": {
        "read": ["env.*", "tooling.*", "debugging.*"],
        "write": "debugging.fixes",
    },
    "agent_debugger": {
        "read": ["env.*", "agents.*", "debugging.*"],
        "write": "agents.debugging",
    },
    "pipeline_debugger": {
        "read": ["env.*", "tooling.*", "agents.*", "debugging.*"],
        "write": "debugging.pipeline_fixes",
    },
    "tasks": {
        "read": ["env.*", "tooling.*", "tasks.*"],
        "write": "tasks.workflow",
    },
}

# ── Configuration helpers (read live from central cfg) ────────────────────────
def _max_token() -> int:
    return cfg.memory.get("max_token", 12000)

def _context_window() -> int:
    return cfg.memory.get("context_window", 1000000)

def _top_k() -> int:
    return cfg.memory.get("top_k", 5)

def _st_ttl_days() -> int:
    return cfg.memory.get("short_term_ttl_days", 7)

def _working_memory_active_turns() -> int:
    return cfg.memory.get("working_memory_active_turns", 8)

def _compact_token_threshold() -> int:
    return cfg.memory.get("compact_token_threshold", 1500)

def _max_memory_entry_chars() -> int:
    return cfg.memory.get("max_memory_entry_chars", 600)

# ── Path helpers ─────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__))


def _token_count(text: str) -> int:
    return len(text) // 4


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


# ── MemoryStore ──────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Thread-safe memory manager supporting domain namespaces,
    Neo4j Knowledge Graph integration, and file-backed episodic short-term storage.
    """

    def __init__(
        self,
        client: BaseLLMClient | Any | None = None,
        namespace: str = "interpreter",
        read_topics: list[str] | None = None,
        write_topic: str | None = None,
    ):
        self.namespace = namespace
        self._client = client or get_llm_client()
        self._lock = threading.Lock()

        # Topic subscriptions for Knowledge Graph
        topic_cfg = _DEFAULT_TOPICS.get(namespace, {
            "read": ["env.*", "tooling.*"],
            "write": f"{namespace}.general",
        })
        self.read_topics: list[str] = read_topics if read_topics is not None else list(topic_cfg["read"])
        self.write_topic: str = write_topic if write_topic is not None else topic_cfg["write"]

        # Neo4j Knowledge Graph
        self.kg: Any = None
        if Neo4jKnowledgeGraph is not None:
            try:
                self.kg = Neo4jKnowledgeGraph()
                if not self.kg.is_available():
                    self.kg = None
            except Exception as e:
                log_it(f"Neo4jKnowledgeGraph initialization failed: {e}", _ENTITY)
                self.kg = None

        # Paths scoped to this namespace
        self._ns_dir = os.path.join(_BASE, namespace)
        self._st_json = os.path.join(self._ns_dir, "short_term", "json")
        self._lt_json = os.path.join(self._ns_dir, "long_term", "json")
        self._st_cursor = os.path.join(self._ns_dir, "short_term", ".cursor")
        self._lt_cursor = os.path.join(self._ns_dir, "long_term", ".cursor")
        self._wm_json = os.path.join(self._ns_dir, "working_memory.json")

        for d in (self._st_json, self._lt_json):
            os.makedirs(d, exist_ok=True)

        # Working memory: list of turn dicts
        self._working: list[dict] = []
        self.last_user_input_ts: float = 0.0

        # Ensure default long-term JSON files exist for this namespace
        default_files = {
            "interpreter": ("behaviours.json", "facts.json"),
            "toolbuilder": ("patterns.json",),
            "debugger": ("fixes.json",),
            "agent_debugger": ("fixes.json",),
            "pipeline_debugger": ("fixes.json",),
            "tasks": ("events.json",),
        }.get(namespace, ("facts.json",))

        for fname in default_files:
            path = os.path.join(self._lt_json, fname)
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump([], f)

        # Load persisted working memory if present
        self.reload_working_memory()

        log_it(f"MemoryStore initialised (namespace={self.namespace!r}).", _ENTITY)

    # ── Public: working memory ────────────────────────────────────────────────

    def _persist_working_memory_unlocked(self):
        """Atomically persist current working memory to disk for worker subprocesses."""
        try:
            tmp_file = f"{self._wm_json}.tmp"
            payload = {
                "last_user_input_ts": self.last_user_input_ts,
                "working": self._working,
            }
            with open(tmp_file, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_file, self._wm_json)
        except Exception as e:
            log_it(f"Failed to persist working memory to {self._wm_json}: {e}", _ENTITY)

    def reload_working_memory(self):
        """Reload working memory state from disk (safe across subprocesses)."""
        if not os.path.exists(self._wm_json):
            return
        with self._lock:
            try:
                with open(self._wm_json, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.last_user_input_ts = data.get("last_user_input_ts", 0.0)
                        self._working = data.get("working", [])
                    elif isinstance(data, list):
                        self._working = data
                log_it(f"Reloaded {len(self._working)} working memory turns from {self._wm_json}.", _ENTITY)
            except Exception as e:
                log_it(f"Failed to reload working memory from {self._wm_json}: {e}", _ENTITY)

    def add_turn(
        self,
        role: str,
        content: str,
        emotion: str = "neutral",
        emotion_strength: str = "low",
        intent_strength: float = 0.0,
        directive: bool = False,
        tool_failure: bool = False,
    ):
        """
        Record a new conversational turn into working memory.
        Computes and caches the embedding vector immediately on write.
        Triggers compaction if working memory exceeds the token budget.
        """
        now = time.time()
        if role == "user":
            self.last_user_input_ts = now

        # Cache the embedding vector on the turn dict
        vector = embed(content, self._client)

        turn = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": now,
            "emotion": emotion,
            "emotion_strength": emotion_strength,
            "intent_strength": intent_strength,
            "directive": directive,
            "tool_failure": tool_failure,
            "embedding": vector,
        }

        with self._lock:
            self._working.append(turn)
            self._maybe_compact()
            self._persist_working_memory_unlocked()

        _METRICS_EMITTER.log({
            "event_type": "turn_added",
            "working_tokens_count": self._working_tokens(),
            "compaction_triggered": False,
            "tokens_freed": 0,
            "turns_evaluated": 1,
            "turns_promoted": 0,
            "facts_consolidated": 0,
        })

        log_it(f"add_turn ({self.namespace}): role={role!r} tokens≈{_token_count(content)}", _ENTITY)

    def get_working_memory(self) -> list[dict]:
        """Return a shallow copy of the working memory list (thread-safe)."""
        with self._lock:
            return list(self._working)

    def get_working_memory_since(self, since_ts: float) -> list[dict]:
        """Return working memory entries added after *since_ts*."""
        with self._lock:
            return [t for t in self._working if t["timestamp"] > since_ts]

    def clear_working_memory(self):
        """Clear working memory turns in-memory and on disk."""
        with self._lock:
            self._working = []
            self.last_user_input_ts = 0.0
            self._persist_working_memory_unlocked()
            log_it(f"Cleared working memory for namespace {self.namespace!r}", _ENTITY)

    # ── Public: retrieval ────────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        extra_namespaces: list[str] | None = None,
        top_k: int | None = None,
    ) -> str:
        """
        Build the retrieval-augmented context string for the LLM prompt.
        Queries Neo4j for active structured facts, and file-backed tiers for episodic turns.
        """
        k = top_k if top_k is not None else _top_k()
        query_vec = embed(query, self._client)
        sections: list[str] = []

        max_chars = _max_memory_entry_chars()
        def _truncate(text: str) -> str:
            return text[:max_chars] + "... [truncated]" if len(text) > max_chars else text

        # 1. Neo4j Active Knowledge (scoped to self.read_topics)
        kg_entries = []
        if self.kg and self.kg.is_available():
            try:
                facts = self.kg.query_active_facts(
                    query,
                    allowed_topics=self.read_topics,
                    client=self._client,
                    top_k=k,
                )
                if facts:
                    kg_entries = [_truncate(f) for f in facts]
                    sections.append(f"### Active Knowledge ({self.namespace} subscribed)")
                    sections.extend([f"- {f}" for f in kg_entries])
            except Exception as exc:
                log_it(f"Neo4j query_active_facts failed: {exc}", _ENTITY)

        # 2. Long-term memories from JSON files
        lt_entries = [_truncate(e) for e in self._query_json_dir(self._lt_json, query_vec, k)]
        if lt_entries:
            sections.append(f"### Long-term memories ({self.namespace})")
            sections.extend(lt_entries)

        # 3. Cross-namespace peer memories from peer JSON files
        if extra_namespaces:
            for extra_ns in extra_namespaces:
                try:
                    peer_lt_path = os.path.join(_BASE, extra_ns, "long_term", "json")
                    if os.path.exists(peer_lt_path):
                        peer_entries = [_truncate(e) for e in self._query_json_dir(peer_lt_path, query_vec, k)]
                        if peer_entries:
                            sections.append(f"### Reference memories ({extra_ns})")
                            sections.extend(peer_entries)
                except Exception as exc:
                    log_it(f"Peer memory query ({extra_ns}) failed: {exc}", _ENTITY)

        # 4. Short-term memories from rolling TTL window JSON files
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=_st_ttl_days())).strftime("%Y-%m-%d")
        st_entries = [
            _truncate(e)
            for e in self._query_short_term_json(query_vec, k, min_date=cutoff)
        ]
        if st_entries:
            sections.append(f"### Short-term memories ({self.namespace}, last {_st_ttl_days()} days)")
            sections.extend(st_entries)

        # 5. Working memory: bounded active turns + session summaries
        max_active = _working_memory_active_turns()
        with self._lock:
            summaries = [
                f"[summary] {_truncate(t['content'])}"
                for t in self._working
                if t.get("role") == "system"
            ]
            recent_turns = [
                t for t in self._working if t.get("role") != "system"
            ][-max_active:]
            recent_lines = [
                f"[{t['role']}] {_truncate(t['content'])}" for t in recent_turns
            ]
            wm_lines = summaries + recent_lines

        if wm_lines:
            sections.append(f"### Working memory ({self.namespace} session)")
            sections.extend(wm_lines)

        context = "\n".join(sections)
        log_it(
            f"retrieve_context ({self.namespace}): kg={len(kg_entries)} lt={len(lt_entries)} st={len(st_entries)} "
            f"wm={len(wm_lines)} turns",
            _ENTITY,
        )
        return context

    # ── Public: write to persistent tiers ────────────────────────────────────

    def write_short_term(self, entry: dict):
        """
        Write *entry* to today's short-term JSON file.
        *entry* must have at least: id, content, timestamp.
        """
        today = _today()
        entry.setdefault("date", today)
        if "embedding" not in entry:
            entry["embedding"] = embed(entry["content"], self._client)

        json_path = os.path.join(self._st_json, f"{today}.json")
        self._append_to_json(json_path, entry)
        log_it(f"write_short_term ({self.namespace}): id={entry['id']!r} date={today}", _ENTITY)

    def write_long_term(self, entry: dict, ltype: str):
        """
        Write *entry* to long-term storage JSON file.
        Supported ltype: 'behaviour', 'fact', 'pattern', 'fix', 'event'.
        """
        entry.setdefault("id", str(uuid.uuid4()))
        entry.setdefault("date", _today())
        entry["ltype"] = ltype
        if "embedding" not in entry:
            entry["embedding"] = embed(entry["content"], self._client)

        # JSON file naming
        if ltype in ("behaviour", "fact", "pattern", "event"):
            fname = f"{ltype}s.json"
        elif ltype == "fix":
            fname = "fixes.json"
        else:
            fname = f"{ltype}.json"

        json_path = os.path.join(self._lt_json, fname)
        self._append_to_json(json_path, entry)
        log_it(f"write_long_term ({self.namespace}): id={entry['id']!r} type={ltype!r}", _ENTITY)

    def write_pattern(self, tool_name: str, signature: str, summary: str):
        """Helper to write an established tool building pattern."""
        content = f"Tool Pattern [{tool_name}]: {signature}\nConvention / Summary: {summary}"
        self.write_long_term({"content": content, "tool": tool_name}, ltype="pattern")
        if self.kg and self.kg.is_available():
            try:
                self.kg.add_tool_definition(tool_name, signature, summary, client=self._client)
            except Exception as exc:
                log_it(f"Failed to record tool pattern to Neo4j: {exc}", _ENTITY)

    def write_fix(self, tool_name: str, error_snippet: str, fix_summary: str):
        """Helper to write a tool debugging fix pair."""
        content = f"Tool Debug Fix [{tool_name}]: Error: {error_snippet}\nFix: {fix_summary}"
        self.write_long_term({"content": content, "tool": tool_name}, ltype="fix")
        if self.kg and self.kg.is_available():
            try:
                self.kg.add_tool_fix(tool_name, error_snippet, fix_summary, client=self._client)
            except Exception as exc:
                log_it(f"Failed to record tool fix to Neo4j: {exc}", _ENTITY)

    def write_agent_fix(self, agent_name: str, failure_mode: str, fix_summary: str):
        """Helper to write an agent prompt/constraint debugging fix pair."""
        content = f"Agent Debug Fix [{agent_name}]: Failure Mode: {failure_mode}\nPrompt Remedy: {fix_summary}"
        self.write_long_term({"content": content, "agent": agent_name}, ltype="fix")
        if self.kg and self.kg.is_available():
            try:
                self.kg.add_agent_fix(agent_name, failure_mode, fix_summary, client=self._client)
            except Exception as exc:
                log_it(f"Failed to record agent fix to Neo4j: {exc}", _ENTITY)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        topic: str | None = None,
        is_functional: bool = True,
    ):
        """Add fact triple to Knowledge Graph using write_topic by default."""
        target_topic = topic or self.write_topic
        if self.kg and self.kg.is_available():
            try:
                self.kg.add_fact(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    topic=target_topic,
                    is_functional=is_functional,
                    client=self._client,
                )
            except Exception as exc:
                log_it(f"Failed to write fact to Neo4j: {exc}", _ENTITY)

    # ── Cursor helpers for background workers ─────────────────────────────────

    @staticmethod
    def read_cursor(cursor_path: str) -> float:
        """Return the float timestamp stored in *cursor_path*, or 0.0."""
        try:
            with open(cursor_path) as f:
                return float(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0.0

    @staticmethod
    def write_cursor(cursor_path: str, ts: float):
        """Persist *ts* to *cursor_path*."""
        with open(cursor_path, "w") as f:
            f.write(str(ts))

    # ── Properties for workers ────────────────────────────────────────────────

    @property
    def st_cursor_path(self) -> str:
        return self._st_cursor

    @property
    def lt_cursor_path(self) -> str:
        return self._lt_cursor

    @property
    def st_json_dir(self) -> str:
        return self._st_json

    @property
    def lt_json_dir(self) -> str:
        return self._lt_json

    @property
    def wm_json_path(self) -> str:
        return self._wm_json

    # ── Compaction (called under lock) ────────────────────────────────────────

    def _token_budget(self) -> int:
        return int(0.5 * _max_token())

    def _working_tokens(self) -> int:
        return sum(_token_count(t["content"]) for t in self._working)

    def _maybe_compact(self):
        """If working memory exceeds the token budget, compact the oldest half."""
        if self._working_tokens() <= self._token_budget():
            return

        half = len(self._working) // 2
        to_compact = self._working[:half]
        self._working = self._working[half:]

        log_it(
            f"Compaction triggered: summarising {len(to_compact)} turns.", _ENTITY
        )

        block = "\n".join(
            f"[{t['role']}] {t['content']}" for t in to_compact
        )
        summary_prompt = (
            "Summarise the following conversation excerpt concisely (≤120 words), "
            "preserving any important facts, decisions, or preferences:\n\n" + block
        )
        try:
            if hasattr(self._client, "generate"):
                summary_text = self._client.generate(summary_prompt)
            else:
                summary_text = self._client.models.generate_content(
                    model="gemini-2.5-flash", contents=summary_prompt
                ).text.strip()
        except Exception as exc:
            log_it(f"Compaction LLM call failed: {exc}", _ENTITY)
            summary_text = "[compacted session excerpt — LLM summary unavailable]"

        summary_entry = {
            "id": str(uuid.uuid4()),
            "role": "system",
            "content": summary_text,
            "timestamp": time.time(),
            "emotion": "neutral",
            "emotion_strength": "low",
            "directive": False,
            "tool_failure": False,
        }
        self._write_short_term_unlocked(summary_entry)

        tokens_freed = sum(_token_count(t["content"]) for t in to_compact)
        _METRICS_EMITTER.log({
            "event_type": "compaction",
            "working_tokens_count": self._working_tokens(),
            "compaction_triggered": True,
            "tokens_freed": tokens_freed,
            "turns_evaluated": len(to_compact),
            "turns_promoted": 0,
            "facts_consolidated": 0,
        })

    def _write_short_term_unlocked(self, entry: dict):
        """Same as write_short_term but assumes the caller holds _lock."""
        today = _today()
        entry.setdefault("date", today)
        if "embedding" not in entry:
            entry["embedding"] = embed(entry["content"], self._client)
        json_path = os.path.join(self._st_json, f"{today}.json")
        self._append_to_json(json_path, entry)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _append_to_json(path: str, entry: dict):
        """Append *entry* to a JSON array file (creates it if missing)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(entry)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _query_short_term_json(
        self,
        query_vec: list[float],
        top_k: int,
        min_date: str | None = None,
    ) -> list[str]:
        """Scan short-term JSON files and return top-K entries by cosine similarity."""
        try:
            if not os.path.exists(self._st_json):
                return []

            files = sorted(f for f in os.listdir(self._st_json) if f.endswith(".json"))
            candidates = []

            for fname in files:
                date_str = fname.replace(".json", "")
                if min_date and date_str < min_date:
                    continue
                file_path = os.path.join(self._st_json, fname)
                try:
                    with open(file_path, "r") as f:
                        entries = json.load(f)
                        for item in entries:
                            if not isinstance(item, dict) or "content" not in item:
                                continue
                            emb = item.get("embedding")
                            if not emb:
                                emb = embed(item["content"], self._client)
                            sim = cosine_similarity(query_vec, emb)
                            candidates.append((sim, item["content"]))
                except Exception:
                    continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            return [text for _, text in candidates[:top_k]]
        except Exception as exc:
            log_it(f"Query short-term JSON failed: {exc}", _ENTITY)
            return []

    def _query_json_dir(
        self,
        directory: str,
        query_vec: list[float],
        top_k: int,
    ) -> list[str]:
        """Scan directory of JSON files and return top-K entries by cosine similarity."""
        try:
            if not os.path.exists(directory):
                return []

            candidates = []
            for fname in os.listdir(directory):
                if not fname.endswith(".json"):
                    continue
                file_path = os.path.join(directory, fname)
                try:
                    with open(file_path, "r") as f:
                        entries = json.load(f)
                        for item in entries:
                            if not isinstance(item, dict) or "content" not in item:
                                continue
                            emb = item.get("embedding")
                            if not emb:
                                emb = embed(item["content"], self._client)
                            sim = cosine_similarity(query_vec, emb)
                            candidates.append((sim, item["content"]))
                except Exception:
                    continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            return [text for _, text in candidates[:top_k]]
        except Exception as exc:
            log_it(f"Query JSON dir failed: {exc}", _ENTITY)
            return []


def clear_all_working_memories():
    """Scan memories/ and clear working_memory.json across all namespaces."""
    mem_root = "memories"
    if not os.path.exists(mem_root):
        return
    for root, _, files in os.walk(mem_root):
        if "working_memory.json" in files:
            wm_path = os.path.join(root, "working_memory.json")
            try:
                with open(wm_path, "w") as f:
                    json.dump({"last_user_input_ts": 0.0, "working": []}, f)
                log_it(f"Reset working memory file at {wm_path}", _ENTITY)
            except Exception as e:
                log_it(f"Failed to reset {wm_path}: {e}", _ENTITY)
