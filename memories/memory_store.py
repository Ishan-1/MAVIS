"""
memories/memory_store.py
Central three-tier memory manager for MAVIS.

Architecture (from FUTURE.md):
  ┌──────────────────────────────────────────────────────────────────┐
  │ Working memory  (in-process Python list, session lifetime)       │
  │   Each turn: {role, content, timestamp, emotion,                 │
  │               emotion_strength, intent_strength, embedding,      │
  │               tool_failure}                                      │
  │   Token cap: min(MAX_TOKEN, 0.4 * CONTEXT_WINDOW)               │
  │   Overflow → compaction (LLM summary) → written to short-term   │
  ├──────────────────────────────────────────────────────────────────┤
  │ Short-term memory (ChromaDB + JSON, 7-day rolling window)        │
  │   memories/short_term/chroma/   ← vector DB                     │
  │   memories/short_term/json/     ← YYYY-MM-DD.json               │
  ├──────────────────────────────────────────────────────────────────┤
  │ Long-term memory  (ChromaDB + JSON, permanent)                   │
  │   memories/long_term/chroma/    ← vector DB                     │
  │   memories/long_term/json/behaviours.json                        │
  │   memories/long_term/json/facts.json                             │
  └──────────────────────────────────────────────────────────────────┘

Embedding caching:
  - Each turn dict stores its embedding vector at write time.
  - The short-term worker reuses these cached vectors for repetition
    detection — no re-embedding on each 15-min tick.
  - ChromaDB stores embeddings for short/long-term on disk; retrieval
    queries only embed the incoming query text, never existing memories.

Token counting:
  - Approximated as len(text) // 4.
  - MAX_TOKEN default: 12 000 (overridable via MAX_TOKEN in .env).
  - CONTEXT_WINDOW default: 1 000 000 (Gemini 2.5 Flash).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import chromadb
from google import genai

from core.config import cfg
from core.helpers import log_it
from core.llm import get_llm_client, BaseLLMClient
from memories.embedding import embed, cosine_similarity

_ENTITY = "memory_store"

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
    Thread-safe three-tier memory manager supporting domain namespaces.

    Namespaces isolate memories on disk (e.g. 'interpreter', 'toolbuilder', 'debugger').
    """

    def __init__(self, client: BaseLLMClient | Any | None = None, namespace: str = "interpreter"):
        self.namespace = namespace
        self._client = client or get_llm_client()
        self._lock = threading.Lock()

        # Paths scoped to this namespace
        self._ns_dir = os.path.join(_BASE, namespace)
        self._st_chroma = os.path.join(self._ns_dir, "short_term", "chroma")
        self._st_json = os.path.join(self._ns_dir, "short_term", "json")
        self._lt_chroma = os.path.join(self._ns_dir, "long_term", "chroma")
        self._lt_json = os.path.join(self._ns_dir, "long_term", "json")
        self._st_cursor = os.path.join(self._ns_dir, "short_term", ".cursor")
        self._lt_cursor = os.path.join(self._ns_dir, "long_term", ".cursor")
        self._wm_json = os.path.join(self._ns_dir, "working_memory.json")

        for d in (self._st_chroma, self._st_json, self._lt_chroma, self._lt_json):
            os.makedirs(d, exist_ok=True)

        # Working memory: list of turn dicts
        self._working: list[dict] = []
        self.last_user_input_ts: float = 0.0

        # ChromaDB clients
        self._st_chroma_client = chromadb.PersistentClient(path=self._st_chroma)
        self._lt_chroma_client = chromadb.PersistentClient(path=self._lt_chroma)

        # Collections scoped per namespace
        self._st_col = self._st_chroma_client.get_or_create_collection(
            name=f"{namespace}_short_term",
            metadata={"hnsw:space": "cosine"},
        )
        self._lt_col = self._lt_chroma_client.get_or_create_collection(
            name=f"{namespace}_long_term",
            metadata={"hnsw:space": "cosine"},
        )

        # Ensure default long-term JSON files exist for this namespace
        default_files = {
            "interpreter": ("behaviours.json", "facts.json"),
            "toolbuilder": ("patterns.json",),
            "debugger": ("fixes.json",),
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
            log_it(f"Failed to persist working memory: {e}", _ENTITY)

    def reload_working_memory(self):
        """Reload working memory and last_user_input_ts from disk (used by worker subprocesses)."""
        if not os.path.exists(self._wm_json):
            return
        try:
            with open(self._wm_json, "r") as f:
                payload = json.load(f)
            with self._lock:
                self.last_user_input_ts = payload.get("last_user_input_ts", self.last_user_input_ts)
                self._working = payload.get("working", self._working)
        except Exception as e:
            log_it(f"Failed to reload working memory: {e}", _ENTITY)

    def add_turn(
        self,
        role: str,
        content: str,
        emotion: str = "neutral",
        emotion_strength: str = "low",
        directive: bool = False,
        tool_failure: bool = False,
        intent_strength: float | None = None,
    ):
        """
        Append a turn to working memory and check the token cap.

        The embedding vector is computed HERE (once) and stored on the turn
        dict so the short-term worker can reuse it for repetition detection
        without any extra API calls.
        """
        embedding = embed(content, self._client)

        resolved_directive = (
            directive if intent_strength is None else (directive or intent_strength > 0.85)
        )

        turn = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "emotion": emotion,
            "emotion_strength": emotion_strength,
            "directive": resolved_directive,
            "tool_failure": tool_failure,
            "embedding": embedding,  # cached here — never re-embedded
        }

        with self._lock:
            self._working.append(turn)
            if role == "user":
                self.last_user_input_ts = turn["timestamp"]

            self._maybe_compact()
            self._persist_working_memory_unlocked()

        log_it(f"add_turn ({self.namespace}): role={role!r} tokens≈{_token_count(content)}", _ENTITY)

    def get_working_memory(self) -> list[dict]:
        """Return a shallow copy of the working memory list (thread-safe)."""
        with self._lock:
            return list(self._working)

    def get_working_memory_since(self, since_ts: float) -> list[dict]:
        """Return working memory entries added after *since_ts*."""
        with self._lock:
            return [t for t in self._working if t["timestamp"] > since_ts]

    # ── Public: retrieval ────────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        extra_namespaces: list[str] | None = None,
        top_k: int | None = None,
    ) -> str:
        """
        Build the retrieval-augmented context string for the LLM prompt.

        Supports cross-namespace subscriptions via *extra_namespaces*
        (e.g., toolbuilder querying debugger fix memories).
        """
        k = top_k if top_k is not None else _top_k()
        query_vec = embed(query, self._client)
        sections: list[str] = []

        max_chars = _max_memory_entry_chars()
        def _truncate(text: str) -> str:
            return text[:max_chars] + "... [truncated]" if len(text) > max_chars else text

        # Long-term top-K for this namespace
        lt_entries = [_truncate(e) for e in self._query_chroma(self._lt_col, query_vec, k)]
        if lt_entries:
            sections.append(f"### Long-term memories ({self.namespace})")
            sections.extend(lt_entries)

        # Cross-namespace peer memories
        if extra_namespaces:
            for extra_ns in extra_namespaces:
                try:
                    peer_lt_path = os.path.join(_BASE, extra_ns, "long_term", "chroma")
                    if os.path.exists(peer_lt_path):
                        peer_chroma = chromadb.PersistentClient(path=peer_lt_path)
                        peer_col = peer_chroma.get_or_create_collection(
                            name=f"{extra_ns}_long_term",
                            metadata={"hnsw:space": "cosine"},
                        )
                        peer_entries = [_truncate(e) for e in self._query_chroma(peer_col, query_vec, k)]
                        if peer_entries:
                            sections.append(f"### Reference memories ({extra_ns})")
                            sections.extend(peer_entries)
                except Exception as exc:
                    log_it(f"Peer memory query ({extra_ns}) failed: {exc}", _ENTITY)

        # Short-term top-K (scoped to rolling TTL window)
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=_st_ttl_days())).strftime(
            "%Y-%m-%d"
        )
        st_entries = [
            _truncate(e)
            for e in self._query_chroma(
                self._st_col, query_vec, k, where={"date": {"$gte": cutoff}}
            )
        ]
        if st_entries:
            sections.append(f"### Short-term memories ({self.namespace}, last {_st_ttl_days()} days)")
            sections.extend(st_entries)

        # Working memory: bounded active turns + session summaries
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
            f"retrieve_context ({self.namespace}): lt={len(lt_entries)} st={len(st_entries)} "
            f"wm={len(wm_lines)} turns",
            _ENTITY,
        )
        return context

    # ── Public: write to persistent tiers ────────────────────────────────────

    def write_short_term(self, entry: dict):
        """
        Write *entry* to today's short-term JSON file and ChromaDB collection.
        *entry* must have at least: id, content, timestamp.
        """
        today = _today()
        entry.setdefault("date", today)

        # JSON (human-readable reference)
        json_path = os.path.join(self._st_json, f"{today}.json")
        self._append_to_json(json_path, entry)

        # ChromaDB — embed if no vector present
        vector = entry.get("embedding") or embed(entry["content"], self._client)
        self._st_col.upsert(
            ids=[entry["id"]],
            embeddings=[vector],
            documents=[entry["content"]],
            metadatas=[{"date": today, "role": entry.get("role", "system")}],
        )
        log_it(f"write_short_term ({self.namespace}): id={entry['id']!r} date={today}", _ENTITY)

    def write_long_term(self, entry: dict, ltype: str):
        """
        Write *entry* to long-term storage.
        Supported ltype: 'behaviour', 'fact', 'pattern', 'fix', 'event'.
        """
        entry.setdefault("id", str(uuid.uuid4()))
        entry.setdefault("date", _today())
        entry["ltype"] = ltype

        # JSON file naming
        if ltype in ("behaviour", "fact", "pattern", "event"):
            fname = f"{ltype}s.json"
        elif ltype == "fix":
            fname = "fixes.json"
        else:
            fname = f"{ltype}.json"

        json_path = os.path.join(self._lt_json, fname)
        self._append_to_json(json_path, entry)

        # ChromaDB
        vector = entry.get("embedding") or embed(entry["content"], self._client)
        self._lt_col.upsert(
            ids=[entry["id"]],
            embeddings=[vector],
            documents=[entry["content"]],
            metadatas=[{"type": ltype, "date": entry["date"]}],
        )
        log_it(f"write_long_term ({self.namespace}): id={entry['id']!r} type={ltype!r}", _ENTITY)

    def write_pattern(self, tool_name: str, signature: str, summary: str):
        """Helper to write an established tool building pattern."""
        content = f"Tool Pattern [{tool_name}]: {signature}\nConvention / Summary: {summary}"
        self.write_long_term({"content": content, "tool": tool_name}, ltype="pattern")

    def write_fix(self, tool_name: str, error_snippet: str, fix_summary: str):
        """Helper to write a tool debugging fix pair."""
        content = f"Tool Debug Fix [{tool_name}]: Error: {error_snippet}\nFix: {fix_summary}"
        self.write_long_term({"content": content, "tool": tool_name}, ltype="fix")

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

        # Build a compact text block for the LLM to summarise
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
        # Write summary to short-term (releases lock is fine — write_short_term
        # is called outside the lock path but we're already under it here, so
        # call the internal method directly)
        self._write_short_term_unlocked(summary_entry)

    def _write_short_term_unlocked(self, entry: dict):
        """Same as write_short_term but assumes the caller holds _lock."""
        today = _today()
        entry.setdefault("date", today)
        json_path = os.path.join(self._st_json, f"{today}.json")
        self._append_to_json(json_path, entry)
        vector = embed(entry["content"], self._client)
        self._st_col.upsert(
            ids=[entry["id"]],
            embeddings=[vector],
            documents=[entry["content"]],
            metadatas=[{"date": today, "role": entry.get("role", "system")}],
        )

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

        # Strip the embedding vector from JSON — it's large and stored in Chroma
        serialisable = {k: v for k, v in entry.items() if k != "embedding"}
        data.append(serialisable)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _query_chroma(
        collection,
        query_vec: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[str]:
        """Query a ChromaDB collection and return the document strings."""
        try:
            kwargs: dict = {
                "query_embeddings": [query_vec],
                "n_results": top_k,
                "include": ["documents"],
            }
            if where:
                kwargs["where"] = where
            results = collection.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            return [d for d in docs if d]
        except Exception as exc:
            log_it(f"ChromaDB query failed: {exc}", _ENTITY)
            return []
