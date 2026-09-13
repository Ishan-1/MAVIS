"""
core/tool_retriever.py
Dynamic tool categorization and semantic retrieval for MAVIS using SQLite and cosine similarity.
Classifies tools into three discrete classes:
- "generalizable": universal primitives (always included in prompt)
- "repurposable": domain-adaptable tools (retrieved via semantic search)
- "specialized": single-purpose, bespoke tools (retrieved via semantic search)
"""
from __future__ import annotations

import os
import json
import sqlite3
from typing import Any

from core.config import cfg
from core.helpers import log_it
from core.llm import get_llm_client, BaseLLMClient
from memories.embedding import embed, cosine_similarity

_ENTITY = "tool_retriever"
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_PATH = os.path.join(_BASE_DIR, "data", "tools_registry.db")

VALID_GENERALIZABILITY_CLASSES = ("specialized", "repurposable", "generalizable")

# Default generalizability classes for baseline built-in tools
_DEFAULT_GENERALIZABILITY: dict[str, str] = {
    "get_current_datetime": "generalizable",
    "extract_date_from_datetime": "generalizable",
    "parse_natural_date_to_yyyymmdd": "generalizable",
    "read_file_contents": "generalizable",
    "get_user_display_name": "generalizable",
    "set_user_display_name": "generalizable",
    "search_news": "repurposable",
}


def normalize_generalizability_class(val: any) -> str:
    """Map string or legacy float to 'specialized', 'repurposable', or 'generalizable'."""
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in VALID_GENERALIZABILITY_CLASSES:
            return val_clean
        if val_clean in ("general", "core", "utility"):
            return "generalizable"
        if val_clean in ("domain", "adaptable"):
            return "repurposable"
        if val_clean in ("niche", "custom", "specific"):
            return "specialized"
    if isinstance(val, (int, float)):
        if val >= 0.75:
            return "generalizable"
        if val >= 0.40:
            return "repurposable"
        return "specialized"
    return "repurposable"


class ToolRetriever:
    """
    Indexes available tools into SQLite with discrete generalizability classes.
    Supplies ALL 'generalizable' tools + top-K domain tools ('repurposable' + 'specialized').
    """

    def __init__(self, client: BaseLLMClient | None = None, chroma_path: str | None = None, db_path: str | None = None):
        self._client = client or get_llm_client()
        target = db_path or chroma_path or _DEFAULT_DB_PATH
        if os.path.isdir(target) or not target.endswith(".db"):
            os.makedirs(target, exist_ok=True)
            self.db_path = os.path.join(target, "tools_registry.db")
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            self.db_path = target

        self._init_db()
        self._generalizability_cache: dict[str, str] = dict(_DEFAULT_GENERALIZABILITY)
        self._load_cached_metadata()

    def _init_db(self):
        """Create tools_registry table if not exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tools_registry (
                        func_name TEXT PRIMARY KEY,
                        key TEXT NOT NULL,
                        description TEXT,
                        generalizability TEXT NOT NULL,
                        embedding TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as exc:
            log_it(f"Failed to initialize SQLite tool registry: {exc}", _ENTITY)

    def _tool_id(self, key: str) -> str:
        return key.split("(")[0].strip()

    def _load_cached_metadata(self):
        """Pre-populate in-memory generalizability scores from SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT func_name, generalizability FROM tools_registry")
                for func_name, gen_class in cursor.fetchall():
                    self._generalizability_cache[func_name] = normalize_generalizability_class(gen_class)
        except Exception as exc:
            log_it(f"Failed loading tool metadata from SQLite: {exc}", _ENTITY)

    def sync_tools(self, commands_dict: dict[str, any]):
        """Index any missing tools into the SQLite collection."""
        if not commands_dict:
            return

        try:
            existing_ids = set()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT func_name FROM tools_registry")
                existing_ids = {row[0] for row in cursor.fetchall()}

            for key, val in commands_dict.items():
                tool_id = self._tool_id(key)
                if isinstance(val, dict):
                    desc_text = val.get("description", "")
                    gen_class = val.get("generalizability") or self._generalizability_cache.get(tool_id, "repurposable")
                else:
                    desc_text = str(val)
                    gen_class = self._generalizability_cache.get(tool_id, "repurposable")

                if tool_id not in existing_ids:
                    self.index_tool(key, desc_text, generalizability=gen_class)
                else:
                    self._generalizability_cache[tool_id] = normalize_generalizability_class(gen_class)
        except Exception as exc:
            log_it(f"Warning: tool sync failed: {exc}", _ENTITY)

    def index_tool(
        self,
        key: str,
        description: str | dict,
        generalizability: str | float = "repurposable",
    ):
        """Index a tool signature, description, and generalizability class."""
        tool_id = self._tool_id(key)
        if isinstance(description, dict):
            desc_text = description.get("description", "")
            gen_class = normalize_generalizability_class(
                description.get("generalizability", generalizability)
            )
        else:
            desc_text = str(description)
            gen_class = normalize_generalizability_class(generalizability)

        self._generalizability_cache[tool_id] = gen_class
        doc_text = f"Tool: {tool_id}\nSignature: {key}\nDescription: {desc_text}"

        try:
            vector = embed(doc_text, self._client)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tools_registry (
                        func_name, key, description, generalizability, embedding
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    tool_id,
                    key,
                    desc_text,
                    gen_class,
                    json.dumps(vector),
                ))
                conn.commit()

            log_it(f"Indexed tool '{tool_id}' (class={gen_class})", _ENTITY)
        except Exception as exc:
            log_it(f"Failed to index tool '{tool_id}': {exc}", _ENTITY)

    def get_tool_generalizability(self, key: str) -> str:
        """Return the generalizability class for a tool key/name."""
        tool_id = self._tool_id(key)
        return self._generalizability_cache.get(tool_id, "repurposable")

    def get_relevant_tools(
        self,
        query: str,
        commands_dict: dict[str, any],
        specific_top_k: int | None = None,
    ) -> dict[str, any]:
        """
        Return candidate tools for the planner:
        - ALL 'generalizable' tools are guaranteed in full.
        - Top-K domain tools ('repurposable' and 'specialized') retrieved via vector search.
        - If total tools <= threshold (default 8), returns all tools without filtering.
        """
        if not commands_dict:
            return {}

        thresh = cfg.memory.get("tool_retrieval_threshold", 8)
        if len(commands_dict) <= thresh:
            return commands_dict

        # Sync before querying to ensure all tools exist in database
        self.sync_tools(commands_dict)

        k = (
            specific_top_k
            if specific_top_k is not None
            else cfg.memory.get("specific_tools_top_k", 5)
        )

        general_tools: dict[str, any] = {}
        domain_tools: dict[str, any] = {}

        for key, val in commands_dict.items():
            tool_id = self._tool_id(key)
            if isinstance(val, dict) and "generalizability" in val:
                gen_class = normalize_generalizability_class(val["generalizability"])
            else:
                gen_class = self.get_tool_generalizability(tool_id)

            if gen_class == "generalizable":
                general_tools[key] = val
            else:
                domain_tools[key] = val

        # If there are no domain tools, return general
        if not domain_tools:
            return general_tools

        try:
            query_vec = embed(query, self._client)

            # Load domain tool vectors from SQLite
            rows = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT func_name, key, embedding FROM tools_registry")
                rows = cursor.fetchall()

            scored = []
            for func_name, key_name, emb_json in rows:
                if key_name in domain_tools:
                    try:
                        tool_vec = json.loads(emb_json)
                        sim = cosine_similarity(query_vec, tool_vec)
                        scored.append((sim, key_name))
                    except Exception:
                        continue

            scored.sort(key=lambda x: x[0], reverse=True)
            matched_domain_keys = [k_name for _, k_name in scored[:k]]

            combined = dict(general_tools)
            for k_name in matched_domain_keys:
                combined[k_name] = domain_tools[k_name]

            log_it(
                f"Selected {len(combined)}/{len(commands_dict)} tools for query '{query[:35]}': "
                f"{len(general_tools)} generalizable + {len(matched_domain_keys)} domain",
                _ENTITY,
            )
            return combined

        except Exception as exc:
            log_it(f"Semantic tool retrieval failed ({exc}), falling back to full registry", _ENTITY)
            return commands_dict
