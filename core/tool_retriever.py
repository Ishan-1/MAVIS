"""
core/tool_retriever.py
Dynamic tool categorization and semantic retrieval for MAVIS.
Classifies tools into three discrete classes:
- "generalizable": universal primitives (always included in prompt)
- "repurposable": domain-adaptable tools (retrieved via semantic search)
- "specialized": single-purpose, bespoke tools (retrieved via semantic search)
"""
from __future__ import annotations

import os
import chromadb
from chromadb.config import Settings
from core.config import cfg
from core.helpers import log_it
from core.llm import get_llm_client, BaseLLMClient
from memories.embedding import embed

_ENTITY = "tool_retriever"
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_PATH = os.path.join(_BASE_DIR, "data", "tools_chroma")

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
    Indexes available tools into ChromaDB with discrete generalizability classes.
    Supplies ALL 'generalizable' tools + top-K domain tools ('repurposable' + 'specialized').
    """

    def __init__(self, client: BaseLLMClient | None = None, chroma_path: str = _CHROMA_PATH):
        self._client = client or get_llm_client()
        os.makedirs(chroma_path, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._chroma.get_or_create_collection(
            name="tools_registry",
            metadata={"hnsw:space": "cosine"},
        )
        self._generalizability_cache: dict[str, str] = dict(_DEFAULT_GENERALIZABILITY)
        self._load_cached_metadata()

    def _tool_id(self, key: str) -> str:
        return key.split("(")[0].strip()

    def _load_cached_metadata(self):
        """Pre-populate in-memory generalizability scores from Chroma collection."""
        try:
            items = self._col.get(include=["metadatas"])
            if items and items.get("metadatas"):
                for meta in items["metadatas"]:
                    if meta and "func_name" in meta and "generalizability" in meta:
                        self._generalizability_cache[meta["func_name"]] = normalize_generalizability_class(
                            meta["generalizability"]
                        )
        except Exception as exc:
            log_it(f"Failed loading tool metadata from Chroma: {exc}", _ENTITY)

    def sync_tools(self, commands_dict: dict[str, any]):
        """Index any missing tools into the Chroma collection."""
        if not commands_dict:
            return

        existing_ids = set(self._col.get(include=[])["ids"])
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
            self._col.upsert(
                ids=[tool_id],
                embeddings=[vector],
                documents=[doc_text],
                metadatas=[{
                    "func_name": tool_id,
                    "key": key,
                    "description": desc_text,
                    "generalizability": gen_class,
                }],
            )
            log_it(
                f"Indexed tool '{tool_id}' (class={gen_class})",
                _ENTITY,
            )
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

        # Sync before querying to ensure all tools exist in Chroma
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

        # If there are no domain tools or few tools, return general + domain
        if not domain_tools:
            return general_tools

        try:
            query_vec = embed(query, self._client)
            results = self._col.query(
                query_embeddings=[query_vec],
                n_results=min(k + len(general_tools), len(commands_dict)),
                include=["metadatas"],
            )

            matched_domain_keys: list[str] = []
            if results and results.get("metadatas"):
                for meta_list in results["metadatas"]:
                    for meta in meta_list:
                        if meta and "key" in meta:
                            k_name = meta["key"]
                            if k_name in domain_tools and k_name not in matched_domain_keys:
                                matched_domain_keys.append(k_name)
                                if len(matched_domain_keys) >= k:
                                    break

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
