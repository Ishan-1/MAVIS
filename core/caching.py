"""
core/caching.py
Semantic cache management for MAVIS DAG pipelines and execution results.
"""
from __future__ import annotations

import os
import json
import time
from typing import Any
import chromadb
from chromadb.config import Settings
from core.helpers import log_it
from core.llm import get_llm_client, BaseLLMClient
from memories.embedding import embed
from core.config import cfg

_ENTITY = "caching"
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_PATH = os.path.join(_BASE_DIR, "data", "pipeline_cache_chroma")

class CacheManager:
    """
    Manages semantic caching of DAG pipelines and results using ChromaDB.
    """
    def __init__(self, client: BaseLLMClient | None = None, chroma_path: str = _CHROMA_PATH):
        self._client = client or get_llm_client()
        os.makedirs(chroma_path, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._chroma.get_or_create_collection(
            name="pipeline_cache",
            metadata={"hnsw:space": "cosine"},
        )
    
    def _verify_cache_with_llm(self, new_query: str, cached_query: str, pipeline_str: str, ttl_valid: bool) -> bool:
        """Use LLM to verify if a cache hit between 0.85 and 0.95 is valid."""
        prompt = (
            f"A user asked: '{new_query}'.\n"
            f"We have a cached pipeline originally generated for the query: '{cached_query}'.\n"
            f"The pipeline is:\n{pipeline_str}\n\n"
            "Does the cached pipeline fundamentally answer the new query? "
            "Reply with exactly YES or NO."
        )
        try:
            response = self._client.generate(prompt, json_mode=False, system_instruction="You are a strict caching validator. Reply YES or NO.").strip().upper()
            return response.startswith("YES")
        except Exception as e:
            log_it(f"LLM cache verification failed: {e}", _ENTITY)
            return False

    def _extract_parameters(self, new_query: str, cached_query: str, pipeline: list[dict]) -> list[dict]:
        """Extract parameters from new query and inject them into generalized pipeline."""
        prompt = (
            f"A user asked: '{new_query}'.\n"
            f"We have a generalized pipeline originally generated for: '{cached_query}'.\n"
            f"The original pipeline is:\n{json.dumps(pipeline, indent=2)}\n\n"
            "Extract the parameters from the new query and update the pipeline's 'params' fields where necessary to fit the new query. "
            "Return the full updated pipeline as a JSON array of dicts."
        )
        try:
            response = self._client.generate(prompt, json_mode=True, system_instruction="Return ONLY a JSON array representing the updated pipeline.")
            return json.loads(response)
        except Exception as e:
            log_it(f"Parameter extraction failed: {e}", _ENTITY)
            return pipeline

    def check_cache(self, query: str) -> dict | None:
        """
        Check if the query matches a cached entry.
        Returns dict with 'pipeline' (and optionally 'result') if cache hit, else None.
        """
        try:
            query_vec = embed(query, self._client)
            results = self._col.query(
                query_embeddings=[query_vec],
                n_results=3,
                include=["metadatas", "distances"]
            )
            
            if not results or not results.get("metadatas") or not results["metadatas"][0]:
                return None
                
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            
            for i, distance in enumerate(distances):
                similarity = 1.0 - distance
                meta = metadatas[i]
                cached_query = meta.get("query", "")
                
                pipeline = json.loads(meta.get("pipeline", "[]"))
                result = json.loads(meta.get("result", "null"))
                ttl_timestamp = meta.get("ttl_timestamp", 0)
                generalizability = meta.get("generalizability", "specialized")
                
                ttl_valid = time.time() < ttl_timestamp
                
                if similarity > 0.95:
                    pass # Automatic hit
                elif similarity >= 0.85:
                    if not self._verify_cache_with_llm(query, cached_query, json.dumps(pipeline), ttl_valid):
                        continue # Try next result
                else:
                    return None # Distances are sorted ascending, so if this < 0.85, the rest are too
                
                # Cache Hit Confirmed
                final_pipeline = pipeline
                if generalizability == "generalized" and similarity < 1.0:
                    final_pipeline = self._extract_parameters(query, cached_query, pipeline)
                
                ttl_seconds = max(0, ttl_timestamp - meta.get("inserted_at", int(time.time())))
                
                if ttl_valid and result is not None:
                    log_it(f"Full Cache Hit for '{query}' (similarity: {similarity:.2f})", _ENTITY)
                    return {"pipeline": final_pipeline, "result": result, "ttl_valid": True, "ttl": ttl_seconds, "generalizability": generalizability}
                else:
                    log_it(f"Pipeline Cache Hit for '{query}' (similarity: {similarity:.2f}), result expired", _ENTITY)
                    return {"pipeline": final_pipeline, "result": None, "ttl_valid": False, "ttl": ttl_seconds, "generalizability": generalizability}
                    
            return None
            
        except Exception as e:
            log_it(f"Cache check failed: {e}", _ENTITY)
            return None

    def save_cache(self, query: str, pipeline: list[dict], result: Any, ttl_seconds: int = 300, generalizability: str = "specialized"):
        """Save a pipeline and its execution result to the cache."""
        try:
            doc_id = str(hash(query))
            query_vec = embed(query, self._client)
            ttl_timestamp = int(time.time() + ttl_seconds)
            
            # Serialize result cleanly
            try:
                serialized_result = json.dumps(result)
            except Exception:
                serialized_result = json.dumps(str(result))
                
            self._col.upsert(
                ids=[doc_id],
                embeddings=[query_vec],
                documents=[query],
                metadatas=[{
                    "query": query,
                    "pipeline": json.dumps(pipeline),
                    "result": serialized_result,
                    "ttl_timestamp": ttl_timestamp,
                    "generalizability": generalizability,
                    "inserted_at": int(time.time())
                }],
            )
            log_it(f"Cached pipeline and result for '{query}' (TTL: {ttl_seconds}s, Gen: {generalizability})", _ENTITY)
        except Exception as e:
            log_it(f"Failed to save cache: {e}", _ENTITY)
            
    def evict_expired(self):
        """Remove only the *result* payload from entries where TTL has expired, preserving the pipeline."""
        try:
            current_time = time.time()
            all_data = self._col.get(include=["metadatas"])
            if not all_data or not all_data.get("ids"):
                return
                
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                ttl_timestamp = meta.get("ttl_timestamp", 0)
                result = meta.get("result", "null")
                if current_time > ttl_timestamp and result != "null":
                    # Evict result but keep pipeline
                    meta["result"] = "null"
                    self._col.update(
                        ids=[doc_id],
                        metadatas=[meta]
                    )
            log_it("Expired cache results evicted.", _ENTITY)
        except Exception as e:
            log_it(f"Failed to evict expired cache: {e}", _ENTITY)
            
    def lru_evict(self, limit: int = 1000):
        """Evict oldest full entries if collection size exceeds limit."""
        try:
            count = self._col.count()
            if count <= limit:
                return
                
            all_data = self._col.get(include=["metadatas"])
            items = []
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                items.append((doc_id, meta.get("inserted_at", 0)))
                
            items.sort(key=lambda x: x[1]) # Sort by inserted_at ascending (oldest first)
            
            to_delete = count - limit
            delete_ids = [item[0] for item in items[:to_delete]]
            
            self._col.delete(ids=delete_ids)
            log_it(f"LRU Evicted {len(delete_ids)} oldest cache entries.", _ENTITY)
        except Exception as e:
            log_it(f"Failed LRU eviction: {e}", _ENTITY)

# Global singleton
cache_manager = CacheManager()
