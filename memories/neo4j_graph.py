"""
memories/neo4j_graph.py
Neo4j Knowledge Graph backend for MAVIS with topic subscriptions,
native vector indexing, and deterministic temporal invalidation.
"""
from __future__ import annotations

import re
from typing import Any
from datetime import datetime, timezone

from core.config import cfg
from core.helpers import log_it

_ENTITY = "neo4j_graph"

try:
    import neo4j
    from neo4j import GraphDatabase, Driver
    _NEO4J_INSTALLED = True
except ImportError:
    neo4j = None
    GraphDatabase = None
    Driver = None
    _NEO4J_INSTALLED = False


def _slugify(text: str) -> str:
    """Normalize text into a valid entity ID slug."""
    clean = re.sub(r"[^a-zA-Z0-9_:\-]", "_", text.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:128]


def _normalize_predicate(pred: str) -> str:
    """Sanitize relationship predicate for Cypher type."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", pred.strip().upper())
    return clean if clean else "RELATED_TO"


class Neo4jKnowledgeGraph:
    """
    Manages structured entity relationships and vector index in Neo4j.
    Enforces topic-based visibility and temporal functional invalidation.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        if not _NEO4J_INSTALLED:
            log_it("neo4j library is not installed.", _ENTITY)
            self._driver: Driver | None = None
            return

        conf = cfg.neo4j
        self.uri = uri or conf.get("uri", "bolt://localhost:7687")
        self.user = user or conf.get("user", "neo4j")
        self.password = password or conf.get("password", "password")
        self.database = database or conf.get("database", "neo4j")
        self.dim = conf.get("vector_dimensions", 768)

        self._driver: Driver | None = None
        self._vector_index_ready: bool = False

        self._connect()
        if self._driver:
            self._init_schema()

    def _connect(self) -> bool:
        """Establish connection to Neo4j."""
        if not _NEO4J_INSTALLED:
            return False
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=300,
            )
            self._driver.verify_connectivity()
            log_it(f"Connected to Neo4j at {self.uri} (database={self.database}).", _ENTITY)
            return True
        except Exception as exc:
            log_it(f"Failed to connect to Neo4j at {self.uri}: {exc}", _ENTITY)
            self._driver = None
            return False

    def is_available(self) -> bool:
        """Check if Neo4j is available and connected."""
        if not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        """Close driver connection."""
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def _init_schema(self):
        """Initialize constraints and vector index in Neo4j."""
        if not self.is_available():
            return
        try:
            with self._driver.session(database=self.database) as session:
                # Unique constraint on Entity id
                session.run(
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
                )
                # Attempt to create vector index (Neo4j 5.11+)
                try:
                    session.run(
                        f"""
                        CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                        FOR (e:Entity) ON (e.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {self.dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """
                    )
                    self._vector_index_ready = True
                    log_it("Neo4j vector index 'entity_embeddings' verified.", _ENTITY)
                except Exception as ve:
                    log_it(f"Vector index creation notice: {ve}", _ENTITY)
                    self._vector_index_ready = False
        except Exception as exc:
            log_it(f"Schema initialization failed: {exc}", _ENTITY)

    def _get_embedding(self, text: str, client: Any = None) -> list[float] | None:
        """Helper to get text embedding vector."""
        try:
            from memories.embedding import embed
            return embed(text, client)
        except Exception as e:
            log_it(f"Embedding failed for '{text[:40]}': {e}", _ENTITY)
            return None

    # ── Node and Relationship Ingestion ────────────────────────────────────────

    def upsert_entity(
        self,
        entity_id: str,
        name: str,
        topic: str,
        entity_type: str = "Concept",
        embedding: list[float] | None = None,
    ) -> str:
        """
        Create or update an Entity node.
        """
        if not self.is_available():
            return entity_id

        cypher = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET
            e.name = $name,
            e.topic = $topic,
            e.type = $type,
            e.created_at = datetime()
        ON MATCH SET
            e.name = $name,
            e.topic = $topic,
            e.type = $type,
            e.updated_at = datetime()
        WITH e
        WHERE $embedding IS NOT NULL
        SET e.embedding = $embedding
        RETURN e.id AS id
        """
        try:
            with self._driver.session(database=self.database) as session:
                res = session.run(
                    cypher,
                    id=entity_id,
                    name=name,
                    topic=topic,
                    type=entity_type,
                    embedding=embedding,
                )
                record = res.single()
                return record["id"] if record else entity_id
        except Exception as exc:
            log_it(f"upsert_entity failed for {entity_id}: {exc}", _ENTITY)
            return entity_id

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        topic: str,
        is_functional: bool = True,
        confidence: float = 1.0,
        client: Any = None,
    ):
        """
        Add a relationship triple between subject and object.
        If is_functional=True, prior active relationships of the same predicate
        for the subject will be superseded (is_active=False).
        """
        if not self.is_available():
            return

        pred = _normalize_predicate(predicate)
        subj_id = f"entity:{_slugify(subject)}"
        obj_id = f"entity:{_slugify(obj)}"

        subj_vec = self._get_embedding(subject, client)
        obj_vec = self._get_embedding(obj, client)

        self.upsert_entity(subj_id, subject, topic, "Entity", subj_vec)
        self.upsert_entity(obj_id, obj, topic, "Entity", obj_vec)

        # Plain Cypher for edge creation and functional superseding
        plain_cypher = f"""
        MATCH (s:Entity {{id: $subj_id}})
        MATCH (o:Entity {{id: $obj_id}})
        OPTIONAL MATCH (s)-[old_r:{pred}]->()
        WHERE $is_functional = true AND old_r.is_active = true
        SET old_r.is_active = false, old_r.superseded_at = datetime()
        WITH s, o
        CREATE (s)-[new_r:{pred} {{
            topic: $topic,
            is_functional: $is_functional,
            is_active: true,
            confidence: $confidence,
            created_at: datetime()
        }}]->(o)
        RETURN new_r
        """

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    plain_cypher,
                    subj_id=subj_id,
                    obj_id=obj_id,
                    topic=topic,
                    is_functional=is_functional,
                    confidence=confidence,
                )
                log_it(f"Neo4j: Added fact ({subject}) -[:{pred}]-> ({obj}) [topic={topic}]", _ENTITY)
        except Exception as exc:
            log_it(f"add_fact failed: {exc}", _ENTITY)

    # ── Deterministic Subagent Helpers (Zero LLM) ──────────────────────────────

    def add_tool_definition(
        self,
        tool_name: str,
        signature: str,
        summary: str,
        required_binaries: list[str] | None = None,
        required_libs: list[str] | None = None,
        client: Any = None,
    ):
        """Record verified tool capabilities and dependencies in tooling.tools topic."""
        if not self.is_available():
            return

        tool_id = f"tool:{_slugify(tool_name)}"
        tool_vec = self._get_embedding(f"{tool_name} {signature} {summary}", client)
        self.upsert_entity(tool_id, tool_name, "tooling.tools", "Tool", tool_vec)

        # Record capability
        cap_id = f"cap:{_slugify(summary[:40])}"
        self.upsert_entity(cap_id, summary, "tooling.tools", "Capability")
        self.add_fact(tool_name, "HAS_CAPABILITY", summary, "tooling.tools", is_functional=True)

        # Record binary requirements in env.binaries
        for b in required_binaries or []:
            bin_id = f"bin:{_slugify(b)}"
            self.upsert_entity(bin_id, b, "env.binaries", "Binary")
            self.add_fact(tool_name, "REQUIRES_BINARY", b, "tooling.tools", is_functional=False)

        # Record library requirements in tooling.libraries
        for lib in required_libs or []:
            lib_id = f"lib:{_slugify(lib)}"
            self.upsert_entity(lib_id, lib, "tooling.libraries", "Library")
            self.add_fact(tool_name, "REQUIRES_LIBRARY", lib, "tooling.tools", is_functional=False)

    def add_tool_fix(
        self,
        tool_name: str,
        error_snippet: str,
        fix_summary: str,
        client: Any = None,
    ):
        """Record tool failure mode and resolved fix chain in debugging.fixes topic."""
        if not self.is_available():
            return

        tool_id = f"tool:{_slugify(tool_name)}"
        err_id = f"error:{_slugify(error_snippet[:40])}"
        fix_id = f"fix:{_slugify(fix_summary[:40])}"

        self.upsert_entity(tool_id, tool_name, "tooling.tools", "Tool")
        self.upsert_entity(err_id, error_snippet[:100], "debugging.fixes", "Error")
        self.upsert_entity(fix_id, fix_summary[:200], "debugging.fixes", "Fix")

        self.add_fact(tool_name, "FAILED_WITH", error_snippet[:100], "debugging.fixes", is_functional=False)
        self.add_fact(error_snippet[:100], "RESOLVED_BY", fix_summary[:200], "debugging.fixes", is_functional=False)

    def add_agent_fix(
        self,
        agent_name: str,
        failure_reason: str,
        prompt_remedy: str,
        client: Any = None,
    ):
        """Record agent prompt failure and judge remedy in agents.debugging topic."""
        if not self.is_available():
            return

        agent_id = f"agent:{_slugify(agent_name)}"
        fail_id = f"failure:{_slugify(failure_reason[:40])}"
        remedy_id = f"remedy:{_slugify(prompt_remedy[:40])}"

        self.upsert_entity(agent_id, agent_name, "agents.definitions", "Agent")
        self.upsert_entity(fail_id, failure_reason[:100], "agents.debugging", "FailureMode")
        self.upsert_entity(remedy_id, prompt_remedy[:200], "agents.debugging", "PromptRemedy")

        self.add_fact(agent_name, "FAILED_WITH", failure_reason[:100], "agents.debugging", is_functional=False)
        self.add_fact(failure_reason[:100], "RESOLVED_BY", prompt_remedy[:200], "agents.debugging", is_functional=False)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def query_active_facts(
        self,
        query_text: str,
        allowed_topics: list[str],
        client: Any = None,
        top_k: int = 5,
        limit: int = 10,
    ) -> list[str]:
        """
        Hybrid dense-graph retrieval:
        1. Finds seed entities via vector search (if index available) or text matching.
        2. Filters seed entities and active edges strictly by allowed_topics.
        3. Returns formatted active triples.
        """
        if not self.is_available() or not allowed_topics:
            return []

        # Prepare topic prefix filters for Cypher
        # e.g. "user.*" -> "user.", "user" -> "user"
        prefixes = [t[:-1] if t.endswith("*") else t for t in allowed_topics]

        query_vec = self._get_embedding(query_text, client)

        # Cypher query with vector search
        if query_vec and self._vector_index_ready:
            cypher = """
            CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_vec)
            YIELD node AS seed, score
            WHERE ANY(p IN $prefixes WHERE seed.topic STARTS WITH p OR seed.topic = p)
            MATCH (seed)-[r]->(target:Entity)
            WHERE r.is_active = true
              AND ANY(p IN $prefixes WHERE r.topic STARTS WITH p OR r.topic = p)
            RETURN DISTINCT seed.name AS subject, type(r) AS predicate, target.name AS object, r.topic AS topic, score
            ORDER BY score DESC
            LIMIT $limit
            """
            params = {
                "top_k": top_k,
                "query_vec": query_vec,
                "prefixes": prefixes,
                "limit": limit,
            }
        else:
            # Fallback Cypher without vector index: match entities where query contains name or topic
            cypher = """
            MATCH (seed:Entity)-[r]->(target:Entity)
            WHERE r.is_active = true
              AND ANY(p IN $prefixes WHERE seed.topic STARTS WITH p OR seed.topic = p)
              AND ANY(p IN $prefixes WHERE r.topic STARTS WITH p OR r.topic = p)
              AND (toLower($query) CONTAINS toLower(seed.name) OR toLower($query) CONTAINS toLower(target.name))
            RETURN DISTINCT seed.name AS subject, type(r) AS predicate, target.name AS object, r.topic AS topic, 1.0 AS score
            LIMIT $limit
            """
            params = {
                "query": query_text,
                "prefixes": prefixes,
                "limit": limit,
            }

        facts: list[str] = []
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                for record in result:
                    subj = record["subject"]
                    pred = record["predicate"].replace("_", " ").lower()
                    obj = record["object"]
                    facts.append(f"{subj} {pred} {obj}")
        except Exception as exc:
            log_it(f"query_active_facts failed: {exc}", _ENTITY)

        return facts
