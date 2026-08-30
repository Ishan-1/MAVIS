"""
memories/embedding.py
Thin wrapper around Gemini text-embedding-004.

Design:
  - Called once per new working-memory turn (at add_turn time).
  - Called once per user query (at retrieve_context time).
  - All previously stored embeddings live inside ChromaDB on disk;
    this module is never called for already-stored memories.
"""

from google import genai
from core.helpers import log_it
import os

_ENTITY = "embedding"
_MODEL = "text-embedding-004"


def get_client() -> genai.Client:
    """Return a Gemini client, reusing env config from main."""
    return genai.Client(vertexai=True, api_key=os.getenv("VERTEX_API_KEY"))


def embed(text: str, client: genai.Client | None = None) -> list[float]:
    """
    Embed *text* using Gemini text-embedding-004 and return the vector.

    Args:
        text:   The string to embed. Truncated to ~8 000 chars if longer.
        client: Optional pre-created genai.Client. If None, one is created.

    Returns:
        A list[float] embedding vector (length 768 for text-embedding-004).

    Raises:
        RuntimeError: if the Gemini call fails after one attempt.
    """
    if client is None:
        client = get_client()

    # Defensive truncation — embedding model has a token limit
    if len(text) > 8_000:
        text = text[:8_000]

    try:
        result = client.models.embed_content(model=_MODEL, contents=text)
        # The SDK returns EmbedContentResponse; extract the values list.
        vector = result.embeddings[0].values
        log_it(f"Embedded text (len={len(text)}) → vector dim={len(vector)}.", _ENTITY)
        return list(vector)
    except Exception as exc:
        log_it(f"Embedding failed: {exc}", _ENTITY)
        raise RuntimeError(f"Embedding call failed: {exc}") from exc


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
