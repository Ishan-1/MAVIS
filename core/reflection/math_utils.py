"""
core/reflection/math_utils.py
Mathematical utilities for vector density, semantic variance,
inverse weighting, and temperature-controlled softmax topic selection.
"""
from __future__ import annotations

import math
import random
from typing import Sequence
from memories.embedding import cosine_similarity


def compute_vector_density(
    query_vec: list[float],
    memory_vecs: Sequence[list[float]],
    threshold: float = 0.75,
) -> int:
    """
    Measure Vector Density: count of memory vectors with cosine similarity >= threshold.
    """
    if not query_vec or not memory_vecs:
        return 0

    density = 0
    for vec in memory_vecs:
        if len(vec) != len(query_vec):
            continue
        sim = cosine_similarity(query_vec, vec)
        if sim >= threshold:
            density += 1
    return density


def compute_semantic_variance(memory_vecs: Sequence[list[float]]) -> float:
    """
    Measure Semantic Variance: average pairwise cosine distance (1 - sim) among memories.
    Returns 0.0 if fewer than 2 vectors are provided.
    """
    n = len(memory_vecs)
    if n < 2:
        return 0.0

    total_distance = 0.0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            vec_a = memory_vecs[i]
            vec_b = memory_vecs[j]
            if len(vec_a) != len(vec_b):
                continue
            sim = cosine_similarity(vec_a, vec_b)
            # Clamp similarity to [-1.0, 1.0] for floating point stability
            sim = max(-1.0, min(1.0, sim))
            dist = 1.0 - sim
            total_distance += dist
            pair_count += 1

    if pair_count == 0:
        return 0.0
    return total_distance / pair_count


def evaluate_pivot(
    density: int,
    variance: float,
    density_rut_threshold: int = 5,
    variance_rut_threshold: float = 0.25,
    density_focus_threshold: int = 3,
) -> tuple[bool, str]:
    """
    Determine whether the agent should pivot to adjacent concepts or explore directly.

    Returns:
        (should_pivot, reason)
    """
    if density >= density_rut_threshold and variance < variance_rut_threshold:
        return True, f"Rut detected (density={density} >= {density_rut_threshold}, variance={variance:.3f} < {variance_rut_threshold})"

    if density < density_focus_threshold:
        return False, f"Under-explored seed (density={density} < {density_focus_threshold})"

    # Balanced knowledge: pivot towards adjacent frontier to expand boundary
    return True, f"Balanced foundation (density={density}, variance={variance:.3f}), expanding frontier"


def compute_novelty_weights(
    candidate_vecs: Sequence[list[float]],
    memory_vecs: Sequence[list[float]],
) -> list[float]:
    """
    For each candidate concept vector, calculate novelty weight: W_i = 1 - max(Similarity).
    If memory_vecs is empty, returns 1.0 for all candidates.
    """
    if not candidate_vecs:
        return []

    if not memory_vecs:
        return [1.0 for _ in candidate_vecs]

    weights: list[float] = []
    for c_vec in candidate_vecs:
        max_sim = -1.0
        for m_vec in memory_vecs:
            if len(c_vec) != len(m_vec):
                continue
            sim = cosine_similarity(c_vec, m_vec)
            if sim > max_sim:
                max_sim = sim
        # Clamp max_sim into [-1.0, 1.0]
        max_sim = max(-1.0, min(1.0, max_sim))
        weights.append(max(0.0, 1.0 - max_sim))

    return weights


def temperature_softmax(weights: list[float], temperature: float = 0.7) -> list[float]:
    """
    Normalize weights into a probability distribution using a temperature parameter.
    P(x_i) = exp(W_i / T) / sum(exp(W_j / T))
    """
    if not weights:
        return []

    t = max(1e-4, float(temperature))
    scaled = [w / t for w in weights]
    max_scaled = max(scaled)  # numerical stability shift
    exp_vals = [math.exp(s - max_scaled) for s in scaled]
    sum_exp = sum(exp_vals)

    if sum_exp == 0.0:
        return [1.0 / len(weights) for _ in weights]

    return [e / sum_exp for e in exp_vals]


def sample_concept(
    candidates: list[str],
    probabilities: list[float],
    rng: random.Random | None = None,
) -> str:
    """
    Sample a candidate based on the calculated probability distribution.
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty.")
    if len(candidates) != len(probabilities):
        raise ValueError("Candidates and probabilities must have identical lengths.")

    r = rng or random
    return r.choices(candidates, weights=probabilities, k=1)[0]
