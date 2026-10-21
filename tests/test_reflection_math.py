"""
tests/test_reflection_math.py
Unit tests for core/reflection/math_utils.py and safe inspection in tools_adapter.py.
"""
import pytest
from core.reflection.math_utils import (
    compute_vector_density,
    compute_semantic_variance,
    evaluate_pivot,
    compute_novelty_weights,
    temperature_softmax,
    sample_concept,
)
from core.reflection.tools_adapter import read_local_context


def test_vector_density():
    # Identity vector
    v = [1.0, 0.0, 0.0]
    memories = [
        [1.0, 0.0, 0.0],  # sim = 1.0 (>= 0.75)
        [0.9, 0.1, 0.0],  # sim ~ 0.99 (>= 0.75)
        [0.0, 1.0, 0.0],  # sim = 0.0 (< 0.75)
    ]
    density = compute_vector_density(v, memories, threshold=0.75)
    assert density == 2


def test_semantic_variance():
    # Identical vectors -> variance = 0
    m_identical = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert compute_semantic_variance(m_identical) == pytest.approx(0.0)

    # Orthogonal vectors -> distance = 1.0
    m_orthogonal = [[1.0, 0.0], [0.0, 1.0]]
    assert compute_semantic_variance(m_orthogonal) == pytest.approx(1.0)


def test_evaluate_pivot():
    # High density, low variance -> rut detected -> should pivot
    should_p, reason = evaluate_pivot(density=6, variance=0.10)
    assert should_p is True
    assert "Rut detected" in reason

    # Low density -> under-explored -> should not pivot
    should_p, reason = evaluate_pivot(density=1, variance=0.50)
    assert should_p is False
    assert "Under-explored" in reason


def test_novelty_weights_and_softmax():
    # Candidates vs memories
    memories = [[1.0, 0.0, 0.0]]
    c1 = [1.0, 0.0, 0.0]  # sim = 1.0 -> weight = 0.0
    c2 = [0.0, 1.0, 0.0]  # sim = 0.0 -> weight = 1.0

    weights = compute_novelty_weights([c1, c2], memories)
    assert len(weights) == 2
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(1.0)

    # Softmax with low temperature should concentrate heavily on c2
    probs = temperature_softmax(weights, temperature=0.2)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] > probs[0]
    assert probs[1] > 0.95

    # Sample concept
    chosen = sample_concept(["c1", "c2"], probs)
    assert chosen in ["c1", "c2"]


def test_tools_adapter_local_security():
    # Forbidden .env access
    status, msg = read_local_context(".env")
    assert status == -1
    assert "Access denied" in msg

    # Forbidden path traversal
    status, msg = read_local_context("../../etc/passwd")
    assert status == -1
    assert "Access denied" in msg

    # Forbidden unapproved directory
    status, msg = read_local_context(".git/config")
    assert status == -1
    assert "Access denied" in msg

    # Approved directory: docs/Benchmark.md
    status, content = read_local_context("docs/Benchmark.md", max_chars=100)
    assert status == 0
    assert len(content) > 0
