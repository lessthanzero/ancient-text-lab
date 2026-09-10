"""Tests for information-theoretic solvability and Shannon unicity bounds."""

from __future__ import annotations

import numpy as np
import pytest
from ancient_text_lab.sequence import TokenSequence, TransitionMatrix, transition_counts
from ancient_text_lab.solvability import (
    assess_corpus_solvability,
    shannon_unicity_distance,
    topological_entropy,
)


def test_shannon_unicity_distance_basic() -> None:
    # 50 signs, entropy 4.0 bits (max = log2(50) = 5.643 bits)
    d_bits, u_tokens = shannon_unicity_distance(signary_size=50, entropy_per_token=4.0)
    assert d_bits > 0.0
    assert u_tokens > 0.0
    # Higher entropy (lower redundancy) -> larger unicity distance
    d2, u2 = shannon_unicity_distance(signary_size=50, entropy_per_token=5.0)
    assert d2 < d_bits
    assert u2 > u_tokens


def test_shannon_unicity_distance_validation() -> None:
    with pytest.raises(ValueError, match="signary_size must be at least 2"):
        shannon_unicity_distance(signary_size=1, entropy_per_token=2.0)
    with pytest.raises(ValueError, match="entropy_per_token must be positive"):
        shannon_unicity_distance(signary_size=10, entropy_per_token=0.0)


def test_topological_entropy_calculation() -> None:
    vocab = ("A", "B", "C")
    # Directed cycle: A -> B -> C -> A
    counts = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int64)
    matrix = TransitionMatrix(vocabulary=vocab, counts=counts)
    lambda_max, h_top, sparsity = topological_entropy(matrix)

    assert np.isclose(lambda_max, 1.0)
    assert np.isclose(h_top, 0.0)
    assert np.isclose(sparsity, 1.0 - (3 / 9))


def test_assess_corpus_solvability_underdetermined() -> None:
    # Small toy corpus: 10 tokens across 5 signs
    seqs = [
        TokenSequence(unit_id="u1", tokens=("A", "B", "C")),
        TokenSequence(unit_id="u2", tokens=("B", "C", "D", "E")),
        TokenSequence(unit_id="u3", tokens=("A", "D", "E")),
    ]
    matrix = transition_counts(seqs)
    assessment = assess_corpus_solvability(seqs, matrix)

    assert assessment.corpus_token_count == 10
    assert assessment.signary_size == 5
    # For 5 signs with unicity distance ~ 10-20 tokens, let's verify underdetermined status
    assert not assessment.is_information_sufficient
    assert "Mathematically under-determined" in assessment.limitations_note


def test_assess_corpus_solvability_sufficient() -> None:
    # Sufficiently large corpus with non-uniform frequencies (Zipf-like)
    seqs = [
        TokenSequence(unit_id=f"u{i}", tokens=("A", "A", "A", "B", "A", "C")) for i in range(100)
    ]
    matrix = transition_counts(seqs)
    # Set a modest key equivocation
    assessment = assess_corpus_solvability(seqs, matrix, key_equivocation_bits=50.0)
    assert assessment.corpus_token_count == 600
    assert assessment.is_information_sufficient
    assert "Information-theoretically sufficient" in assessment.limitations_note
