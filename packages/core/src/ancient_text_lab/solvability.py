"""Information-theoretic solvability metrics and Shannon unicity bounds.

Provides mathematically rigorous bounds on whether a given corpus or document
contains sufficient information density to be deciphered uniquely without external
bilinguals or unconstrained priors.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ancient_text_lab.sequence import TokenSequence, TransitionMatrix


@dataclass(frozen=True, slots=True)
class SolvabilityAssessment:
    """Rigorous information-theoretic report on corpus decipherability."""

    corpus_token_count: int
    signary_size: int
    empirical_entropy_per_token: float
    redundancy: float
    unicity_distance: float
    is_information_sufficient: bool
    topological_entropy: float
    perron_frobenius_eigenvalue: float
    transition_sparsity: float
    limitations_note: str


def compute_empirical_token_entropy(sequences: Iterable[TokenSequence]) -> tuple[int, int, float]:
    """Compute token count, signary size, and empirical 1-gram Shannon entropy in bits."""
    counts: dict[str, int] = {}
    total = 0
    for seq in sequences:
        for token in seq.tokens:
            counts[token] = counts.get(token, 0) + 1
            total += 1

    if total == 0 or not counts:
        return 0, 0, 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return total, len(counts), entropy


def shannon_unicity_distance(
    signary_size: int,
    entropy_per_token: float,
    key_equivocation_bits: float | None = None,
) -> tuple[float, float]:
    """Calculate language redundancy and Shannon Unicity Distance U = H(K) / D.

    Parameters
    ----------
    signary_size : int
        Number of distinct signs/syllabograms in the writing system (|Sigma|).
    entropy_per_token : float
        Empirical Shannon entropy per sign in bits (H_L).
    key_equivocation_bits : float, optional
        Prior uncertainty of the cipher/sign-to-phoneme mapping H(K).
        Default is log2(signary_size!), corresponding to a general substitution permutation.

    Returns
    -------
    tuple[float, float]
        (redundancy D, unicity_distance U in tokens)
    """
    if signary_size < 2:
        raise ValueError("signary_size must be at least 2")
    if entropy_per_token <= 0:
        raise ValueError("entropy_per_token must be positive")

    max_entropy = math.log2(signary_size)
    if entropy_per_token >= max_entropy:
        # Near zero redundancy: unicity distance tends to infinity
        redundancy_rate = 1e-6
    else:
        redundancy_rate = 1.0 - (entropy_per_token / max_entropy)

    # Redundancy in bits per sign: D = R * log2(|Sigma|)
    d_bits = redundancy_rate * max_entropy

    if key_equivocation_bits is None:
        # Sterling's approximation or exact log2(n!) for permutation keys:
        # log2(n!) = sum_{k=1}^n log2(k)
        key_equivocation_bits = sum(math.log2(k) for k in range(1, signary_size + 1))

    unicity_tokens = key_equivocation_bits / d_bits
    return d_bits, unicity_tokens


def topological_entropy(matrix: TransitionMatrix) -> tuple[float, float, float]:
    """Calculate Perron-Frobenius eigenvalue, topological entropy, and transition sparsity.

    Topological entropy H_top = log2(lambda_max) bounds the topological complexity
    of the sequence grammar (Chomsky Type 3 language capacity).

    Returns
    -------
    tuple[float, float, float]
        (perron_frobenius_eigenvalue, topological_entropy_bits, sparsity_ratio)
    """
    n = len(matrix.vocabulary)
    if n == 0:
        return 0.0, 0.0, 1.0

    # Binary adjacency matrix: 1 if transition observed, else 0
    adj = (matrix.counts > 0).astype(float)
    sparsity = 1.0 - (float(np.count_nonzero(adj)) / (n * n))

    # Eigenvalues of adjacency matrix
    eigvals = np.linalg.eigvals(adj)
    real_parts = eigvals.real[np.isreal(eigvals) | (np.abs(eigvals.imag) < 1e-8)]

    if len(real_parts) == 0 or np.all(real_parts <= 0):
        lambda_max = 1.0
    else:
        lambda_max = float(np.max(real_parts))

    h_top = math.log2(max(lambda_max, 1.0))
    return lambda_max, h_top, sparsity


def assess_corpus_solvability(
    sequences: Iterable[TokenSequence],
    matrix: TransitionMatrix,
    *,
    key_equivocation_bits: float | None = None,
) -> SolvabilityAssessment:
    """Assess whether a corpus satisfies the information-theoretic criteria for unique decipherment."""
    materialized = tuple(sequences)
    total_tokens, signary_size, h_empirical = compute_empirical_token_entropy(materialized)

    if signary_size < 2 or total_tokens == 0:
        return SolvabilityAssessment(
            corpus_token_count=total_tokens,
            signary_size=signary_size,
            empirical_entropy_per_token=h_empirical,
            redundancy=0.0,
            unicity_distance=float("inf"),
            is_information_sufficient=False,
            topological_entropy=0.0,
            perron_frobenius_eigenvalue=0.0,
            transition_sparsity=1.0,
            limitations_note="Corpus contains insufficient tokens or vocabulary to evaluate.",
        )

    d_bits, u_tokens = shannon_unicity_distance(
        signary_size, h_empirical, key_equivocation_bits=key_equivocation_bits
    )
    lambda_max, h_top, sparsity = topological_entropy(matrix)

    is_sufficient = total_tokens >= u_tokens

    if not is_sufficient:
        note = (
            f"Mathematically under-determined: total tokens ({total_tokens}) < Shannon unicity "
            f"distance ({u_tokens:.1f}). Purely internal decipherment without bilingual or external "
            f"genealogical anchor yields infinitely many statistically indistinguishable solutions."
        )
    else:
        note = (
            f"Information-theoretically sufficient: total tokens ({total_tokens}) exceeds "
            f"unicity distance ({u_tokens:.1f}). Unique combinatorial solution is mathematically bounded."
        )

    return SolvabilityAssessment(
        corpus_token_count=total_tokens,
        signary_size=signary_size,
        empirical_entropy_per_token=h_empirical,
        redundancy=d_bits,
        unicity_distance=u_tokens,
        is_information_sufficient=is_sufficient,
        topological_entropy=h_top,
        perron_frobenius_eigenvalue=lambda_max,
        transition_sparsity=sparsity,
        limitations_note=note,
    )
