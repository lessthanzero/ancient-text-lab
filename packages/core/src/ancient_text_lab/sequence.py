"""Deterministic, script-neutral token sequence primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ancient_text_lab.models import TextUnit


@dataclass(frozen=True, slots=True)
class TokenSequence:
    unit_id: str
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id must not be empty")
        if not self.tokens or any(not token for token in self.tokens):
            raise ValueError("tokens must be a non-empty sequence of non-empty identifiers")


@dataclass(frozen=True, slots=True)
class TransitionMatrix:
    vocabulary: tuple[str, ...]
    counts: np.ndarray

    def __post_init__(self) -> None:
        if self.counts.shape != (len(self.vocabulary), len(self.vocabulary)):
            raise ValueError("counts shape must match vocabulary")
        if np.any(self.counts < 0):
            raise ValueError("transition counts cannot be negative")


def normalize_token_sequences(units: Iterable[TextUnit]) -> tuple[TokenSequence, ...]:
    """Convert normalized text units to ordered, explicit-boundary token sequences."""
    ordered = sorted(units, key=lambda unit: unit.position)
    return tuple(TokenSequence(unit_id=unit.id, tokens=unit.signs) for unit in ordered)


def transition_counts(
    sequences: Iterable[TokenSequence], vocabulary: Sequence[str] | None = None
) -> TransitionMatrix:
    """Count adjacent token pairs without crossing unit boundaries."""
    materialized = tuple(sequences)
    observed = {token for sequence in materialized for token in sequence.tokens}
    vocab = tuple(sorted(observed) if vocabulary is None else vocabulary)
    if len(vocab) != len(set(vocab)):
        raise ValueError("vocabulary must not contain duplicate tokens")
    unknown = observed.difference(vocab)
    if unknown:
        raise ValueError(f"vocabulary omits observed tokens: {sorted(unknown)}")

    index = {token: position for position, token in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)), dtype=np.int64)
    for sequence in materialized:
        for left, right in zip(sequence.tokens, sequence.tokens[1:], strict=False):
            counts[index[left], index[right]] += 1
    return TransitionMatrix(vocabulary=vocab, counts=counts)


def ppmi(matrix: TransitionMatrix, *, smoothing: float = 0.0) -> np.ndarray:
    """Return a non-negative PPMI matrix using an explicit additive smoothing policy."""
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    counts = matrix.counts.astype(float) + smoothing
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    joint = counts / total
    expected = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log2(np.divide(joint, expected, out=np.ones_like(joint), where=expected > 0))
    return np.maximum(values, 0.0)


def shuffle_within_units(
    sequences: Iterable[TokenSequence], *, iterations: int, seed: int
) -> tuple[tuple[TokenSequence, ...], ...]:
    """Generate seeded frequency-preserving null samples while retaining unit lengths."""
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    materialized = tuple(sequences)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        samples.append(
            tuple(
                TokenSequence(unit_id=sequence.unit_id, tokens=tuple(rng.permutation(sequence.tokens)))
                for sequence in materialized
            )
        )
    return tuple(samples)

