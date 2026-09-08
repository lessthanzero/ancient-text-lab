"""Deterministic behavior checks for script-neutral sequence primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ancient_text_lab import NormalizedDocument, ppmi, shuffle_within_units, transition_counts
from ancient_text_lab.sequence import normalize_token_sequences

ROOT = Path(__file__).resolve().parents[1]


def load_sequences(sample_name: str):
    raw = json.loads((ROOT / "data" / "samples" / sample_name / "document.json").read_text())
    return normalize_token_sequences(NormalizedDocument.model_validate(raw).units)


@pytest.mark.parametrize("sample_name", ["linear-a", "phaistos-disc"])
def test_transition_counts_respect_explicit_unit_boundaries(sample_name: str) -> None:
    sequences = load_sequences(sample_name)
    matrix = transition_counts(sequences)

    assert matrix.counts.sum() == sum(len(sequence.tokens) - 1 for sequence in sequences)
    assert matrix.vocabulary == tuple(sorted(matrix.vocabulary))


def test_ppmi_has_explicit_smoothing_and_handles_empty_counts() -> None:
    sequences = load_sequences("linear-a")
    matrix = transition_counts(sequences)

    result = ppmi(matrix, smoothing=0.1)
    assert result.shape == matrix.counts.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    with pytest.raises(ValueError, match="non-negative"):
        ppmi(matrix, smoothing=-0.1)


@pytest.mark.parametrize("sample_name", ["linear-a", "phaistos-disc"])
def test_seeded_null_samples_are_reproducible_and_preserve_boundaries(sample_name: str) -> None:
    sequences = load_sequences(sample_name)
    first = shuffle_within_units(sequences, iterations=3, seed=42)
    second = shuffle_within_units(sequences, iterations=3, seed=42)

    assert first == second
    for null_sample in first:
        assert [item.unit_id for item in null_sample] == [item.unit_id for item in sequences]
        assert [len(item.tokens) for item in null_sample] == [len(item.tokens) for item in sequences]
        assert sorted(token for item in null_sample for token in item.tokens) == sorted(
            token for item in sequences for token in item.tokens
        )


def test_transition_counts_rejects_an_incomplete_vocabulary() -> None:
    sequences = load_sequences("linear-a")
    with pytest.raises(ValueError, match="omits observed"):
        transition_counts(sequences, vocabulary=("A01",))
