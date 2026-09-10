"""Tests for the Alice Kober Linear B grid challenge and metrics."""

from __future__ import annotations

import pytest
from ancient_text_lab.kober import (
    LINEAR_B_GROUND_TRUTH_GRID,
    evaluate_kober_grid,
    induce_kober_stem_alternations,
)


def test_ground_truth_grid_completeness() -> None:
    # 54 core canonical signs
    assert len(LINEAR_B_GROUND_TRUTH_GRID) >= 50
    # Every sign maps to (consonant, vowel)
    for c, v in LINEAR_B_GROUND_TRUTH_GRID.values():
        assert len(v) == 1
        assert v in {"a", "e", "i", "o", "u"}
        assert len(c) >= 1


def test_evaluate_kober_grid_perfect() -> None:
    # Build perfect clustering from ground truth
    signs = sorted(LINEAR_B_GROUND_TRUTH_GRID.keys())
    c_labels = sorted({row[0] for row in LINEAR_B_GROUND_TRUTH_GRID.values()})
    v_labels = sorted({row[1] for row in LINEAR_B_GROUND_TRUTH_GRID.values()})

    c_map = {label: i for i, label in enumerate(c_labels)}
    v_map = {label: i for i, label in enumerate(v_labels)}

    pred_c = {sign: c_map[LINEAR_B_GROUND_TRUTH_GRID[sign][0]] for sign in signs}
    pred_v = {sign: v_map[LINEAR_B_GROUND_TRUTH_GRID[sign][1]] for sign in signs}

    report = evaluate_kober_grid(pred_c, pred_v)

    assert report.evaluated_signs == len(signs)
    assert pytest.approx(report.consonant_pairwise_f1) == 1.0
    assert pytest.approx(report.vowel_pairwise_f1) == 1.0
    assert pytest.approx(report.overall_kober_f1) == 1.0


def test_induce_kober_stem_alternations() -> None:
    # Triplet: da-to, da-te, da-ti
    # Another word: ko-no-so, ko-no-se
    words = [
        ("da", "to"),  # stem len 1
        ("da", "te"),
        ("ko", "no", "so"),  # stem len 2: ko-no -> so
        ("ko", "no", "se"),  # stem len 2: ko-no -> se
        ("pa", "i", "to"),  # isolated word: pa-i -> to
    ]
    alternations = induce_kober_stem_alternations(words, min_stem_length=2)

    assert "ko-no" in alternations
    assert alternations["ko-no"] == {"so", "se"}
    assert "pa-i" not in alternations  # only 1 suffix
    assert "da" not in alternations  # stem len is 1, min_stem_length=2
