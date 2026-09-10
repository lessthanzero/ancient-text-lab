"""Tests for synthetic lacunae masking and Bayesian restoration engine."""

from __future__ import annotations

import pytest
from ancient_text_lab.evaluation import BenchmarkCase, GoldLabel, Prediction
from ancient_text_lab.masking import generate_synthetic_lacunae
from ancient_text_lab.restoration import (
    BayesianTransitionInfiller,
    evaluate_restoration_predictions,
)
from ancient_text_lab.sequence import TokenSequence, transition_counts


def test_generate_synthetic_lacunae() -> None:
    seqs = [
        TokenSequence(unit_id="u1", tokens=("A", "B", "C", "D", "E")),
        TokenSequence(unit_id="u2", tokens=("B", "C", "D", "E", "F")),
    ]
    masked = generate_synthetic_lacunae(
        seqs,
        source_artifact_id="test-tablet",
        mask_rate=0.5,
        seed=42,
    )
    assert len(masked.benchmark_cases) == len(masked.gold_labels)
    assert len(masked.benchmark_cases) > 0

    # Ensure no reserved gold keys in features
    for case in masked.benchmark_cases:
        assert "gold" not in case.features
        assert "target" not in case.features
        assert "value" not in case.features
        assert "left_context" in case.features
        assert "right_context" in case.features


def test_evaluate_restoration_predictions_scoring() -> None:
    cases = [
        BenchmarkCase(
            id="c1", split="test", group_id="g1", features={}, source_artifact_ids=("art",)
        ),
        BenchmarkCase(
            id="c2", split="test", group_id="g1", features={}, source_artifact_ids=("art",)
        ),
        BenchmarkCase(
            id="c3", split="test", group_id="g2", features={}, source_artifact_ids=("art",)
        ),
    ]
    gold = [
        GoldLabel(
            case_id="c1", value="A", confidence=1.0, source_artifact_ids=("art",), rationale="test"
        ),
        GoldLabel(
            case_id="c2", value="B", confidence=1.0, source_artifact_ids=("art",), rationale="test"
        ),
        GoldLabel(
            case_id="c3", value="C", confidence=1.0, source_artifact_ids=("art",), rationale="test"
        ),
    ]
    preds = [
        Prediction(case_id="c1", value="A", score=0.9, model_version="m1"),  # Correct (+1.0)
        Prediction(case_id="c2", value=None, score=0.0, model_version="m1"),  # Abstained (0.0)
        Prediction(case_id="c3", value="X", score=0.4, model_version="m1"),  # False (-2.0)
    ]

    report = evaluate_restoration_predictions(cases, gold, preds, penalty_wrong=2.0)

    assert report.evaluated_cases == 3
    assert pytest.approx(report.exact_match) == 1 / 3
    assert pytest.approx(report.coverage) == 2 / 3
    assert pytest.approx(report.abstention_rate) == 1 / 3
    # Reward: (1.0 + 0.0 - 2.0) / 3 = -1.0 / 3
    assert pytest.approx(report.risk_calibrated_score) == -1.0 / 3
    assert pytest.approx(report.mean_correct_confidence) == 0.9


def test_bayesian_transition_infiller_learning() -> None:
    # Build strong repetitive bigram: A -> B -> C
    seqs = [TokenSequence(unit_id=f"u{i}", tokens=("A", "B", "C")) for i in range(50)]
    matrix = transition_counts(seqs)

    infiller = BayesianTransitionInfiller(matrix, entropy_threshold_bits=1.5)

    # Given left context ["A"] and right context ["C"], the missing token MUST be "B"
    case = BenchmarkCase(
        id="c1",
        split="test",
        group_id="g1",
        features={"left_context": ["A"], "right_context": ["C"]},
        source_artifact_ids=("art",),
    )
    pred = infiller.predict_case(case)
    assert pred.value == "B"
    assert pred.score is not None and pred.score > 0.8


def test_bayesian_transition_infiller_abstention() -> None:
    # When context is completely uninformative or high entropy, infiller should abstain
    seqs = [
        TokenSequence(unit_id="u1", tokens=("A", "B")),
        TokenSequence(unit_id="u2", tokens=("A", "C")),
        TokenSequence(unit_id="u3", tokens=("A", "D")),
        TokenSequence(unit_id="u4", tokens=("A", "E")),
    ]
    matrix = transition_counts(seqs)

    # Very strict entropy threshold
    infiller = BayesianTransitionInfiller(matrix, entropy_threshold_bits=0.5)

    case = BenchmarkCase(
        id="c1",
        split="test",
        group_id="g1",
        features={"left_context": ["A"], "right_context": []},
        source_artifact_ids=("art",),
    )
    pred = infiller.predict_case(case)
    # High entropy -> value should be None (abstained)
    assert pred.value is None
