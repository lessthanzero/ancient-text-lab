"""Tests for blind benchmark contracts and their small project fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ancient_text_lab import (
    BenchmarkCase,
    BenchmarkManifest,
    GoldLabel,
    Prediction,
    SourceArtifact,
    evaluate_exact_match,
    validate_benchmark_cases,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = SourceArtifact(
    id="fixture-source",
    uri="fixture://benchmark/source",
    sha256="a621c958672d21b4c24102750fcdce21b7f1c0d230c2b47a833497756f4f20c0",
    media_type="application/json",
    license="MIT",
    artifact_kind="synthetic_fixture",
)


def manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id="fixture-benchmark",
        version="1",
        task="fixture task",
        split_seed=42,
        source_artifacts=(ARTIFACT,),
        limitations="Synthetic data only.",
    )


def cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            id="train",
            split="train",
            group_id="family-a",
            features={"form": "A"},
            source_artifact_ids=(ARTIFACT.id,),
        ),
        BenchmarkCase(
            id="test",
            split="test",
            group_id="family-b",
            features={"form": "B"},
            source_artifact_ids=(ARTIFACT.id,),
        ),
    )


def labels() -> tuple[GoldLabel, ...]:
    return (
        GoldLabel(
            case_id="train",
            value="A",
            confidence=1,
            source_artifact_ids=(ARTIFACT.id,),
            rationale="fixture",
        ),
        GoldLabel(
            case_id="test",
            value="B",
            confidence=1,
            source_artifact_ids=(ARTIFACT.id,),
            rationale="fixture",
        ),
    )


def test_evaluation_reports_abstention_and_exact_match() -> None:
    report = evaluate_exact_match(
        manifest(),
        cases(),
        labels(),
        (Prediction(case_id="test", value="B", score=0.8, model_version="v1"),),
    )
    assert report.evaluated_cases == 1
    assert report.exact_match == 1.0
    assert report.coverage == 1.0
    assert report.mean_correct_score == 0.8


def test_evaluation_keeps_abstentions_in_the_denominator() -> None:
    report = evaluate_exact_match(manifest(), cases(), labels(), ())
    assert report.exact_match == 0.0
    assert report.coverage == 0.0


def test_benchmark_rejects_gold_leakage_and_group_overlap() -> None:
    with pytest.raises(ValueError, match="reserved gold"):
        BenchmarkCase(
            id="bad",
            split="test",
            group_id="g",
            features={"label": "leak"},
            source_artifact_ids=(ARTIFACT.id,),
        )

    overlapping = (
        cases()[0],
        BenchmarkCase(
            id="test-2",
            split="test",
            group_id="family-a",
            features={},
            source_artifact_ids=(ARTIFACT.id,),
        ),
    )
    with pytest.raises(ValueError, match="share group IDs"):
        validate_benchmark_cases(
            manifest(),
            overlapping,
            (
                labels()[0],
                GoldLabel(
                    case_id="test-2",
                    value="B",
                    confidence=1,
                    source_artifact_ids=(ARTIFACT.id,),
                    rationale="fixture",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("project", "relative_path"),
    [
        ("linear-b", "benchmarks/transfer/public_cases.jsonl"),
        ("linear-b", "benchmarks/kober_grid/public_cases.jsonl"),
        ("etruscan", "benchmarks/morphology/public_cases.jsonl"),
    ],
)
def test_project_public_cases_validate_against_benchmark_schema(
    project: str, relative_path: str
) -> None:
    schema = json.loads((ROOT / "data/schemas/v1/benchmark.schema.json").read_text())
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    for line in (ROOT / "projects" / project / relative_path).read_text().splitlines():
        validator.validate(json.loads(line))
