"""Leakage-resistant contracts and metrics for research benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, model_validator

from ancient_text_lab.models import ContractModel, SourceArtifact


class BenchmarkManifest(ContractModel):
    """Metadata for a benchmark whose inputs and gold labels are kept separate."""

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task: str = Field(min_length=1)
    split_seed: int
    source_artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    limitations: str = Field(min_length=1)


class BenchmarkCase(ContractModel):
    """A model-visible case. Gold answers must never appear in ``features``."""

    id: str = Field(min_length=1)
    split: str = Field(pattern=r"^(train|test)$")
    group_id: str = Field(min_length=1)
    features: dict[str, object] = Field(default_factory=dict)
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_gold_like_feature_names(self) -> BenchmarkCase:
        reserved = {"gold", "gold_label", "label", "target", "answer"}
        overlap = reserved.intersection(self.features)
        if overlap:
            raise ValueError(
                f"model-visible features contain reserved gold keys: {sorted(overlap)}"
            )
        return self


class GoldLabel(ContractModel):
    """A withheld expected answer with its own evidence trail."""

    case_id: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class Prediction(ContractModel):
    case_id: str = Field(min_length=1)
    value: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str = Field(min_length=1)


class EvaluationReport(ContractModel):
    benchmark_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    evaluated_cases: int = Field(ge=0)
    exact_match: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    mean_correct_score: float | None = Field(default=None, ge=0.0, le=1.0)


def validate_benchmark_cases(
    manifest: BenchmarkManifest,
    cases: Iterable[BenchmarkCase],
    gold_labels: Iterable[GoldLabel],
) -> tuple[tuple[BenchmarkCase, ...], tuple[GoldLabel, ...]]:
    """Validate IDs, provenance, and group-disjoint train/test splits."""
    materialized_cases = tuple(cases)
    materialized_gold = tuple(gold_labels)
    case_ids = [case.id for case in materialized_cases]
    gold_ids = [label.case_id for label in materialized_gold]
    if len(case_ids) != len(set(case_ids)) or len(gold_ids) != len(set(gold_ids)):
        raise ValueError("benchmark case and gold-label IDs must each be unique")
    if set(case_ids) != set(gold_ids):
        raise ValueError("every benchmark case must have exactly one gold label")

    artifact_ids = {artifact.id for artifact in manifest.source_artifacts}
    referenced = {
        artifact_id
        for record in (*materialized_cases, *materialized_gold)
        for artifact_id in record.source_artifact_ids
    }
    unknown = referenced.difference(artifact_ids)
    if unknown:
        raise ValueError(f"benchmark references undeclared source artifacts: {sorted(unknown)}")

    train_groups = {case.group_id for case in materialized_cases if case.split == "train"}
    test_groups = {case.group_id for case in materialized_cases if case.split == "test"}
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise ValueError(f"train and test splits share group IDs: {sorted(overlap)}")
    return materialized_cases, materialized_gold


def evaluate_exact_match(
    manifest: BenchmarkManifest,
    cases: Iterable[BenchmarkCase],
    gold_labels: Iterable[GoldLabel],
    predictions: Iterable[Prediction],
    *,
    split: str = "test",
) -> EvaluationReport:
    """Score exact labels and abstentions without silently dropping predictions."""
    materialized_cases, materialized_gold = validate_benchmark_cases(manifest, cases, gold_labels)
    scoped_cases = tuple(case for case in materialized_cases if case.split == split)
    gold_by_case = {label.case_id: label for label in materialized_gold}
    prediction_by_case = {prediction.case_id: prediction for prediction in predictions}
    unknown_predictions = set(prediction_by_case).difference({case.id for case in scoped_cases})
    if unknown_predictions:
        raise ValueError(
            f"predictions contain cases outside requested split: {sorted(unknown_predictions)}"
        )

    answered = [prediction_by_case.get(case.id) for case in scoped_cases]
    non_abstained = [
        prediction for prediction in answered if prediction and prediction.value is not None
    ]
    correct = [
        prediction
        for case, prediction in zip(scoped_cases, answered, strict=True)
        if prediction and prediction.value == gold_by_case[case.id].value
    ]
    correct_scores = [prediction.score for prediction in correct if prediction.score is not None]
    count = len(scoped_cases)
    return EvaluationReport(
        benchmark_id=manifest.id,
        split=split,
        evaluated_cases=count,
        exact_match=(len(correct) / count) if count else 0.0,
        coverage=(len(non_abstained) / count) if count else 0.0,
        mean_correct_score=(sum(correct_scores) / len(correct_scores)) if correct_scores else None,
    )
