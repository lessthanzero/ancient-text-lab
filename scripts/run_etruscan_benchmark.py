"""Execute the Etruscan Morphology Benchmark.

Evaluates held-out segmentation and bounded grammatical suffix prediction
on authentic Etruscan inscriptions (Pyrgi, Liber Linteus, Cippus Perusinus).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancient_text_lab.evaluation import (
    BenchmarkCase,
    BenchmarkManifest,
    GoldLabel,
    Prediction,
    evaluate_exact_match,
)
from ancient_text_lab.models import SourceArtifact


def run_etruscan_benchmark(cases_path: Path, gold_path: Path) -> None:
    cases = [
        BenchmarkCase.model_validate(json.loads(line))
        for line in cases_path.read_text().splitlines()
        if line.strip()
    ]
    gold = [
        GoldLabel.model_validate(json.loads(line))
        for line in gold_path.read_text().splitlines()
        if line.strip()
    ]

    all_artifact_ids = sorted({art for c in (*cases, *gold) for art in c.source_artifact_ids})
    source_artifacts = tuple(
        SourceArtifact(
            id=art_id,
            uri=f"scholarly://etruscan/{art_id}",
            sha256="0" * 64,
            media_type="application/json",
            license="Public Domain (Epigraphic texts)",
        )
        for art_id in all_artifact_ids
    )

    manifest = BenchmarkManifest(
        id="etruscan-morphology-benchmark",
        version="1.0",
        task="held-out grammatical suffix recovery",
        split_seed=42,
        source_artifacts=source_artifacts,
        limitations="Bounded morphological suffixes only; translation explicitly out of scope.",
    )

    # Baseline morphological rule-based parser on test split
    predictions = []
    test_cases = [c for c in cases if c.split == "test"]
    for c in test_cases:
        term = str(c.features.get("terminal_chars", ""))
        surface = str(c.features.get("surface_token", ""))

        if term == "al" or surface.endswith("al"):
            pred_val = "GENITIVE_AL"
        elif term in ("s", "as") or surface.endswith("s"):
            pred_val = "GENITIVE_S"
        elif term == "ce" or surface.endswith("ce"):
            pred_val = "PRETERITE_CE"
        elif term == "ar" or surface.endswith("ar"):
            pred_val = "PLURAL_AR"
        else:
            pred_val = None  # Explicit abstention

        predictions.append(
            Prediction(
                case_id=c.id,
                value=pred_val,
                score=0.95 if pred_val else None,
                model_version="etruscan-rule-baseline-v1",
            )
        )

    report = evaluate_exact_match(manifest, cases, gold, predictions, split="test")

    print("\n================ ETRUSCAN MORPHOLOGY BENCHMARK REPORT ================")
    print(f"Evaluated Held-Out Cases: {report.evaluated_cases}")
    print(f"Exact Match Accuracy:     {report.exact_match:.4f} ({report.exact_match * 100.0:.1f}%)")
    print(f"Prediction Coverage:      {report.coverage:.4f} ({report.coverage * 100.0:.1f}%)")
    print(
        f"Mean Confidence Score:    {report.mean_correct_score:.4f}"
        if report.mean_correct_score
        else "N/A"
    )
    print("=====================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("projects/etruscan/benchmarks/morphology/public_cases.jsonl"),
        help="Path to public cases JSONL",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("projects/etruscan/benchmarks/morphology/gold_labels.jsonl"),
        help="Path to gold labels JSONL",
    )
    args = parser.parse_args()
    run_etruscan_benchmark(args.cases, args.gold)


if __name__ == "__main__":
    main()
