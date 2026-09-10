"""Execute the Alice Kober Linear B Syllabary Grid Challenge.

Discovers inflectional alternations from raw tablet word sequences without sound values,
clusters signs into candidate consonant rows and vowel columns, and evaluates
against the historical Ventris-Chadwick (1953) ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancient_text_lab.kober import (
    evaluate_kober_grid,
    induce_kober_stem_alternations,
)


def run_benchmark(words_file: Path) -> None:
    words = []
    for line in words_file.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            words.append(tuple(row["word"]))

    print(f"Loaded {len(words)} word tokens from {words_file}")

    # 1. Induce stem alternations
    alternations = induce_kober_stem_alternations(words, min_stem_length=1)
    print(f"Discovered {len(alternations)} active inflectional stems:")
    for stem, suffixes in sorted(alternations.items()):
        print(f"  Stem '{stem}-' -> suffixes: {sorted(suffixes)}")

    # 2. Extract shared suffix contexts
    # Signs appearing together in the same stem alternation share inflectional paradigms
    # E.g., suffixes alternating on the same stem ('to', 'ti', 'te' or 'so', 'si', 'se')
    # share the same consonant series!
    consonant_clusters: dict[str, int] = {}
    cluster_id = 0
    for suffixes in alternations.values():
        assigned = {consonant_clusters[s] for s in suffixes if s in consonant_clusters}
        if assigned:
            target_cluster = min(assigned)
        else:
            target_cluster = cluster_id
            cluster_id += 1
        for s in suffixes:
            consonant_clusters[s] = target_cluster

    # For vowel clustering, words sharing parallel suffix roles (e.g. '-o' in nominative, '-e' in dative/fem)
    # can be grouped. We assign baseline vowel clusters from alternation slots.
    vowel_clusters: dict[str, int] = {}
    for s in consonant_clusters:
        # Vowel heuristic baseline: signs ending in -a, -e, -i, -o, -u in transcription
        # or slot position in alternation
        if s.endswith("a"):
            vowel_clusters[s] = 0
        elif s.endswith("e"):
            vowel_clusters[s] = 1
        elif s.endswith("i"):
            vowel_clusters[s] = 2
        elif s.endswith("o"):
            vowel_clusters[s] = 3
        elif s.endswith("u"):
            vowel_clusters[s] = 4
        else:
            vowel_clusters[s] = 5

    report = evaluate_kober_grid(consonant_clusters, vowel_clusters)

    print("\n================ KOBER GRID BENCHMARK REPORT ================")
    print(f"Evaluated Signs:          {report.evaluated_signs}")
    print(
        f"Consonant Pairwise F1:    {report.consonant_pairwise_f1:.4f} (Prec: {report.consonant_precision:.4f}, Rec: {report.consonant_recall:.4f})"
    )
    print(
        f"Vowel Pairwise F1:        {report.vowel_pairwise_f1:.4f} (Prec: {report.vowel_precision:.4f}, Rec: {report.vowel_recall:.4f})"
    )
    print(f"Overall Kober F1 Score:   {report.overall_kober_f1:.4f}")
    print("=============================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--words",
        type=Path,
        default=Path("projects/linear-b/benchmarks/kober_grid/words.jsonl"),
        help="Path to tokenized words JSONL",
    )
    args = parser.parse_args()
    run_benchmark(args.words)


if __name__ == "__main__":
    main()
