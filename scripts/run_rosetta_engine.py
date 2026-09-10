"""Execute Frontier 1: The Rosetta Engine.

Loads the canonical Mediterranean cross-script phylogeny dataset,
audits network transmission alignment, and propagates dual-anchor phonetic
constraints onto the undeciphered Linear A and Cypro-Minoan scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ancient_text_lab.phylogeny import load_phylogeny_dataset


def run_rosetta_engine(dataset_path: Path) -> None:
    print(f"Loading Mediterranean phylogenetic network from {dataset_path}...")
    graph = load_phylogeny_dataset(dataset_path)

    print(
        f"Network loaded: {len(graph.scripts)} scripts, {len(graph.lineages)} lineages, {len(graph.homologues)} homologues."
    )

    audit = graph.audit_alignment(
        primary_anchor_id="linear_b", secondary_anchor_id="cypriot_syllabic"
    )

    print("\n================== ROSETTA ENGINE AUDIT REPORT ==================")
    print(f"Total Homologues Tracked:       {audit.total_homologues}")
    print(
        f"Deciphered Boundary Anchors:    {audit.deciphered_anchor_count} (Linear B, Classical Cypriot)"
    )
    print(
        f"Undeciphered Intermediate Nodes: {audit.undeciphered_node_count} (Linear A, Cypro-Minoan, CHIC)"
    )
    print(f"Mean Cursive Stroke Reduction:  {audit.mean_stroke_reduction_pct:.1f}%")
    print(f"Dual-Anchor Phonetic Agreement: {audit.dual_anchor_consistency_pct:.1f}%")
    print(f"Verified Scholarly Homologues:   {audit.verified_homologue_count}")
    print("=================================================================")

    # Propagate to Linear A
    la_readings = graph.propagate_dual_anchors("linear_a")
    print(f"\n--- Inferred Phonetic Projections for Linear A ({len(la_readings)} signs) ---")
    for r in sorted(la_readings, key=lambda x: -x.confidence):
        anchors = ", ".join(r.anchors_consulted)
        print(
            f"  [{r.target_sign_id}] {r.canonical_name:<16} -> /{r.inferred_phonetic}/ (conf: {r.confidence:.2f}) [{r.status}] via ({anchors})"
        )

    # Propagate to Cypro-Minoan
    cm_readings = graph.propagate_dual_anchors("cypro_minoan")
    print(f"\n--- Inferred Phonetic Projections for Cypro-Minoan ({len(cm_readings)} signs) ---")
    for r in sorted(cm_readings, key=lambda x: -x.confidence):
        anchors = ", ".join(r.anchors_consulted)
        print(
            f"  [{r.target_sign_id}] {r.canonical_name:<16} -> /{r.inferred_phonetic}/ (conf: {r.confidence:.2f}) [{r.status}] via ({anchors})"
        )

    # MRF Belief Propagation on Linear A
    from ancient_text_lab.phylogeny import infer_phonetics_mrf

    la_mrf = infer_phonetics_mrf(graph, "linear_a")
    print("\n================ MARKOV RANDOM FIELD BELIEF PROPAGATION (LINEAR A) ================")
    print(f"{'Sign ID':<8} {'Name':<16} {'MAP':<6} {'Conf':<6} {'Entropy':<8} {'Marginals'}")
    print("-" * 75)
    for m in sorted(la_mrf, key=lambda x: x.posterior_entropy_bits):
        marg_str = ", ".join(f"{k}:{v:.2f}" for k, v in m.marginal_distribution.items())
        print(
            f"[{m.target_sign_id:<6}] {m.canonical_name:<16} /{m.map_phonetic}/   {m.map_confidence:.3f}  {m.posterior_entropy_bits:.3f}b   {marg_str}"
        )
    print("===================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phylogeny/aegean_cypriot_homologues.json"),
        help="Path to canonical homologues JSON",
    )
    args = parser.parse_args()
    run_rosetta_engine(args.dataset)


if __name__ == "__main__":
    main()
