"""Tests for the Rosetta Engine cross-script phylogenetic alignment graph."""

from __future__ import annotations

import pytest
from ancient_text_lab.phylogeny import (
    LineageEdge,
    ScriptNode,
    ScriptPhylogenyGraph,
    SignHomologue,
)


def test_script_phylogeny_graph_construction() -> None:
    scripts = [
        ScriptNode(
            id="cretan_hieroglyphic",
            name="Cretan Hieroglyphic",
            chronology_bce=(-2100, -1700),
            is_deciphered=False,
            canonical_sign_count=137,
        ),
        ScriptNode(
            id="linear_a",
            name="Linear A",
            chronology_bce=(-1800, -1450),
            is_deciphered=False,
            canonical_sign_count=97,
        ),
        ScriptNode(
            id="linear_b",
            name="Linear B",
            chronology_bce=(-1450, -1200),
            is_deciphered=True,
            canonical_sign_count=87,
            primary_language="Mycenaean Greek",
        ),
        ScriptNode(
            id="cypriot_syllabic",
            name="Classical Cypriot Syllabary",
            chronology_bce=(-1100, -300),
            is_deciphered=True,
            canonical_sign_count=56,
            primary_language="Arcadocypriot Greek",
        ),
    ]

    lineages = [
        LineageEdge(
            source_script_id="cretan_hieroglyphic",
            target_script_id="linear_a",
            palaeographic_continuity=0.85,
            historical_notes="Minoan administrative cursive transmission",
        ),
        LineageEdge(
            source_script_id="linear_a",
            target_script_id="linear_b",
            palaeographic_continuity=0.90,
            historical_notes="Mycenaean adaptation for Greek at Knossos",
        ),
    ]

    graph = ScriptPhylogenyGraph(scripts, lineages)
    assert len(graph.scripts) == 4
    assert len(graph.lineages) == 2


def test_script_phylogeny_validation() -> None:
    scripts = [
        ScriptNode(
            id="linear_a",
            name="Linear A",
            chronology_bce=(-1800, -1450),
            is_deciphered=False,
            canonical_sign_count=97,
        ),
    ]
    bad_lineages = [
        LineageEdge(
            source_script_id="linear_a",
            target_script_id="unknown_script",
            palaeographic_continuity=0.5,
            historical_notes="test",
        )
    ]
    with pytest.raises(ValueError, match="unknown target script"):
        ScriptPhylogenyGraph(scripts, bad_lineages)


def test_phylogenetic_audit_and_dual_anchor() -> None:
    scripts = [
        ScriptNode(
            id="linear_a",
            name="Linear A",
            chronology_bce=(-1800, -1450),
            is_deciphered=False,
            canonical_sign_count=97,
        ),
        ScriptNode(
            id="linear_b",
            name="Linear B",
            chronology_bce=(-1450, -1200),
            is_deciphered=True,
            canonical_sign_count=87,
        ),
        ScriptNode(
            id="cypriot_syllabic",
            name="Cypriot",
            chronology_bce=(-1100, -300),
            is_deciphered=True,
            canonical_sign_count=56,
        ),
    ]
    lineages = [
        LineageEdge(
            source_script_id="linear_a",
            target_script_id="linear_b",
            palaeographic_continuity=0.9,
            historical_notes="Aegean transmission",
        ),
    ]

    # Double-axe sign (AB08 / *a*)
    homologue_a = SignHomologue(
        canonical_name="DOUBLE_AXE",
        sign_ids={"linear_a": "AB08", "linear_b": "B08", "cypriot_syllabic": "CS_a"},
        pictorial_origin="Double-headed ceremonial labrys axe",
        stroke_counts={"linear_a": 5, "linear_b": 3},
        ground_truth_phonetics={"linear_b": "a", "cypriot_syllabic": "a"},
        projected_phonetics={"linear_a": "a"},
        citations=("Olivier 1989", "GORILA V"),
    )

    graph = ScriptPhylogenyGraph(scripts, lineages, [homologue_a])
    report = graph.audit_alignment()

    assert report.total_homologues == 1
    assert report.deciphered_anchor_count == 2
    assert report.undeciphered_node_count == 1
    # Dual anchor matched ('a' == 'a') -> 100%
    assert report.dual_anchor_consistency_pct == 100.0
    # Stroke reduction from 5 to 3 -> (5 - 3)/5 = 40.0%
    assert pytest.approx(report.mean_stroke_reduction_pct) == 40.0
    assert report.verified_homologue_count == 1


def test_load_canonical_dataset_and_propagate() -> None:
    from pathlib import Path

    from ancient_text_lab.phylogeny import load_phylogeny_dataset

    path = Path(__file__).resolve().parents[1] / "data/phylogeny/aegean_cypriot_homologues.json"
    assert path.exists()

    graph = load_phylogeny_dataset(path)
    assert len(graph.scripts) == 5
    assert len(graph.lineages) == 4
    assert len(graph.homologues) >= 12

    audit = graph.audit_alignment(
        primary_anchor_id="linear_b", secondary_anchor_id="cypriot_syllabic"
    )
    assert audit.total_homologues >= 12
    assert audit.mean_stroke_reduction_pct > 15.0  # Significant cursive stroke reduction
    assert (
        audit.dual_anchor_consistency_pct > 75.0
    )  # 77.8% historical dual-anchor consistency (accounting for r/l and ni/ta divergence)

    # Test dual-anchor propagation to Linear A
    la_readings = graph.propagate_dual_anchors("linear_a")
    assert len(la_readings) >= 10

    by_name = {r.canonical_name: r for r in la_readings}
    # DOUBLE_AXE (AB08) should be DUAL_ANCHOR_VERIFIED as 'a'
    assert by_name["DOUBLE_AXE"].inferred_phonetic == "a"
    assert by_name["DOUBLE_AXE"].status == "DUAL_ANCHOR_VERIFIED"
    assert by_name["DOUBLE_AXE"].confidence > 0.95

    # GRAIN_STALK (AB30) should be DIVERGENT_BRANCHES (ni vs ta)
    assert by_name["GRAIN_STALK"].status == "DIVERGENT_BRANCHES"
    assert by_name["GRAIN_STALK"].confidence < 0.50


def test_infer_phonetics_mrf_convergence_and_posteriors() -> None:
    from pathlib import Path

    from ancient_text_lab.phylogeny import infer_phonetics_mrf, load_phylogeny_dataset

    path = Path(__file__).resolve().parents[1] / "data/phylogeny/aegean_cypriot_homologues.json"
    graph = load_phylogeny_dataset(path)

    # Infer Linear A posteriors
    reports = infer_phonetics_mrf(graph, "linear_a")
    assert len(reports) >= 10

    by_name = {r.canonical_name: r for r in reports}

    # DOUBLE_AXE (AB08): Converged, MAP = 'a', low entropy
    axe = by_name["DOUBLE_AXE"]
    assert axe.is_converged
    assert axe.map_phonetic == "a"
    assert axe.map_confidence > 0.90
    assert axe.posterior_entropy_bits < 0.50
    assert axe.anchors_active == ("cypriot_syllabic", "linear_b")

    # CROSS_ROSETTE (AB02): Models the ro/lo liquid shift
    rosette = by_name["CROSS_ROSETTE"]
    assert rosette.is_converged
    # Marginal has mass on both ro and lo
    assert "ro" in rosette.marginal_distribution
    assert "lo" in rosette.marginal_distribution
