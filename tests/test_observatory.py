"""Tests for Frontier 2: The Decipherment Observatory."""

from __future__ import annotations

from pathlib import Path

from ancient_text_lab.observatory import (
    DeciphermentClaim,
    DeciphermentObservatory,
    EpistemicVerdict,
)
from ancient_text_lab.phylogeny import load_phylogeny_dataset
from ancient_text_lab.sequence import TokenSequence


def test_observatory_detects_underdetermined_corpus() -> None:
    # Small toy corpus of 15 tokens -> under-determined
    seqs = [
        TokenSequence(unit_id="u1", tokens=("A", "B", "C", "D")),
        TokenSequence(unit_id="u2", tokens=("B", "C", "D", "E")),
        TokenSequence(unit_id="u3", tokens=("A", "D", "E")),
    ]
    claim = DeciphermentClaim(
        claim_id="amateur-phaistos-claim",
        script_id="phaistos_disc",
        proponent="Anonymous Blogger",
        proposed_language="Proto-Semitic",
        proposed_sign_mapping={"A": "ba", "B": "el", "C": "yam"},
        corpus_sequences=seqs,
        citations=("Self-published 2026",),
    )

    observatory = DeciphermentObservatory()
    certificate = observatory.audit_claim(claim)

    assert certificate.verdict == EpistemicVerdict.UNDERDETERMINED
    assert not certificate.is_solvable_in_principle
    assert "UNDERDETERMINED: Total surviving tokens" in certificate.summary_verdict_rationale


def test_observatory_falsifies_contradicting_dual_anchors() -> None:
    path = Path(__file__).resolve().parents[1] / "data/phylogeny/aegean_cypriot_homologues.json"
    graph = load_phylogeny_dataset(path)

    # Large enough corpus so unicity passes
    seqs = [
        TokenSequence(unit_id=f"u{i}", tokens=("AB08", "AB04", "AB37", "AB28", "AB01"))
        for i in range(150)
    ]
    # Proponent claims AB08 (DOUBLE_AXE) is "mu", directly contradicting dual-anchor consensus "a"
    claim = DeciphermentClaim(
        claim_id="erroneous-linear-a-claim",
        script_id="linear_a",
        proponent="Speculative Linguist",
        proposed_language="Basque",
        proposed_sign_mapping={"AB08": "mu", "AB04": "te"},
        corpus_sequences=seqs,
        citations=("Blog post 2026",),
    )

    observatory = DeciphermentObservatory(phylogeny_graph=graph)
    certificate = observatory.audit_claim(claim)

    assert certificate.verdict == EpistemicVerdict.FALSIFIED
    assert (
        "FALSIFIED: Claim directly contradicts verified dual-anchor boundary conditions"
        in certificate.summary_verdict_rationale
    )
