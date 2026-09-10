"""Execute Frontier 2: The Decipherment Observatory CLI.

Audits any proposed ancient script decipherment through the 4-stage epistemic sieve:
1. Shannon Unicity Distance Sieve
2. 100,000-Permutation Monte Carlo Null Sieve
3. Masked Synthetic Lacunae Infilling Sieve
4. Dual-Anchor Phylogenetic Consistency Sieve
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ancient_text_lab.observatory import (
    DeciphermentClaim,
    DeciphermentObservatory,
)
from ancient_text_lab.phylogeny import load_phylogeny_dataset
from ancient_text_lab.sequence import TokenSequence


def run_demo_audits() -> None:
    data_path = Path("data/phylogeny/aegean_cypriot_homologues.json")
    graph = load_phylogeny_dataset(data_path) if data_path.exists() else None
    observatory = DeciphermentObservatory(phylogeny_graph=graph)

    print("=================================================================")
    print("      THE DECIPHERMENT OBSERVATORY: AUTOMATED CLAIM AUDITOR     ")
    print("=================================================================")

    # Case A: An under-determined claim on small corpus
    small_seqs = [
        TokenSequence(unit_id=f"doc_{i}", tokens=("S01", "S02", "S05", "S12")) for i in range(10)
    ]
    claim_a = DeciphermentClaim(
        claim_id="phaistos-proto-semitic-hypothesis",
        script_id="phaistos_disc",
        proponent="Speculative Researcher",
        proposed_language="Proto-Semitic",
        proposed_sign_mapping={"S01": "ba", "S02": "el", "S05": "mot", "S12": "yam"},
        corpus_sequences=small_seqs,
        citations=("Amateur Monograph 2026",),
    )

    cert_a = observatory.audit_claim(claim_a)
    print(f"\n[CLAIM 1] {cert_a.claim_id}")
    print(f"  Target Script:       {cert_a.script_id}")
    print(f"  Proposed Language:   {cert_a.proposed_language}")
    print(
        f"  Corpus Tokens:       {cert_a.corpus_token_count} (Unicity Bound U: {cert_a.unicity_distance:.1f})"
    )
    print(f"  Verdict:             >>> {cert_a.verdict.value} <<<")
    print(f"  Rationale:           {cert_a.summary_verdict_rationale}")

    # Case B: A claim contradicting dual-anchor consensus
    large_seqs = [
        TokenSequence(unit_id=f"doc_{i}", tokens=("AB08", "AB04", "AB37", "AB28", "AB01", "AB77"))
        for i in range(120)
    ]
    claim_b = DeciphermentClaim(
        claim_id="linear-a-basque-substrate-hypothesis",
        script_id="linear_a",
        proponent="Comparative Essayist",
        proposed_language="Archaic Vasconic",
        # Claims AB08 is "gai" instead of dual-anchor verified "a"
        proposed_sign_mapping={"AB08": "gai", "AB04": "te", "AB37": "ti", "AB28": "i"},
        corpus_sequences=large_seqs,
        citations=("Online Forum Post 2026",),
    )

    cert_b = observatory.audit_claim(claim_b)
    print(f"\n[CLAIM 2] {cert_b.claim_id}")
    print(f"  Target Script:       {cert_b.script_id}")
    print(f"  Proposed Language:   {cert_b.proposed_language}")
    print(f"  Phylo Consistency:   {cert_b.phylogenetic_consistency_pct:.1f}%")
    print(f"  Verdict:             >>> {cert_b.verdict.value} <<<")
    print(f"  Rationale:           {cert_b.summary_verdict_rationale}")
    print("=================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", default=True, help="Run demonstration audits"
    )
    args = parser.parse_args()
    if args.demo:
        run_demo_audits()


if __name__ == "__main__":
    main()
