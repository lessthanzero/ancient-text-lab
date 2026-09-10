"""Execute Frontier 3: The Epigraphic Foundation Model CLI.

Demonstrates self-supervised epigraphic representation learning across ancient Mediterranean scripts:
- Ingests formulaic contexts from Linear B, Linear A, Etruscan, and Latin.
- Fits continuous dense representations (PPMI + Truncated SVD).
- Evaluates zero-shot functional archetype projection (Ledgers vs Votives vs Funerary).
- Demonstrates position-aware archetype modulation and entropy-gated abstention.
"""

from __future__ import annotations

from ancient_text_lab.foundation import (
    EpigraphicCarrier,
    EpigraphicContextEncoder,
    EpigraphicEmbeddingModel,
    EpigraphicGenre,
    InscriptionContext,
    TokenPosition,
)


def run_foundation_demo() -> None:
    print("=================================================================")
    print("   FRONTIER 3: THE EPIGRAPHIC FOUNDATION MODEL (MELM PIPELINE)   ")
    print("=================================================================")

    # Curate multi-script Mediterranean epigraphic corpus
    contexts = [
        # Linear B administrative records (Pylos rations & personnel)
        InscriptionContext(
            document_id="PY_An_01",
            script="linear_b",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["KO-WO", "VIR", "5"],
                ["KO-WA", "MUL", "8"],
                ["TO-SO", "WHEAT", "120", "1/2"],
            ],
        ),
        InscriptionContext(
            document_id="KN_Fp_01",
            script="linear_b",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["ME-NO", "DE-LE-U-KO", "OIL", "20"],
                ["PA-SI-TE-O-I", "OIL", "10"],
            ],
        ),
        # Linear A administrative ledgers (Haghia Triada)
        InscriptionContext(
            document_id="HT_095",
            script="linear_a",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["DA-DU-MA-TO", "GRAIN", "30"],
                ["KU-RO", "GRAIN", "50", "1/4"],
            ],
        ),
        # Linear A peak sanctuary votive inscriptions
        InscriptionContext(
            document_id="PK_Za_11",
            script="linear_a",
            genre=EpigraphicGenre.VOTIVE_DEDICATION,
            carrier=EpigraphicCarrier.STONE_STELE,
            lines=[
                ["A-TA-I-*301-WA-JA", "JA-SA-SA-RA-ME", "U-NA-KA-NA-SI"],
                ["I-PI-NA-MA", "SI-RU-TE", "JA-SA-SA-RA-ME"],
            ],
        ),
        # Etruscan funerary stelae & ritual texts
        InscriptionContext(
            document_id="CIPPUS_PERUSINUS",
            script="etruscan",
            genre=EpigraphicGenre.JURIDICAL_TREATY,
            carrier=EpigraphicCarrier.STONE_STELE,
            lines=[
                ["tepras", "velthinas", "rasnes", "spural"],
                ["larthal", "velthinas", "clenar"],
            ],
        ),
        # Latin archaic votive & dedicatory formula
        InscriptionContext(
            document_id="CIL_I_01",
            script="archaic_latin",
            genre=EpigraphicGenre.VOTIVE_DEDICATION,
            carrier=EpigraphicCarrier.BRONZE_PLAQUE,
            lines=[
                ["MARTE", "DONUM", "DEDIT"],
                ["APOLONEI", "SACRUM", "FECIT"],
            ],
        ),
    ]

    print(f"\n[1] Training Epigraphic Context Encoder on {len(contexts)} multi-script documents...")
    encoder = EpigraphicContextEncoder(embedding_dim=16, entropy_threshold_bits=1.85)
    encoder.fit_contexts(contexts)

    # Ingest embeddings
    embed_model = EpigraphicEmbeddingModel(embedding_dim=4, window_size=2)
    embed_model.fit(contexts)
    print(f"    Vocabulary Size:        {len(embed_model.vocab)} distinct tokens")
    print(f"    Dense Latent Space:     {embed_model.embedding_dim} dimensions (SVD)")

    # Test Archetype Projections
    test_queries = [
        ("GRAIN", TokenPosition.LINE_MEDIAL, "Expected: COMMODITY_RECORD"),
        ("1/4", TokenPosition.LINE_FINAL, "Expected: NUMERICAL_OR_FRACTION"),
        ("JA-SA-SA-RA-ME", TokenPosition.LINE_MEDIAL, "Expected: DEITY_OR_SACRED_EPITHET"),
        ("velthinas", TokenPosition.LINE_INITIAL, "Expected: PROPER_NAME_OR_PATRONYMIC"),
        ("KU-RO", TokenPosition.LINE_INITIAL, "Expected: SYNTACTIC_FORMULA_OR_VERB"),
        ("UNKNOWN_GLYPH_X", TokenPosition.ISOLATED, "Expected: ABSTAINED (High Entropy)"),
    ]

    print("\n[2] Zero-Shot Functional Archetype Projection & Positional Modulation:")
    print("-----------------------------------------------------------------")
    for token, pos, expected in test_queries:
        prof = encoder.project_token_archetype(token, position=pos)
        status = "ABSTAINED" if prof.abstained else f"{prof.archetype} ({prof.confidence:.1%})"
        print(
            f"  Token: {token:17} | Pos: {pos.value:12} | Pred: {status:28} | H: {prof.entropy_bits:.2f} bits | {expected}"
        )

    print("\n[3] Semantic Latent Space Nearest Neighbors (Cosine Similarity):")
    print("-----------------------------------------------------------------")
    seed_tokens = ["GRAIN", "OIL", "JA-SA-SA-RA-ME"]
    for seed in seed_tokens:
        neighbors = embed_model.most_similar(seed, top_k=3)
        formatted = ", ".join(f"{nbr} ({sim:+.2f})" for nbr, sim in neighbors)
        print(f"  Anchor: {seed:15} -> {formatted}")

    print("\n=================================================================")
    print("           EPIGRAPHIC FOUNDATION MODEL BENCHMARK COMPLETE         ")
    print("=================================================================")


if __name__ == "__main__":
    run_foundation_demo()
