"""Tests for Frontier 3: The Epigraphic Foundation Model."""

from __future__ import annotations

from ancient_text_lab.foundation import (
    EpigraphicCarrier,
    EpigraphicContextEncoder,
    EpigraphicEmbeddingModel,
    EpigraphicGenre,
    InscriptionContext,
    TokenPosition,
)


def test_epigraphic_context_encoder_training_and_projection() -> None:
    contexts = [
        # Administrative accounting tablet (Linear B / Linear A style)
        InscriptionContext(
            document_id="HT_001",
            script="linear_a",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["KU-RO", "GRAIN", "10", "1/2"],
                ["A-DU", "OIL", "5"],
            ],
        ),
        # Votive libation vessel (peak sanctuary dedication)
        InscriptionContext(
            document_id="IO_Za_002",
            script="linear_a",
            genre=EpigraphicGenre.VOTIVE_DEDICATION,
            carrier=EpigraphicCarrier.STONE_STELE,
            lines=[
                ["A-TA-I-*301-WA-JA", "JA-SA-SA-RA-ME", "U-NA-KA-NA-SI"],
            ],
        ),
        # Funerary epitaph (Etruscan stele)
        InscriptionContext(
            document_id="CIPPUS_001",
            script="etruscan",
            genre=EpigraphicGenre.FUNERARY_EPITAPH,
            carrier=EpigraphicCarrier.STONE_STELE,
            lines=[
                ["larthal", "velthinas", "clenar", "svalce"],
            ],
        ),
    ]

    encoder = EpigraphicContextEncoder(entropy_threshold_bits=2.0)
    encoder.fit_contexts(contexts)

    # 1. Test commodity / accounting token projection
    prof_grain = encoder.project_token_archetype("GRAIN")
    assert not prof_grain.abstained
    assert prof_grain.archetype in ("COMMODITY_RECORD", "NUMERICAL_OR_FRACTION")
    assert prof_grain.confidence > 0.40

    # 2. Test votive deity epithet projection (JA-SA-SA-RA-ME)
    prof_votive = encoder.project_token_archetype("JA-SA-SA-RA-ME")
    assert not prof_votive.abstained
    assert prof_votive.archetype == "DEITY_OR_SACRED_EPITHET"
    assert prof_votive.confidence > 0.55

    # 3. Test onomastic / patronymic projection (velthinas)
    prof_patronymic = encoder.project_token_archetype("velthinas")
    assert not prof_patronymic.abstained
    assert prof_patronymic.archetype == "PROPER_NAME_OR_PATRONYMIC"

    # 4. Out of vocabulary token -> max entropy, abstained
    prof_unknown = encoder.project_token_archetype("TOTALLY_UNKNOWN_TOKEN")
    assert prof_unknown.abstained
    assert prof_unknown.archetype == "UNKNOWN"
    assert prof_unknown.entropy_bits > 2.0


def test_extract_token_sequences() -> None:
    contexts = [
        InscriptionContext(
            document_id="DOC1",
            script="test",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[["A", "B"], ["C", "D"]],
        )
    ]
    encoder = EpigraphicContextEncoder()
    seqs = encoder.extract_token_sequences(contexts)

    assert len(seqs) == 2
    assert seqs[0].unit_id == "DOC1_L0"
    assert seqs[0].tokens == ("A", "B")
    assert seqs[1].unit_id == "DOC1_L1"
    assert seqs[1].tokens == ("C", "D")


def test_positional_archetype_modulation() -> None:
    contexts = [
        InscriptionContext(
            document_id="TABLET_POS",
            script="linear_a",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["KU-RO", "GRAIN", "10"],
            ],
        ),
    ]
    encoder = EpigraphicContextEncoder(entropy_threshold_bits=2.5)
    encoder.fit_contexts(contexts)

    # Line-final token strongly boosts NUMERICAL_OR_FRACTION
    prof_final = encoder.project_token_archetype("10", position=TokenPosition.LINE_FINAL)
    assert not prof_final.abstained
    assert prof_final.archetype == "NUMERICAL_OR_FRACTION"

    # Line-initial token boosts SYNTACTIC_FORMULA_OR_VERB
    prof_initial = encoder.project_token_archetype("KU-RO", position=TokenPosition.LINE_INITIAL)
    assert not prof_initial.abstained
    assert prof_initial.archetype in (
        "SYNTACTIC_FORMULA_OR_VERB",
        "COMMODITY_RECORD",
        "PROPER_NAME_OR_PATRONYMIC",
    )


def test_epigraphic_embedding_model() -> None:
    contexts = [
        InscriptionContext(
            document_id="PY_01",
            script="linear_b",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["KO-WO", "VIR", "10"],
                ["KO-WA", "MUL", "15"],
                ["TO-SO", "WHEAT", "100"],
            ],
        ),
        InscriptionContext(
            document_id="HT_02",
            script="linear_a",
            genre=EpigraphicGenre.ADMINISTRATIVE_LEDGER,
            carrier=EpigraphicCarrier.CLAY_TABLET,
            lines=[
                ["KU-RO", "GRAIN", "50"],
                ["DA-DU-MA-TO", "GRAIN", "25"],
            ],
        ),
    ]

    model = EpigraphicEmbeddingModel(embedding_dim=4, window_size=2, seed=42)
    model.fit(contexts)

    # Check embedding retrieval
    emb_wheat = model.get_embedding("WHEAT")
    assert emb_wheat is not None
    assert emb_wheat.shape == (4,)

    emb_oov = model.get_embedding("NONEXISTENT_GLYPH")
    assert emb_oov is None

    # Check cosine similarity
    sim = model.cosine_similarity("WHEAT", "TO-SO")
    assert -1.0 <= sim <= 1.0

    # Check nearest neighbors
    neighbors = model.most_similar("WHEAT", top_k=3)
    assert len(neighbors) > 0
    assert all(isinstance(score, float) for _, score in neighbors)
