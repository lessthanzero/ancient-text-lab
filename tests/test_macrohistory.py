"""Tests for Macro-Historical Intelligence Engines (Palantir for History)."""

from __future__ import annotations

from ancient_text_lab.macrohistory import (
    CausalEdge,
    CausalNode,
    EvidenceDataType,
    FragmentSourceTier,
    LateBronzeAgeCausalEngine,
    LostTextReconstructor,
    MacroNetworkReconstructor,
    MultimodalTradeEdge,
    StressFactorType,
    TextualFragment,
    TradeNode,
)


def test_late_bronze_age_causal_collapse_simulation() -> None:
    engine = LateBronzeAgeCausalEngine()

    # Define key nodes in the 1200 BCE Mediterranean system
    engine.add_node(
        CausalNode(
            node_id="megadrought_climate",
            factor_type=StressFactorType.CLIMATIC_MEGADROUGHT,
            baseline_stress=0.85,
            resilience_capacity=0.10,
        )
    )
    engine.add_node(
        CausalNode(
            node_id="palace_grain_supply",
            factor_type=StressFactorType.COMMODITY_SUPPLY_SEVERANCE,
            baseline_stress=0.20,
            resilience_capacity=0.30,
        )
    )
    engine.add_node(
        CausalNode(
            node_id="mycenaean_palace_economy",
            factor_type=StressFactorType.INTERNAL_SYSTEMIC_FRAGILITY,
            baseline_stress=0.25,
            resilience_capacity=0.40,
        )
    )

    # Climate shock severely degrades grain supply, which topples palace economy
    engine.add_edge(
        CausalEdge(
            source_id="megadrought_climate",
            target_id="palace_grain_supply",
            coupling_weight=0.90,
            delay_years=2.0,
        )
    )
    engine.add_edge(
        CausalEdge(
            source_id="palace_grain_supply",
            target_id="mycenaean_palace_economy",
            coupling_weight=0.85,
            delay_years=3.0,
        )
    )

    # Baseline simulation: cascading collapse
    res_base = engine.simulate_crisis(years=15)
    assert res_base.final_system_stress > 0.50

    # Counterfactual intervention: do(palace_grain_supply = 0.0) via maritime relief
    res_intervened = engine.simulate_crisis(
        years=15,
        interventions={"palace_grain_supply": 0.0},
    )
    # Mitigated stress under intervention
    assert res_intervened.final_system_stress < res_base.final_system_stress


def test_lost_text_reconstruction_lattice() -> None:
    reconstructor = LostTextReconstructor(work_title="Epic Cycle: Cypria", total_estimated_books=11)

    frag1 = TextualFragment(
        fragment_id="cypria_f1",
        author="Stasinus",
        work_title="Cypria",
        source_author="Proclus Chrestomathy",
        source_tier=FragmentSourceTier.PARAPHRASE_OR_SUMMARY,
        greek_or_original_text="Zeus plans the Trojan war to relieve Earth of overpopulation.",
        estimated_book_number=1,
        metrical_scheme="dactylic_hexameter",
    )
    frag2 = TextualFragment(
        fragment_id="cypria_f2",
        author="Stasinus",
        work_title="Cypria",
        source_author="Athenaeus Deipnosophistae",
        source_tier=FragmentSourceTier.VERBATIM_DIRECT_QUOTE,
        greek_or_original_text="ei gar tis thneton phresin elpetai...",
        estimated_book_number=1,
        metrical_scheme="dactylic_hexameter",
    )

    reconstructor.add_fragment(frag1)
    reconstructor.add_fragment(frag2)

    lattice = reconstructor.compile_lattice()
    assert lattice.work_title == "Epic Cycle: Cypria"
    assert lattice.total_estimated_books == 11
    assert lattice.attested_fragments_count == 2
    assert lattice.estimated_survival_ratio > 0.0


def test_ancient_macro_network_topology() -> None:
    net = MacroNetworkReconstructor()

    # Add core Mediterranean nodes circa 1300 BCE
    net.add_node(
        TradeNode(
            node_id="Mycenae", region="Aegean", coordinates=(37.73, 22.75), settlement_type="Palace"
        )
    )
    net.add_node(
        TradeNode(
            node_id="Enkomi",
            region="Cyprus",
            coordinates=(35.15, 33.88),
            settlement_type="Mining_Harbor",
        )
    )
    net.add_node(
        TradeNode(
            node_id="Ugarit",
            region="Levant",
            coordinates=(35.60, 35.78),
            settlement_type="Harbor_Emporium",
        )
    )

    # Add multimodal trade edges
    net.add_edge(
        MultimodalTradeEdge(
            source_node="Enkomi",
            target_node="Ugarit",
            evidence_types=(
                EvidenceDataType.LEAD_ISOTOPE_ANALYSIS,
                EvidenceDataType.EPIGRAPHIC_CORRESPONDENCE,
            ),
            commodity_types=("copper_oxhide_ingots", "grain"),
            empirical_weight=9.5,
        )
    )
    net.add_edge(
        MultimodalTradeEdge(
            source_node="Mycenae",
            target_node="Ugarit",
            evidence_types=(EvidenceDataType.CERAMIC_PETROGRAPHY, EvidenceDataType.SHIPWRECK_CARGO),
            commodity_types=("lh_iiib_pottery", "perfumed_oil"),
            empirical_weight=7.2,
        )
    )

    metrics = net.compute_network_metrics()
    assert metrics["total_emporia_nodes"] == 3
    assert metrics["total_provenanced_routes"] == 2
    assert metrics["average_connectivity_degree"] > 1.0
    assert len(metrics["primary_trade_hubs"]) > 0
    # Ugarit is the primary hub connected to both Enkomi and Mycenae
    assert metrics["primary_trade_hubs"][0][0] == "Ugarit"
