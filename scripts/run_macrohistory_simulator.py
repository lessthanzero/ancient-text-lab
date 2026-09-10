"""Execute Frontiers 6, 7, 8: Macro-Historical Intelligence Simulator CLI.

Demonstrates the Palantir for History platform:
- Frontier 6: Late Bronze Age Collapse SCM Simulation & Counterfactual Interventions.
- Frontier 7: Lost Text Reconstruction Lattice for ancient fragmentary works.
- Frontier 8: Ancient Macro-Network Topology Reconstruction from archaeological layers.
"""

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


def run_macrohistory_demo() -> None:
    print("=================================================================")
    print("   PALANTIR FOR HISTORY: MACRO-HISTORICAL CAUSAL INTELLIGENCE   ")
    print("=================================================================")

    # -------------------------------------------------------------------------
    # FRONTIER 6: CAUSAL LBA COLLAPSE ENGINE
    # -------------------------------------------------------------------------
    print("\n[FRONTIER 6] Late Bronze Age Collapse Polycrisis SCM (~1200 BCE):")
    engine = LateBronzeAgeCausalEngine()

    engine.add_node(CausalNode("megadrought", StressFactorType.CLIMATIC_MEGADROUGHT, 0.90, 0.10))
    engine.add_node(CausalNode("tin_copper_severance", StressFactorType.COMMODITY_SUPPLY_SEVERANCE, 0.30, 0.20))
    engine.add_node(CausalNode("sea_peoples_raids", StressFactorType.MIGRATORY_MILITARY_RAIDS, 0.40, 0.30))
    engine.add_node(CausalNode("palace_centralized_collapse", StressFactorType.INTERNAL_SYSTEMIC_FRAGILITY, 0.20, 0.35))

    engine.add_edge(CausalEdge("megadrought", "tin_copper_severance", 0.75, 2.0))
    engine.add_edge(CausalEdge("megadrought", "sea_peoples_raids", 0.85, 3.0))
    engine.add_edge(CausalEdge("tin_copper_severance", "palace_centralized_collapse", 0.90, 1.0))
    engine.add_edge(CausalEdge("sea_peoples_raids", "palace_centralized_collapse", 0.95, 1.0))

    sim_base = engine.simulate_crisis(years=20)
    print(f"  Baseline Simulation (20 Years): System Stress = {sim_base.final_system_stress:.2%}")
    print(f"  Collapse Triggered:            {sim_base.is_catastrophic_collapse}")
    print(f"  Primary Destabilizer Vector:   {sim_base.dominant_destabilizer}")

    # Counterfactual intervention: do(tin_copper_severance = 0.10)
    sim_cf = engine.simulate_crisis(years=20, interventions={"tin_copper_severance": 0.10})
    print(f"  Counterfactual do(Trade Relief): System Stress = {sim_cf.final_system_stress:.2%} (Stress Reduction: {sim_base.final_system_stress - sim_cf.final_system_stress:.2%})")

    # -------------------------------------------------------------------------
    # FRONTIER 7: LOST TEXT RECONSTRUCTION LATTICE
    # -------------------------------------------------------------------------
    print("\n[FRONTIER 7] Probabilistic Philological Fragment Lattice (The Lost Classics):")
    reconstructor = LostTextReconstructor("Manetho: Aegyptiaca (History of Egypt)", total_estimated_books=3)
    reconstructor.add_fragment(
        TextualFragment(
            fragment_id="f1",
            author="Manetho",
            work_title="Aegyptiaca",
            source_author="Josephus Contra Apionem",
            source_tier=FragmentSourceTier.VERBATIM_DIRECT_QUOTE,
            greek_or_original_text="Hyksos shepherds captured the land without a battle...",
            estimated_book_number=2,
            metrical_scheme=None,
        )
    )
    reconstructor.add_fragment(
        TextualFragment(
            fragment_id="f2",
            author="Manetho",
            work_title="Aegyptiaca",
            source_author="Eusebius Chronicon",
            source_tier=FragmentSourceTier.PARAPHRASE_OR_SUMMARY,
            greek_or_original_text="Dynasty 1 to 11 kings list and regnal totals...",
            estimated_book_number=1,
            metrical_scheme=None,
        )
    )
    lattice = reconstructor.compile_lattice()
    print(f"  Lost Masterpiece:              {lattice.work_title}")
    print(f"  Estimated Architectural Books: {lattice.total_estimated_books}")
    print(f"  Attested Preserved Fragments:  {lattice.attested_fragments_count}")
    print(f"  Estimated Textual Recovery:    {lattice.estimated_survival_ratio:.2%}")

    # -------------------------------------------------------------------------
    # FRONTIER 8: ANCIENT MACRO-NETWORK TOPOLOGY
    # -------------------------------------------------------------------------
    print("\n[FRONTIER 8] Mediterranean Maritime Hypergraph (~1350 BCE):")
    net = MacroNetworkReconstructor()
    nodes = [
        TradeNode("Mycenae", "Aegean", (37.73, 22.75), "Palace"),
        TradeNode("Knossos", "Crete", (35.29, 25.16), "Palace"),
        TradeNode("Enkomi", "Cyprus", (35.15, 33.88), "Mining_Harbor"),
        TradeNode("Ugarit", "Levant", (35.60, 35.78), "Harbor_Emporium"),
        TradeNode("Amarna", "Egypt", (27.64, 30.90), "Imperial_Capital"),
        TradeNode("Hattusa", "Anatolia", (40.01, 34.61), "Inland_Capital"),
    ]
    for n in nodes:
        net.add_node(n)

    # Add verified archaeological routes
    net.add_edge(MultimodalTradeEdge("Enkomi", "Ugarit", (EvidenceDataType.LEAD_ISOTOPE_ANALYSIS,), ("copper_oxhides",), 9.8))
    net.add_edge(MultimodalTradeEdge("Mycenae", "Ugarit", (EvidenceDataType.CERAMIC_PETROGRAPHY, EvidenceDataType.SHIPWRECK_CARGO), ("lh_iiib_vessels", "olive_oil"), 8.4))
    net.add_edge(MultimodalTradeEdge("Ugarit", "Amarna", (EvidenceDataType.EPIGRAPHIC_CORRESPONDENCE,), ("diplomatic_gifts", "cedar"), 9.1))
    net.add_edge(MultimodalTradeEdge("Knossos", "Amarna", (EvidenceDataType.CERAMIC_PETROGRAPHY,), ("minoan_faience",), 6.5))

    metrics = net.compute_network_metrics()
    print(f"  Total Emporia & Capitals:      {metrics['total_emporia_nodes']}")
    print(f"  Provenanced Evidence Routes:   {metrics['total_provenanced_routes']}")
    print(f"  Critical Network Chokepoint:   {metrics['primary_trade_hubs'][0][0]} (Weighted flow: {metrics['primary_trade_hubs'][0][1]:.1f})")
    print("=================================================================")


if __name__ == "__main__":
    run_macrohistory_demo()
