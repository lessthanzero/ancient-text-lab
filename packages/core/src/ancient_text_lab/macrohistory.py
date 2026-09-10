"""Macro-Historical Intelligence Engines (Palantir for History).

Frontiers 6, 7, and 8:
- Frontier 6: Causal Modeling of the Late Bronze Age Collapse (Polycrisis SCMs).
- Frontier 7: Philological Fragment Lattice & Lost Text Reconstruction.
- Frontier 8: Multimodal Macro-Network Topology Reconstruction (1500-1100 BCE).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# =============================================================================
# FRONTIER 6: CAUSAL MODELING OF THE LATE BRONZE AGE COLLAPSE (~1200 BCE)
# =============================================================================


class StressFactorType(str, Enum):
    """Coupled stress drivers in the Late Bronze Age polycrisis."""

    CLIMATIC_MEGADROUGHT = "climatic_megadrought"  # Speleothem / pollen aridification
    COMMODITY_SUPPLY_SEVERANCE = "commodity_severance"  # Tin / copper / grain blockage
    INTERNAL_SYSTEMIC_FRAGILITY = "internal_fragility"  # Hyper-centralized palace bureaucracy
    MIGRATORY_MILITARY_RAIDS = "migratory_raids"  # 'Sea Peoples' / Lukka / Shardana
    SEISMIC_DESTRUCTION = "seismic_destruction"  # Earthquake storms (Schaeffer/Nur)
    EPIDEMIC_OUTBREAK = "epidemic_outbreak"  # Hittite plague / urban epidemics


@dataclass(frozen=True, slots=True)
class CausalNode:
    """A variable node in the macro-historical causal Directed Acyclic Graph (DAG)."""

    node_id: str
    factor_type: StressFactorType
    baseline_stress: float  # [0.0, 1.0]
    resilience_capacity: float  # [0.0, 1.0]


@dataclass(frozen=True, slots=True)
class CausalEdge:
    """A directed causal influence link with transmission weight."""

    source_id: str
    target_id: str
    coupling_weight: float  # [0.0, 1.0]
    delay_years: float


@dataclass(frozen=True, slots=True)
class CollapseSimulationResult:
    """Result of a dynamic stress propagation or counterfactual do-calculus run."""

    simulation_id: str
    time_steps_years: int
    final_system_stress: float
    is_catastrophic_collapse: bool
    dominant_destabilizer: str
    node_trajectories: Mapping[str, Sequence[float]]


class LateBronzeAgeCausalEngine:
    """Structural Causal Model (SCM) simulating the Mediterranean polycrisis."""

    def __init__(self) -> None:
        self.nodes: dict[str, CausalNode] = {}
        self.edges: list[CausalEdge] = []

    def add_node(self, node: CausalNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: CausalEdge) -> None:
        self.edges.append(edge)

    def simulate_crisis(
        self,
        *,
        years: int = 50,
        interventions: Mapping[str, float] | None = None,
    ) -> CollapseSimulationResult:
        """Run dynamic stress propagation over time with optional counterfactual interventions."""
        interventions = interventions or {}

        # Initialize stress state
        current_stress: dict[str, float] = {}
        for nid, node in self.nodes.items():
            current_stress[nid] = interventions.get(nid, node.baseline_stress)

        trajectories: dict[str, list[float]] = {nid: [val] for nid, val in current_stress.items()}

        for _ in range(years):
            next_stress = dict(current_stress)

            for edge in self.edges:
                src_val = current_stress.get(edge.source_id, 0.0)
                # Stress propagates non-linearly if it exceeds resilience
                target_node = self.nodes[edge.target_id]
                effective_impact = max(0.0, src_val - target_node.resilience_capacity * 0.5)
                transmitted = effective_impact * edge.coupling_weight
                next_stress[edge.target_id] = min(
                    1.0, next_stress[edge.target_id] + transmitted * 0.2
                )

            # Apply hard do(X) interventions
            for nid, val in interventions.items():
                next_stress[nid] = val

            current_stress = next_stress
            for nid, val in current_stress.items():
                trajectories[nid].append(val)

        # Evaluate aggregate system state
        mean_stress = float(np.mean(list(current_stress.values())))
        dominant = max(current_stress.items(), key=lambda item: item[1])[0]

        return CollapseSimulationResult(
            simulation_id="LBA_1200_BCE_SIM",
            time_steps_years=years,
            final_system_stress=mean_stress,
            is_catastrophic_collapse=mean_stress > 0.70,
            dominant_destabilizer=dominant,
            node_trajectories=trajectories,
        )


# =============================================================================
# FRONTIER 7: PHILOLOGICAL FRAGMENT RECONSTRUCTION ENGINE
# =============================================================================


class FragmentSourceTier(str, Enum):
    """Reliability tier for fragments of lost ancient works."""

    VERBATIM_DIRECT_QUOTE = "verbatim_quote"  # e.g. Athenaeus quoting Sophocles
    PARAPHRASE_OR_SUMMARY = "paraphrase_summary"  # e.g. Proclus summary of Epic Cycle
    SCHOLIA_MARGINALIA = "scholia_marginalia"  # Commentary on Homer citing lost work
    PAPROLOGICAL_SCRAP = "papyrological_scrap"  # Damaged papyrus physical fragment


@dataclass(frozen=True, slots=True)
class TextualFragment:
    """An attested fragment of a lost ancient composition."""

    fragment_id: str
    author: str
    work_title: str
    source_author: str  # e.g., "Athenaeus", "Eusebius", "Stobaeus"
    source_tier: FragmentSourceTier
    greek_or_original_text: str
    estimated_book_number: int | None
    metrical_scheme: str | None


@dataclass(frozen=True, slots=True)
class LostWorkReconstructionLattice:
    """Probabilistic structural skeleton of an ancient lost book."""

    work_title: str
    total_estimated_books: int
    attested_fragments_count: int
    reconstructed_outline: Sequence[tuple[str, Sequence[TextualFragment]]]
    estimated_survival_ratio: float


class LostTextReconstructor:
    """Reconstructs the probable architecture of lost classical works from fragment distributions."""

    def __init__(self, work_title: str, total_estimated_books: int = 1) -> None:
        self.work_title = work_title
        self.total_estimated_books = total_estimated_books
        self.fragments: list[TextualFragment] = []

    def add_fragment(self, fragment: TextualFragment) -> None:
        self.fragments.append(fragment)

    def compile_lattice(self) -> LostWorkReconstructionLattice:
        """Partition fragments into structural sequence and estimate surviving volume."""
        book_bins: dict[str, list[TextualFragment]] = {}
        for b in range(1, self.total_estimated_books + 1):
            book_bins[f"Book_{b}"] = []

        unassigned: list[TextualFragment] = []
        for frag in self.fragments:
            if frag.estimated_book_number and f"Book_{frag.estimated_book_number}" in book_bins:
                book_bins[f"Book_{frag.estimated_book_number}"].append(frag)
            else:
                unassigned.append(frag)

        outline = [(k, tuple(v)) for k, v in book_bins.items()]
        if unassigned:
            outline.append(("Unassigned_Fragments", tuple(unassigned)))

        # Rough survival volume calculation: typical ancient book ~ 800-1000 lines
        est_total_lines = self.total_estimated_books * 900
        surviving_words = sum(len(f.greek_or_original_text.split()) for f in self.fragments)
        est_surviving_lines = surviving_words / 6.0  # Approx 6 words per hexameter line
        ratio = min(1.0, est_surviving_lines / max(1.0, float(est_total_lines)))

        return LostWorkReconstructionLattice(
            work_title=self.work_title,
            total_estimated_books=self.total_estimated_books,
            attested_fragments_count=len(self.fragments),
            reconstructed_outline=outline,
            estimated_survival_ratio=float(ratio),
        )


# =============================================================================
# FRONTIER 8: ANCIENT MACRO-NETWORK TOPOLOGY RECONSTRUCTION (1500–1100 BCE)
# =============================================================================


class EvidenceDataType(str, Enum):
    """Archaeological and scientific evidence sources for maritime macro-networks."""

    LEAD_ISOTOPE_ANALYSIS = "lead_isotope"  # Traces copper oxhide ingots to mines
    CERAMIC_PETROGRAPHY = "ceramic_petrography"  # Clay fabric origins (e.g. Mycenaean LH IIIB)
    SHIPWRECK_CARGO = "shipwreck_cargo"  # Direct snapshot of maritime transit (Uluburun)
    EPIGRAPHIC_CORRESPONDENCE = "epigraphic_text"  # Amarna tablets, Ugaritic archives
    METEOROLOGICAL_CURRENTS = "ocean_currents"  # Prevailing winds & counter-clockwise currents


@dataclass(frozen=True, slots=True)
class TradeNode:
    """An ancient emporium, palace center, or harbor node."""

    node_id: str
    region: str  # e.g., "Aegean", "Cyprus", "Levant", "Egypt"
    coordinates: tuple[float, float]  # (latitude, longitude)
    settlement_type: str  # e.g., "Palace", "Harbor_Emporium", "Mining_Center"


@dataclass(frozen=True, slots=True)
class MultimodalTradeEdge:
    """A provenanced interaction route between two Bronze Age centers."""

    source_node: str
    target_node: str
    evidence_types: tuple[EvidenceDataType, ...]
    commodity_types: tuple[str, ...]
    empirical_weight: float  # Relative interaction volume supported by physical data


class MacroNetworkReconstructor:
    """Reconstructs the Bronze Age Mediterranean trade hypergraph from multi-modal evidence."""

    def __init__(self) -> None:
        self.nodes: dict[str, TradeNode] = {}
        self.edges: list[MultimodalTradeEdge] = []

    def add_node(self, node: TradeNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: MultimodalTradeEdge) -> None:
        self.edges.append(edge)

    def compute_network_metrics(self) -> dict[str, Any]:
        """Calculate network connectivity, trade centrality, and vulnerability bottlenecks."""
        degrees: dict[str, int] = {nid: 0 for nid in self.nodes}
        weighted_flow: dict[str, float] = {nid: 0.0 for nid in self.nodes}

        for edge in self.edges:
            degrees[edge.source_node] = degrees.get(edge.source_node, 0) + 1
            degrees[edge.target_node] = degrees.get(edge.target_node, 0) + 1
            weighted_flow[edge.source_node] = (
                weighted_flow.get(edge.source_node, 0.0) + edge.empirical_weight
            )
            weighted_flow[edge.target_node] = (
                weighted_flow.get(edge.target_node, 0.0) + edge.empirical_weight
            )

        # Top hub centers by weighted trade flow
        ranked_hubs = sorted(weighted_flow.items(), key=lambda item: item[1], reverse=True)

        return {
            "total_emporia_nodes": len(self.nodes),
            "total_provenanced_routes": len(self.edges),
            "primary_trade_hubs": ranked_hubs[:5],
            "average_connectivity_degree": float(np.mean(list(degrees.values())))
            if degrees
            else 0.0,
        }
