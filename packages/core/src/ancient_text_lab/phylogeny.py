"""The Rosetta Engine: Cross-script phylogenetic graph & transmission modeling.

Models evolutionary transmission lineages across ancient writing systems,
evaluating cross-script sign homologues with dual-anchor constraints on known deciphered endpoints.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScriptNode:
    """A writing system node in the phylogenetic tree."""

    id: str
    name: str
    chronology_bce: tuple[int, int]  # (start_bce, end_bce) e.g. (-1800, -1450)
    is_deciphered: bool
    canonical_sign_count: int
    primary_language: str | None = None  # e.g. "Mycenaean Greek" or None if unknown


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """A historical transmission edge between a source script and daughter script."""

    source_script_id: str
    target_script_id: str
    palaeographic_continuity: float  # [0.0, 1.0]
    historical_notes: str


@dataclass(frozen=True, slots=True)
class SignHomologue:
    """A cross-script sign correspondence tracking ductus and phonetic projections."""

    canonical_name: str
    sign_ids: Mapping[
        str, str
    ]  # script_id -> sign_identifier (e.g. {"linear_a": "AB08", "linear_b": "B08"})
    pictorial_origin: str
    stroke_counts: Mapping[str, int]  # script_id -> stroke count
    ground_truth_phonetics: Mapping[str, str]  # readings on deciphered endpoints
    projected_phonetics: Mapping[str, str]  # projected readings on undeciphered nodes
    citations: tuple[str, ...]
    confidence_tier: str = "E4 (Direct Formal Homology)"


@dataclass(frozen=True, slots=True)
class PhylogeneticAuditReport:
    """Audit report of cross-script alignment and dual-anchor consistency."""

    total_homologues: int
    scripts_in_network: int
    deciphered_anchor_count: int
    undeciphered_node_count: int
    mean_stroke_reduction_pct: float
    dual_anchor_consistency_pct: float
    verified_homologue_count: int
    epistemic_notes: str


@dataclass(frozen=True, slots=True)
class PropagatedReading:
    """Inferred phonetic reading on an intermediate undeciphered script node."""

    canonical_name: str
    target_script_id: str
    target_sign_id: str
    inferred_phonetic: str
    confidence: float
    status: str  # "DUAL_ANCHOR_VERIFIED", "SINGLE_ANCHOR_PROJECTED", "DIVERGENT_BRANCHES"
    anchors_consulted: tuple[str, ...]


def calculate_cursive_drift(source_strokes: int, target_strokes: int) -> float:
    """Calculate the stroke reduction percentage from parent to daughter sign."""
    if source_strokes <= 0:
        return 0.0
    return (source_strokes - target_strokes) / source_strokes


class ScriptPhylogenyGraph:
    """Phylogenetic graph orchestrating cross-script alignment and anchor propagation."""

    def __init__(
        self,
        scripts: Iterable[ScriptNode],
        lineages: Iterable[LineageEdge],
        homologues: Iterable[SignHomologue] = (),
    ) -> None:
        self._scripts: dict[str, ScriptNode] = {s.id: s for s in scripts}
        self._lineages: list[LineageEdge] = list(lineages)
        self._homologues: list[SignHomologue] = list(homologues)

        # Validate lineage references
        for edge in self._lineages:
            if edge.source_script_id not in self._scripts:
                raise ValueError(f"unknown source script: {edge.source_script_id}")
            if edge.target_script_id not in self._scripts:
                raise ValueError(f"unknown target script: {edge.target_script_id}")

    @property
    def scripts(self) -> Mapping[str, ScriptNode]:
        return self._scripts

    @property
    def lineages(self) -> Sequence[LineageEdge]:
        return tuple(self._lineages)

    @property
    def homologues(self) -> Sequence[SignHomologue]:
        return tuple(self._homologues)

    def add_homologue(self, homologue: SignHomologue) -> None:
        self._homologues.append(homologue)

    def audit_alignment(
        self,
        *,
        primary_anchor_id: str = "linear_b",
        secondary_anchor_id: str = "cypriot_syllabic",
    ) -> PhylogeneticAuditReport:
        """Audit the network of homologues against known deciphered anchor scripts."""
        deciphered = [s for s in self._scripts.values() if s.is_deciphered]
        undeciphered = [s for s in self._scripts.values() if not s.is_deciphered]

        if not self._homologues:
            return PhylogeneticAuditReport(
                total_homologues=0,
                scripts_in_network=len(self._scripts),
                deciphered_anchor_count=len(deciphered),
                undeciphered_node_count=len(undeciphered),
                mean_stroke_reduction_pct=0.0,
                dual_anchor_consistency_pct=0.0,
                verified_homologue_count=0,
                epistemic_notes="No homologues registered in network.",
            )

        dual_anchor_checks = 0
        dual_anchor_matches = 0
        stroke_reductions = []
        verified_count = 0

        for h in self._homologues:
            # Check dual anchors if present
            has_primary = primary_anchor_id in h.ground_truth_phonetics
            has_secondary = secondary_anchor_id in h.ground_truth_phonetics

            if has_primary and has_secondary:
                dual_anchor_checks += 1
                val1 = h.ground_truth_phonetics[primary_anchor_id].lower()
                val2 = h.ground_truth_phonetics[secondary_anchor_id].lower()
                if val1 == val2:
                    dual_anchor_matches += 1

            # Check stroke reduction between earlier and later scripts
            for edge in self._lineages:
                src_count = h.stroke_counts.get(edge.source_script_id)
                tgt_count = h.stroke_counts.get(edge.target_script_id)
                if src_count and tgt_count and src_count > 0:
                    stroke_reductions.append(calculate_cursive_drift(src_count, tgt_count))

            if len(h.citations) > 0 and len(h.ground_truth_phonetics) > 0:
                verified_count += 1

        dual_pct = (
            (dual_anchor_matches / dual_anchor_checks * 100.0) if dual_anchor_checks > 0 else 100.0
        )
        mean_reduction = (
            (sum(stroke_reductions) / len(stroke_reductions) * 100.0) if stroke_reductions else 0.0
        )

        notes = (
            f"Phylogenetic network contains {len(self._scripts)} scripts and {len(self._homologues)} "
            f"homologues. Mean cursive stroke reduction across lineages: {mean_reduction:.1f}%."
        )

        return PhylogeneticAuditReport(
            total_homologues=len(self._homologues),
            scripts_in_network=len(self._scripts),
            deciphered_anchor_count=len(deciphered),
            undeciphered_node_count=len(undeciphered),
            mean_stroke_reduction_pct=mean_reduction,
            dual_anchor_consistency_pct=dual_pct,
            verified_homologue_count=verified_count,
            epistemic_notes=notes,
        )

    def propagate_dual_anchors(
        self,
        target_script_id: str,
        *,
        primary_anchor_id: str = "linear_b",
        secondary_anchor_id: str = "cypriot_syllabic",
    ) -> tuple[PropagatedReading, ...]:
        """Propagate phonetic values from deciphered boundary anchors onto an intermediate undeciphered script."""
        if target_script_id not in self._scripts:
            raise ValueError(f"unknown target script: {target_script_id}")

        results: list[PropagatedReading] = []

        for h in self._homologues:
            target_sign = h.sign_ids.get(target_script_id)
            if not target_sign:
                continue

            has_primary = primary_anchor_id in h.ground_truth_phonetics
            has_secondary = secondary_anchor_id in h.ground_truth_phonetics

            val_primary = h.ground_truth_phonetics.get(primary_anchor_id)
            val_secondary = h.ground_truth_phonetics.get(secondary_anchor_id)

            if has_primary and has_secondary:
                if val_primary.lower() == val_secondary.lower():  # type: ignore[union-attr]
                    # Perfect dual-anchor agreement
                    results.append(
                        PropagatedReading(
                            canonical_name=h.canonical_name,
                            target_script_id=target_script_id,
                            target_sign_id=target_sign,
                            inferred_phonetic=val_primary.lower(),  # type: ignore[union-attr]
                            confidence=0.98,
                            status="DUAL_ANCHOR_VERIFIED",
                            anchors_consulted=(primary_anchor_id, secondary_anchor_id),
                        )
                    )
                else:
                    # Divergence between branches
                    results.append(
                        PropagatedReading(
                            canonical_name=h.canonical_name,
                            target_script_id=target_script_id,
                            target_sign_id=target_sign,
                            inferred_phonetic=f"{val_primary.lower()}/{val_secondary.lower()}",  # type: ignore[union-attr]
                            confidence=0.45,
                            status="DIVERGENT_BRANCHES",
                            anchors_consulted=(primary_anchor_id, secondary_anchor_id),
                        )
                    )
            elif has_primary:
                # Single Aegean anchor
                results.append(
                    PropagatedReading(
                        canonical_name=h.canonical_name,
                        target_script_id=target_script_id,
                        target_sign_id=target_sign,
                        inferred_phonetic=val_primary.lower(),  # type: ignore[union-attr]
                        confidence=0.85,
                        status="SINGLE_ANCHOR_PROJECTED",
                        anchors_consulted=(primary_anchor_id,),
                    )
                )
            elif has_secondary:
                # Single Cypriot anchor
                results.append(
                    PropagatedReading(
                        canonical_name=h.canonical_name,
                        target_script_id=target_script_id,
                        target_sign_id=target_sign,
                        inferred_phonetic=val_secondary.lower(),  # type: ignore[union-attr]
                        confidence=0.80,
                        status="SINGLE_ANCHOR_PROJECTED",
                        anchors_consulted=(secondary_anchor_id,),
                    )
                )

        return tuple(results)


def load_phylogeny_dataset(path: Path) -> ScriptPhylogenyGraph:
    """Load a canonical phylogeny dataset JSON file into a ScriptPhylogenyGraph."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    scripts = [
        ScriptNode(
            id=s["id"],
            name=s["name"],
            chronology_bce=tuple(s["chronology_bce"]),  # type: ignore[arg-type]
            is_deciphered=s["is_deciphered"],
            canonical_sign_count=s["canonical_sign_count"],
            primary_language=s.get("primary_language"),
        )
        for s in raw["scripts"]
    ]

    lineages = [
        LineageEdge(
            source_script_id=e["source_script_id"],
            target_script_id=e["target_script_id"],
            palaeographic_continuity=e["palaeographic_continuity"],
            historical_notes=e["historical_notes"],
        )
        for e in raw["lineages"]
    ]

    homologues = [
        SignHomologue(
            canonical_name=h["canonical_name"],
            sign_ids=h["sign_ids"],
            pictorial_origin=h["pictorial_origin"],
            stroke_counts=h["stroke_counts"],
            ground_truth_phonetics=h["ground_truth_phonetics"],
            projected_phonetics=h.get("projected_phonetics", {}),
            citations=tuple(h["citations"]),
            confidence_tier=h.get("confidence_tier", "E4 (Direct Formal Homology)"),
        )
        for h in raw["homologues"]
    ]

    return ScriptPhylogenyGraph(scripts, lineages, homologues)


@dataclass(frozen=True, slots=True)
class BeliefPropagationInferenceReport:
    """Rigorous probabilistic inference report for an undeciphered sign under MRF message passing."""

    canonical_name: str
    target_script_id: str
    target_sign_id: str
    map_phonetic: str
    map_confidence: float
    marginal_distribution: Mapping[str, float]
    posterior_entropy_bits: float
    anchors_active: tuple[str, ...]
    is_converged: bool


def infer_phonetics_mrf(
    graph: ScriptPhylogenyGraph,
    target_script_id: str,
    *,
    max_iterations: int = 20,
    convergence_tol: float = 1e-5,
) -> tuple[BeliefPropagationInferenceReport, ...]:
    """Infer phonetic posteriors using Markov Random Field Belief Propagation across the transmission DAG.

    Solves the dual-anchor boundary value problem simultaneously across Aegean and Cypriot lineages.
    """
    import math

    if target_script_id not in graph.scripts:
        raise ValueError(f"unknown target script: {target_script_id}")

    reports: list[BeliefPropagationInferenceReport] = []

    for h in graph.homologues:
        target_sign = h.sign_ids.get(target_script_id)
        if not target_sign:
            continue

        # Collect state space for this homologue: all candidate readings across ground truth + projected
        state_set = set()
        for v in h.ground_truth_phonetics.values():
            state_set.add(v.lower())
        for v in h.projected_phonetics.values():
            state_set.add(v.lower())

        if not state_set:
            continue

        states = sorted(state_set)
        s_dim = len(states)
        s_idx = {s: i for i, s in enumerate(states)}

        # Identify nodes present in this homologue
        present_nodes = [s_id for s_id in graph.scripts if s_id in h.sign_ids]

        # Initial node potentials psi_i(x)
        potentials: dict[str, list[float]] = {}
        active_anchors: list[str] = []

        for node_id in present_nodes:
            is_anchor = graph.scripts[node_id].is_deciphered and node_id in h.ground_truth_phonetics
            if is_anchor:
                active_anchors.append(node_id)
                gt_val = h.ground_truth_phonetics[node_id].lower()
                pot = [1e-4] * s_dim
                if gt_val in s_idx:
                    pot[s_idx[gt_val]] = 0.999
                # Normalize
                tot = sum(pot)
                potentials[node_id] = [p / tot for p in pot]
            else:
                # Undeciphered: uniform prior
                potentials[node_id] = [1.0 / s_dim] * s_dim

        # Lineages present in this homologue's sub-graph
        active_edges = [
            e
            for e in graph.lineages
            if e.source_script_id in potentials and e.target_script_id in potentials
        ]

        # Initialize messages m_{u -> v} = uniform
        messages: dict[tuple[str, str], list[float]] = {}
        for e in active_edges:
            messages[(e.source_script_id, e.target_script_id)] = [1.0 / s_dim] * s_dim
            messages[(e.target_script_id, e.source_script_id)] = [1.0 / s_dim] * s_dim

        # Belief propagation message passing
        converged = False
        for _ in range(max_iterations):
            max_delta = 0.0
            new_messages: dict[tuple[str, str], list[float]] = {}

            for (u, v), m_old in messages.items():
                # Incoming product: psi_u(x_u) * prod_{k in N(u) \ {v}} m_{k -> u}(x_u)
                prod = list(potentials[u])
                for (k, target_node), m_in in messages.items():
                    if target_node == u and k != v:
                        for s_i in range(s_dim):
                            prod[s_i] *= m_in[s_i]

                # Transmission potential: psi(x_u, x_v)
                # Diagonal = high continuity, off-diagonal = low transition
                m_new = [0.0] * s_dim
                for j in range(s_dim):
                    val_j = 0.0
                    for i in range(s_dim):
                        # Transition matrix
                        if i == j:
                            t_prob = 0.90
                        elif {states[i], states[j]} == {"ro", "lo"}:
                            t_prob = 0.60  # Known liquid shift
                        else:
                            t_prob = 0.10 / max(1, s_dim - 1)
                        val_j += prod[i] * t_prob
                    m_new[j] = val_j

                # Normalize message
                tot_m = sum(m_new)
                if tot_m > 0:
                    m_new = [m / tot_m for m in m_new]

                delta = max(abs(a - b) for a, b in zip(m_new, m_old, strict=True))
                max_delta = max(max_delta, delta)
                new_messages[(u, v)] = m_new

            messages = new_messages
            if max_delta < convergence_tol:
                converged = True
                break

        # Compute marginal at target_script_id
        marginal = list(potentials[target_script_id])
        for (k, target_node), m_in in messages.items():
            if target_node == target_script_id:
                for s_i in range(s_dim):
                    marginal[s_i] *= m_in[s_i]

        tot_marg = sum(marginal)
        if tot_marg > 0:
            marginal = [m / tot_marg for m in marginal]

        # MAP state
        best_i = int(max(range(s_dim), key=lambda i: marginal[i]))
        best_state = states[best_i]
        best_conf = marginal[best_i]

        # Posterior entropy in bits
        entropy = -sum(p * math.log2(p) for p in marginal if p > 0)

        marg_dict = {states[i]: round(marginal[i], 4) for i in range(s_dim)}

        reports.append(
            BeliefPropagationInferenceReport(
                canonical_name=h.canonical_name,
                target_script_id=target_script_id,
                target_sign_id=target_sign,
                map_phonetic=best_state,
                map_confidence=best_conf,
                marginal_distribution=marg_dict,
                posterior_entropy_bits=entropy,
                anchors_active=tuple(sorted(active_anchors)),
                is_converged=converged,
            )
        )

    return tuple(reports)
