"""Synthetic damage masking protocols for epigraphic restoration benchmarks.

Generates realistic damage masks (isolated lacunae, broken tablet edges) on complete
documents without leaking withheld gold answers into model-visible cases.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from ancient_text_lab.evaluation import BenchmarkCase, GoldLabel
from ancient_text_lab.sequence import TokenSequence


class DamageType(str, Enum):
    """Realistic physical degradation patterns observed on clay and stone."""

    ISOLATED_LACUNA = "isolated_lacuna"  # Pinhole abrasion / gouge: 1 token lost
    EDGE_FRACTURE = "edge_fracture"  # Broken tablet corner: consecutive edge tokens lost


@dataclass(frozen=True, slots=True)
class MaskedCorpus:
    """A masked corpus prepared for benchmark evaluation without gold leakage."""

    masked_sequences: tuple[TokenSequence, ...]
    benchmark_cases: tuple[BenchmarkCase, ...]
    gold_labels: tuple[GoldLabel, ...]


def generate_synthetic_lacunae(
    sequences: Iterable[TokenSequence],
    *,
    source_artifact_id: str,
    split: str = "test",
    mask_rate: float = 0.15,
    edge_fracture_prob: float = 0.30,
    seed: int = 42,
    mask_token: str = "[LACUNA]",
) -> MaskedCorpus:
    """Create realistic synthetic damage masks across token sequences.

    Produces model-visible BenchmarkCases (containing only surrounding context)
    and separate, withheld GoldLabels.
    """
    rng = random.Random(seed)
    materialized = tuple(sequences)

    masked_seqs: list[TokenSequence] = []
    cases: list[BenchmarkCase] = []
    gold: list[GoldLabel] = []

    case_counter = 0

    for seq in materialized:
        tokens = list(seq.tokens)
        n = len(tokens)
        if n == 0:
            continue

        mask_indices: set[int] = set()

        # Decide whether to simulate an edge fracture on this sequence
        if rng.random() < edge_fracture_prob and n >= 3:
            # 1 to 2 tokens from start or end
            span = rng.randint(1, min(2, n - 1))
            if rng.random() < 0.5:
                # Start fracture
                mask_indices.update(range(span))
            else:
                # End fracture
                mask_indices.update(range(n - span, n))

        # Additional isolated random sign masks
        for i in range(n):
            if i not in mask_indices and rng.random() < mask_rate:
                mask_indices.add(i)

        new_tokens = list(tokens)
        for idx in sorted(mask_indices):
            case_id = f"lacuna-{source_artifact_id}-{seq.unit_id}-pos{idx}-{case_counter}"
            case_counter += 1

            original_token = tokens[idx]
            new_tokens[idx] = mask_token

            # Model visible features: surrounding context window only
            left_context = tuple(tokens[max(0, idx - 2) : idx])
            right_context = tuple(tokens[idx + 1 : min(n, idx + 3)])

            cases.append(
                BenchmarkCase(
                    id=case_id,
                    split=split,
                    group_id=seq.unit_id,
                    features={
                        "position": idx,
                        "unit_length": n,
                        "left_context": list(left_context),
                        "right_context": list(right_context),
                    },
                    source_artifact_ids=(source_artifact_id,),
                )
            )

            gold.append(
                GoldLabel(
                    case_id=case_id,
                    value=original_token,
                    confidence=1.0,
                    source_artifact_ids=(source_artifact_id,),
                    rationale="Ground-truth unmasked sign.",
                )
            )

        masked_seqs.append(TokenSequence(unit_id=seq.unit_id, tokens=tuple(new_tokens)))

    return MaskedCorpus(
        masked_sequences=tuple(masked_seqs),
        benchmark_cases=tuple(cases),
        gold_labels=tuple(gold),
    )
