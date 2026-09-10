"""Deterministic Diophantine and rational accounting ledger infiller.

Recovers missing quantities and damaged ledger entries from balanced accounting
tablets with provable mathematical certainty (residual Delta = 0.0).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """A single commodity or ration entry on an administrative ledger."""

    item_id: str
    commodity: str
    quantity: Fraction | None
    is_damaged: bool = False

    def __post_init__(self) -> None:
        if self.is_damaged and self.quantity is not None:
            raise ValueError("damaged entry must not have a pre-assigned quantity")
        if not self.is_damaged and self.quantity is None:
            raise ValueError("intact entry must have a non-null quantity")


@dataclass(frozen=True, slots=True)
class LedgerSolution:
    """Deterministic recovery result for damaged entries on a ledger."""

    recovered_quantities: dict[str, Fraction]
    residual: Fraction
    is_unique: bool
    degrees_of_freedom: int
    rationale: str


def solve_single_ledger_lacuna(
    entries: Iterable[LedgerEntry],
    target_total: Fraction,
) -> LedgerSolution:
    """Solve for exactly one damaged quantity given a verified ledger total."""
    materialized = tuple(entries)
    damaged = [e for e in materialized if e.is_damaged]

    if len(damaged) == 0:
        actual_total = sum(
            (e.quantity for e in materialized if e.quantity is not None), Fraction(0)
        )
        return LedgerSolution(
            recovered_quantities={},
            residual=target_total - actual_total,
            is_unique=True,
            degrees_of_freedom=0,
            rationale="Ledger has zero damaged entries; verified total consistency.",
        )

    if len(damaged) > 1:
        raise ValueError(
            f"solve_single_ledger_lacuna requires at most 1 damaged entry, got {len(damaged)}"
        )

    target_entry = damaged[0]
    known_sum = sum(
        (e.quantity for e in materialized if not e.is_damaged and e.quantity is not None),
        Fraction(0),
    )
    recovered = target_total - known_sum

    if recovered < 0:
        return LedgerSolution(
            recovered_quantities={target_entry.item_id: recovered},
            residual=Fraction(0),
            is_unique=False,
            degrees_of_freedom=0,
            rationale=f"Inconsistent ledger: known sum ({known_sum}) exceeds target total ({target_total}).",
        )

    return LedgerSolution(
        recovered_quantities={target_entry.item_id: recovered},
        residual=Fraction(0),
        is_unique=True,
        degrees_of_freedom=0,
        rationale=f"Exact deterministic rational recovery: {target_total} - {known_sum} = {recovered}.",
    )


def solve_diophantine_lattice(
    coefficients: Sequence[int],
    target_sum: int,
    *,
    allowed_values: Sequence[int] | None = None,
    max_val: int = 100,
) -> list[tuple[int, ...]]:
    """Find all non-negative integer solutions to sum(c_i * x_i) = target_sum.

    Parameters
    ----------
    coefficients : Sequence[int]
        Positive integer coefficients for unknown quantities.
    target_sum : int
        Target integer sum.
    allowed_values : Sequence[int], optional
        Explicit discrete allowed values (e.g. allowable fractional units or counts).
    max_val : int
        Search ceiling for each variable when allowed_values is None.

    Returns
    -------
    list[tuple[int, ...]]
        All valid non-negative integer solution vectors.
    """
    if not coefficients:
        return []
    if any(c <= 0 for c in coefficients):
        raise ValueError("all coefficients must be positive integers")

    solutions: list[tuple[int, ...]] = []

    def _backtrack(idx: int, current_sum: int, current_vec: list[int]) -> None:
        if idx == len(coefficients):
            if current_sum == target_sum:
                solutions.append(tuple(current_vec))
            return

        c = coefficients[idx]
        candidates = allowed_values if allowed_values is not None else range(max_val + 1)

        for val in candidates:
            next_sum = current_sum + c * val
            if next_sum > target_sum:
                if allowed_values is None or sorted(allowed_values) == list(allowed_values):
                    break
                continue
            current_vec.append(val)
            _backtrack(idx + 1, next_sum, current_vec)
            current_vec.pop()

    _backtrack(0, 0, [])
    return solutions
