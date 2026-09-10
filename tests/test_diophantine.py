"""Tests for deterministic Diophantine and rational accounting solver."""

from __future__ import annotations

from fractions import Fraction

import pytest
from ancient_text_lab.diophantine import (
    LedgerEntry,
    solve_diophantine_lattice,
    solve_single_ledger_lacuna,
)


def test_solve_single_ledger_lacuna_exact() -> None:
    # Model a Minoan grain ledger (HT style):
    # Entry 1: 10 + 1/2
    # Entry 2: damaged [...]
    # Entry 3: 5 + 1/4
    # KU-RO (Total): 25 + 3/4
    entries = [
        LedgerEntry(item_id="e1", commodity="GRAIN", quantity=Fraction(21, 2)),
        LedgerEntry(item_id="e2", commodity="GRAIN", quantity=None, is_damaged=True),
        LedgerEntry(item_id="e3", commodity="GRAIN", quantity=Fraction(21, 4)),
    ]
    target_total = Fraction(103, 4)  # 25.75

    solution = solve_single_ledger_lacuna(entries, target_total)

    assert solution.is_unique
    assert solution.degrees_of_freedom == 0
    assert solution.residual == Fraction(0)
    # Known: 10.5 + 5.25 = 15.75. Target: 25.75. Recovered: 10.0 (Fraction(10, 1))
    assert solution.recovered_quantities["e2"] == Fraction(10, 1)


def test_solve_single_ledger_lacuna_validation() -> None:
    with pytest.raises(ValueError, match="damaged entry must not have a pre-assigned quantity"):
        LedgerEntry(item_id="e1", commodity="GRAIN", quantity=Fraction(5), is_damaged=True)

    with pytest.raises(ValueError, match="intact entry must have a non-null quantity"):
        LedgerEntry(item_id="e1", commodity="GRAIN", quantity=None, is_damaged=False)

    # Multiple damaged entries error
    entries = [
        LedgerEntry(item_id="e1", commodity="GRAIN", quantity=None, is_damaged=True),
        LedgerEntry(item_id="e2", commodity="GRAIN", quantity=None, is_damaged=True),
    ]
    with pytest.raises(ValueError, match="requires at most 1 damaged entry"):
        solve_single_ledger_lacuna(entries, Fraction(20))


def test_solve_diophantine_lattice() -> None:
    # 2 unknown fractional counters x1, x2 with scale factor 2 and 4, target = 10
    # 2*x1 + 4*x2 = 10 -> x1 + 2*x2 = 5
    # Non-negative integer solutions: (5, 0), (3, 1), (1, 2)
    sols = solve_diophantine_lattice([2, 4], 10)
    assert (5, 0) in sols
    assert (3, 1) in sols
    assert (1, 2) in sols
    assert len(sols) == 3
