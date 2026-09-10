"""Linear B Syllabary Grid ('The Alice Kober Challenge') benchmark & metrics.

Evaluates unsupervised recovery of consonant-row and vowel-column structures from
inflectional alternations against the historical Ventris-Kober ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Canonical Ventris-Chadwick 1953 Linear B syllabary phonological grid:
# Sign name -> (consonant_series, vowel_series)
# '-' represents pure vowel (initial vowel without preceding consonant)
LINEAR_B_GROUND_TRUTH_GRID: dict[str, tuple[str, str]] = {
    # Pure vowels
    "a": ("-", "a"),
    "e": ("-", "e"),
    "i": ("-", "i"),
    "o": ("-", "o"),
    "u": ("-", "u"),
    # D-series
    "da": ("d", "a"),
    "de": ("d", "e"),
    "di": ("d", "i"),
    "do": ("d", "o"),
    "du": ("d", "u"),
    # J-series
    "ja": ("j", "a"),
    "je": ("j", "e"),
    "jo": ("j", "o"),
    # K-series
    "ka": ("k", "a"),
    "ke": ("k", "e"),
    "ki": ("k", "i"),
    "ko": ("k", "o"),
    "ku": ("k", "u"),
    # M-series
    "ma": ("m", "a"),
    "me": ("m", "e"),
    "mi": ("m", "i"),
    "mo": ("m", "o"),
    "mu": ("m", "u"),
    # N-series
    "na": ("n", "a"),
    "ne": ("n", "e"),
    "ni": ("n", "i"),
    "no": ("n", "o"),
    "nu": ("n", "u"),
    # P-series
    "pa": ("p", "a"),
    "pe": ("p", "e"),
    "pi": ("p", "i"),
    "po": ("p", "o"),
    "pu": ("p", "u"),
    # Q-series (labiovelar)
    "qa": ("q", "a"),
    "qe": ("q", "e"),
    "qi": ("q", "i"),
    "qo": ("q", "o"),
    # R-series (liquid r/l)
    "ra": ("r", "a"),
    "re": ("r", "e"),
    "ri": ("r", "i"),
    "ro": ("r", "o"),
    "ru": ("r", "u"),
    # S-series
    "sa": ("s", "a"),
    "se": ("s", "e"),
    "si": ("s", "i"),
    "so": ("s", "o"),
    "su": ("s", "u"),
    # T-series
    "ta": ("t", "a"),
    "te": ("t", "e"),
    "ti": ("t", "i"),
    "to": ("t", "o"),
    "tu": ("t", "u"),
    # W-series
    "wa": ("w", "a"),
    "we": ("w", "e"),
    "wi": ("w", "i"),
    "wo": ("w", "o"),
    # Z-series
    "za": ("z", "a"),
    "ze": ("z", "e"),
    "zo": ("z", "o"),
}


@dataclass(frozen=True, slots=True)
class KoberGridReport:
    """Benchmark report measuring unsupervised syllabic grid reconstruction."""

    evaluated_signs: int
    consonant_pairwise_f1: float
    consonant_precision: float
    consonant_recall: float
    vowel_pairwise_f1: float
    vowel_precision: float
    vowel_recall: float
    overall_kober_f1: float


def _pairwise_cluster_metrics(
    predicted_clusters: Mapping[str, int],
    ground_truth_labels: Mapping[str, str],
) -> tuple[float, float, float]:
    """Calculate pairwise clustering Precision, Recall, and F1."""
    common_signs = sorted(set(predicted_clusters).intersection(ground_truth_labels))
    n = len(common_signs)
    if n < 2:
        return 0.0, 0.0, 0.0

    tp = 0
    fp = 0
    fn = 0

    for i in range(n):
        s_i = common_signs[i]
        c_pred_i = predicted_clusters[s_i]
        c_gold_i = ground_truth_labels[s_i]

        for j in range(i + 1, n):
            s_j = common_signs[j]
            pred_same = c_pred_i == predicted_clusters[s_j]
            gold_same = c_gold_i == ground_truth_labels[s_j]

            if pred_same and gold_same:
                tp += 1
            elif pred_same and not gold_same:
                fp += 1
            elif not pred_same and gold_same:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_kober_grid(
    predicted_consonant_groups: Mapping[str, int],
    predicted_vowel_groups: Mapping[str, int],
    ground_truth_grid: Mapping[str, tuple[str, str]] | None = None,
) -> KoberGridReport:
    """Evaluate predicted consonant and vowel clustering against the Ventris ground truth."""
    grid = LINEAR_B_GROUND_TRUTH_GRID if ground_truth_grid is None else ground_truth_grid

    gt_consonants = {sign: row[0] for sign, row in grid.items()}
    gt_vowels = {sign: row[1] for sign, row in grid.items()}

    c_prec, c_rec, c_f1 = _pairwise_cluster_metrics(predicted_consonant_groups, gt_consonants)
    v_prec, v_rec, v_f1 = _pairwise_cluster_metrics(predicted_vowel_groups, gt_vowels)

    overall = (c_f1 + v_f1) / 2.0
    common = set(predicted_consonant_groups).intersection(predicted_vowel_groups).intersection(grid)

    return KoberGridReport(
        evaluated_signs=len(common),
        consonant_pairwise_f1=c_f1,
        consonant_precision=c_prec,
        consonant_recall=c_rec,
        vowel_pairwise_f1=v_f1,
        vowel_precision=v_prec,
        vowel_recall=v_rec,
        overall_kober_f1=overall,
    )


def induce_kober_stem_alternations(
    words: Iterable[Sequence[str]],
    *,
    min_stem_length: int = 2,
) -> dict[str, set[str]]:
    """Discover inflectional suffix alternations from unvocalized token sequences.

    Returns a mapping: stem -> set of observed suffix signs.
    For example: 'da-to' vs 'da-te' yields stem 'da-' with suffixes {'to', 'te'}.
    """
    stems: dict[str, set[str]] = {}
    for word in words:
        if len(word) >= min_stem_length + 1:
            stem = "-".join(word[:-1])
            suffix = word[-1]
            stems.setdefault(stem, set()).add(suffix)

    # Filter only stems with active alternations (at least 2 distinct suffixes)
    return {stem: suffixes for stem, suffixes in stems.items() if len(suffixes) >= 2}
