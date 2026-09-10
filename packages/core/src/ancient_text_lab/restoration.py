"""Bayesian restoration engine and risk-calibrated evaluation for lacunae benchmarks.

Combines bigram/trigram transition priors with entropy-gated abstention thresholds,
ensuring false guesses are penalized and high-uncertainty contexts trigger explicit refusal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ancient_text_lab.evaluation import BenchmarkCase, GoldLabel, Prediction
from ancient_text_lab.sequence import TransitionMatrix


@dataclass(frozen=True, slots=True)
class RestorationReport:
    """Evaluation metrics for lacunae restoration under explicit risk calibration."""

    evaluated_cases: int
    exact_match: float
    coverage: float
    abstention_rate: float
    risk_calibrated_score: float
    penalty_wrong: float
    mean_correct_confidence: float | None


def evaluate_restoration_predictions(
    cases: Iterable[BenchmarkCase],
    gold_labels: Iterable[GoldLabel],
    predictions: Iterable[Prediction],
    *,
    penalty_wrong: float = 2.0,
) -> RestorationReport:
    """Score restoration predictions with strict penalty for ungrounded hallucinations.

    Scoring formula per case:
    - Correct guess: +1.0
    - Principled abstention (value is None): 0.0
    - False guess: -penalty_wrong
    """
    materialized_cases = tuple(cases)
    gold_by_id = {g.case_id: g.value for g in gold_labels}
    pred_by_id = {p.case_id: p for p in predictions}

    n = len(materialized_cases)
    if n == 0:
        return RestorationReport(
            evaluated_cases=0,
            exact_match=0.0,
            coverage=0.0,
            abstention_rate=0.0,
            risk_calibrated_score=0.0,
            penalty_wrong=penalty_wrong,
            mean_correct_confidence=None,
        )

    correct_count = 0
    abstained_count = 0
    false_count = 0
    correct_scores: list[float] = []

    for case in materialized_cases:
        gold_val = gold_by_id[case.id]
        pred = pred_by_id.get(case.id)

        if pred is None or pred.value is None:
            abstained_count += 1
        elif pred.value == gold_val:
            correct_count += 1
            if pred.score is not None:
                correct_scores.append(pred.score)
        else:
            false_count += 1

    total_reward = (1.0 * correct_count) + (0.0 * abstained_count) - (penalty_wrong * false_count)
    risk_score = total_reward / n

    return RestorationReport(
        evaluated_cases=n,
        exact_match=correct_count / n,
        coverage=(n - abstained_count) / n,
        abstention_rate=abstained_count / n,
        risk_calibrated_score=risk_score,
        penalty_wrong=penalty_wrong,
        mean_correct_confidence=(
            sum(correct_scores) / len(correct_scores) if correct_scores else None
        ),
    )


class BayesianTransitionInfiller:
    """Infill damaged signs using transition priors and entropy-gated abstention."""

    def __init__(
        self,
        matrix: TransitionMatrix,
        *,
        smoothing: float = 0.1,
        entropy_threshold_bits: float = 2.5,
        model_version: str = "bayesian-infiller-v1",
    ) -> None:
        self.vocabulary = matrix.vocabulary
        self.vocab_size = len(matrix.vocabulary)
        self.vocab_index = {token: i for i, token in enumerate(matrix.vocabulary)}
        self.entropy_threshold_bits = entropy_threshold_bits
        self.model_version = model_version

        # Additive smoothed transition probability matrix P(right | left)
        smoothed = matrix.counts.astype(float) + smoothing
        row_sums = smoothed.sum(axis=1, keepdims=True)
        self.transition_probs = smoothed / np.where(row_sums > 0, row_sums, 1.0)

        # Unigram prior P(s)
        unigram_counts = matrix.counts.sum(axis=1) + smoothing
        self.unigram_probs = unigram_counts / unigram_counts.sum()

    def predict_case(self, case: BenchmarkCase) -> Prediction:
        """Predict or abstain for a single damaged context."""
        left_ctx = case.features.get("left_context", [])
        right_ctx = case.features.get("right_context", [])

        left_token = left_ctx[-1] if isinstance(left_ctx, list) and left_ctx else None
        right_token = right_ctx[0] if isinstance(right_ctx, list) and right_ctx else None

        # Calculate posterior likelihood vector over all candidate signs s in vocabulary
        posterior = np.copy(self.unigram_probs)

        if left_token is not None and left_token in self.vocab_index:
            l_idx = self.vocab_index[left_token]
            posterior *= self.transition_probs[l_idx, :]

        if right_token is not None and right_token in self.vocab_index:
            r_idx = self.vocab_index[right_token]
            # P(right | s)
            posterior *= self.transition_probs[:, r_idx]

        total = posterior.sum()
        if total == 0:
            return Prediction(
                case_id=case.id,
                value=None,
                score=0.0,
                model_version=self.model_version,
            )

        posterior /= total

        # Compute Shannon entropy in bits: H = - sum p * log2(p)
        non_zero = posterior[posterior > 0]
        entropy = -float(np.sum(non_zero * np.log2(non_zero)))

        # Abstain if entropy exceeds threshold (too uncertain)
        if entropy > self.entropy_threshold_bits:
            return Prediction(
                case_id=case.id,
                value=None,
                score=0.0,
                model_version=self.model_version,
            )

        best_idx = int(np.argmax(posterior))
        best_token = self.vocabulary[best_idx]
        confidence = float(posterior[best_idx])

        return Prediction(
            case_id=case.id,
            value=best_token,
            score=confidence,
            model_version=self.model_version,
        )

    def predict_all(self, cases: Iterable[BenchmarkCase]) -> tuple[Prediction, ...]:
        """Generate predictions for an iterable of benchmark cases."""
        return tuple(self.predict_case(c) for c in cases)
