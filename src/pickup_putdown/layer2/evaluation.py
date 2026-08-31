"""Convert validated Layer 2 predictions to Task 8 canonical format.

Calls the shared evaluator. Preserves source timestamps and provenance.

Does not fabricate runtime, VRAM, or evaluation results when live
inference has not run.
"""

from __future__ import annotations

import logging
from typing import Any

from pickup_putdown.evaluation import (
    Criterion,
    EvaluationEvent,
    EvaluationPrediction,
    evaluate_class_aware,
)
from pickup_putdown.layer2.merge_predictions import merge_predictions
from pickup_putdown.layer2.schemas import Layer2Prediction, Layer2WindowResponse

logger = logging.getLogger(__name__)


def predictions_to_canonical(
    predictions: list[Layer2Prediction],
) -> list[EvaluationPrediction]:
    """Convert Layer 2 predictions to evaluation-ready format.

    Expands two-item events into separate rows per the repository convention.
    Preserves source timestamps and provenance via pred_id.
    """
    results: list[EvaluationPrediction] = []
    for pred in predictions:
        for row in pred.to_canonical():
            results.append(
                EvaluationPrediction(
                    clip_id=row["clip_id"],
                    type=row["type"],
                    t_start=row["t_start"],
                    t_end=row["t_end"],
                    pred_id=row["pred_id"],
                    score=row["score"],
                    model=row["model"],
                )
            )
    return results


def evaluate_layer2(
    predictions: list[Layer2Prediction],
    ground_truth: list[EvaluationEvent],
    *,
    criterion: Criterion | None = None,
    matcher: str = "hungarian",
) -> dict[str, Any]:
    """Run Task 8 evaluation on Layer 2 predictions.

    Converts predictions to canonical format and calls the shared
    evaluator. Returns metrics dict compatible with existing reports.

    Does NOT fabricate results. If predictions or ground_truth is
    empty, returns a zero-metrics dict with a note.
    """
    crit = criterion or Criterion()

    if not predictions:
        logger.warning("No predictions to evaluate")
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": "no predictions",
        }

    if not ground_truth:
        logger.warning("No ground truth to evaluate against")
        return {
            "tp": 0,
            "fp": len(predictions),
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": "no ground truth",
        }

    canonical_preds = predictions_to_canonical(predictions)
    result = evaluate_class_aware(
        ground_truth,
        canonical_preds,
        criterion=crit,
        matcher=matcher,
    )

    total = result.tp + result.fp
    precision = result.tp / total if total > 0 else 0.0
    recall = result.tp / (result.tp + result.fn) if (result.tp + result.fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": len(result.matched),
        "unmatched_gt": len(result.unmatched_gt),
        "unmatched_pred": len(result.unmatched_pred),
    }


def evaluate_from_responses(
    responses: list[Layer2WindowResponse],
    ground_truth: list[EvaluationEvent],
    *,
    merge_threshold_s: float = 1.0,
    criterion: Criterion | None = None,
    matcher: str = "hungarian",
) -> dict[str, Any]:
    """Full pipeline: merge window responses -> evaluate.

    Convenience wrapper that merges predictions from window responses
    and runs evaluation.
    """
    merged = merge_predictions(responses, merge_threshold_s=merge_threshold_s)
    return evaluate_layer2(
        merged,
        ground_truth,
        criterion=criterion,
        matcher=matcher,
    )
