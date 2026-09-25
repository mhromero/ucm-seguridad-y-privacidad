from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score


def threshold_predictions(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probabilities >= threshold).astype(np.float32)


def multilabel_metrics(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = threshold_predictions(predicted_probabilities, threshold)

    return {
        "subset_accuracy": float(accuracy_score(true_labels, predictions)),
        "micro_f1": float(f1_score(true_labels, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        "sample_jaccard": float(
            jaccard_score(true_labels, predictions, average="samples", zero_division=0)
        ),
        "hamming_accuracy": float(1.0 - hamming_loss(true_labels, predictions)),
    }


def teacher_student_metrics(
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
    threshold: float = 0.5,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    teacher_predictions = threshold_predictions(teacher_probabilities, threshold)
    student_predictions = threshold_predictions(student_probabilities, threshold)

    per_label_agreement = (teacher_predictions == student_predictions).mean(axis=0)
    result: dict[str, Any] = {
        "exact_agreement": float((teacher_predictions == student_predictions).all(axis=1).mean()),
        "labelwise_agreement": float((teacher_predictions == student_predictions).mean()),
        "probability_mae": float(np.abs(teacher_probabilities - student_probabilities).mean()),
        "teacher_positive_rate": float(teacher_predictions.mean()),
        "student_positive_rate": float(student_predictions.mean()),
        "per_label_agreement": {
            (label_names[idx] if label_names else str(idx)): float(value)
            for idx, value in enumerate(per_label_agreement)
        },
    }
    return result


def build_report(
    *,
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
    true_labels: np.ndarray | None,
    threshold: float,
    label_names: list[str] | None,
) -> dict[str, Any]:
    report = {
        "teacher_vs_student": teacher_student_metrics(
            teacher_probabilities,
            student_probabilities,
            threshold=threshold,
            label_names=label_names,
        )
    }

    if true_labels is not None:
        report["teacher_vs_truth"] = multilabel_metrics(
            true_labels,
            teacher_probabilities,
            threshold=threshold,
        )
        report["student_vs_truth"] = multilabel_metrics(
            true_labels,
            student_probabilities,
            threshold=threshold,
        )

    return report
