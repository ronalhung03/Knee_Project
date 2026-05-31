from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .constants import CLASS_NAMES, NUM_CLASSES


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    qwk: float
    mae: float
    mse: float
    confusion: list[list[int]]
    per_grade_accuracy: dict[str, float]


def _per_grade_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    result: dict[str, float] = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        total = matrix[idx].sum()
        result[class_name] = float(matrix[idx, idx] / total) if total else 0.0
    return result


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    probabilities: np.ndarray | None = None,
) -> Metrics:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    auc: float | None = None
    if probabilities is not None:
        try:
            y_bin = label_binarize(y_true_arr, classes=list(range(NUM_CLASSES)))
            auc = float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr"))
        except ValueError:
            auc = None
    return Metrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        precision=float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        auc=auc,
        qwk=float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic")),
        mae=float(mean_absolute_error(y_true_arr, y_pred_arr)),
        mse=float(mean_squared_error(y_true_arr, y_pred_arr)),
        confusion=confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(NUM_CLASSES))).tolist(),
        per_grade_accuracy=_per_grade_accuracy(y_true_arr, y_pred_arr),
    )

