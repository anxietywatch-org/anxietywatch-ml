"""
Evaluation metrics for AnxietyWatch ML.

Computes standard classification metrics for baseline model validation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: list[str] = None
    threshold: float = 0.5

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "balanced_accuracy",
                "specificity",
                "false_positive_rate",
            ]


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: dict[str, Optional[float]]
    confusion_matrix: np.ndarray
    n_samples: int
    n_positive: int
    threshold: float
    metrics_available: dict[str, bool]

    def __str__(self) -> str:
        lines = [
            f"Evaluation Results (n={self.n_samples}, positive={self.n_positive})",
            f"Threshold: {self.threshold}",
            "-" * 40,
        ]
        for name, value in self.metrics.items():
            available = self.metrics_available.get(name, True)
            if not available:
                lines.append(f"  {name}: N/A (not computable)")
            elif value is None:
                lines.append(f"  {name}: N/A")
            else:
                lines.append(f"  {name}: {value:.4f}")
        lines.append("-" * 40)
        lines.append("Confusion Matrix:")
        lines.append(f"  TN={self.confusion_matrix[0,0]} FP={self.confusion_matrix[0,1]}")
        lines.append(f"  FN={self.confusion_matrix[1,0]} TP={self.confusion_matrix[1,1]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "metrics_available": self.metrics_available,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "threshold": self.threshold,
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    """
    Compute evaluation metrics.

    Handles single-class scenarios gracefully without warnings.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC, AP)
        config: Evaluation configuration
    """
    config = config or EvaluationConfig()

    metrics = {}
    metrics_available = {}
    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_samples - n_positive

    # Determine if both classes are present in y_true
    has_both_classes = n_positive > 0 and n_negative > 0
    has_proba = y_proba is not None and len(y_proba.shape) == 2 and y_proba.shape[1] == 2

    # Confusion matrix (single-class y_true is forced to 2x2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        # Handle case where only one class present
        cm = np.array([[cm[0,0] if cm.shape[0] > 0 else 0, 0], [0, 0]])
    tn, fp, fn, tp = cm.ravel()

    # Availability rules: a metric is unavailable (NaN + metrics_available=False)
    # when its denominator does not exist for the split, so no artificial 0.0
    # or balanced_accuracy over a half-missing evaluation is ever produced.
    has_pred_pos = (tp + fp) > 0   # precision defined
    has_true_pos = (tp + fn) > 0   # recall defined
    has_true_neg = (tn + fp) > 0   # specificity / FPR defined

    # Basic metrics
    if "accuracy" in config.metrics:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_available["accuracy"] = True

    if "precision" in config.metrics:
        if has_pred_pos:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics_available["precision"] = True
        else:
            metrics["precision"] = float("nan")
            metrics_available["precision"] = False

    if "recall" in config.metrics:
        if has_true_pos:
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics_available["recall"] = True
        else:
            metrics["recall"] = float("nan")
            metrics_available["recall"] = False

    if "f1" in config.metrics:
        if has_pred_pos and has_true_pos:
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
            metrics_available["f1"] = True
        else:
            metrics["f1"] = float("nan")
            metrics_available["f1"] = False

    # Probability-based metrics (require both classes in y_true)
    if "roc_auc" in config.metrics:
        if has_both_classes and has_proba:
            try:
                proba_positive = y_proba[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, proba_positive)
                metrics_available["roc_auc"] = True
            except ValueError:
                metrics["roc_auc"] = None
                metrics_available["roc_auc"] = False
        else:
            metrics["roc_auc"] = None
            metrics_available["roc_auc"] = False

    if "average_precision" in config.metrics:
        if has_both_classes and has_proba:
            try:
                proba_positive = y_proba[:, 1]
                metrics["average_precision"] = average_precision_score(y_true, proba_positive)
                metrics_available["average_precision"] = True
            except ValueError:
                metrics["average_precision"] = None
                metrics_available["average_precision"] = False
        else:
            metrics["average_precision"] = None
            metrics_available["average_precision"] = False

    # Confusion-matrix derived metrics
    tpr = tp / (tp + fn) if has_true_pos else float("nan")
    tnr = tn / (tn + fp) if has_true_neg else float("nan")

    if "balanced_accuracy" in config.metrics:
        if has_true_pos and has_true_neg:
            metrics["balanced_accuracy"] = (tpr + tnr) / 2.0
            metrics_available["balanced_accuracy"] = True
        else:
            metrics["balanced_accuracy"] = float("nan")
            metrics_available["balanced_accuracy"] = False

    if "specificity" in config.metrics:
        if has_true_neg:
            metrics["specificity"] = tnr
            metrics_available["specificity"] = True
        else:
            metrics["specificity"] = float("nan")
            metrics_available["specificity"] = False

    if "false_positive_rate" in config.metrics:
        if has_true_neg:
            metrics["false_positive_rate"] = 1.0 - tnr
            metrics_available["false_positive_rate"] = True
        else:
            metrics["false_positive_rate"] = float("nan")
            metrics_available["false_positive_rate"] = False

    logger.info(f"Evaluation: {metrics} (available: {metrics_available})")
    return EvaluationResult(
        metrics=metrics,
        metrics_available=metrics_available,
        confusion_matrix=cm,
        n_samples=n_samples,
        n_positive=n_positive,
        threshold=config.threshold,
    )


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    """Evaluate using a specific probability threshold."""
    if y_proba.shape[1] != 2:
        raise ValueError("y_proba must have 2 columns for binary classification")

    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    eval_config = config or EvaluationConfig()
    eval_config.threshold = threshold
    return evaluate(y_true, y_pred, y_proba, eval_config)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find optimal threshold by maximizing a metric."""
    from sklearn.metrics import precision_recall_curve

    if y_proba.shape[1] != 2:
        raise ValueError("y_proba must have 2 columns")

    proba_positive = y_proba[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, proba_positive)

    if metric == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])  # Last threshold is 1.0 with no predictions
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    raise ValueError(f"Unsupported metric for threshold optimization: {metric}")


def create_evaluator(config: dict) -> EvaluationConfig:
    """Factory function to create evaluator config from config dict."""
    eval_cfg = config.get("evaluation", {})
    return EvaluationConfig(
        metrics=eval_cfg.get("metrics", [
            "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"
        ]),
        threshold=eval_cfg.get("threshold", 0.5),
    )