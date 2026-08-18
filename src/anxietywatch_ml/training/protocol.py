"""Training & evaluation protocol for real ground-truth datasets (004-A).

Bridges the :class:`~anxietywatch_ml.ground_truth.builder.GroundTruthDataset`
(003-A) into the group-aware training pipeline with strict no-leakage rules:

- readiness checks raise clear errors on empty / misaligned / single-class
  datasets BEFORE anything is fitted;
- train / val / test are split **by user** (group-aware), so a user's rows
  never appear in more than one split;
- every learned preprocessing step is fitted on TRAIN only and applied to
  val / test through the serialized bundle;
- a ``DummyClassifier(prior)`` baseline AND two ``LogisticRegression``
  variants (``class_weight=None`` and ``class_weight="balanced"``) are all
  trained on the SAME group-by-user split for a like-for-like comparison;
- the decision threshold is selected on the validation split (fallback: train)
  and NEVER on the test split;
- the LR "winner" is chosen on validation metrics only: TEST METRICS NEVER
  SELECT A WINNER. Any statement about which variant is better is a
  validation-only comparison;
- the serialized artifact is a :class:`TrainedModelBundle` with BOTH the split
  group identifiers and the row indices stripped: fitted pipeline and
  estimator only, NO user / session / device / event IDs, NO train/val/test
  indices, NO dataset rows, NO raw telemetry and NO metadata table.

This module does NOT deploy to Azure. It only produces the protocol and the
reproducible artifact.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from anxietywatch_ml.evaluation.metrics import (
    EvaluationResult,
    create_evaluator,
    evaluate_with_threshold,
    find_best_threshold,
)
from anxietywatch_ml.evaluation.splitting import SplitResult
from anxietywatch_ml.ground_truth.builder import GroundTruthDataset
from anxietywatch_ml.pipelines.model_pipeline import (
    ModelPipelineConfig,
    TrainedModelBundle,
    evaluate_pipeline,
    load_trained_bundle,
    save_trained_bundle,
    train_with_pipeline,
    transform_for_inference,
)

DEFAULT_MIN_ROWS = 10
DEFAULT_MIN_USERS = 2
GROUP_COLUMN = "user_id"
DEFAULT_MODEL_VERSION = "0.1.0"
DEFAULT_TARGET = "target_support_requested"


class DatasetReadinessError(ValueError):
    """Raised when a GroundTruthDataset is not safe to train on."""


@dataclass
class DatasetReadinessReport:
    """Result of the pre-training readiness checks.

    ``ready`` is True only when every check passed. ``errors`` lists every
    failing check so the caller knows exactly why training is refused.
    """

    ready: bool
    errors: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.ready:
            return "dataset readiness: OK"
        return "dataset readiness: FAILED\n" + "\n".join(f"  - {e}" for e in self.errors)


@dataclass
class SplitCounts:
    """Row/user/class counts for one train/val/test split."""

    n_rows: int
    n_users: int
    class_counts: dict
    n_positive: int


@dataclass
class GroundTruthTrainingResult:
    """Everything produced by one ground-truth training protocol run.

    ``dummy_metrics``, ``unweighted_metrics`` and ``balanced_metrics`` map
    split name (``train``/``val``/``test``) to the default-threshold
    :class:`EvaluationResult`. ``*_test_at_threshold`` re-evaluates the test
    split with the threshold selected on val (never on test).

    ``selected_variant`` is ``"unweighted"`` or ``"balanced"`` and is decided
    on VALIDATION metrics only (``selection_source``/``selection_metric``);
    test metrics never select a winner. ``selected_test_at_threshold`` is the
    final test evaluation of the selected variant.
    """

    dataset_readiness: DatasetReadinessReport
    split: SplitResult
    split_counts: dict[str, SplitCounts]
    user_intersections: dict[str, int]
    feature_names: list[str]
    dummy_bundle: TrainedModelBundle
    dummy_metrics: dict[str, EvaluationResult | None]
    unweighted_bundle: TrainedModelBundle
    unweighted_metrics: dict[str, EvaluationResult | None]
    unweighted_threshold: float
    unweighted_threshold_source: str
    unweighted_test_at_threshold: EvaluationResult
    balanced_bundle: TrainedModelBundle
    balanced_metrics: dict[str, EvaluationResult | None]
    balanced_threshold: float
    balanced_threshold_source: str
    balanced_test_at_threshold: EvaluationResult
    selected_variant: str
    selection_source: str
    selection_metric: str
    selected_test_at_threshold: EvaluationResult


def check_dataset_ready(
    dataset: GroundTruthDataset,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_users: int = DEFAULT_MIN_USERS,
) -> DatasetReadinessReport:
    """Run the readiness checks without raising.

    Checks (each produces one error when it fails):

    - dataset is not empty;
    - ``X``, ``y`` and ``metadata`` have the same number of rows;
    - ``y`` contains both classes;
    - ``metadata`` exposes the group column (``user_id``);
    - there are at least ``min_users`` distinct users (group split by user);
    - there are at least ``min_rows`` rows.
    """
    checks: dict[str, bool] = {}
    errors: list[str] = []

    empty = dataset.X is None or dataset.y is None or len(dataset.X) == 0
    checks["not_empty"] = not empty
    if empty:
        errors.append("dataset is empty: no rows to train on")

    aligned = (
        len(dataset.X) == len(dataset.y) == len(dataset.metadata)
    )
    checks["aligned"] = aligned
    if not aligned:
        errors.append(
            "X, y and metadata are misaligned "
            f"(X={len(dataset.X)}, y={len(dataset.y)}, "
            f"metadata={len(dataset.metadata)})"
        )

    n_classes = len({int(v) for v in dataset.y.dropna().unique()}) if len(dataset.y) else 0
    checks["both_classes"] = n_classes >= 2
    if n_classes < 2:
        errors.append(
            f"y has a single class (found {n_classes}); both classes are "
            "required for a meaningful train/test protocol"
        )

    has_group = GROUP_COLUMN in dataset.metadata.columns
    checks["has_group_column"] = has_group
    if not has_group:
        errors.append(f"metadata is missing the group column '{GROUP_COLUMN}'")

    n_users = dataset.metadata[GROUP_COLUMN].nunique() if has_group and len(dataset.metadata) else 0
    checks["enough_users"] = n_users >= min_users
    if n_users < min_users:
        errors.append(
            f"group split by user requires at least {min_users} distinct users, "
            f"found {n_users}"
        )

    checks["enough_rows"] = len(dataset.X) >= min_rows
    if len(dataset.X) < min_rows:
        errors.append(f"dataset too small: {len(dataset.X)} rows (min {min_rows})")

    return DatasetReadinessReport(ready=not errors, errors=errors, checks=checks)


def assert_dataset_ready(
    dataset: GroundTruthDataset,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_users: int = DEFAULT_MIN_USERS,
) -> DatasetReadinessReport:
    """Run readiness checks and raise :class:`DatasetReadinessError` on failure."""
    report = check_dataset_ready(dataset, min_rows=min_rows, min_users=min_users)
    if not report.ready:
        raise DatasetReadinessError(str(report))
    return report


def train_ground_truth(
    dataset: GroundTruthDataset,
    config: dict,
    output_path: Path | str | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_users: int = DEFAULT_MIN_USERS,
    model_version: str = DEFAULT_MODEL_VERSION,
    target: str = DEFAULT_TARGET,
) -> GroundTruthTrainingResult:
    """Run the full 004-A protocol on a GroundTruthDataset.

    Fits a ``DummyClassifier(prior)`` baseline and two ``LogisticRegression``
    variants (``class_weight=None`` and ``class_weight="balanced"``) on the
    SAME group-by-user split, evaluates all three on train/val/test, selects
    each LR variant's decision threshold on val (fallback train), and picks
    the LR "winner" from VALIDATION metrics only. Test metrics never select a
    winner; the selected variant's test evaluation is the final estimate.
    Optionally persists the selected bundle (with group identifiers and row
    indices stripped) to ``output_path``, carrying non-personal inference
    metadata (``model_version``, ``target``, ``threshold``,
    ``threshold_source``, ``feature_names``) in ``runtime_config``.
    """
    readiness = assert_dataset_ready(dataset, min_rows=min_rows, min_users=min_users)

    X, y = dataset.X, dataset.y
    groups = dataset.metadata[GROUP_COLUMN].astype(str)

    training_cfg = config.get("training", {})
    shared = ModelPipelineConfig(
        group_by="user",
        test_size=training_cfg.get("test_size", 0.2),
        val_size=training_cfg.get("val_size", 0.1),
        random_state=training_cfg.get("random_state", config.get("random_seed", 42)),
    )

    def with_model_type(model_type: str) -> ModelPipelineConfig:
        return ModelPipelineConfig(
            model_type=model_type,
            group_by=shared.group_by,
            test_size=shared.test_size,
            val_size=shared.val_size,
            random_state=shared.random_state,
        )

    def runtime_with_class_weight(class_weight) -> dict:
        runtime = deepcopy(config)
        logistic_cfg = dict(runtime["model"].get("logistic_regression", {}))
        logistic_cfg["class_weight"] = class_weight
        runtime["model"]["logistic_regression"] = logistic_cfg
        return runtime

    dummy_bundle = train_with_pipeline(
        X, y,
        group_column=groups,
        config=with_model_type("dummy"),
        runtime_config=deepcopy(config),
    )
    unweighted_bundle = train_with_pipeline(
        X, y,
        group_column=groups,
        config=with_model_type("logistic_regression"),
        runtime_config=runtime_with_class_weight(None),
    )
    balanced_bundle = train_with_pipeline(
        X, y,
        group_column=groups,
        config=with_model_type("logistic_regression"),
        runtime_config=runtime_with_class_weight("balanced"),
    )

    dummy_metrics = _split_results(evaluate_pipeline(dummy_bundle, X, y))
    unweighted_metrics = _split_results(evaluate_pipeline(unweighted_bundle, X, y))
    balanced_metrics = _split_results(evaluate_pipeline(balanced_bundle, X, y))

    unweighted_threshold, unweighted_source = _select_threshold(unweighted_bundle, X, y)
    balanced_threshold, balanced_source = _select_threshold(balanced_bundle, X, y)
    assert unweighted_source == balanced_source

    unweighted_test_at_threshold = _evaluate_at_threshold(
        unweighted_bundle, X, y, unweighted_threshold, config
    )
    balanced_test_at_threshold = _evaluate_at_threshold(
        balanced_bundle, X, y, balanced_threshold, config
    )

    selection_source = unweighted_source
    unweighted_val_at_threshold = _evaluate_at_threshold(
        unweighted_bundle, X, y, unweighted_threshold, config, selection_source
    )
    balanced_val_at_threshold = _evaluate_at_threshold(
        balanced_bundle, X, y, balanced_threshold, config, selection_source
    )
    selected_variant = _select_variant(unweighted_val_at_threshold, balanced_val_at_threshold)
    if selected_variant == "balanced":
        selected_test_at_threshold = balanced_test_at_threshold
    else:
        selected_test_at_threshold = unweighted_test_at_threshold

    split = unweighted_bundle.split_result
    split_counts = _split_counts(dataset, split)
    user_intersections = _user_intersections(split)

    if output_path is not None:
        selected_bundle = (
            balanced_bundle if selected_variant == "balanced" else unweighted_bundle
        )
        selected_threshold = (
            balanced_threshold if selected_variant == "balanced" else unweighted_threshold
        )
        save_bundle_with_metadata(
            selected_bundle,
            output_path,
            {
                "model_version": model_version,
                "target": target,
                "threshold": float(selected_threshold),
                "threshold_source": selection_source,
                "feature_names": list(X.columns),
            },
        )

    return GroundTruthTrainingResult(
        dataset_readiness=readiness,
        split=split,
        split_counts=split_counts,
        user_intersections=user_intersections,
        feature_names=list(X.columns),
        dummy_bundle=dummy_bundle,
        dummy_metrics=dummy_metrics,
        unweighted_bundle=unweighted_bundle,
        unweighted_metrics=unweighted_metrics,
        unweighted_threshold=unweighted_threshold,
        unweighted_threshold_source=unweighted_source,
        unweighted_test_at_threshold=unweighted_test_at_threshold,
        balanced_bundle=balanced_bundle,
        balanced_metrics=balanced_metrics,
        balanced_threshold=balanced_threshold,
        balanced_threshold_source=balanced_source,
        balanced_test_at_threshold=balanced_test_at_threshold,
        selected_variant=selected_variant,
        selection_source=selection_source,
        selection_metric="f1",
        selected_test_at_threshold=selected_test_at_threshold,
    )


def load_ground_truth_bundle(path: Path | str) -> TrainedModelBundle:
    """Load a saved ground-truth training artifact."""
    return load_trained_bundle(path)


def save_bundle_with_metadata(
    bundle: TrainedModelBundle,
    path: Path | str,
    metadata: dict,
) -> None:
    """Persist a bundle whose ``runtime_config`` carries inference metadata.

    ``metadata`` (threshold, model version, target, feature schema) travels in
    the serialized artifact via ``runtime_config["model"]``. It must never
    contain user IDs or raw telemetry; ``save_trained_bundle`` still strips
    split identifiers and row indices before writing.
    """
    enriched = deepcopy(bundle)
    model_meta = dict(enriched.runtime_config.get("model", {}))
    model_meta.update(metadata)
    enriched.runtime_config["model"] = model_meta
    save_trained_bundle(enriched, path)


def _split_results(eval_by_split: dict) -> dict[str, EvaluationResult | None]:
    """Map the evaluate_pipeline dict to split name -> EvaluationResult."""
    return {
        name: (entry["result"] if entry is not None else None)
        for name, entry in eval_by_split.items()
    }


def _select_variant(
    unweighted_val: EvaluationResult | None,
    balanced_val: EvaluationResult | None,
) -> str:
    """Pick the LR variant with the better VALIDATION F1 at its own threshold.

    Test metrics never participate in the choice. NaN (unavailable) scores
    count as missing. On ties the unweighted variant is kept (simpler model).
    Any statement about a "winner" is a validation-only comparison.
    """
    def _f1(result):
        if result is None:
            return None
        value = result.metrics.get("f1")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return value

    score_unw = _f1(unweighted_val)
    score_bal = _f1(balanced_val)
    if score_bal is not None and score_unw is not None and score_bal > score_unw:
        return "balanced"
    return "unweighted"


def _evaluate_at_threshold(
    bundle: TrainedModelBundle,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    config: dict,
    split_name: str = "test",
) -> EvaluationResult:
    """Evaluate a split at an explicit threshold through the fitted bundle."""
    proba = _split_proba(bundle, X, y, split_name)
    if proba is None:
        return EvaluationResult(
            metrics={},
            confusion_matrix=np.zeros((2, 2), dtype=int),
            n_samples=0,
            n_positive=0,
            threshold=threshold,
            metrics_available={},
        )
    indices = getattr(bundle.split_result, f"{split_name}_indices")
    y_split = y.iloc[indices].values
    return evaluate_with_threshold(y_split, proba, threshold, create_evaluator(config))


def _select_threshold(
    bundle: TrainedModelBundle, X: pd.DataFrame, y: pd.Series
) -> tuple[float, str]:
    """Select the decision threshold on val (fallback: train), never on test."""
    for split_name in ("val", "train"):
        indices = getattr(bundle.split_result, f"{split_name}_indices")
        if len(indices) == 0:
            continue
        proba = _split_proba(bundle, X, y, split_name)
        y_split = y.iloc[indices].values
        if proba is None:
            continue
        if (y_split == 1).any() and (y_split == 0).any():
            threshold, _ = find_best_threshold(y_split, proba)
            return threshold, split_name
    return 0.5, "val"


def _split_proba(bundle: TrainedModelBundle, X: pd.DataFrame, y: pd.Series, split_name: str):
    """Positive-class probabilities for a split through the fitted bundle."""
    indices = getattr(bundle.split_result, f"{split_name}_indices")
    if len(indices) == 0:
        return None
    X_split = X.iloc[indices]
    X_transformed = transform_for_inference(bundle, X_split)
    return bundle.model.predict_proba(X_transformed)


def _split_counts(dataset: GroundTruthDataset, split: SplitResult) -> dict[str, SplitCounts]:
    """Row/user/class counts per split."""
    counts: dict[str, SplitCounts] = {}
    for name in ("train", "val", "test"):
        indices = getattr(split, f"{name}_indices")
        y_split = dataset.y.iloc[indices]
        groups_split = dataset.metadata[GROUP_COLUMN].iloc[indices]
        class_counts = y_split.value_counts().sort_index()
        counts[name] = SplitCounts(
            n_rows=len(indices),
            n_users=int(groups_split.nunique()),
            class_counts={str(k): int(v) for k, v in class_counts.items()},
            n_positive=int(y_split.sum()),
        )
    return counts


def _user_intersections(split: SplitResult) -> dict[str, int]:
    """Pairwise intersection sizes of user sets across splits (must be 0)."""
    sets = {
        name: set(getattr(split, f"{name}_groups"))
        for name in ("train", "val", "test")
    }
    return {
        "train_vs_val": len(sets["train"] & sets["val"]),
        "train_vs_test": len(sets["train"] & sets["test"]),
        "val_vs_test": len(sets["val"] & sets["test"]),
    }