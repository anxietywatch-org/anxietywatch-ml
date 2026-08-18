"""
Group-aware model pipeline for AnxietyWatch ML.

All learned preprocessing is fitted only on the training partition and is
serialized together with the fitted estimator for reproducible inference.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anxietywatch_ml.evaluation.splitting import (
    GroupBy,
    SplitResult,
    get_group_column,
    group_aware_split,
)
from anxietywatch_ml.models.baseline import BaselineModel, create_model
from anxietywatch_ml.models.serialization import load_artifact, save_artifact


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select numeric features that have at least one observed TRAIN value."""

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.feature_names is not None:
            numeric_cols = [
                col
                for col in self.feature_names
                if col in numeric_cols
            ]

        self.dropped_all_missing_ = [
            col
            for col in numeric_cols
            if X[col].isna().all()
        ]
        self.feature_names_ = [
            col
            for col in numeric_cols
            if col not in self.dropped_all_missing_
        ]

        if not self.feature_names_:
            raise ValueError("No usable numeric features available in training data.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [
            col
            for col in self.feature_names_
            if col not in X.columns
        ]
        if missing:
            raise ValueError(
                f"Inference feature matrix is missing columns: {missing}"
            )
        return X[self.feature_names_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class NaNIndicator(BaseEstimator, TransformerMixin):
    """Add indicators so semantic missingness remains visible to the model."""

    def __init__(self, features_to_monitor: list[str]):
        self.features_to_monitor = features_to_monitor

    def fit(self, X: pd.DataFrame, y=None):
        self.monitored_features_ = [
            col
            for col in self.features_to_monitor
            if col in X.columns
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.monitored_features_:
            X[f"{col}_was_nan"] = X[col].isna().astype(int)
        return X


class ModelInputImputer(BaseEstimator, TransformerMixin):
    """Fit TRAIN-only fill values and guarantee a finite estimator matrix."""

    def __init__(self):
        self.zero_fill_features = {
            "ibi_available",
            "ibi_coverage_ratio",
        }

    def fit(self, X: pd.DataFrame, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        self.fill_values_: dict[str, float] = {}

        for col in X.columns:
            if col in self.zero_fill_features:
                self.fill_values_[col] = 0.0
                continue

            median = X[col].median(skipna=True)
            self.fill_values_[col] = (
                0.0
                if pd.isna(median)
                else float(median)
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy().replace([np.inf, -np.inf], np.nan)

        for col, fill_value in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)

        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                "Model input still contains NaN or Inf after preprocessing."
            )

        return X


@dataclass
class ModelPipelineConfig:
    """Configuration for group-aware training."""

    model_type: str = "logistic_regression"
    group_by: Literal["session", "user"] = "session"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class TrainedModelBundle:
    """Complete reproducible inference artifact."""

    preprocessing_pipeline: Pipeline
    model: BaselineModel
    split_result: SplitResult
    config: ModelPipelineConfig
    runtime_config: dict


BUNDLE_ARTIFACT_TYPE = "trained_model_bundle"


def create_model_pipeline(config: ModelPipelineConfig) -> Pipeline:
    hrv_features = ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50"]
    ibi_availability = ["ibi_available", "ibi_coverage_ratio"]

    return Pipeline(
        [
            ("feature_selector", FeatureSelector()),
            (
                "nan_indicator",
                NaNIndicator(
                    features_to_monitor=hrv_features + ibi_availability
                ),
            ),
            ("model_input_imputer", ModelInputImputer()),
            ("scaler", StandardScaler()),
        ]
    )


def train_with_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    group_column: Optional[pd.Series] = None,
    config: Optional[ModelPipelineConfig] = None,
    runtime_config: Optional[dict] = None,
) -> TrainedModelBundle:
    if config is None:
        config = ModelPipelineConfig()

    if group_column is None:
        group_column = get_group_column(X, config.group_by)

    split_result = group_aware_split(
        X,
        y,
        group_column,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        group_by=config.group_by,
    )

    X_train = X.iloc[split_result.train_indices]
    y_train = y.iloc[split_result.train_indices]

    preprocessing_pipeline = create_model_pipeline(config)
    preprocessing_pipeline.fit(X_train)
    X_train_transformed = preprocessing_pipeline.transform(X_train)

    model_config = deepcopy(runtime_config or {})
    model_config.setdefault("model", {})
    model_config["model"]["type"] = config.model_type
    model_config.setdefault("random_seed", config.random_state)

    model = create_model(model_config)
    model.fit(X_train_transformed, y_train)

    return TrainedModelBundle(
        preprocessing_pipeline=preprocessing_pipeline,
        model=model,
        split_result=split_result,
        config=config,
        runtime_config=deepcopy(runtime_config or model_config),
    )


def transform_for_inference(
    bundle: TrainedModelBundle,
    X: pd.DataFrame,
) -> pd.DataFrame | np.ndarray:
    return bundle.preprocessing_pipeline.transform(X)


def evaluate_pipeline(
    bundle: TrainedModelBundle,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    from anxietywatch_ml.evaluation.metrics import create_evaluator, evaluate

    results = {}
    evaluator_config = create_evaluator(bundle.runtime_config)

    for split_name, indices in (
        ("train", bundle.split_result.train_indices),
        ("val", bundle.split_result.val_indices),
        ("test", bundle.split_result.test_indices),
    ):
        if len(indices) == 0:
            results[split_name] = None
            continue

        X_split = X.iloc[indices]
        y_split = y.iloc[indices]
        X_transformed = transform_for_inference(bundle, X_split)

        y_pred = bundle.model.predict(X_transformed)
        y_proba = bundle.model.predict_proba(X_transformed)
        evaluation = evaluate(
            y_split.values,
            y_pred,
            y_proba,
            evaluator_config,
        )

        results[split_name] = {
            "result": evaluation,
            "metrics": evaluation.metrics,
            "metrics_available": evaluation.metrics_available,
            "n_samples": len(y_split),
            "n_positive": int(y_split.sum()),
        }

    return results


def save_trained_bundle(
    bundle: TrainedModelBundle,
    path: Path | str,
) -> None:
    save_artifact(
        bundle,
        path,
        artifact_type=BUNDLE_ARTIFACT_TYPE,
    )


def load_trained_bundle(path: Path | str) -> TrainedModelBundle:
    return load_artifact(
        path,
        expected_type=TrainedModelBundle,
        expected_artifact_type=BUNDLE_ARTIFACT_TYPE,
    )
