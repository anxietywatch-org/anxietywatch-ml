"""
Model pipeline with preprocessing for AnxietyWatch ML.

Ensures preprocessing is fit ONLY on training data and applied to validation/test.
"""

from dataclasses import dataclass
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
from anxietywatch_ml.models.baseline import BaselineModel


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select only numeric features for modeling."""

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        numeric_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if self.feature_names is not None:
            numeric_cols = [
                col
                for col in self.feature_names
                if col in numeric_cols
            ]

        # A feature that contains absolutely no observed values in TRAIN
        # cannot be learned by the estimator.
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
            raise ValueError(
                "No usable numeric features available in training data."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class NaNIndicator(BaseEstimator, TransformerMixin):
    """
    Add binary indicator columns for NaN values in specified features.
    Useful for features where NaN has semantic meaning (e.g., missing IBI).
    """

    def __init__(self, features_to_monitor: list[str]):
        self.features_to_monitor = features_to_monitor
        self.monitored_features_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.monitored_features_ = [c for c in self.features_to_monitor if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.monitored_features_:
            if col in X.columns:
                X[f"{col}_was_nan"] = X[col].isna().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        extra = [f"{c}_was_nan" for c in self.monitored_features_]
        return np.array(list(input_features) + extra if input_features is not None else extra)


class ModelInputImputer(BaseEstimator, TransformerMixin):
    """
    Convert semantic feature missingness into a finite estimator matrix.

    IMPORTANT:
    The original feature DataFrame keeps NaN values.

    Imputation here exists ONLY at the estimator boundary and is fitted
    exclusively on training data.

    Missingness indicators are added before this transformer, so an
    imputed numerical placeholder must not be interpreted as a real
    physiological measurement.
    """

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

            if pd.isna(median):
                # Normally completely-empty features should already
                # have been removed by FeatureSelector.
                self.fill_values_[col] = 0.0
            else:
                self.fill_values_[col] = float(median)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)

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
    """Configuration for the full model pipeline."""
    model_type: str = "logistic_regression"
    group_by: Literal["session", "user"] = "session"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    impute_strategy: str = "median"  # for non-HRV/IBI features


def create_model_pipeline(config: ModelPipelineConfig) -> Pipeline:
    """
    Create a full sklearn pipeline with preprocessing + model.

    The pipeline handles:
    1. Feature selection (numeric only)
    2. NaN indicators for semantically meaningful missingness
    3. HRV/IBI-aware imputation (fit on train, transform on test)
    4. Standard scaling
    5. Model

    This pipeline should be fit ONLY on training data.
    """
    # HRV features that should keep NaN
    hrv_features = ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50"]
    ibi_availability = ["ibi_available", "ibi_coverage_ratio"]
    features_to_monitor = hrv_features + ibi_availability

    steps = [
        ("feature_selector", FeatureSelector()),
        ("nan_indicator", NaNIndicator(features_to_monitor=features_to_monitor)),
        ("model_input_imputer", ModelInputImputer()),
        ("scaler", StandardScaler()),
    ]

    # Model will be added by the training function
    return Pipeline(steps)


@dataclass
class TrainedModelBundle:
    """Container for fitted preprocessing pipeline and model."""
    preprocessing_pipeline: Pipeline
    model: BaselineModel
    split_result: SplitResult
    config: ModelPipelineConfig


def train_with_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    group_column: Optional[pd.Series] = None,
    config: Optional[ModelPipelineConfig] = None,
) -> TrainedModelBundle:
    """
    Train model with group-aware split and proper preprocessing.

    Args:
        X: Feature matrix
        y: Labels
        group_column: Optional pre-computed group column (session_id or user_id).
                     If not provided, will be extracted from X using config.group_by.
        config: Pipeline configuration

    Returns:
        TrainedModelBundle with fitted preprocessing pipeline, model, and split info
    """
    from anxietywatch_ml.models.baseline import create_model
    from anxietywatch_ml.evaluation.splitting import group_aware_split, get_group_column

    if config is None:
        config = ModelPipelineConfig()

    # Group-aware split
    if group_column is None:
        group_column = get_group_column(X, config.group_by)
    split_result = group_aware_split(
        X, y, group_column,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        group_by=config.group_by,
    )

    # Create and fit preprocessing pipeline on TRAIN data only
    preprocessing_pipeline = create_model_pipeline(config)
    X_train = X.iloc[split_result.train_indices]
    y_train = y.iloc[split_result.train_indices]
    preprocessing_pipeline.fit(X_train)

    # Transform all splits
    X_train_transformed = preprocessing_pipeline.transform(X_train)
    y_train = y.iloc[split_result.train_indices]

    # Create and fit model on transformed training data
    model = create_model({
        "model": {
            "type": config.model_type,
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 200,
                "class_weight": "balanced",
                "random_state": config.random_state,
            },
            "dummy": {
                "strategy": "prior",
                "constant": 0,
            },
        }
    })
    model.fit(X_train_transformed, y_train)

    return TrainedModelBundle(
        preprocessing_pipeline=preprocessing_pipeline,
        model=model,
        split_result=split_result,
        config=config,
    )


def evaluate_pipeline(
    bundle: TrainedModelBundle,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Evaluate pipeline on train/val/test splits.

    Returns dict with metrics for each split.
    """
    from anxietywatch_ml.evaluation.metrics import evaluate, create_evaluator

    results = {}
    evaluator_config = create_evaluator({})

    for split_name, indices in [
        ("train", bundle.split_result.train_indices),
        ("val", bundle.split_result.val_indices),
        ("test", bundle.split_result.test_indices),
    ]:
        if len(indices) == 0:
            results[split_name] = None
            continue

        X_split = X.iloc[indices]
        y_split = y.iloc[indices]

        # Transform using fitted preprocessing pipeline
        X_transformed = bundle.preprocessing_pipeline.transform(X_split)

        # Predict using fitted model
        y_pred = bundle.model.predict(X_transformed)
        y_proba = bundle.model.predict_proba(X_transformed)

        metrics_result = evaluate(y_split.values, y_pred, y_proba)
        results[split_name] = {
            "metrics": metrics_result.metrics,
            "metrics_available": metrics_result.metrics_available,
            "n_samples": len(y_split),
            "n_positive": int(y_split.sum()),
        }

    return results