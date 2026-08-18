"""
Baseline models for AnxietyWatch ML.

IMPORTANT: These are INFRASTRUCTURE BASELINES only.
They are NOT clinical models and do NOT detect anxiety.
They exist solely to validate the ML pipeline plumbing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anxietywatch_ml.models.serialization import save_model, load_model, PickleModelSerializer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for baseline model."""
    model_type: str = "baseline"  # baseline | logistic_regression | dummy
    # LogisticRegression params
    C: float = 1.0
    max_iter: int = 200
    class_weight: str = "balanced"
    random_state: int = 42
    # DummyClassifier params
    dummy_strategy: str = "prior"
    dummy_constant: int = 0


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaselineModel":
        pass


class SklearnBaselineModel(BaselineModel, BaseEstimator):
    """
    Wrapper around scikit-learn classifiers for baseline models.

    This is an INFRASTRUCTURE BASELINE - NOT A CLINICAL MODEL.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False
        self._feature_names: list[str] = []

    def _create_pipeline(self) -> Pipeline:
        """Create the appropriate pipeline based on config."""
        if self.config.model_type == "logistic_regression":
            clf = LogisticRegression(
                C=self.config.C,
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                solver="lbfgs",
            )
        elif self.config.model_type == "dummy":
            clf = DummyClassifier(
                strategy=self.config.dummy_strategy,
                constant=self.config.dummy_constant,
                random_state=self.config.random_state,
            )
        else:  # baseline = dummy with prior
            clf = DummyClassifier(
                strategy="prior",
                random_state=self.config.random_state,
            )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series) -> "SklearnBaselineModel":
        """Fit the baseline model."""
        logger.info(f"Fitting {self.config.model_type} baseline model on {X.shape[0]} samples, {X.shape[1]} features")

        # Store feature names for consistency
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
            X_df = X
        else:
            # numpy array - create default feature names
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self._feature_names)

        # Handle NaN/inf in features
        X_clean = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.pipeline = self._create_pipeline()
        self.pipeline.fit(X_clean, y)
        self._is_fitted = True

        logger.info(f"Model fitted. Classes: {self.pipeline.classes_}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to DataFrame if needed
        if hasattr(X, 'columns'):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self._feature_names)

        # Ensure same feature order
        X_df = X_df[self._feature_names]
        X_clean = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        return self.pipeline.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities, ensuring 2 columns for binary classification."""
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to DataFrame if needed
        if hasattr(X, 'columns'):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self._feature_names)

        X_df = X_df[self._feature_names]
        X_clean = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        proba = self.pipeline.predict_proba(X_clean)

        # Ensure binary classification always returns 2 columns
        if proba.shape[1] == 1:
            # Only one class present in training - add zero column for missing class
            classes = self.pipeline.classes_
            if classes[0] == 0:
                # Missing class 1
                proba = np.column_stack([proba, np.zeros(len(proba))])
            else:
                # Missing class 0
                proba = np.column_stack([np.zeros(len(proba)), proba])

        return proba

    def save(self, path: Path) -> None:
        """Save model to disk using serialization interface."""
        # The serializer expects a BaselineModel with _is_fitted attribute
        save_model(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "SklearnBaselineModel":
        """Load model from disk using serialization interface."""
        return load_model(path, cls)
        model._feature_names = data["feature_names"]
        model._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return model


def create_model(config: dict) -> BaselineModel:
    """Factory function to create baseline model from config."""
    model_cfg = config.get("model", {})
    model_config = ModelConfig(
        model_type=model_cfg.get("type", "baseline"),
        C=model_cfg.get("logistic_regression", {}).get("C", 1.0),
        max_iter=model_cfg.get("logistic_regression", {}).get("max_iter", 200),
        class_weight=model_cfg.get("logistic_regression", {}).get("class_weight", "balanced"),
        random_state=model_cfg.get("logistic_regression", {}).get("random_state", 42),
        dummy_strategy=model_cfg.get("dummy", {}).get("strategy", "prior"),
        dummy_constant=model_cfg.get("dummy", {}).get("constant", 0),
    )
    return SklearnBaselineModel(model_config)