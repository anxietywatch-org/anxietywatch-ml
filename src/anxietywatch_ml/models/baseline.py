"""
Baseline models for AnxietyWatch ML.

IMPORTANT: These are INFRASTRUCTURE BASELINES only.
They are NOT clinical models and do NOT detect anxiety.
They exist solely to validate the ML pipeline plumbing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from anxietywatch_ml.models.serialization import load_model, save_model

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for baseline model."""

    model_type: str = "baseline"  # baseline | logistic_regression | dummy
    C: float = 1.0
    max_iter: int = 200
    class_weight: str | None = "balanced"
    random_state: int = 42
    dummy_strategy: str = "prior"
    dummy_constant: int = 0


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series,
    ) -> "BaselineModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaselineModel":
        raise NotImplementedError


class SklearnBaselineModel(BaselineModel, BaseEstimator):
    """Wrapper around scikit-learn classifiers used as infrastructure baselines."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False
        self._feature_names: list[str] = []

    def _create_pipeline(self) -> Pipeline:
        if self.config.model_type == "logistic_regression":
            classifier = LogisticRegression(
                C=self.config.C,
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                solver="lbfgs",
            )
        elif self.config.model_type == "dummy":
            classifier = DummyClassifier(
                strategy=self.config.dummy_strategy,
                constant=self.config.dummy_constant,
                random_state=self.config.random_state,
            )
        else:
            classifier = DummyClassifier(
                strategy="prior",
                random_state=self.config.random_state,
            )

        # Scaling/imputation belong to model_pipeline.py.  Keeping the
        # estimator wrapper free of another StandardScaler prevents applying
        # preprocessing twice when the model is inside TrainedModelBundle.
        return Pipeline([("classifier", classifier)])

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if hasattr(X, "columns"):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=self._feature_names)

        if self._feature_names:
            frame = frame[self._feature_names]

        # Standalone baseline-model tests may call this wrapper without the
        # external model pipeline.  The production bundle path should already
        # be finite before reaching this class.
        return frame.replace([np.inf, -np.inf], np.nan).fillna(0)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series,
    ) -> "SklearnBaselineModel":
        logger.info(
            "Fitting %s baseline model on %d samples, %d features",
            self.config.model_type,
            X.shape[0],
            X.shape[1],
        )

        if hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = [
                f"feature_{index}"
                for index in range(X.shape[1])
            ]

        X_clean = self._as_frame(X)
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(X_clean, y)
        self._is_fitted = True

        logger.info("Model fitted. Classes: %s", self.pipeline.classes_)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.pipeline.predict(self._as_frame(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        proba = self.pipeline.predict_proba(self._as_frame(X))

        if proba.shape[1] == 1:
            classes = self.pipeline.classes_
            if classes[0] == 0:
                proba = np.column_stack([proba, np.zeros(len(proba))])
            else:
                proba = np.column_stack([np.zeros(len(proba)), proba])

        return proba

    def save(self, path: Path) -> None:
        save_model(self, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SklearnBaselineModel":
        return load_model(path, cls)


def create_model(config: dict) -> BaselineModel:
    """Factory function to create baseline model from configuration."""

    model_cfg = config.get("model", {})
    logistic_cfg = model_cfg.get("logistic_regression", {})
    dummy_cfg = model_cfg.get("dummy", {})

    model_config = ModelConfig(
        model_type=model_cfg.get("type", "baseline"),
        C=logistic_cfg.get("C", 1.0),
        max_iter=logistic_cfg.get("max_iter", 200),
        class_weight=logistic_cfg.get("class_weight", "balanced"),
        random_state=logistic_cfg.get(
            "random_state",
            config.get("random_seed", 42),
        ),
        dummy_strategy=dummy_cfg.get("strategy", "prior"),
        dummy_constant=dummy_cfg.get("constant", 0),
    )
    return SklearnBaselineModel(model_config)
