"""Prediction pipeline for a serialized AnxietyWatch ML TrainedModelBundle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from anxietywatch_ml.data.validation import (
    log_validation_result,
    validate_dataframe,
)
from anxietywatch_ml.evaluation.metrics import EvaluationResult, evaluate
from anxietywatch_ml.features.builder import create_feature_builder
from anxietywatch_ml.pipelines.model_pipeline import (
    TrainedModelBundle,
    load_trained_bundle,
    transform_for_inference,
)
from anxietywatch_ml.preprocessing.pipeline import WindowedData, create_pipeline

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of prediction pipeline."""

    predictions: pd.DataFrame
    window_metadata: list[dict]
    metrics: Optional[EvaluationResult] = None


class PredictionPipeline:
    """Load one complete training bundle and reuse its fitted preprocessing."""

    def __init__(self, config: dict, model_path: Path):
        self.model_path = model_path
        self.bundle: TrainedModelBundle = load_trained_bundle(model_path)

        # The artifact's training-time data/feature configuration is the source
        # of truth for inference.  This prevents an external config file from
        # silently changing feature semantics after training.
        self.config = self.bundle.runtime_config or config
        self.preprocessing = create_pipeline(self.config)
        self.feature_builder = create_feature_builder(self.config)

    def _predict_features(self, X: pd.DataFrame) -> tuple:
        X_transformed = transform_for_inference(self.bundle, X)
        predictions = self.bundle.model.predict(X_transformed)
        probabilities = self.bundle.model.predict_proba(X_transformed)[:, 1]
        return predictions, probabilities

    def run(
        self,
        batches: list,
        ground_truth: Optional[pd.Series] = None,
    ) -> PredictionResult:
        logger.info("Running prediction on %d batches", len(batches))

        windowed_data = self.preprocessing.run(batches)
        X = self.feature_builder.build(windowed_data.windows)
        predictions, probabilities = self._predict_features(X)

        results = []
        for index, (prediction, probability, meta) in enumerate(
            zip(
                predictions,
                probabilities,
                windowed_data.window_metadata,
            )
        ):
            results.append(
                {
                    "window_id": index,
                    "user_id": meta["user_id"],
                    "device_id": meta["device_id"],
                    "session_id": meta["session_id"],
                    "window_start": meta["window_start"],
                    "window_end": meta["window_end"],
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "n_samples": meta["n_samples"],
                }
            )

        metrics = None
        if ground_truth is not None:
            if len(ground_truth) != len(predictions):
                raise ValueError(
                    "Ground truth length does not match prediction length"
                )
            metrics = evaluate(
                ground_truth.values,
                predictions,
                self.bundle.model.predict_proba(
                    transform_for_inference(self.bundle, X)
                ),
            )

        return PredictionResult(
            predictions=pd.DataFrame(results),
            window_metadata=windowed_data.window_metadata,
            metrics=metrics,
        )

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        ground_truth: Optional[pd.Series] = None,
    ) -> PredictionResult:
        validation = validate_dataframe(df)
        log_validation_result(validation, "Input DataFrame validation")
        if not validation.is_valid:
            raise ValueError(f"Invalid input DataFrame: {validation.errors}")

        windowed_data = self._dataframe_to_windows(df)
        X = self.feature_builder.build(windowed_data.windows)
        predictions, probabilities = self._predict_features(X)

        results = []
        for index, (prediction, probability, meta) in enumerate(
            zip(
                predictions,
                probabilities,
                windowed_data.window_metadata,
            )
        ):
            results.append(
                {
                    "window_id": index,
                    "user_id": meta["user_id"],
                    "device_id": meta["device_id"],
                    "session_id": meta["session_id"],
                    "window_start": meta["window_start"],
                    "window_end": meta["window_end"],
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "n_samples": meta["n_samples"],
                }
            )

        metrics = None
        if ground_truth is not None:
            if len(ground_truth) != len(predictions):
                raise ValueError(
                    "Ground truth length does not match prediction length"
                )
            proba_matrix = self.bundle.model.predict_proba(
                transform_for_inference(self.bundle, X)
            )
            metrics = evaluate(
                ground_truth.values,
                predictions,
                proba_matrix,
            )

        return PredictionResult(
            predictions=pd.DataFrame(results),
            window_metadata=windowed_data.window_metadata,
            metrics=metrics,
        )

    def _dataframe_to_windows(self, df: pd.DataFrame) -> WindowedData:
        # Transitional CSV/Parquet path.  Raw batch inference remains the
        # canonical path for the MVP because this simplified conversion treats
        # one session as one window.
        windows = []
        metadata = []

        for session_id, session_df in df.groupby("session_id"):
            session_df = session_df.sort_values("timestamp").reset_index(drop=True)
            windows.append(session_df)
            metadata.append(
                {
                    "session_id": session_id,
                    "user_id": (
                        session_df["user_id"].iloc[0]
                        if "user_id" in session_df.columns
                        else None
                    ),
                    "device_id": (
                        session_df["device_id"].iloc[0]
                        if "device_id" in session_df.columns
                        else None
                    ),
                    "window_start": session_df["timestamp"].min(),
                    "window_end": session_df["timestamp"].max(),
                    "window_index": 0,
                    "n_samples": len(session_df),
                }
            )

        return WindowedData(
            windows=windows,
            window_metadata=metadata,
            original_batches=[],
        )
