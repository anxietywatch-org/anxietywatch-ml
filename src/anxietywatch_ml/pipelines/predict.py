"""
Prediction pipeline for AnxietyWatch ML.

Loads a trained model and runs inference on new data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.validation import ValidationResult, validate_dataframe, log_validation_result
from anxietywatch_ml.evaluation.metrics import EvaluationResult, evaluate
from anxietywatch_ml.features.builder import FeatureBuilder, create_feature_builder
from anxietywatch_ml.models.baseline import BaselineModel
from anxietywatch_ml.preprocessing.pipeline import PreprocessingPipeline, create_pipeline, WindowedData

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of prediction pipeline."""
    predictions: pd.DataFrame  # Columns: window_id, user_id, session_id, prediction, probability, timestamp
    window_metadata: list[dict]
    metrics: Optional[EvaluationResult] = None


class PredictionPipeline:
    """
    Prediction pipeline for AnxietyWatch ML.

    Loads a trained model and applies it to new telemetry data.
    """

    def __init__(self, config: dict, model_path: Path):
        self.config = config
        self.model_path = model_path

        # Initialize components
        self.preprocessing = create_pipeline(config)
        self.feature_builder = create_feature_builder(config)
        self.model = BaselineModel.load(model_path)

    def run(
        self,
        batches: list,
        ground_truth: Optional[pd.Series] = None,
    ) -> PredictionResult:
        """
        Run prediction on telemetry batches.

        Args:
            batches: List of TelemetryBatch objects
            ground_truth: Optional true labels for evaluation
        """
        logger.info(f"Running prediction on {len(batches)} batches")

        # Preprocessing
        windowed_data = self.preprocessing.run(batches)

        # Feature engineering
        X = self.feature_builder.build(windowed_data.windows)

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Build results DataFrame
        results = []
        for i, (pred, prob, meta) in enumerate(zip(predictions, probabilities, windowed_data.window_metadata)):
            results.append({
                "window_id": i,
                "user_id": meta["user_id"],
                "device_id": meta["device_id"],
                "session_id": meta["session_id"],
                "window_start": meta["window_start"],
                "window_end": meta["window_end"],
                "prediction": int(pred),
                "probability": float(prob),
                "n_samples": meta["n_samples"],
            })

        results_df = pd.DataFrame(results)

        # Evaluate if ground truth provided
        metrics = None
        if ground_truth is not None:
            if len(ground_truth) == len(predictions):
                metrics = evaluate(ground_truth.values, predictions, probabilities)
                logger.info(f"Evaluation metrics: {metrics.metrics}")
            else:
                logger.warning(f"Ground truth length ({len(ground_truth)}) != predictions ({len(predictions)})")

        return PredictionResult(
            predictions=results_df,
            window_metadata=windowed_data.window_metadata,
            metrics=metrics,
        )

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        ground_truth: Optional[pd.Series] = None,
    ) -> PredictionResult:
        """
        Run prediction from a pre-flattened DataFrame.

        Expected columns: timestamp, heart_rate_bpm, ibi_ms, skin_temperature_celsius,
        quality_heart_rate, quality_ibi, quality_wearing_state, user_id, session_id, device_id
        """
        # Validate
        result = validate_dataframe(df)
        log_validation_result(result, "Input DataFrame validation")
        if not result.is_valid:
            raise ValueError(f"Invalid input DataFrame: {result.errors}")

        # Convert DataFrame to WindowedData format
        # This is a simplified path - in practice you'd use the full preprocessing
        windowed_data = self._dataframe_to_windows(df)

        # Feature engineering
        X = self.feature_builder.build(windowed_data.windows)

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Build results
        results = []
        for i, (pred, prob, meta) in enumerate(zip(predictions, probabilities, windowed_data.window_metadata)):
            results.append({
                "window_id": i,
                "user_id": meta["user_id"],
                "device_id": meta["device_id"],
                "session_id": meta["session_id"],
                "window_start": meta["window_start"],
                "window_end": meta["window_end"],
                "prediction": int(pred),
                "probability": float(prob),
                "n_samples": meta["n_samples"],
            })

        results_df = pd.DataFrame(results)

        # Evaluate
        metrics = None
        if ground_truth is not None:
            if len(ground_truth) == len(predictions):
                metrics = evaluate(ground_truth.values, predictions, probabilities)

        return PredictionResult(
            predictions=results_df,
            window_metadata=windowed_data.window_metadata,
            metrics=metrics,
        )

    def _dataframe_to_windows(self, df: pd.DataFrame) -> WindowedData:
        """Convert flat DataFrame to WindowedData format."""
        # Simple approach: treat each session as one window
        windows = []
        metadata = []

        for session_id, session_df in df.groupby("session_id"):
            session_df = session_df.sort_values("timestamp").reset_index(drop=True)
            windows.append(session_df)
            metadata.append({
                "session_id": session_id,
                "user_id": session_df["user_id"].iloc[0] if "user_id" in session_df.columns else None,
                "device_id": session_df["device_id"].iloc[0] if "device_id" in session_df.columns else None,
                "window_start": session_df["timestamp"].min(),
                "window_end": session_df["timestamp"].max(),
                "window_index": 0,
                "n_samples": len(session_df),
            })

        return WindowedData(
            windows=windows,
            window_metadata=metadata,
            original_batches=[],
        )


def run_prediction(
    model_path: str,
    config_path: Optional[str] = None,
    input_data: Optional[str] = None,
) -> PredictionResult:
    """Entry point for running prediction pipeline."""
    config = load_config(config_path)
    pipeline = PredictionPipeline(config, Path(model_path))

    # For now, generate synthetic data for demo
    from anxietywatch_ml.data.synthetic import create_generator
    generator = create_generator(config)
    batches = generator.generate_dataset()

    return pipeline.run(batches)