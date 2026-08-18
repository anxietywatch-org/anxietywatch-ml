"""
Training pipeline for AnxietyWatch ML.

Orchestrates the full training flow:
synthetic data -> preprocessing -> features -> model training -> evaluation
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.synthetic import SyntheticTelemetryGenerator, create_generator
from anxietywatch_ml.data.validation import validate_batch, log_validation_result
from anxietywatch_ml.evaluation.metrics import EvaluationConfig, EvaluationResult, evaluate, create_evaluator
from anxietywatch_ml.features.builder import FeatureBuilder, create_feature_builder
from anxietywatch_ml.models.baseline import BaselineModel, create_model
from anxietywatch_ml.preprocessing.pipeline import PreprocessingPipeline, create_pipeline, WindowedData

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of training pipeline."""
    model: BaselineModel
    train_metrics: EvaluationResult
    val_metrics: Optional[EvaluationResult]
    test_metrics: EvaluationResult
    feature_names: list[str]
    n_train: int
    n_val: int
    n_test: int


class TrainingPipeline:
    """
    Full training pipeline for AnxietyWatch ML.

    This pipeline uses SYNTHETIC DATA ONLY.
    It does NOT train a clinical anxiety detector.
    """

    def __init__(self, config: dict):
        self.config = config
        self.random_seed = config.get("random_seed", 42)

        # Initialize components
        self.generator = create_generator(config)
        self.preprocessing = create_pipeline(config)
        self.feature_builder = create_feature_builder(config)
        self.model = create_model(config)
        self.evaluator_config = create_evaluator(config)

        # Training splits
        train_cfg = config.get("training", {})
        self.test_size = train_cfg.get("test_size", 0.2)
        self.val_size = train_cfg.get("val_size", 0.1)
        self.stratify = train_cfg.get("stratify", True)

    def run(self, model_output_path: Optional[Path] = None) -> TrainingResult:
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("Starting AnxietyWatch ML Training Pipeline")
        logger.info("DATA: SYNTHETIC - NOT CLINICAL")
        logger.info("=" * 60)

        # 1. Generate synthetic data
        logger.info("Step 1/6: Generating synthetic telemetry data...")
        batches, anomaly_sessions = self.generator.generate_dataset()
        logger.info(f"Generated {len(batches)} batches, {sum(anomaly_sessions.values())} anomaly sessions")

        # Validate a sample of batches
        for batch in batches[:5]:
            result = validate_batch(batch)
            log_validation_result(result, "Batch validation")

        # 2. Preprocessing
        logger.info("Step 2/6: Preprocessing and windowing...")
        windowed_data = self.preprocessing.run(batches)

        # 3. Feature engineering
        logger.info("Step 3/6: Building features...")
        X = self.feature_builder.build(windowed_data.windows)

        # 4. Create labels (synthetic - based on anomaly sessions)
        logger.info("Step 4/6: Creating synthetic labels...")
        y = self._create_labels(windowed_data, anomaly_sessions)

        # 5. Train/val/test split
        logger.info("Step 5/6: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y if self.stratify else None,
        )

        if self.val_size > 0:
            val_relative_size = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_relative_size,
                random_state=self.random_seed,
                stratify=y_train if self.stratify else None,
            )
        else:
            X_val, y_val = None, None

        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)}")

        # 6. Train model
        logger.info("Step 6/6: Training model...")
        self.model.fit(X_train, y_train)

        # 7. Evaluate
        logger.info("Evaluating on train set...")
        train_proba = self.model.predict_proba(X_train)
        train_pred = self.model.predict(X_train)
        train_metrics = evaluate(y_train, train_pred, train_proba, self.evaluator_config)

        val_metrics = None
        if X_val is not None:
            logger.info("Evaluating on validation set...")
            val_proba = self.model.predict_proba(X_val)
            val_pred = self.model.predict(X_val)
            val_metrics = evaluate(y_val, val_pred, val_proba, self.evaluator_config)

        logger.info("Evaluating on test set...")
        test_proba = self.model.predict_proba(X_test)
        test_pred = self.model.predict(X_test)
        test_metrics = evaluate(y_test, test_pred, test_proba, self.evaluator_config)

        # Log results
        logger.info("\n" + str(train_metrics))
        if val_metrics:
            logger.info("\n" + str(val_metrics))
        logger.info("\n" + str(test_metrics))

        # Save model if path provided
        if model_output_path:
            self.model.save(model_output_path)

        return TrainingResult(
            model=self.model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_names=list(X.columns),
            n_train=len(X_train),
            n_val=len(X_val) if X_val is not None else 0,
            n_test=len(X_test),
        )

    def _create_labels(
        self,
        windowed_data: WindowedData,
        anomaly_sessions: Optional[dict[UUID, bool]] = None,
    ) -> pd.Series:
        """
        Create synthetic labels for training.

        Uses the anomaly session flags from the synthetic generator.
        If anomaly_sessions is not provided, falls back to HR-based heuristic.
        """
        labels = []

        for i, (window, meta) in enumerate(zip(windowed_data.windows, windowed_data.window_metadata)):
            session_id_str = meta.get("session_id")
            # Convert string session_id to UUID for lookup in anomaly_sessions
            session_id_uuid = UUID(session_id_str) if session_id_str else None
            
            if anomaly_sessions is not None and session_id_uuid in anomaly_sessions:
                # Use the ground truth anomaly flag from synthetic generator
                label = 1 if anomaly_sessions[session_id_uuid] else 0
            else:
                # Fallback: HR-based heuristic (for backward compatibility)
                hr = window["heart_rate_bpm"].dropna()
                if len(hr) > 0:
                    mean_hr = hr.mean()
                    label = 1 if mean_hr > 100 else 0
                else:
                    label = 0
            labels.append(label)

        return pd.Series(labels, name="label")


def run_training(config_path: Optional[str] = None, model_output: Optional[str] = None) -> TrainingResult:
    """Entry point for running training pipeline."""
    config = load_config(config_path)
    pipeline = TrainingPipeline(config)
    output_path = Path(model_output) if model_output else None
    return pipeline.run(output_path)