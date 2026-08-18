"""
Training pipeline for AnxietyWatch ML.

The bootstrap trains only on synthetic data.  It produces a complete
TrainedModelBundle containing TRAIN-fitted preprocessing plus the estimator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID

import pandas as pd

from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.validation import log_validation_result, validate_batch
from anxietywatch_ml.evaluation.metrics import EvaluationResult
from anxietywatch_ml.features.builder import create_feature_builder
from anxietywatch_ml.pipelines.model_pipeline import (
    ModelPipelineConfig,
    TrainedModelBundle,
    evaluate_pipeline,
    save_trained_bundle,
    train_with_pipeline,
)
from anxietywatch_ml.preprocessing.pipeline import WindowedData, create_pipeline
from anxietywatch_ml.data.synthetic import create_generator

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of the reproducible training pipeline."""

    bundle: TrainedModelBundle
    train_metrics: EvaluationResult
    val_metrics: Optional[EvaluationResult]
    test_metrics: EvaluationResult
    feature_names: list[str]
    n_train: int
    n_val: int
    n_test: int

    @property
    def model(self):
        """Compatibility accessor for callers that still need the estimator."""
        return self.bundle.model


class TrainingPipeline:
    """Single source of truth for AnxietyWatch ML bootstrap training."""

    def __init__(self, config: dict):
        self.config = config
        self.random_seed = config.get("random_seed", 42)
        self.generator = create_generator(config)
        self.preprocessing = create_pipeline(config)
        self.feature_builder = create_feature_builder(config)

    def run(self, model_output_path: Optional[Path] = None) -> TrainingResult:
        logger.info("=" * 60)
        logger.info("Starting AnxietyWatch ML Training Pipeline")
        logger.info("DATA: SYNTHETIC - NOT CLINICAL")
        logger.info("=" * 60)

        logger.info("Step 1/6: Generating synthetic telemetry data...")
        batches, anomaly_sessions = self.generator.generate_dataset()
        logger.info(
            "Generated %d batches, %d anomaly sessions",
            len(batches),
            sum(anomaly_sessions.values()),
        )

        for batch in batches[:5]:
            validation = validate_batch(batch)
            log_validation_result(validation, "Batch validation")

        logger.info("Step 2/6: Preprocessing and windowing...")
        windowed_data = self.preprocessing.run(batches)

        logger.info("Step 3/6: Building features...")
        X = self.feature_builder.build(windowed_data.windows)

        logger.info("Step 4/6: Creating synthetic labels...")
        y = self._create_labels(windowed_data, anomaly_sessions)

        logger.info("Step 5/6: Group-aware split and TRAIN-only preprocessing...")
        training_cfg = self.config.get("training", {})
        group_by = training_cfg.get("group_by", "session")
        group_key = "session_id" if group_by == "session" else "user_id"

        groups = pd.Series(
            [meta[group_key] for meta in windowed_data.window_metadata],
            index=X.index,
            name=group_key,
        )

        pipeline_config = ModelPipelineConfig(
            model_type=self.config.get("model", {}).get("type", "baseline"),
            group_by=group_by,
            test_size=training_cfg.get("test_size", 0.2),
            val_size=training_cfg.get("val_size", 0.1),
            random_state=training_cfg.get(
                "random_state",
                self.random_seed,
            ),
        )

        bundle = train_with_pipeline(
            X=X,
            y=y,
            group_column=groups,
            config=pipeline_config,
            runtime_config=self.config,
        )

        logger.info("Step 6/6: Evaluating train/val/test...")
        evaluation = evaluate_pipeline(bundle, X, y)

        train_metrics = evaluation["train"]["result"]
        val_metrics = (
            evaluation["val"]["result"]
            if evaluation["val"] is not None
            else None
        )
        test_metrics = evaluation["test"]["result"]

        logger.info("\n%s", train_metrics)
        if val_metrics is not None:
            logger.info("\n%s", val_metrics)
        logger.info("\n%s", test_metrics)

        if model_output_path is not None:
            save_trained_bundle(bundle, model_output_path)
            logger.info("Training bundle saved to %s", model_output_path)

        split = bundle.split_result
        return TrainingResult(
            bundle=bundle,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_names=list(X.columns),
            n_train=len(split.train_indices),
            n_val=len(split.val_indices),
            n_test=len(split.test_indices),
        )

    def _create_labels(
        self,
        windowed_data: WindowedData,
        anomaly_sessions: Optional[dict[UUID, bool]] = None,
    ) -> pd.Series:
        """Map synthetic session ground truth to windows without heuristic fallback."""

        if anomaly_sessions is None:
            raise ValueError(
                "Synthetic training requires anomaly_sessions ground truth."
            )

        labels: list[int] = []
        for meta in windowed_data.window_metadata:
            session_id = meta.get("session_id")
            if not session_id:
                raise ValueError("Window metadata is missing session_id")

            session_uuid = UUID(str(session_id))
            if session_uuid not in anomaly_sessions:
                raise ValueError(
                    f"No synthetic ground truth found for session {session_id}"
                )

            labels.append(1 if anomaly_sessions[session_uuid] else 0)

        return pd.Series(labels, index=range(len(labels)), name="label")


def run_training(
    config_path: Optional[str] = None,
    model_output: Optional[str] = None,
) -> TrainingResult:
    config = load_config(config_path)
    output_path = Path(model_output) if model_output else None
    return TrainingPipeline(config).run(output_path)
