"""Event-anchored raw-window inference for the prototype serving service.

The ML service owns windowing: a ``PredictWindowRequest`` carries the raw
telemetry covering the detector event period (possibly spanning several backend
batches - there is no ``batchId``), and this module flattens, sorts, trims to
``[detectedAt - 60s, detectedAt]``, validates data quality and reuses the
EXACT canonical preprocessing/feature-building path shared with training, then
delegates to the existing :class:`GroundTruthPredictor`.

Mirrors the ground-truth dataset builder semantics:
- window selection ``[T-60s, T]`` inclusive (ground_truth.builder._select_window)
- ``min_samples_per_window = 10`` and ``min_hr_ratio = 0.3`` (configs/base.yaml
  ``ground_truth`` and PreprocessingConfig defaults)
"""

from datetime import timedelta
from typing import Optional

import pandas as pd

from anxietywatch_ml.features.builder import FeatureBuilder
from anxietywatch_ml.preprocessing.pipeline import PreprocessingPipeline
from anxietywatch_ml.serving.contracts import (
    FEATURE_SCHEMA,
    PredictResponse,
    PredictWindowRequest,
)
from anxietywatch_ml.serving.predictor import GroundTruthPredictor, PredictorError

# Canonical event window anchored at detectedAt (matches ground_truth.builder
# and configs/base.yaml ground_truth.window_size_seconds).
WINDOW_SIZE_SECONDS = 60.0
# Data-quality gates (configs/base.yaml ground_truth + PreprocessingConfig).
MIN_WINDOW_SAMPLES = 10
MIN_HR_RATIO = 0.3


class EventWindowProcessor:
    """Compute features from a raw event window and produce a prediction."""

    def __init__(
        self,
        predictor: Optional[GroundTruthPredictor] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.predictor = predictor
        self._prep = preprocessing or PreprocessingPipeline()
        self._feature_builder = feature_builder or FeatureBuilder()

    def build_features(self, request: PredictWindowRequest) -> dict:
        """Build the 16-feature vector for an event-anchored raw window.

        This is the parity surface tested against the offline
        GroundTruthDatasetBuilder path. No model is involved.
        """
        flat = self._prep.flatten_samples(
            request.samples,
            user_id=request.user_id,
            device_id=request.device_id,
            session_id=request.session_id,
        )
        flat = flat.sort_values("timestamp").reset_index(drop=True)

        t_end = request.detected_at
        t_start = t_end - timedelta(seconds=WINDOW_SIZE_SECONDS)
        window = flat[(flat["timestamp"] >= t_start) & (flat["timestamp"] <= t_end)].copy()

        if window.empty:
            raise PredictorError(
                "no telemetry samples fall within the "
                "[detectedAt - 60s, detectedAt] window"
            )
        if len(window) < MIN_WINDOW_SAMPLES:
            raise PredictorError(
                f"insufficient window data: {len(window)} samples < "
                f"{MIN_WINDOW_SAMPLES} required"
            )
        hr_ratio = float(window["heart_rate_bpm"].notna().mean())
        if hr_ratio < MIN_HR_RATIO:
            raise PredictorError(
                f"insufficient heart-rate coverage: {hr_ratio:.3f} < "
                f"{MIN_HR_RATIO} required"
            )

        cleaned = self._prep.clean_window(window)
        windows = self._feature_builder.build([cleaned])
        row = windows.iloc[0]

        features = {}
        for name in FEATURE_SCHEMA:
            value = row.get(name)
            features[name] = None if pd.isna(value) else float(value)
        return features

    def predict(self, request: PredictWindowRequest) -> PredictResponse:
        """Build the window features and predict WITHOUT fitting anything."""
        if self.predictor is None:
            raise PredictorError("predictor is not available")
        return self.predictor.predict(self.build_features(request))