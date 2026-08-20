"""Event-anchored raw-window inference for the prototype serving service.

The ML service owns windowing: a ``PredictWindowRequest`` carries the raw
telemetry covering the detector event period (possibly spanning several backend
batches - there is no ``batchId``), and this module flattens, sorts, trims to
``[detectedAt - window_size, detectedAt]``, validates data quality and reuses
the EXACT canonical preprocessing/feature-building path shared with training,
then delegates to the existing :class:`GroundTruthPredictor`.

Training-serving skew elimination
---------------------------------
The processor NEVER hardcodes a window contract. In production it is built via
:meth:`EventWindowProcessor.from_bundle`, which reads the training-time config
embedded in the serialized bundle (``runtime_config``) and derives the window
contract (``GroundTruthBuilderConfig``), the preprocessing pipeline
(``create_pipeline``) and the feature builder (``create_feature_builder``)
through the SAME factories the offline
:func:`~anxietywatch_ml.ground_truth.builder.create_ground_truth_builder` uses.
A bundle retrained with a different window (e.g. 90s / 20 samples) is therefore
served with that exact contract automatically.
"""

from datetime import timedelta
from typing import Optional

import pandas as pd

from anxietywatch_ml.features.builder import FeatureBuilder, create_feature_builder
from anxietywatch_ml.ground_truth.builder import (
    GroundTruthBuilderConfig,
    create_ground_truth_builder_config,
)
from anxietywatch_ml.pipelines.model_pipeline import TrainedModelBundle
from anxietywatch_ml.preprocessing.pipeline import PreprocessingPipeline, create_pipeline
from anxietywatch_ml.serving.contracts import (
    FEATURE_SCHEMA,
    PredictResponse,
    PredictWindowRequest,
)
from anxietywatch_ml.serving.predictor import GroundTruthPredictor, PredictorError


class EventWindowProcessor:
    """Compute features from a raw event window and produce a prediction."""

    def __init__(
        self,
        predictor: Optional[GroundTruthPredictor] = None,
        window_config: Optional[GroundTruthBuilderConfig] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.predictor = predictor
        # Defaults mirror the offline GroundTruthDatasetBuilder defaults exactly
        # (same canonical objects), so a config-free processor and a config-free
        # builder can never diverge.
        self._window_config = window_config or GroundTruthBuilderConfig()
        self._prep = preprocessing or PreprocessingPipeline()
        self._feature_builder = feature_builder or FeatureBuilder()

    @property
    def window_config(self) -> GroundTruthBuilderConfig:
        """The window contract this processor enforces (single shared source)."""
        return self._window_config

    @classmethod
    def from_bundle(
        cls,
        bundle: TrainedModelBundle,
        predictor: Optional[GroundTruthPredictor] = None,
    ) -> "EventWindowProcessor":
        """Build a processor from the training-time config in a serialized bundle.

        The bundle captures the full ``config`` dict at training time in
        ``runtime_config`` (ground_truth / window / features / preprocessing).
        Deriving the window contract, preprocessing and feature builder from
        that embedded config via the SAME factories as the offline builder is
        what guarantees training-serving parity for any retrained artifact.
        """
        runtime_config = bundle.runtime_config or {}
        return cls(
            predictor=predictor,
            window_config=create_ground_truth_builder_config(runtime_config),
            preprocessing=create_pipeline(runtime_config),
            feature_builder=create_feature_builder(runtime_config),
        )

    def build_features(self, request: PredictWindowRequest) -> dict:
        """Build the 16-feature vector for an event-anchored raw window.

        This is the parity surface tested against the offline
        GroundTruthDatasetBuilder path. No model is involved.
        """
        window_size = float(self._window_config.window_size_seconds)
        min_samples = int(self._window_config.min_samples_per_window)
        min_hr_ratio = float(self._window_config.min_hr_ratio)

        flat = self._prep.flatten_samples(
            request.samples,
            user_id=request.user_id,
            device_id=request.device_id,
            session_id=request.session_id,
        )
        flat = flat.sort_values("timestamp").reset_index(drop=True)

        t_end = request.detected_at
        t_start = t_end - timedelta(seconds=window_size)
        window = flat[(flat["timestamp"] >= t_start) & (flat["timestamp"] <= t_end)].copy()

        if window.empty:
            raise PredictorError(
                "no telemetry samples fall within the "
                f"[detectedAt - {window_size:.0f}s, detectedAt] window"
            )
        if len(window) < min_samples:
            raise PredictorError(
                f"insufficient window data: {len(window)} samples < "
                f"{min_samples} required"
            )
        hr_ratio = float(window["heart_rate_bpm"].notna().mean())
        if hr_ratio < min_hr_ratio:
            raise PredictorError(
                f"insufficient heart-rate coverage: {hr_ratio:.3f} < "
                f"{min_hr_ratio} required"
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