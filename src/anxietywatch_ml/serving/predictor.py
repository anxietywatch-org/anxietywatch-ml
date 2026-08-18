"""Reusable predictor for the prototype inference service.

Loads a ``TrainedModelBundle`` ONCE, validates the feature schema, converts a
request into a DataFrame, applies the existing ``transform_for_inference``
pipeline (never duplicated, never refit), extracts the positive-class
probability and applies the threshold that came from training metadata.
"""

import pandas as pd

from anxietywatch_ml.pipelines.model_pipeline import (
    TrainedModelBundle,
    transform_for_inference,
)
from anxietywatch_ml.serving.contracts import PredictResponse
from anxietywatch_ml.training import load_ground_truth_bundle

MODEL_METADATA_KEYS = (
    "model_version",
    "target",
    "threshold",
    "threshold_source",
    "feature_names",
)


class PredictorError(ValueError):
    """Raised when a prediction cannot be produced (validation or loading)."""


def _read_metadata(bundle: TrainedModelBundle, key: str):
    model_meta = (bundle.runtime_config or {}).get("model", {})
    return model_meta.get(key)


def _require_metadata(bundle: TrainedModelBundle, key: str):
    value = _read_metadata(bundle, key)
    if value is None:
        raise PredictorError(
            f"model artifact is missing required inference metadata: '{key}'. "
            "Retrain with train_ground_truth(output_path=...) so the bundle "
            "carries model_version/target/threshold/feature_names."
        )
    return value


class GroundTruthPredictor:
    """Inference wrapper over a trained GroundTruth bundle.

    The threshold comes from training metadata (``runtime_config.model``);
    there is no silent 0.5 fallback and no threshold recomputation.
    """

    def __init__(self, bundle: TrainedModelBundle):
        self.bundle = bundle
        self.model_version = str(_require_metadata(bundle, "model_version"))
        self.target = str(_require_metadata(bundle, "target"))
        self.threshold = float(_require_metadata(bundle, "threshold"))
        self.threshold_source = str(_require_metadata(bundle, "threshold_source"))
        self.feature_names = list(_require_metadata(bundle, "feature_names"))

        self._feature_set = set(self.feature_names)
        if len(self.feature_names) != len(self._feature_set):
            raise PredictorError("feature schema contains duplicate names")

    @classmethod
    def from_path(cls, path) -> "GroundTruthPredictor":
        """Load the bundle once and build a predictor around it."""
        return cls(load_ground_truth_bundle(path))

    def validate_features(self, features: dict) -> None:
        """Ensure the request satisfies the structural feature schema."""
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise PredictorError(
                f"feature schema violation: missing required features {missing}"
            )
        extra = sorted(set(features) - self._feature_set)
        if extra:
            raise PredictorError(
                f"feature schema violation: unexpected features {extra} "
                "(detector/identity metadata is never accepted)"
            )

    def _build_frame(self, features: dict) -> pd.DataFrame:
        self.validate_features(features)
        row = {name: features.get(name) for name in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)

    def predict(self, features: dict) -> PredictResponse:
        """Produce a prediction WITHOUT fitting anything."""
        X = self._build_frame(features)
        X_transformed = transform_for_inference(self.bundle, X)
        proba = self.bundle.model.predict_proba(X_transformed)
        if proba.shape[1] != 2:
            raise PredictorError("model did not produce binary probabilities")
        support_probability = float(proba[0, 1])
        prediction = int(support_probability >= self.threshold)
        return PredictResponse(
            prediction=prediction,
            support_probability=support_probability,
            threshold=self.threshold,
            model_version=self.model_version,
            target=self.target,
        )