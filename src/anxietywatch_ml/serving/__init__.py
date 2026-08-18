"""Prototype inference service (005-A).

HTTP inference over a trained GroundTruth bundle: contracts, predictor and
FastAPI application. The model is loaded once; preprocessing is reused via
``transform_for_inference`` and never refit.
"""

from anxietywatch_ml.serving.app import DEFAULT_MODEL_PATH, app, create_app
from anxietywatch_ml.serving.contracts import (
    FEATURE_SCHEMA,
    FORBIDDEN_FEATURES,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from anxietywatch_ml.serving.predictor import (
    GroundTruthPredictor,
    PredictorError,
)
from anxietywatch_ml.serving.train_demo import train_demo_model

__all__ = [
    "DEFAULT_MODEL_PATH",
    "FEATURE_SCHEMA",
    "FORBIDDEN_FEATURES",
    "GroundTruthPredictor",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "PredictorError",
    "app",
    "create_app",
    "train_demo_model",
]