"""Prototype inference service (005-A).

HTTP inference over a trained GroundTruth bundle: contracts, predictor and
FastAPI application. The model is loaded once; preprocessing is reused via
``transform_for_inference`` and never refit.

007-B1: event-anchored raw-window inference (``/predict/window``) reuses the
canonical preprocessing/feature-building path via ``EventWindowProcessor``.
"""

from anxietywatch_ml.serving.app import API_KEY_ENV, DEFAULT_MODEL_PATH, app, create_app
from anxietywatch_ml.serving.contracts import (
    FEATURE_SCHEMA,
    FORBIDDEN_FEATURES,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictWindowRequest,
)
from anxietywatch_ml.serving.predictor import (
    GroundTruthPredictor,
    PredictorError,
)
from anxietywatch_ml.serving.train_demo import train_demo_model
from anxietywatch_ml.serving.window_inference import (
    MIN_HR_RATIO,
    MIN_WINDOW_SAMPLES,
    WINDOW_SIZE_SECONDS,
    EventWindowProcessor,
)

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODEL_PATH",
    "EventWindowProcessor",
    "FEATURE_SCHEMA",
    "FORBIDDEN_FEATURES",
    "GroundTruthPredictor",
    "HealthResponse",
    "MIN_HR_RATIO",
    "MIN_WINDOW_SAMPLES",
    "PredictRequest",
    "PredictResponse",
    "PredictWindowRequest",
    "PredictorError",
    "WINDOW_SIZE_SECONDS",
    "app",
    "create_app",
    "train_demo_model",
]