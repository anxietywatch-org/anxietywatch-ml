"""FastAPI prototype inference service (v0.1).

The model is loaded ONCE at application startup and reused for every request.
If no model is available the service reports ``model_loaded=false`` in
``/health`` (it does not fake being healthy) and ``/predict`` answers 503.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from anxietywatch_ml.serving.contracts import HealthResponse, PredictRequest, PredictResponse
from anxietywatch_ml.serving.predictor import GroundTruthPredictor, PredictorError

DEFAULT_MODEL_PATH = os.environ.get(
    "ANXIETYWATCH_MODEL_PATH",
    "models/prototype_v0.1.0.pkl",
)


def create_app(model_path: str | None = None) -> FastAPI:
    """Build the service. ``model_path=None`` reads the environment default."""
    path = model_path or DEFAULT_MODEL_PATH

    predictor = None
    model_loaded = False
    model_version = "unknown"
    if path and Path(path).exists():
        try:
            predictor = GroundTruthPredictor.from_path(path)
            model_loaded = True
            model_version = predictor.model_version
        except Exception:  # noqa: BLE001 - surface as degraded, not crash
            # Invalid/corrupt artifact: never claim healthy.
            predictor = None
            model_loaded = False
            model_version = "unknown"

    app = FastAPI(title="AnxietyWatch Prototype Inference", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_version=model_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="model not loaded: no valid inference artifact available",
            )
        try:
            return predictor.predict(request.model_dump())
        except PredictorError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:  # noqa: BLE001 - never leak an internal stack trace
            raise HTTPException(status_code=500, detail="inference failed")

    return app


app = create_app()