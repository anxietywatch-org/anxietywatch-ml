"""FastAPI prototype inference service (v0.1).

The model is loaded ONCE at application startup and reused for every request.

Inference endpoints (``/predict``, ``/predict/window``) are authenticated with
an API key sent as ``X-Api-Key`` and configured via ``ANXIETYWATCH_API_KEY``.
Comparison is constant-time (``secrets.compare_digest``). By default the key is
NOT configured, so inference is refused with 503 ("inference authentication is
not configured") rather than ever becoming public. ``/health`` stays
unauthenticated so probes work. The key is never logged or echoed.

Production startup is fail-fast: when ``ANXIETYWATCH_REQUIRE_MODEL`` is enabled
(as set in the container image) the process aborts if the configured artifact
cannot be loaded. There is never a silent fallback to an untrained model.

In local/dev (``ANXIETYWATCH_REQUIRE_MODEL`` unset) the service starts in a
degraded state: ``/health`` reports ``model_loaded=false`` with HTTP 503 and
the inference endpoints answer 503.
"""

import os
import secrets
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from anxietywatch_ml.serving.contracts import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictWindowRequest,
)
from anxietywatch_ml.serving.predictor import GroundTruthPredictor, PredictorError
from anxietywatch_ml.serving.window_inference import EventWindowProcessor

DEFAULT_MODEL_PATH = os.environ.get(
    "ANXIETYWATCH_MODEL_PATH",
    "models/prototype_v0.1.0.pkl",
)

API_KEY_ENV = "ANXIETYWATCH_API_KEY"


def _require_model_flag(default: bool = False) -> bool:
    """Parse ``ANXIETYWATCH_REQUIRE_MODEL`` (1/true/yes/on => required)."""
    raw = os.getenv("ANXIETYWATCH_REQUIRE_MODEL")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_predictor(path: str):
    """Load the trained bundle ONCE. Returns ``(predictor, model_version)``."""
    predictor = GroundTruthPredictor.from_path(path)
    return predictor, predictor.model_version


def _api_key_guard(configured_key: Optional[str]):
    """Build a FastAPI dependency enforcing ``X-Api-Key`` authentication.

    - No key configured: 503 (inference intentionally unavailable, never public).
    - Missing header: 401.
    - Wrong key: 401 (constant-time comparison, no echo of the supplied value).
    """

    def require_api_key(
        x_api_key: Annotated[Optional[str], Header(alias="X-Api-Key")] = None,
    ) -> None:
        if configured_key is None:
            raise HTTPException(
                status_code=503,
                detail="inference authentication is not configured",
            )
        if x_api_key is None:
            raise HTTPException(status_code=401, detail="missing API key")
        if not secrets.compare_digest(x_api_key, configured_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    return require_api_key


def create_app(
    model_path: str | None = None,
    require_model: bool | None = None,
    api_key: str | None = None,
) -> FastAPI:
    """Build the service.

    ``model_path=None`` reads ``ANXIETYWATCH_MODEL_PATH``.
    ``require_model=None`` reads ``ANXIETYWATCH_REQUIRE_MODEL`` (default False).
    ``api_key=None`` reads ``ANXIETYWATCH_API_KEY``. Inference endpoints are
    always behind API-key authentication; if no key is configured they respond
    503 rather than becoming public. ``api_key`` is injectable for tests.
    When the model is required but cannot be loaded, startup raises so the
    process exits instead of serving traffic without a model.
    """
    path = model_path or DEFAULT_MODEL_PATH
    required = _require_model_flag() if require_model is None else require_model
    configured_key = os.getenv(API_KEY_ENV) if api_key is None else api_key

    predictor = None
    model_loaded = False
    model_version = "unknown"
    if path and Path(path).exists():
        try:
            predictor, model_version = _load_predictor(path)
            model_loaded = predictor is not None
        except Exception:  # noqa: BLE001 - surface as degraded, not crash
            # Invalid/corrupt artifact: never claim healthy.
            predictor = None
            model_loaded = False
            model_version = "unknown"

    if required and not model_loaded:
        raise RuntimeError(
            "cannot start AnxietyWatch inference: model artifact is required "
            f"(ANXIETYWATCH_REQUIRE_MODEL=true) but could not be loaded from "
            f"{path!r}. Refusing to serve without a trained model."
        )

    app = FastAPI(title="AnxietyWatch Prototype Inference", version="0.1.0")

    require_api_key = _api_key_guard(configured_key)
    window_processor = EventWindowProcessor(predictor) if predictor is not None else None

    @app.get("/health", response_model=HealthResponse)
    def health():
        body = HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_version=model_version,
        )
        if model_loaded:
            return body
        # A running process without a model is NOT ready: signal it with a
        # non-2xx so HTTP probes (Azure Container Apps) stop routing traffic.
        return JSONResponse(status_code=503, content=body.model_dump())

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
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

    @app.post(
        "/predict/window",
        response_model=PredictResponse,
        dependencies=[Depends(require_api_key)],
    )
    def predict_window(request: PredictWindowRequest):
        if window_processor is None:
            raise HTTPException(
                status_code=503,
                detail="model not loaded: no valid inference artifact available",
            )
        try:
            return window_processor.predict(request)
        except PredictorError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:  # noqa: BLE001 - never leak an internal stack trace
            raise HTTPException(status_code=500, detail="inference failed")

    return app


app = create_app()