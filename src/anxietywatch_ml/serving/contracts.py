"""HTTP contracts for the AnxietyWatch prototype inference service.

The request is the 16 current GroundTruthDataset features. Every feature is
optional (``None``) where the pipeline supports semantic NaN. Detector and
identity metadata are NOT part of the request: ``extra="forbid"`` rejects
``detector_score``, ``detector_state``, ``rules_version``, ``response``,
``user_id``, ``session_id``, ``device_id``, ``event_id`` with a 422 instead of
silently leaking them into ``X``.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

FEATURE_SCHEMA = [
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_slope_bpm_per_min",
    "hrv_rmssd",
    "hrv_sdnn",
    "ibi_available",
    "ibi_coverage_ratio",
    "skin_temp_mean",
    "quality_good_ratio",
    "quality_fair_ratio",
    "quality_poor_ratio",
    "valid_sample_ratio",
    "window_duration_seconds",
    "sample_count",
]

# Detector / identity metadata that must never enter the feature matrix.
FORBIDDEN_FEATURES = [
    "detector_score",
    "detector_state",
    "rules_version",
    "response",
    "user_id",
    "session_id",
    "device_id",
    "event_id",
]


class PredictRequest(BaseModel):
    """Inference request: one ML window of features (NaN allowed)."""

    model_config = ConfigDict(extra="forbid")

    hr_mean: Optional[float] = None
    hr_std: Optional[float] = None
    hr_min: Optional[float] = None
    hr_max: Optional[float] = None
    hr_slope_bpm_per_min: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    ibi_available: Optional[float] = None
    ibi_coverage_ratio: Optional[float] = None
    skin_temp_mean: Optional[float] = None
    quality_good_ratio: Optional[float] = None
    quality_fair_ratio: Optional[float] = None
    quality_poor_ratio: Optional[float] = None
    valid_sample_ratio: Optional[float] = None
    window_duration_seconds: Optional[float] = None
    sample_count: Optional[float] = None


class PredictResponse(BaseModel):
    """Inference response.

    ``prediction = 1`` means "the model predicts SUPPORT_REQUESTED for an
    event that already passed the Watch detector". It does NOT mean "the user
    has anxiety".
    """

    prediction: int
    support_probability: float
    threshold: float
    model_version: str
    target: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str