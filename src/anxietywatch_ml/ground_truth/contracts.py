"""Ground-truth contracts for the AnxietyWatch ML dataset builder.

Normalized internal schemas (snake_case) for the durable event documents
persisted by the backend:

- ``suspected_events``  -> :class:`SuspectedEvent`
- ``event_decisions``   -> :class:`EventDecision`

The backend stores serialized .NET records (PascalCase) plus ``_id``,
``receivedAt`` and an auth ``userId`` (camelCase). Adapters normalize the
documents via :mod:`anxietywatch_ml.contracts.normalize`, which accepts the
three spellings (PascalCase / camelCase / snake_case) recursively and enforces
the canonical authenticated ``userId`` rule. Unrecognized keys are ignored
(extra="ignore").
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anxietywatch_ml.contracts.normalize import normalize_keys, resolve_identity

_IGNORE_UNKNOWN = ConfigDict(extra="ignore")


class SuspectedEventFeatures(BaseModel):
    """Watch-computed features snapshot at detection time.

    This is EXCLUDED from the model feature matrix (exclude_from_X=true);
    it is kept only for parity checks against ML-computed features.
    """

    model_config = _IGNORE_UNKNOWN

    heart_rate_mean: Optional[float] = None
    heart_rate_max: Optional[float] = None
    heart_rate_slope_bpm_per_minute: Optional[float] = None
    heart_rate_delta_from_baseline: Optional[float] = None
    rmssd_millis: Optional[float] = None
    sdnn_millis: Optional[float] = None
    movement_magnitude_mean: Optional[float] = None
    movement_variance: Optional[float] = None
    valid_sample_ratio: float = 0.0
    last_sample_age_seconds: int = 0
    sample_count: int = 0

    @field_validator("valid_sample_ratio")
    @classmethod
    def _ratio_bounds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("valid_sample_ratio must be within [0, 1]")
        return v


class SuspectedEventBaseline(BaseModel):
    """Watch baseline snapshot at detection time.

    EXCLUDED from the model feature matrix (exclude_from_X=true).
    """

    model_config = _IGNORE_UNKNOWN

    sample_count: int = 0
    mean_heart_rate: float = 0.0
    heart_rate_m2: float = 0.0
    updated_at_epoch_millis: int = 0


class SuspectedEvent(BaseModel):
    """A heuristic detection event as stored in ``suspected_events``."""

    model_config = _IGNORE_UNKNOWN

    event_id: UUID
    device_id: UUID
    user_id: Optional[UUID] = None
    session_id: UUID
    sequence: int = Field(default=0, ge=0)
    detected_at: datetime
    state: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rules_version: str
    features: SuspectedEventFeatures
    baseline: SuspectedEventBaseline


class EventDecision(BaseModel):
    """A primary user decision as stored in ``event_decisions``."""

    model_config = _IGNORE_UNKNOWN

    event_id: UUID
    device_id: UUID
    user_id: Optional[UUID] = None
    session_id: UUID
    sequence: int = Field(default=0, ge=0)
    detected_at: datetime
    responded_at: datetime
    response: str

    @field_validator("responded_at")
    @classmethod
    def _responded_after_detected(cls, v: datetime, info) -> datetime:
        detected = info.data.get("detected_at")
        if detected is not None and v < detected:
            raise ValueError("responded_at must be >= detected_at")
        return v


class SuspectedEventAdapter:
    """Normalize a ``suspected_events`` Mongo document to SuspectedEvent."""

    @staticmethod
    def from_backend_dict(data: dict) -> SuspectedEvent:
        normalized = normalize_keys(data, SuspectedEvent)
        normalized["user_id"] = resolve_identity(data)
        return SuspectedEvent.model_validate(normalized)


class EventDecisionAdapter:
    """Normalize an ``event_decisions`` Mongo document to EventDecision."""

    @staticmethod
    def from_backend_dict(data: dict) -> EventDecision:
        normalized = normalize_keys(data, EventDecision)
        normalized["user_id"] = resolve_identity(data)
        return EventDecision.model_validate(normalized)