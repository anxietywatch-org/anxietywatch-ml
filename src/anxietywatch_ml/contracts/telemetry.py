"""
Internal ML telemetry contract.

This is the canonical schema used by the ML pipeline.
It is decoupled from the backend transport DTO (TelemetryBatchRequest).
An adapter should transform backend DTO -> this schema.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anxietywatch_ml.contracts.normalize import normalize_keys, resolve_identity


class SignalQuality(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class WearingState(str, Enum):
    ON_BODY = "onBody"
    OFF_BODY = "offBody"
    UNKNOWN = "unknown"


class TelemetrySampleQuality(BaseModel):
    """Per-channel signal quality for a single telemetry sample.

    Aliases mirror the backend transport spellings (camelCase). Both
    snake_case (internal/normalized) and camelCase (transport) are accepted.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    heart_rate: SignalQuality = Field(alias="heartRate")
    ibi: SignalQuality = Field(alias="ibi")
    wearing_state: WearingState = Field(default=WearingState.UNKNOWN, alias="wearingState")


class TelemetrySample(BaseModel):
    """
    Single telemetry sample as received from the backend.

    Fields matching the backend TelemetrySampleRequest.
    Accelerometer and ambient_temperature_celsius are always None
    in the current pipeline but kept for forward compatibility.
    Aliases mirror the backend transport spellings (camelCase); snake_case
    and camelCase are both accepted.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: datetime
    heart_rate_bpm: Optional[float] = Field(default=None, alias="heartRateBpm")
    ibi_ms: list[float] = Field(default_factory=list, alias="ibiMs")
    accelerometer: Optional[dict] = Field(default=None, alias="accelerometer")  # Always None currently
    skin_temperature_celsius: Optional[float] = Field(default=None, alias="skinTemperatureCelsius")
    ambient_temperature_celsius: Optional[float] = Field(default=None, alias="ambientTemperatureCelsius")  # Always None currently
    quality: TelemetrySampleQuality

    @field_validator("ibi_ms", mode="before")
    @classmethod
    def _ibi_ms_max_len(cls, v: list[float]) -> list[float]:
        if len(v) > 16:
            return v[:16]
        return v

    @field_validator("heart_rate_bpm", mode="before")
    @classmethod
    def _hr_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return None if v <= 0 else v
        return v


class TelemetryBatch(BaseModel):
    """
    Complete telemetry batch as received from the backend.

    This matches the backend TelemetryBatchRequest structure.
    """
    model_config = ConfigDict(extra="forbid")

    batch_id: UUID
    device_id: UUID
    user_id: Optional[UUID] = None
    session_id: UUID
    started_at: datetime
    ended_at: datetime
    sequence: int
    samples: Annotated[list[TelemetrySample], Field(min_length=1, max_length=600)]

    @field_validator("ended_at")
    @classmethod
    def _ended_after_started(cls, v: datetime, info) -> datetime:
        if "started_at" in info.data and v < info.data["started_at"]:
            raise ValueError("ended_at must be >= started_at")
        return v

    @field_validator("sequence")
    @classmethod
    def _sequence_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("sequence must be >= 0")
        return v


class TelemetryBatchAdapter:
    """
    Adapter to convert backend transport DTO to internal ML schema.

    In practice, the backend DTO comes from MongoDB as a dict whose keys may
    be PascalCase (real .NET serialization), camelCase (transport/synthetic)
    or snake_case (internal). Keys are resolved recursively and the canonical
    authenticated ``userId`` is enforced (see contracts.normalize).
    """

    @staticmethod
    def from_backend_dict(data: dict) -> TelemetryBatch:
        """
        Convert a backend telemetry batch dict (from MongoDB) to TelemetryBatch.

        Accepts EventId/eventId/event_id spellings for every field, including
        nested Samples/Quality. Unknown keys (e.g. ``_id``, ``receivedAt``)
        are dropped.
        """
        normalized = normalize_keys(data, TelemetryBatch)
        normalized["user_id"] = resolve_identity(data)
        return TelemetryBatch.model_validate(normalized)