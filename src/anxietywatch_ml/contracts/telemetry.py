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
    model_config = ConfigDict(extra="forbid")

    heart_rate: SignalQuality
    ibi: SignalQuality
    wearing_state: WearingState = WearingState.UNKNOWN


class TelemetrySample(BaseModel):
    """
    Single telemetry sample as received from the backend.

    Fields matching the backend TelemetrySampleRequest.
    Accelerometer and ambient_temperature_celsius are always None
    in the current pipeline but kept for forward compatibility.
    """
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    heart_rate_bpm: Optional[float] = None
    ibi_ms: list[float] = Field(default_factory=list)
    accelerometer: Optional[dict] = None  # Always None currently
    skin_temperature_celsius: Optional[float] = None
    ambient_temperature_celsius: Optional[float] = None  # Always None currently
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
        if v is not None and v <= 0:
            return None
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

    In practice, the backend DTO comes from MongoDB as a dict.
    This adapter handles the transformation and validation.
    """

    @staticmethod
    def from_backend_dict(data: dict) -> TelemetryBatch:
        """
        Convert a backend telemetry batch dict (from MongoDB) to TelemetryBatch.

        Expected keys: batchId, deviceId, userId, sessionId, startedAt, endedAt,
        sequence, samples (list of sample dicts with timestamp, heartRateBpm,
        ibiMs, accelerometer, skinTemperatureCelsius, ambientTemperatureCelsius,
        quality with heartRate, ibi, wearingState)
        """
        # Convert field names from camelCase to snake_case
        converted = {
            "batch_id": data.get("batchId") or data.get("batch_id"),
            "device_id": data.get("deviceId") or data.get("device_id"),
            "user_id": data.get("userId") or data.get("user_id"),
            "session_id": data.get("sessionId") or data.get("session_id"),
            "started_at": data.get("startedAt") or data.get("started_at"),
            "ended_at": data.get("endedAt") or data.get("ended_at"),
            "sequence": data.get("sequence", 0),
            "samples": [],
        }

        for sample in data.get("samples", []):
            quality = sample.get("quality", {})
            converted_sample = {
                "timestamp": sample.get("timestamp"),
                "heart_rate_bpm": sample.get("heartRateBpm") or sample.get("heart_rate_bpm"),
                "ibi_ms": sample.get("ibiMs") or sample.get("ibi_ms") or [],
                "accelerometer": sample.get("accelerometer"),
                "skin_temperature_celsius": sample.get("skinTemperatureCelsius") or sample.get("skin_temperature_celsius"),
                "ambient_temperature_celsius": sample.get("ambientTemperatureCelsius") or sample.get("ambient_temperature_celsius"),
                "quality": {
                    "heart_rate": quality.get("heartRate") or quality.get("heart_rate", "unknown"),
                    "ibi": quality.get("ibi", "unknown"),
                    "wearing_state": quality.get("wearingState") or quality.get("wearing_state", "unknown"),
                },
            }
            converted["samples"].append(converted_sample)

        return TelemetryBatch.model_validate(converted)