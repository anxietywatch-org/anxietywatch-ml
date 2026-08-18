"""
Tests for AnxietyWatch ML contracts.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from uuid import UUID
from anxietywatch_ml.contracts.telemetry import (
    TelemetryBatch,
    TelemetrySample,
    TelemetrySampleQuality,
    SignalQuality,
    WearingState,
    TelemetryBatchAdapter,
)


class TestTelemetrySampleQuality:
    def test_valid_quality(self):
        quality = TelemetrySampleQuality(
            heart_rate=SignalQuality.GOOD,
            ibi=SignalQuality.FAIR,
            wearing_state=WearingState.UNKNOWN,
        )
        assert quality.heart_rate == SignalQuality.GOOD
        assert quality.ibi == SignalQuality.FAIR
        assert quality.wearing_state == WearingState.UNKNOWN

    def test_invalid_heart_rate_quality(self):
        with pytest.raises(ValueError):
            TelemetrySampleQuality(
                heart_rate="excellent",  # Not in enum
                ibi=SignalQuality.GOOD,
                wearing_state=WearingState.ON_BODY,
            )

    def test_invalid_wearing_state(self):
        with pytest.raises(ValueError):
            TelemetrySampleQuality(
                heart_rate=SignalQuality.GOOD,
                ibi=SignalQuality.GOOD,
                wearing_state="wearing",  # Not in enum
            )


class TestTelemetrySample:
    def test_minimal_valid_sample(self):
        sample = TelemetrySample(
            timestamp=datetime.now(timezone.utc),
            heart_rate_bpm=72.0,
            ibi_ms=[],
            accelerometer=None,
            skin_temperature_celsius=None,
            ambient_temperature_celsius=None,
            quality=TelemetrySampleQuality(
                heart_rate=SignalQuality.GOOD,
                ibi=SignalQuality.UNKNOWN,
                wearing_state=WearingState.UNKNOWN,
            ),
        )
        assert sample.heart_rate_bpm == 72.0
        assert sample.ibi_ms == []
        assert sample.accelerometer is None

    def test_sample_with_ibi(self):
        sample = TelemetrySample(
            timestamp=datetime.now(timezone.utc),
            heart_rate_bpm=72.0,
            ibi_ms=[800.0, 810.0, 805.0, 795.0],
            accelerometer=None,
            skin_temperature_celsius=33.5,
            ambient_temperature_celsius=None,
            quality=TelemetrySampleQuality(
                heart_rate=SignalQuality.GOOD,
                ibi=SignalQuality.GOOD,
                wearing_state=WearingState.UNKNOWN,
            ),
        )
        assert len(sample.ibi_ms) == 4

    def test_ibi_max_length(self):
        # Should truncate to 16
        long_ibi = [800.0 + i for i in range(20)]
        sample = TelemetrySample(
            timestamp=datetime.now(timezone.utc),
            heart_rate_bpm=72.0,
            ibi_ms=long_ibi,
            accelerometer=None,
            skin_temperature_celsius=None,
            ambient_temperature_celsius=None,
            quality=TelemetrySampleQuality(
                heart_rate=SignalQuality.GOOD,
                ibi=SignalQuality.GOOD,
                wearing_state=WearingState.UNKNOWN,
            ),
        )
        assert len(sample.ibi_ms) == 16

    def test_heart_rate_negative_becomes_none(self):
        sample = TelemetrySample(
            timestamp=datetime.now(timezone.utc),
            heart_rate_bpm=-10.0,  # Invalid
            ibi_ms=[],
            accelerometer=None,
            skin_temperature_celsius=None,
            ambient_temperature_celsius=None,
            quality=TelemetrySampleQuality(
                heart_rate=SignalQuality.POOR,
                ibi=SignalQuality.UNKNOWN,
                wearing_state=WearingState.UNKNOWN,
            ),
        )
        assert sample.heart_rate_bpm is None

    def test_heart_rate_zero_becomes_none(self):
        sample = TelemetrySample(
            timestamp=datetime.now(timezone.utc),
            heart_rate_bpm=0.0,
            ibi_ms=[],
            accelerometer=None,
            skin_temperature_celsius=None,
            ambient_temperature_celsius=None,
            quality=TelemetrySampleQuality(
                heart_rate=SignalQuality.POOR,
                ibi=SignalQuality.UNKNOWN,
                wearing_state=WearingState.UNKNOWN,
            ),
        )
        assert sample.heart_rate_bpm is None


class TestTelemetryBatch:
    def test_valid_batch(self):
        samples = [
            TelemetrySample(
                timestamp=datetime.now(timezone.utc),
                heart_rate_bpm=72.0,
                ibi_ms=[],
                accelerometer=None,
                skin_temperature_celsius=None,
                ambient_temperature_celsius=None,
                quality=TelemetrySampleQuality(
                    heart_rate=SignalQuality.GOOD,
                    ibi=SignalQuality.UNKNOWN,
                    wearing_state=WearingState.UNKNOWN,
                ),
            )
        ]

        batch = TelemetryBatch(
            batch_id=uuid4(),
            device_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            sequence=1,
            samples=samples,
        )

        assert len(batch.samples) == 1
        assert batch.sequence == 1

    def test_empty_samples_rejected(self):
        with pytest.raises(ValueError):
            TelemetryBatch(
                batch_id=uuid4(),
                device_id=uuid4(),
                user_id=uuid4(),
                session_id=uuid4(),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                sequence=0,
                samples=[],
            )

    def test_ended_before_started_rejected(self):
        samples = [
            TelemetrySample(
                timestamp=datetime.now(timezone.utc),
                heart_rate_bpm=72.0,
                ibi_ms=[],
                accelerometer=None,
                skin_temperature_celsius=None,
                ambient_temperature_celsius=None,
                quality=TelemetrySampleQuality(
                    heart_rate=SignalQuality.GOOD,
                    ibi=SignalQuality.UNKNOWN,
                    wearing_state=WearingState.UNKNOWN,
                ),
            )
        ]

        with pytest.raises(ValueError):
            TelemetryBatch(
                batch_id=uuid4(),
                device_id=uuid4(),
                user_id=uuid4(),
                session_id=uuid4(),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc) - __import__('datetime').timedelta(hours=1),
                sequence=0,
                samples=samples,
            )

    def test_negative_sequence_rejected(self):
        with pytest.raises(ValueError):
            TelemetryBatch(
                batch_id=uuid4(),
                device_id=uuid4(),
                user_id=uuid4(),
                session_id=uuid4(),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                sequence=-1,
                samples=[
                    TelemetrySample(
                        timestamp=datetime.now(timezone.utc),
                        heart_rate_bpm=72.0,
                        ibi_ms=[],
                        accelerometer=None,
                        skin_temperature_celsius=None,
                        ambient_temperature_celsius=None,
                        quality=TelemetrySampleQuality(
                            heart_rate=SignalQuality.GOOD,
                            ibi=SignalQuality.UNKNOWN,
                            wearing_state=WearingState.UNKNOWN,
                        ),
                    )
                ],
            )


class TestTelemetryBatchAdapter:
    def test_from_backend_dict_minimal(self):
        data = {
            "batchId": str(uuid4()),
            "deviceId": str(uuid4()),
            "userId": str(uuid4()),
            "sessionId": str(uuid4()),
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "endedAt": datetime.now(timezone.utc).isoformat(),
            "sequence": 1,
            "samples": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "heartRateBpm": 72.0,
                    "ibiMs": [],
                    "accelerometer": None,
                    "skinTemperatureCelsius": None,
                    "ambientTemperatureCelsius": None,
                    "quality": {
                        "heartRate": "good",
                        "ibi": "unknown",
                        "wearingState": "unknown",
                    },
                }
            ],
        }

        batch = TelemetryBatchAdapter.from_backend_dict(data)

        assert batch.batch_id == UUID(data["batchId"])
        assert batch.device_id == UUID(data["deviceId"])
        assert batch.user_id == UUID(data["userId"])
        assert len(batch.samples) == 1
        assert batch.samples[0].heart_rate_bpm == 72.0

    def test_from_backend_dict_snake_case(self):
        """Test adapter handles snake_case field names too."""
        data = {
            "batch_id": str(uuid4()),
            "device_id": str(uuid4()),
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "sequence": 1,
            "samples": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "heart_rate_bpm": 72.0,
                    "ibi_ms": [],
                    "accelerometer": None,
                    "skin_temperature_celsius": None,
                    "ambient_temperature_celsius": None,
                    "quality": {
                        "heart_rate": "good",
                        "ibi": "unknown",
                        "wearing_state": "unknown",
                    },
                }
            ],
        }

        batch = TelemetryBatchAdapter.from_backend_dict(data)
        assert len(batch.samples) == 1

    def test_missing_optional_fields(self):
        """Test adapter handles missing optional fields."""
        data = {
            "batchId": str(uuid4()),
            "deviceId": str(uuid4()),
            # userId missing
            "sessionId": str(uuid4()),
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "endedAt": datetime.now(timezone.utc).isoformat(),
            "sequence": 1,
            "samples": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    # heartRateBpm missing
                    "ibiMs": [],
                    "quality": {
                        "heartRate": "good",
                        "ibi": "unknown",
                        "wearingState": "unknown",
                    },
                }
            ],
        }

        batch = TelemetryBatchAdapter.from_backend_dict(data)
        assert batch.user_id is None
        assert batch.samples[0].heart_rate_bpm is None