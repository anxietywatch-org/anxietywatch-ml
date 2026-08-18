"""
Tests for validation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from anxietywatch_ml.data.validation import (
    validate_batch,
    validate_dataframe,
    ValidationResult,
)
from anxietywatch_ml.contracts.telemetry import (
    TelemetryBatch,
    TelemetrySample,
    TelemetrySampleQuality,
    SignalQuality,
    WearingState,
)


class TestValidateBatch:
    def create_valid_batch(self) -> TelemetryBatch:
        return TelemetryBatch(
            batch_id=uuid4(),
            device_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            sequence=1,
            samples=[
                TelemetrySample(
                    timestamp=datetime.now(timezone.utc),
                    heart_rate_bpm=72.0,
                    ibi_ms=[800.0, 810.0],
                    accelerometer=None,
                    skin_temperature_celsius=33.5,
                    ambient_temperature_celsius=None,
                    quality=TelemetrySampleQuality(
                        heart_rate=SignalQuality.GOOD,
                        ibi=SignalQuality.GOOD,
                        wearing_state=WearingState.UNKNOWN,
                    ),
                )
            ],
        )

    def test_valid_batch_passes(self):
        batch = self.create_valid_batch()
        result = validate_batch(batch)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_empty_batch_fails(self):
        # Pydantic validates at construction time
        with pytest.raises(Exception) as exc_info:
            TelemetryBatch(
                batch_id=uuid4(),
                device_id=uuid4(),
                user_id=uuid4(),
                session_id=uuid4(),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                sequence=1,
                samples=[],
            )
        assert "too_short" in str(exc_info.value).lower() or "at least 1" in str(exc_info.value).lower()

    def test_unsorted_timestamps_warning(self):
        base = datetime.now(timezone.utc)
        batch = TelemetryBatch(
            batch_id=uuid4(),
            device_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            started_at=base,
            ended_at=base + timedelta(seconds=10),
            sequence=1,
            samples=[
                TelemetrySample(
                    timestamp=base + timedelta(seconds=10),
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
                ),
                TelemetrySample(
                    timestamp=base,
                    heart_rate_bpm=70.0,
                    ibi_ms=[],
                    accelerometer=None,
                    skin_temperature_celsius=None,
                    ambient_temperature_celsius=None,
                    quality=TelemetrySampleQuality(
                        heart_rate=SignalQuality.GOOD,
                        ibi=SignalQuality.UNKNOWN,
                        wearing_state=WearingState.UNKNOWN,
                    ),
                ),
            ],
        )
        result = validate_batch(batch)
        assert result.is_valid
        assert any("not sorted" in w.lower() for w in result.warnings)

    def test_large_time_gap_warning(self):
        base = datetime.now(timezone.utc)
        batch = TelemetryBatch(
            batch_id=uuid4(),
            device_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            started_at=base,
            ended_at=base + timedelta(seconds=600),
            sequence=1,
            samples=[
                TelemetrySample(
                    timestamp=base,
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
                ),
                TelemetrySample(
                    timestamp=base + timedelta(seconds=400),  # > 300s gap
                    heart_rate_bpm=70.0,
                    ibi_ms=[],
                    accelerometer=None,
                    skin_temperature_celsius=None,
                    ambient_temperature_celsius=None,
                    quality=TelemetrySampleQuality(
                        heart_rate=SignalQuality.GOOD,
                        ibi=SignalQuality.UNKNOWN,
                        wearing_state=WearingState.UNKNOWN,
                    ),
                ),
            ],
        )
        result = validate_batch(batch)
        assert result.is_valid
        assert any("gap" in w.lower() for w in result.warnings)

    def test_low_hr_availability_warning(self):
        base = datetime.now(timezone.utc)
        batch = TelemetryBatch(
            batch_id=uuid4(),
            device_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            started_at=base,
            ended_at=base + timedelta(seconds=10),
            sequence=1,
            samples=[
                TelemetrySample(
                    timestamp=base + timedelta(seconds=i),
                    heart_rate_bpm=72.0 if i == 0 else None,  # Only 1/10 have HR
                    ibi_ms=[],
                    accelerometer=None,
                    skin_temperature_celsius=None,
                    ambient_temperature_celsius=None,
                    quality=TelemetrySampleQuality(
                        heart_rate=SignalQuality.GOOD if i == 0 else SignalQuality.UNKNOWN,
                        ibi=SignalQuality.UNKNOWN,
                        wearing_state=WearingState.UNKNOWN,
                    ),
                )
                for i in range(10)
            ],
        )
        result = validate_batch(batch)
        assert result.is_valid
        assert any("availability" in w.lower() for w in result.warnings)

    def test_hr_out_of_range_warning(self):
        batch = self.create_valid_batch()
        # Modify sample to have out-of-range HR
        batch.samples[0].heart_rate_bpm = 250.0
        result = validate_batch(batch)
        assert result.is_valid
        assert any("physiological" in w.lower() for w in result.warnings)

    def test_ibi_hr_inconsistency_warning(self):
        base = datetime.now(timezone.utc)
        batch = TelemetryBatch(
            batch_id=uuid4(),
            device_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            started_at=base,
            ended_at=base + timedelta(seconds=10),
            sequence=1,
            samples=[
                TelemetrySample(
                    timestamp=base,
                    heart_rate_bpm=60.0,  # IBI should be ~1000ms
                    ibi_ms=[500.0, 500.0],  # But we give 500ms (HR=120)
                    accelerometer=None,
                    skin_temperature_celsius=None,
                    ambient_temperature_celsius=None,
                    quality=TelemetrySampleQuality(
                        heart_rate=SignalQuality.GOOD,
                        ibi=SignalQuality.GOOD,
                        wearing_state=WearingState.UNKNOWN,
                    ),
                )
            ],
        )
        result = validate_batch(batch)
        assert result.is_valid
        assert any("inconsistent" in w.lower() for w in result.warnings)


class TestValidateDataFrame:
    def test_valid_dataframe(self):
        df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc) for _ in range(5)],
            "heart_rate_bpm": [70, 72, 71, 73, 70],
            "user_id": [str(uuid4()) for _ in range(5)],
            "session_id": [str(uuid4()) for _ in range(5)],
        })
        result = validate_dataframe(df)
        assert result.is_valid

    def test_missing_required_column(self):
        df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "heart_rate_bpm": [70],
            # user_id missing
            "session_id": [str(uuid4())],
        })
        result = validate_dataframe(df)
        assert not result.is_valid
        assert any("user_id" in e for e in result.errors)

    def test_nan_in_critical_column(self):
        df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc), None],
            "heart_rate_bpm": [70, 72],
            "user_id": [str(uuid4()), str(uuid4())],
            "session_id": [str(uuid4()), str(uuid4())],
        })
        result = validate_dataframe(df)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    def test_non_monotonic_timestamps_warning(self):
        base = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "timestamp": [base + timedelta(seconds=10), base],
            "heart_rate_bpm": [70, 72],
            "user_id": [str(uuid4()), str(uuid4())],
            "session_id": ["sess1", "sess1"],
        })
        result = validate_dataframe(df)
        assert result.is_valid
        assert any("monotonic" in w.lower() for w in result.warnings)

    def test_hr_out_of_range_warning(self):
        df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "heart_rate_bpm": [250],  # Out of range
            "user_id": [str(uuid4())],
            "session_id": [str(uuid4())],
        })
        result = validate_dataframe(df)
        assert result.is_valid
        assert any("physiological" in w.lower() for w in result.warnings)


class TestValidationResult:
    def test_bool_conversion(self):
        valid = ValidationResult(True, [], [])
        invalid = ValidationResult(False, ["error"], [])

        assert bool(valid) is True
        assert bool(invalid) is False

    def test_str_representation(self):
        result = ValidationResult(True, [], ["warning1"])
        s = str(result)
        assert "Valid: True" in s
        assert "Warnings: 1" in s