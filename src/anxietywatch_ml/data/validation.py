"""
Validation utilities for AnxietyWatch ML telemetry data.

Validates both the internal ML contract and provides utilities
for checking data quality before feature engineering.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from anxietywatch_ml.contracts.telemetry import TelemetryBatch, TelemetrySample

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, is_valid: bool, errors: list[str], warnings: list[str]):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        parts = [f"Valid: {self.is_valid}"]
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")
        return "; ".join(parts)


def validate_batch(batch: TelemetryBatch) -> ValidationResult:
    """
    Validate a TelemetryBatch beyond Pydantic model validation.

    Checks for data quality issues that Pydantic doesn't catch.
    """
    errors = []
    warnings = []

    # Check sample count
    if len(batch.samples) == 0:
        errors.append("Batch has no samples")
    elif len(batch.samples) > 600:
        warnings.append(f"Batch has {len(batch.samples)} samples (max recommended 600)")

    # Check time ordering
    timestamps = [s.timestamp for s in batch.samples]
    if timestamps != sorted(timestamps):
        warnings.append("Samples are not sorted by timestamp")

    # Check for large time gaps
    if len(timestamps) > 1:
        gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps)-1)]
        max_gap = max(gaps)
        if max_gap > 300:  # 5 minutes
            warnings.append(f"Large time gap detected: {max_gap:.0f} seconds")

    # Check heart rate availability
    hr_samples = [s for s in batch.samples if s.heart_rate_bpm is not None]
    hr_ratio = len(hr_samples) / len(batch.samples) if batch.samples else 0
    if hr_ratio < 0.5:
        warnings.append(f"Low heart rate availability: {hr_ratio:.1%}")

    # Check heart rate physiological range
    for s in hr_samples:
        if s.heart_rate_bpm < 30 or s.heart_rate_bpm > 220:
            warnings.append(f"Heart rate out of physiological range: {s.heart_rate_bpm} bpm")

    # Check IBI consistency with HR
    for s in batch.samples:
        if s.heart_rate_bpm and s.ibi_ms:
            expected_ibi = 60000.0 / s.heart_rate_bpm
            actual_ibi_mean = np.mean(s.ibi_ms)
            if abs(expected_ibi - actual_ibi_mean) / expected_ibi > 0.2:
                warnings.append(
                    f"IBI mean ({actual_ibi_mean:.0f}ms) inconsistent with HR "
                    f"({s.heart_rate_bpm} bpm -> {expected_ibi:.0f}ms)"
                )

    # Check quality distribution
    quality_counts = {"good": 0, "fair": 0, "poor": 0, "unknown": 0}
    for s in batch.samples:
        quality_counts[s.quality.heart_rate.value] += 1
    if quality_counts["poor"] / len(batch.samples) > 0.3:
        warnings.append(f"High poor-quality ratio: {quality_counts['poor']}/{len(batch.samples)}")

    # Check sequence continuity (would need previous batch context)
    # This is a placeholder for cross-batch validation

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings)


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list[str]] = None) -> ValidationResult:
    """
    Validate a telemetry DataFrame.

    Expected columns: timestamp, heart_rate_bpm, ibi_ms, etc.
    """
    errors = []
    warnings = []

    if required_columns is None:
        required_columns = ["timestamp", "heart_rate_bpm", "user_id", "session_id"]

    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return ValidationResult(False, errors, warnings)

    # Check for NaN in critical columns
    for col in ["timestamp", "user_id", "session_id"]:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Column '{col}' contains NaN values")

    # Check timestamp monotonicity per session
    if "session_id" in df.columns and "timestamp" in df.columns:
        for session_id, group in df.groupby("session_id"):
            if not group["timestamp"].is_monotonic_increasing:
                warnings.append(f"Session {session_id}: timestamps not monotonic")

    # Check heart rate range
    if "heart_rate_bpm" in df.columns:
        hr = df["heart_rate_bpm"].dropna()
        if len(hr) > 0:
            if (hr < 30).any() or (hr > 220).any():
                warnings.append("Heart rate values outside physiological range [30, 220]")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings)


def log_validation_result(result: ValidationResult, context: str = "Validation") -> None:
    """Log validation result with appropriate level."""
    if result.is_valid:
        logger.info(f"{context} passed")
    else:
        logger.error(f"{context} failed: {result.errors}")

    for warning in result.warnings:
        logger.warning(f"{context} warning: {warning}")