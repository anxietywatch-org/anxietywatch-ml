"""
Preprocessing pipeline for AnxietyWatch ML.

Transforms raw telemetry batches into windowed, cleaned time-series
ready for feature engineering.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd

from anxietywatch_ml.contracts.telemetry import TelemetryBatch, TelemetrySample

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    window_size_seconds: int = 60
    stride_seconds: int = 30
    min_samples_per_window: int = 10
    max_gap_seconds: float = 60.0  # Max gap to interpolate
    hr_outlier_std: float = 3.0  # Std deviations for outlier detection


@dataclass
class WindowedData:
    """Container for windowed telemetry data."""
    windows: list[pd.DataFrame]  # Each window is a DataFrame of samples
    window_metadata: list[dict]  # Metadata per window (user_id, session_id, start, end, etc.)
    original_batches: list[TelemetryBatch]


class PreprocessingPipeline:
    """
    Preprocessing pipeline for telemetry data.

    Steps:
    1. Flatten batches to DataFrame
    2. Sort by user/session/time
    3. Handle missing values
    4. Detect and handle outliers
    5. Segment into fixed windows
    6. Filter windows with insufficient data
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def run(self, batches: list[TelemetryBatch]) -> WindowedData:
        """Run the full preprocessing pipeline."""
        logger.info(f"Preprocessing {len(batches)} batches")

        # Step 1: Flatten to DataFrame
        df = self._flatten_batches(batches)
        logger.info(f"Flattened to {len(df)} samples across {df['user_id'].nunique()} users")

        # Step 2: Sort
        df = df.sort_values(["user_id", "session_id", "timestamp"]).reset_index(drop=True)

        # Steps 3-4: Handle missing values and detect outliers (canonical cleaning)
        df = self.clean_window(df)

        # Step 5: Segment into windows
        windows, metadata = self._segment_windows(df)

        # Step 6: Filter valid windows
        valid_windows, valid_metadata = self._filter_windows(windows, metadata)

        logger.info(f"Created {len(valid_windows)} valid windows from {len(windows)} total")

        return WindowedData(
            windows=valid_windows,
            window_metadata=valid_metadata,
            original_batches=batches,
        )

    def _flatten_batches(self, batches: list[TelemetryBatch]) -> pd.DataFrame:
        """Convert list of TelemetryBatch to a flat DataFrame."""
        rows = []
        for batch in batches:
            for sample in batch.samples:
                rows.append(
                    self._row_from_sample(
                        batch_id=batch.batch_id,
                        user_id=batch.user_id,
                        device_id=batch.device_id,
                        session_id=batch.session_id,
                        sequence=batch.sequence,
                        sample=sample,
                    )
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _row_from_sample(
        *,
        batch_id,
        user_id,
        device_id,
        session_id,
        sequence,
        sample: TelemetrySample,
    ) -> dict:
        """Build one flat row from a single sample (shared by all flatten paths)."""
        return {
            "batch_id": str(batch_id) if batch_id is not None else None,
            "user_id": str(user_id) if user_id else None,
            "device_id": str(device_id),
            "session_id": str(session_id),
            "sequence": sequence,
            "timestamp": sample.timestamp,
            "heart_rate_bpm": sample.heart_rate_bpm,
            "ibi_ms": sample.ibi_ms,
            "skin_temperature_celsius": sample.skin_temperature_celsius,
            "quality_heart_rate": sample.quality.heart_rate.value,
            "quality_ibi": sample.quality.ibi.value,
            "quality_wearing_state": sample.quality.wearing_state.value,
        }

    def flatten_samples(
        self,
        samples: list[TelemetrySample],
        *,
        user_id=None,
        device_id,
        session_id,
    ) -> pd.DataFrame:
        """Flatten a raw list of telemetry samples into the canonical flat frame.

        Used by the serving path (event-anchored raw window), where the caller
        already scopes the samples to one device/session. Batch/sequence context
        is not applicable (no backend batch boundaries), so ``batch_id`` and
        ``sequence`` are ``None``/``0``.
        """
        rows = [
            self._row_from_sample(
                batch_id=None,
                user_id=user_id,
                device_id=device_id,
                session_id=session_id,
                sequence=0,
                sample=sample,
            )
            for sample in samples
        ]
        return pd.DataFrame(rows)

    def clean_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply canonical missing-value handling and HR outlier detection.

        Public reusable abstraction shared by the training/ground-truth path
        and the serving path, so both consume identical cleaning semantics.
        """
        df = self._handle_missing_values(df)
        df = self._detect_outliers(df)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        df = df.copy()

        # Forward-fill heart rate within session (short gaps only)
        df["heart_rate_bpm"] = df.groupby("session_id")["heart_rate_bpm"].transform(
            lambda x: x.ffill(limit=int(self.config.max_gap_seconds))
        )

        # For skin temperature, forward fill with longer limit
        if "skin_temperature_celsius" in df.columns:
            df["skin_temperature_celsius"] = df.groupby("session_id")["skin_temperature_celsius"].transform(
                lambda x: x.ffill(limit=300)
            )

        # IBI: can't easily interpolate, leave as-is (list column)
        # Quality: forward fill
        for qcol in ["quality_heart_rate", "quality_ibi", "quality_wearing_state"]:
            if qcol in df.columns:
                df[qcol] = df.groupby("session_id")[qcol].transform(lambda x: x.ffill())

        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and mark heart rate outliers using rolling statistics."""
        df = df.copy()

        # Compute rolling statistics per session without apply (avoids FutureWarning)
        # We'll compute rolling stats on the full dataframe grouped by session_id
        df["hr_rolling_mean"] = df.groupby("session_id")["heart_rate_bpm"].transform(
            lambda x: x.rolling(window=10, center=True, min_periods=3).mean()
        )
        df["hr_rolling_std"] = df.groupby("session_id")["heart_rate_bpm"].transform(
            lambda x: x.rolling(window=10, center=True, min_periods=3).std()
        )

        # Compute z-score
        df["hr_z_score"] = np.abs(df["heart_rate_bpm"] - df["hr_rolling_mean"]) / (
            df["hr_rolling_std"] + 1e-6
        )

        # Mark outliers - only where we have enough samples in the window
        # Count non-NaN HR values per session in rolling window
        hr_count = df.groupby("session_id")["heart_rate_bpm"].transform(
            lambda x: x.rolling(window=10, center=True, min_periods=1).count()
        )
        df["hr_is_outlier"] = (
            (df["hr_z_score"] > self.config.hr_outlier_std) & (hr_count >= 5)
        )

        # Clean up temporary columns
        df = df.drop(columns=["hr_rolling_mean", "hr_rolling_std"])

        return df

    def _segment_windows(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[pd.DataFrame], list[dict]]:
        """Segment data into fixed-size sliding windows per session."""
        windows = []
        metadata = []

        for session_id, session_df in df.groupby("session_id"):
            session_df = session_df.sort_values("timestamp").reset_index(drop=True)
            if len(session_df) < self.config.min_samples_per_window:
                continue

            start_time = session_df["timestamp"].min()
            end_time = session_df["timestamp"].max()
            duration = (end_time - start_time).total_seconds()

            if duration < self.config.window_size_seconds:
                # Single window for short sessions
                windows.append(session_df)
                metadata.append({
                    "session_id": session_id,
                    "user_id": session_df["user_id"].iloc[0],
                    "device_id": session_df["device_id"].iloc[0],
                    "window_start": start_time,
                    "window_end": end_time,
                    "window_index": 0,
                    "n_samples": len(session_df),
                })
                continue

            # Sliding windows
            current_start = start_time
            window_idx = 0
            while current_start + timedelta(seconds=self.config.window_size_seconds) <= end_time:
                current_end = current_start + timedelta(seconds=self.config.window_size_seconds)
                window_mask = (session_df["timestamp"] >= current_start) & (session_df["timestamp"] < current_end)
                window_df = session_df[window_mask].copy()

                if len(window_df) >= self.config.min_samples_per_window:
                    windows.append(window_df)
                    metadata.append({
                        "session_id": session_id,
                        "user_id": session_df["user_id"].iloc[0],
                        "device_id": session_df["device_id"].iloc[0],
                        "window_start": current_start,
                        "window_end": current_end,
                        "window_index": window_idx,
                        "n_samples": len(window_df),
                    })

                current_start += timedelta(seconds=self.config.stride_seconds)
                window_idx += 1

        return windows, metadata

    def _filter_windows(
        self,
        windows: list[pd.DataFrame],
        metadata: list[dict],
    ) -> tuple[list[pd.DataFrame], list[dict]]:
        """Filter out windows that don't meet quality criteria."""
        valid_windows = []
        valid_metadata = []

        for window, meta in zip(windows, metadata):
            # Check minimum samples
            if meta["n_samples"] < self.config.min_samples_per_window:
                continue

            # Check heart rate availability in window
            hr_available = window["heart_rate_bpm"].notna().sum()
            hr_ratio = hr_available / len(window)
            if hr_ratio < 0.3:
                continue

            valid_windows.append(window)
            valid_metadata.append(meta)

        return valid_windows, valid_metadata


def create_pipeline(config: dict) -> PreprocessingPipeline:
    """Factory function to create preprocessing pipeline from config.

    Reads ``config["window"]`` (segmentation) and ``config["preprocessing"]``
    (cleaning: ``max_gap_seconds``, ``hr_outlier_std``). Used by BOTH the
    offline ground-truth builder and the serving raw-window processor so the
    cleaning contract cannot diverge between training and inference.
    """
    window_cfg = config.get("window", {})
    prep_cfg = config.get("preprocessing", {})
    prep_config = PreprocessingConfig(
        window_size_seconds=window_cfg.get("size_seconds", 60),
        stride_seconds=window_cfg.get("stride_seconds", 30),
        min_samples_per_window=window_cfg.get("min_samples_per_window", 10),
        max_gap_seconds=prep_cfg.get("max_gap_seconds", 60.0),
        hr_outlier_std=prep_cfg.get("hr_outlier_std", 3.0),
    )
    return PreprocessingPipeline(prep_config)