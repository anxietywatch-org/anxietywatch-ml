"""
Feature builder for AnxietyWatch ML.

Computes window-level features from preprocessed telemetry windows.
All features are computed from the ACTUALLY AVAILABLE signals.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Heart rate features
    hr_mean: bool = True
    hr_std: bool = True
    hr_min: bool = True
    hr_max: bool = True
    hr_slope_bpm_per_min: bool = True
    hr_delta_from_baseline: bool = False  # Requires baseline - not available

    # HRV features (require IBI)
    hrv_rmssd: bool = True
    hrv_sdnn: bool = True
    hrv_pnn50: bool = False

    # IBI availability features
    ibi_available: bool = True
    ibi_coverage_ratio: bool = True

    # Movement features
    movement_magnitude_mean: bool = False
    movement_variance_mean: bool = False

    # Temperature features
    skin_temp_mean: bool = True
    skin_temp_std: bool = False

    # Quality features
    quality_good_ratio: bool = True
    quality_fair_ratio: bool = True
    quality_poor_ratio: bool = True
    valid_sample_ratio: bool = True

    # Temporal features
    window_duration_seconds: bool = True
    sample_count: bool = True
    last_sample_age_seconds: bool = False


class FeatureBuilder:
    """
    Builds feature vectors from windowed telemetry data.

    Only uses signals that are ACTUALLY AVAILABLE in the current pipeline:
    - heart_rate_bpm (always)
    - ibi_ms (partial - only Samsung devices)
    - skin_temperature_celsius (partial)
    - quality fields (always)
    - accelerometer: NOT AVAILABLE (always None)
    - ambient_temperature: NOT AVAILABLE (always None)
    - wearing_state: NOT AVAILABLE (always "unknown")
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def build(self, windows: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Build feature matrix from list of window DataFrames.

        Returns DataFrame with one row per window, columns = features.
        """
        if not windows:
            return pd.DataFrame()

        feature_rows = []
        for i, window in enumerate(windows):
            features = self._compute_window_features(window)
            feature_rows.append(features)

        df = pd.DataFrame(feature_rows)

        # Feature engineering preserves semantic missingness.
        # Imputation MUST happen later, after train/val/test splitting.
        numeric_cols = df.select_dtypes(include=[np.number]).columns


        df[numeric_cols] = df[numeric_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        )


        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logger.info(
                "Features containing semantic missing values: %s",
                nan_cols,
            )


        logger.info(
            "Built feature matrix: %d windows x %d features",
            df.shape[0],
            df.shape[1],
        )
        return df

    def _compute_window_features(self, window: pd.DataFrame) -> dict:
        """Compute all features for a single window."""
        features = {}

        # Ensure sorted by timestamp
        window = window.sort_values("timestamp")

        # --- Heart Rate Features ---
        hr = window["heart_rate_bpm"].dropna()
        if len(hr) > 0:
            if self.config.hr_mean:
                features["hr_mean"] = hr.mean()
            if self.config.hr_std:
                features["hr_std"] = hr.std() if len(hr) > 1 else 0.0
            if self.config.hr_min:
                features["hr_min"] = hr.min()
            if self.config.hr_max:
                features["hr_max"] = hr.max()
            if self.config.hr_slope_bpm_per_min:
                features["hr_slope_bpm_per_min"] = self._compute_hr_slope(window)
        else:
            # No HR data in window
            for fname in ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_slope_bpm_per_min"]:
                if getattr(self.config, fname, False):
                    features[fname] = np.nan

        # --- HRV Features (from IBI) ---
        # Collect all IBI values in window
        all_ibi = []
        for ibi_list in window["ibi_ms"].dropna():
            if isinstance(ibi_list, list) and len(ibi_list) > 0:
                all_ibi.extend(ibi_list)

        # Only compute HRV if we have sufficient valid IBI samples
        if len(all_ibi) >= 3:
            all_ibi = np.array(all_ibi)
            # Filter physiological range
            all_ibi = all_ibi[(all_ibi >= 250) & (all_ibi <= 2000)]

            if len(all_ibi) >= 3:
                if self.config.hrv_rmssd:
                    features["hrv_rmssd"] = self._compute_rmssd(all_ibi)
                if self.config.hrv_sdnn:
                    # std with ddof=1 requires at least 2 samples, we have >=3
                    features["hrv_sdnn"] = float(all_ibi.std(ddof=1))
                if self.config.hrv_pnn50:
                    features["hrv_pnn50"] = self._compute_pnn50(all_ibi)
            else:
                # Not enough valid IBI after filtering
                self._set_nan(features, ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50"])
        else:
            # Insufficient IBI samples (< 3)
            self._set_nan(features, ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50"])

        # --- IBI Availability Features ---
        # Total samples in window
        n_total = len(window)
        # Count samples with non-empty IBI
        ibi_nonempty = sum(1 for ibi_list in window["ibi_ms"] if isinstance(ibi_list, list) and len(ibi_list) > 0)
        
        if self.config.ibi_available:
            features["ibi_available"] = 1 if ibi_nonempty > 0 else 0
        if self.config.ibi_coverage_ratio:
            features["ibi_coverage_ratio"] = ibi_nonempty / n_total if n_total > 0 else 0.0

        # --- Movement Features ---
        # NOTE: accelerometer is always None in current pipeline
        # We don't have raw accelerometer data
        # The watch computes magnitudeG and variance but doesn't send them to cloud
        # So these will be NaN - keeping for forward compatibility
        if self.config.movement_magnitude_mean:
            features["movement_magnitude_mean"] = np.nan
        if self.config.movement_variance_mean:
            features["movement_variance_mean"] = np.nan

        # --- Temperature Features ---
        temp = window["skin_temperature_celsius"].dropna()
        if len(temp) > 0:
            if self.config.skin_temp_mean:
                features["skin_temp_mean"] = temp.mean()
            if self.config.skin_temp_std:
                features["skin_temp_std"] = temp.std() if len(temp) > 1 else 0.0
        else:
            if self.config.skin_temp_mean:
                features["skin_temp_mean"] = np.nan
            if self.config.skin_temp_std:
                features["skin_temp_std"] = np.nan

        # --- Quality Features ---
        n_total = len(window)
        if n_total > 0:
            if self.config.quality_good_ratio:
                features["quality_good_ratio"] = (
                    window["quality_heart_rate"] == "good"
                ).sum() / n_total
            if self.config.quality_fair_ratio:
                features["quality_fair_ratio"] = (
                    window["quality_heart_rate"] == "fair"
                ).sum() / n_total
            if self.config.quality_poor_ratio:
                features["quality_poor_ratio"] = (
                    window["quality_heart_rate"] == "poor"
                ).sum() / n_total
            if self.config.valid_sample_ratio:
                # Valid = HR present and quality not poor
                valid = window["heart_rate_bpm"].notna() & (window["quality_heart_rate"] != "poor")
                features["valid_sample_ratio"] = valid.sum() / n_total
        else:
            for fname in ["quality_good_ratio", "quality_fair_ratio", "quality_poor_ratio", "valid_sample_ratio"]:
                if getattr(self.config, fname, False):
                    features[fname] = 0.0

        # --- Temporal Features ---
        if self.config.window_duration_seconds:
            start = window["timestamp"].min()
            end = window["timestamp"].max()
            features["window_duration_seconds"] = (end - start).total_seconds()

        if self.config.sample_count:
            features["sample_count"] = n_total

        if self.config.last_sample_age_seconds:
            last_ts = window["timestamp"].max()
            now = pd.Timestamp.now(tz=last_ts.tz if last_ts.tz else "UTC")
            features["last_sample_age_seconds"] = (now - last_ts).total_seconds()

        return features

    def _compute_hr_slope(self, window: pd.DataFrame) -> float:
        """Compute heart rate slope (bpm per minute) using linear regression."""
        hr_data = window[["timestamp", "heart_rate_bpm"]].dropna()
        if len(hr_data) < 2:
            return 0.0

        # Convert timestamp to minutes since window start
        x = (hr_data["timestamp"] - hr_data["timestamp"].min()).dt.total_seconds() / 60.0
        y = hr_data["heart_rate_bpm"].values

        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)

    def _compute_rmssd(self, ibi: np.ndarray) -> float:
        """Compute RMSSD (Root Mean Square of Successive Differences)."""
        if len(ibi) < 2:
            return np.nan
        diff = np.diff(ibi)
        return float(np.sqrt(np.mean(diff ** 2)))

    def _compute_pnn50(self, ibi: np.ndarray) -> float:
        """Compute pNN50 (% of successive differences > 50ms)."""
        if len(ibi) < 2:
            return np.nan
        diff = np.abs(np.diff(ibi))
        return float((diff > 50).sum() / len(diff) * 100)

    def _set_nan(self, features: dict, names: list[str]) -> None:
        """Set multiple features to NaN."""
        for name in names:
            if getattr(self.config, name, False):
                features[name] = np.nan


def create_feature_builder(config: dict) -> FeatureBuilder:
    """Factory function to create feature builder from config."""
    feat_cfg = config.get("features", {})
    feature_config = FeatureConfig(
        hr_mean=feat_cfg.get("hr_mean", True),
        hr_std=feat_cfg.get("hr_std", True),
        hr_min=feat_cfg.get("hr_min", True),
        hr_max=feat_cfg.get("hr_max", True),
        hr_slope_bpm_per_min=feat_cfg.get("hr_slope_bpm_per_min", True),
        hr_delta_from_baseline=feat_cfg.get("hr_delta_from_baseline", False),
        hrv_rmssd=feat_cfg.get("hrv_rmssd", True),
        hrv_sdnn=feat_cfg.get("hrv_sdnn", True),
        hrv_pnn50=feat_cfg.get("hrv_pnn50", False),
        ibi_available=feat_cfg.get("ibi_available", True),
        ibi_coverage_ratio=feat_cfg.get("ibi_coverage_ratio", True),
        movement_magnitude_mean=feat_cfg.get("movement_magnitude_mean", False),
        movement_variance_mean=feat_cfg.get("movement_variance_mean", False),
        skin_temp_mean=feat_cfg.get("skin_temp_mean", True),
        skin_temp_std=feat_cfg.get("skin_temp_std", False),
        quality_good_ratio=feat_cfg.get("quality_good_ratio", True),
        quality_fair_ratio=feat_cfg.get("quality_fair_ratio", True),
        quality_poor_ratio=feat_cfg.get("quality_poor_ratio", True),
        valid_sample_ratio=feat_cfg.get("valid_sample_ratio", True),
        window_duration_seconds=feat_cfg.get("window_duration_seconds", True),
        sample_count=feat_cfg.get("sample_count", True),
        last_sample_age_seconds=feat_cfg.get("last_sample_age_seconds", False),
    )
    return FeatureBuilder(feature_config)