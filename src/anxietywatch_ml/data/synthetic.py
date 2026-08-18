"""
Synthetic telemetry data generator for AnxietyWatch ML pipeline.

This generates data that matches the REAL backend contract (TelemetryBatch).
All data is SYNTHETIC and clearly marked as such.
This does NOT represent real anxiety detection capability.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4
import hashlib

import numpy as np
import pandas as pd

from anxietywatch_ml.contracts.telemetry import (
    TelemetryBatch,
    TelemetrySample,
    TelemetrySampleQuality,
    SignalQuality,
    WearingState,
)

logger = logging.getLogger(__name__)


class SyntheticTelemetryGenerator:
    """
    Generates synthetic telemetry batches matching the AnxietyWatch backend contract.

    All randomness is controlled by a configurable seed for reproducibility.
    """

    def __init__(self, config: dict):
        self.config = config
        self.random_seed = config.get("random_seed", 42)
        self.rng = np.random.default_rng(self.random_seed)

        # Synthetic data parameters
        synth_cfg = config.get("synthetic", {})
        self.n_users = synth_cfg.get("n_users", 10)
        self.n_sessions_per_user = synth_cfg.get("n_sessions_per_user", 5)
        self.samples_per_session = synth_cfg.get("samples_per_session", 300)
        self.sampling_rate_hz = synth_cfg.get("sampling_rate_hz", 1.0)

        self.hr_baseline_mean = synth_cfg.get("hr_baseline_mean", 70.0)
        self.hr_baseline_std = synth_cfg.get("hr_baseline_std", 8.0)
        self.hr_anomaly_shift_bpm = synth_cfg.get("hr_anomaly_shift_bpm", 20.0)
        self.ibi_available_prob = synth_cfg.get("ibi_available_probability", 0.3)
        self.motion_magnitude_mean = synth_cfg.get("motion_magnitude_mean", 0.05)
        self.motion_magnitude_std = synth_cfg.get("motion_magnitude_std", 0.03)
        self.skin_temp_mean = synth_cfg.get("skin_temp_mean", 33.5)
        self.skin_temp_std = synth_cfg.get("skin_temp_std", 0.5)

        self.anomaly_probability = synth_cfg.get("anomaly_probability", 0.15)
        self.label_noise = synth_cfg.get("label_noise", 0.05)

        # Pre-generate user baselines
        self._user_baselines = {}
        self._uuid_counter = 0
        for i in range(self.n_users):
            user_id = self._deterministic_uuid(f"user_{i}")
            self._user_baselines[user_id] = {
                "hr_mean": self.rng.normal(self.hr_baseline_mean, self.hr_baseline_std),
                "hr_std": max(1.0, self.rng.normal(5.0, 1.5)),
            }

    def _deterministic_uuid(self, prefix: str = "") -> UUID:
        """Generate a deterministic UUID based on seed and counter."""
        self._uuid_counter += 1
        seed_str = f"{self.random_seed}_{prefix}_{self._uuid_counter}"
        hash_bytes = hashlib.md5(seed_str.encode()).digest()
        # Convert to UUID format (version 4 style)
        hash_bytes = bytearray(hash_bytes)
        hash_bytes[6] = (hash_bytes[6] & 0x0F) | 0x40  # Version 4
        hash_bytes[8] = (hash_bytes[8] & 0x3F) | 0x80  # Variant 10
        return UUID(bytes=bytes(hash_bytes))

    def _generate_quality(self, is_anomaly: bool) -> SignalQuality:
        """Generate signal quality, slightly worse during anomalies."""
        if is_anomaly:
            probs = [0.4, 0.3, 0.2, 0.1]  # good, fair, poor, unknown
        else:
            probs = [0.7, 0.2, 0.05, 0.05]
        quality_str = self.rng.choice([q.value for q in SignalQuality], p=probs)
        return SignalQuality(quality_str)

    def _generate_ibi_ms(self, heart_rate_bpm: float, available: bool) -> list[float]:
        """Generate synthetic IBI values from heart rate."""
        if not available or heart_rate_bpm is None:
            return []

        # Mean IBI in ms = 60000 / HR
        mean_ibi = 60000.0 / heart_rate_bpm
        # Add realistic variability
        n_ibi = self.rng.integers(1, 17)
        # IBI typically varies by ~50-100ms around mean
        ibi_values = self.rng.normal(mean_ibi, mean_ibi * 0.05, n_ibi)
        # Clip to physiological range
        ibi_values = np.clip(ibi_values, 300, 2000)
        return ibi_values.tolist()

    def _generate_samples(
        self,
        session_start: datetime,
        n_samples: int,
        is_anomaly_session: bool,
        user_baseline: dict,
        ibi_supported: bool,
    ) -> list[TelemetrySample]:
        """
        Generate synthetic samples for one session.


        IMPORTANT:
        `is_anomaly_session` is synthetic ground truth used only to verify
        the ML plumbing. It is NOT a clinical anxiety definition.


        For bootstrap purposes the whole positive session receives the
        configured synthetic signal shift so labels and generated signals
        are intentionally connected.
        """
        samples: list[TelemetrySample] = []
        current_time = session_start


        # Synthetic session-level ground truth.
        in_anomaly = bool(is_anomaly_session)


        for _ in range(n_samples):
            if in_anomaly:
                hr_bpm = self.rng.normal(
                    user_baseline["hr_mean"] + self.hr_anomaly_shift_bpm,
                    user_baseline["hr_std"] * 1.5,
                )
            else:
                hr_bpm = self.rng.normal(
                    user_baseline["hr_mean"],
                    user_baseline["hr_std"],
                )


            hr_bpm = float(np.clip(hr_bpm, 40, 180))


            # Occasional sensor gap.
            if self.rng.random() < 0.02:
                hr_bpm = None


            # IBI capability is device/session-level, not sampled independently
            # for every telemetry sample.
            ibi_ms = (
                self._generate_ibi_ms(hr_bpm, available=True)
                if ibi_supported and hr_bpm is not None
                else []
            )


            skin_temp = self.rng.normal(
                self.skin_temp_mean,
                self.skin_temp_std,
            )
            if self.rng.random() < 0.05:
                skin_temp = None


            hr_quality = self._generate_quality(in_anomaly)
            ibi_quality = self._generate_quality(in_anomaly)


            sample = TelemetrySample(
                timestamp=current_time,
                heart_rate_bpm=hr_bpm,
                ibi_ms=ibi_ms,
                accelerometer=None,
                skin_temperature_celsius=skin_temp,
                ambient_temperature_celsius=None,
                quality=TelemetrySampleQuality(
                    heart_rate=hr_quality,
                    ibi=ibi_quality,
                    wearing_state=WearingState.UNKNOWN,
                ),
            )


            samples.append(sample)
            current_time += timedelta(seconds=1.0 / self.sampling_rate_hz)


        return samples

    def generate_batch(
        self,
        user_id: Optional[UUID] = None,
        device_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        sequence: int = 0,
        is_anomaly_session: bool = False,
        ibi_supported: Optional[bool] = None,
    ) -> TelemetryBatch:
        """Generate a single telemetry batch."""
        if user_id is None:
            user_id = self.rng.choice(list(self._user_baselines.keys()))
        if device_id is None:
            device_id = self._deterministic_uuid("device")
        if session_id is None:
            session_id = self._deterministic_uuid("session")

        user_baseline = self._user_baselines[user_id]
        # Use deterministic session start based on seed and UUID counter
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        days_offset = int(self.rng.integers(0, 30))
        session_start = base_time + timedelta(days=days_offset)

        if ibi_supported is None:
            ibi_supported = bool(
                self.rng.random() < self.ibi_available_prob
            )

        samples = self._generate_samples(
            session_start=session_start,
            n_samples=self.samples_per_session,
            is_anomaly_session=is_anomaly_session,
            user_baseline=user_baseline,
            ibi_supported=ibi_supported,
        )

        batch = TelemetryBatch(
            batch_id=self._deterministic_uuid("batch"),
            device_id=device_id,
            user_id=user_id,
            session_id=session_id,
            started_at=samples[0].timestamp,
            ended_at=samples[-1].timestamp,
            sequence=sequence,
            samples=samples,
        )
        return batch

    def generate_dataset(self) -> tuple[list[TelemetryBatch], dict[UUID, bool]]:
        batches = []
        anomaly_sessions = {}


        for user_id in self._user_baselines:
            device_id = self._deterministic_uuid(f"device_{user_id}")


            # Synthetic device capability.
            # A device either provides IBI or does not.
            device_supports_ibi = bool(
                self.rng.random() < self.ibi_available_prob
            )


            for session_idx in range(self.n_sessions_per_user):
                session_id = self._deterministic_uuid(f"session_{user_id}_{session_idx}")


                is_anomaly = bool(
                    self.rng.random() < self.anomaly_probability
                )


                anomaly_sessions[session_id] = is_anomaly


                batch = self.generate_batch(
                    user_id=user_id,
                    device_id=device_id,
                    session_id=session_id,
                    sequence=session_idx,
                    is_anomaly_session=is_anomaly,
                    ibi_supported=device_supports_ibi,
                )


                batches.append(batch)


        logger.info(
            "Generated synthetic dataset: %d batches, %d anomaly sessions",
            len(batches),
            sum(anomaly_sessions.values()),
        )


        return batches, anomaly_sessions

    def generate_dataframe(self) -> pd.DataFrame:
        """Generate a flat DataFrame for exploration/analysis."""
        batches, _ = self.generate_dataset()
        rows = []
        for batch in batches:
            for sample in batch.samples:
                rows.append({
                    "batch_id": str(batch.batch_id),
                    "user_id": str(batch.user_id) if batch.user_id else None,
                    "device_id": str(batch.device_id),
                    "session_id": str(batch.session_id),
                    "sequence": batch.sequence,
                    "timestamp": sample.timestamp,
                    "heart_rate_bpm": sample.heart_rate_bpm,
                    "ibi_count": len(sample.ibi_ms),
                    "ibi_mean_ms": np.mean(sample.ibi_ms) if sample.ibi_ms else None,
                    "skin_temperature_celsius": sample.skin_temperature_celsius,
                    "quality_heart_rate": sample.quality.heart_rate.value,
                    "quality_ibi": sample.quality.ibi.value,
                    "is_synthetic": True,  # CLEARLY MARKED
                })
        return pd.DataFrame(rows)


def create_generator(config: dict) -> SyntheticTelemetryGenerator:
    """Factory function to create a generator from config."""
    return SyntheticTelemetryGenerator(config)