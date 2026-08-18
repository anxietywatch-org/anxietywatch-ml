"""Synthetic ground-truth documents for the dataset builder.

Generates in-memory JSON documents matching the durable backend collections
(``suspected_events``, ``event_decisions``) and the ``telemetry_batches``
contract, so the dataset builder can be validated end-to-end without a
database connection.

All output is SYNTHETIC, deterministic by seed, and clearly marked. It exists
to validate the plumbing, not to represent real anxiety detection data.
"""

import hashlib
import logging
from datetime import timedelta
from typing import Optional
from uuid import UUID

import numpy as np
from scipy import stats

from anxietywatch_ml.contracts.telemetry import TelemetryBatch
from anxietywatch_ml.data.synthetic import SyntheticTelemetryGenerator

logger = logging.getLogger(__name__)


def batch_to_backend_dict(batch: TelemetryBatch) -> dict:
    """Serialize a TelemetryBatch to a backend Mongo-style document (camelCase)."""
    return {
        "batchId": str(batch.batch_id),
        "deviceId": str(batch.device_id),
        "userId": str(batch.user_id) if batch.user_id else None,
        "sessionId": str(batch.session_id),
        "startedAt": batch.started_at.isoformat(),
        "endedAt": batch.ended_at.isoformat(),
        "sequence": batch.sequence,
        "samples": [
            {
                "timestamp": s.timestamp.isoformat(),
                "heartRateBpm": s.heart_rate_bpm,
                "ibiMs": s.ibi_ms,
                "accelerometer": s.accelerometer,
                "skinTemperatureCelsius": s.skin_temperature_celsius,
                "ambientTemperatureCelsius": s.ambient_temperature_celsius,
                "quality": {
                    "heartRate": s.quality.heart_rate.value,
                    "ibi": s.quality.ibi.value,
                    "wearingState": s.quality.wearing_state.value,
                },
            }
            for s in batch.samples
        ],
    }


class SyntheticGroundTruthGenerator:
    """Generates consistent synthetic telemetry + suspected + decision docs."""

    def __init__(self, config: dict):
        self.config = config
        self.random_seed = config.get("random_seed", 42)
        self.rng = np.random.default_rng(self.random_seed)
        self.telemetry_gen = SyntheticTelemetryGenerator(config)
        gt_cfg = config.get("ground_truth", {})
        self.window_size_seconds = gt_cfg.get("window_size_seconds", 60.0)
        self.support_requested_probability = gt_cfg.get("support_requested_probability", 0.3)
        self.label_noise = gt_cfg.get("label_noise", 0.05)
        self.n_events = gt_cfg.get("n_events", 20)
        self.anomaly_probability = config.get("synthetic", {}).get("anomaly_probability", 0.15)
        self._uuid_counter = 0

    def _deterministic_uuid(self, prefix: str = "") -> UUID:
        self._uuid_counter += 1
        seed_str = f"{self.random_seed}_{prefix}_{self._uuid_counter}"
        hash_bytes = hashlib.md5(seed_str.encode()).digest()
        hash_bytes = bytearray(hash_bytes)
        hash_bytes[6] = (hash_bytes[6] & 0x0F) | 0x40  # Version 4
        hash_bytes[8] = (hash_bytes[8] & 0x3F) | 0x80  # Variant 10
        return UUID(bytes=bytes(hash_bytes))

    def generate_docs(self, n_events: Optional[int] = None) -> dict:
        """Generate in-memory JSON docs for telemetry, suspected events, decisions."""
        n = n_events or self.n_events
        user_ids = list(self.telemetry_gen._user_baselines.keys())
        docs = {"telemetry_batches": [], "suspected_events": [], "event_decisions": []}

        for i in range(n):
            user_id = user_ids[i % len(user_ids)]
            device_id = self._deterministic_uuid(f"device_{i}")
            session_id = self._deterministic_uuid(f"session_{i}")
            is_anomaly = bool(self.rng.random() < self.anomaly_probability)
            ibi_supported = bool(self.rng.random() < self.telemetry_gen.ibi_available_prob)

            batch = self.telemetry_gen.generate_batch(
                user_id=user_id,
                device_id=device_id,
                session_id=session_id,
                sequence=i,
                is_anomaly_session=is_anomaly,
                ibi_supported=ibi_supported,
            )
            docs["telemetry_batches"].append(batch_to_backend_dict(batch))

            event_id = self._deterministic_uuid(f"event_{i}")
            detected_at = batch.samples[self._event_sample_index(len(batch.samples))].timestamp
            response = self._sample_response(is_anomaly)

            docs["suspected_events"].append(
                self._suspected_doc(
                    event_id, batch, detected_at, is_anomaly, user_id, device_id, session_id, i
                )
            )
            docs["event_decisions"].append(
                self._decision_doc(
                    event_id, batch, detected_at, response, user_id, device_id, session_id, i
                )
            )

        logger.info(
            "Generated synthetic ground truth: %d events, %d batches",
            n,
            len(docs["telemetry_batches"]),
        )
        return docs

    def _event_sample_index(self, n_samples: int) -> int:
        """Anchor the detection so [T-60s, T] is fully covered by telemetry."""
        return min(max(120, int(n_samples * 0.6)), n_samples - 1)

    def _sample_response(self, is_anomaly: bool) -> str:
        if is_anomaly and self.rng.random() < (1.0 - self.label_noise):
            return "SUPPORT_REQUESTED"
        if not is_anomaly and self.rng.random() < self.label_noise:
            return "SUPPORT_REQUESTED"
        return "ACTIVITY_CONFIRMED" if self.rng.random() < 0.5 else "USER_OK"

    def _suspected_doc(
        self,
        event_id: UUID,
        batch: TelemetryBatch,
        detected_at,
        is_anomaly: bool,
        user_id,
        device_id,
        session_id,
        sequence: int,
    ) -> dict:
        t_start = detected_at - timedelta(seconds=self.window_size_seconds)
        window = [
            s for s in batch.samples if t_start <= s.timestamp <= detected_at
        ]
        features = self._watch_features(window)
        user_baseline = self.telemetry_gen._user_baselines[user_id]
        score = (
            float(0.62 + 0.33 * self.rng.random())
            if is_anomaly
            else float(self.rng.uniform(0.45, 0.58))
        )
        return {
            "_id": str(event_id),
            "eventId": str(event_id),
            "deviceId": str(device_id),
            "userId": str(user_id),
            "sessionId": str(session_id),
            "sequence": sequence,
            "detectedAt": detected_at.isoformat(),
            "state": "USER_VALIDATION",
            "score": round(score, 4),
            "rulesVersion": "synthetic-rules-v1",
            "features": features,
            "baseline": {
                "sampleCount": 100,
                "meanHeartRate": round(float(user_baseline["hr_mean"]), 4),
                "heartRateM2": 0.0,
                "updatedAtEpochMillis": int(detected_at.timestamp() * 1000),
            },
            "receivedAt": detected_at.isoformat(),
        }

    def _decision_doc(
        self,
        event_id: UUID,
        batch: TelemetryBatch,
        detected_at,
        response: str,
        user_id,
        device_id,
        session_id,
        sequence: int,
    ) -> dict:
        responded_at = detected_at + timedelta(seconds=8)
        return {
            "_id": str(event_id),
            "eventId": str(event_id),
            "deviceId": str(device_id),
            "userId": str(user_id),
            "sessionId": str(session_id),
            "sequence": sequence,
            "detectedAt": detected_at.isoformat(),
            "respondedAt": responded_at.isoformat(),
            "response": response,
            "receivedAt": responded_at.isoformat(),
        }

    @staticmethod
    def _watch_features(window) -> dict:
        """Compute a plausible Watch features snapshot from the synthetic window."""
        hrs = [s.heart_rate_bpm for s in window if s.heart_rate_bpm is not None]
        n = len(window)
        slope = None
        if hrs and len(hrs) >= 2:
            times = [s.timestamp for s in window if s.heart_rate_bpm is not None]
            x = np.array([(t - times[0]).total_seconds() / 60.0 for t in times])
            y = np.array(hrs)
            slope = float(stats.linregress(x, y).slope)
        return {
            "heartRateMean": float(np.mean(hrs)) if hrs else None,
            "heartRateMax": float(np.max(hrs)) if hrs else None,
            "heartRateSlopeBpmPerMinute": slope,
            "heartRateDeltaFromBaseline": None,
            "rmssdMillis": None,
            "sdnnMillis": None,
            "movementMagnitudeMean": None,
            "movementVariance": None,
            "validSampleRatio": round(len(hrs) / n, 4) if n else 0.0,
            "lastSampleAgeSeconds": 0,
            "sampleCount": len(hrs),
        }


def create_ground_truth_generator(config: dict) -> SyntheticGroundTruthGenerator:
    """Factory function to create a ground-truth generator from config."""
    return SyntheticGroundTruthGenerator(config)