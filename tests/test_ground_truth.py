"""
Tests for the ground-truth dataset builder (Phase 3).

Covers contract normalization, label policy, dataset building (window
selection, exclusions), synthetic docs reproducibility, and the end-to-end
smoke (synthetic docs -> X, y, metadata).
"""

import uuid
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from anxietywatch_ml.config import load_config
from anxietywatch_ml.contracts.normalize import IdentityMismatchError
from anxietywatch_ml.contracts.telemetry import (
    SignalQuality,
    TelemetryBatch,
    TelemetryBatchAdapter,
    TelemetrySample,
    TelemetrySampleQuality,
)
from anxietywatch_ml.data.synthetic import create_generator
from anxietywatch_ml.ground_truth.builder import (
    EXCLUDED_METADATA_COLUMNS,
    EXCLUSION_REASONS,
    GroundTruthDatasetBuilder,
    create_ground_truth_builder,
)
from anxietywatch_ml.ground_truth.contracts import (
    EventDecision,
    EventDecisionAdapter,
    SuspectedEvent,
    SuspectedEventAdapter,
)
from anxietywatch_ml.ground_truth.label_policy import (
    PHYSICAL_ACTIVITY,
    PRIMARY_RESPONSES,
    SELF_REPORTED_OK,
    SUPPORT_REQUESTED,
    apply_label_policy,
)
from anxietywatch_ml.ground_truth.synthetic import (
    batch_to_backend_dict,
    create_ground_truth_generator,
)

# Feature columns produced by the default FeatureConfig (features/builder.py).
EXPECTED_FEATURE_COLUMNS = {
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_slope_bpm_per_min",
    "hrv_rmssd",
    "hrv_sdnn",
    "ibi_available",
    "ibi_coverage_ratio",
    "skin_temp_mean",
    "quality_good_ratio",
    "quality_fair_ratio",
    "quality_poor_ratio",
    "valid_sample_ratio",
    "window_duration_seconds",
    "sample_count",
}


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("configs/base.yaml")


@pytest.fixture
def generator(config):
    """Create synthetic ground-truth generator."""
    return create_ground_truth_generator(config)


def _sample_suspected_doc(event_id="11111111-1111-4111-8111-111111111111"):
    return {
        "_id": event_id,
        "eventId": event_id,
        "deviceId": "22222222-2222-4222-8222-222222222222",
        "userId": "33333333-3333-4333-8333-333333333333",
        "sessionId": "44444444-4444-4444-8444-444444444444",
        "sequence": 3,
        "detectedAt": "2026-01-15T10:00:00+00:00",
        "state": "USER_VALIDATION",
        "score": 0.81,
        "rulesVersion": "rules-v1",
        "features": {
            "heartRateMean": 88.5,
            "heartRateMax": 101.0,
            "heartRateSlopeBpmPerMinute": 3.2,
            "heartRateDeltaFromBaseline": 12.0,
            "rmssdMillis": 45.1,
            "sdnnMillis": 60.0,
            "movementMagnitudeMean": None,
            "movementVariance": None,
            "validSampleRatio": 0.95,
            "lastSampleAgeSeconds": 0,
            "sampleCount": 57,
        },
        "baseline": {
            "sampleCount": 100,
            "meanHeartRate": 70.0,
            "heartRateM2": 120.0,
            "updatedAtEpochMillis": 1736927400000,
        },
        "receivedAt": "2026-01-15T10:00:08+00:00",
    }


def _sample_decision_doc(event_id="11111111-1111-4111-8111-111111111111", response="SUPPORT_REQUESTED"):
    return {
        "_id": event_id,
        "eventId": event_id,
        "deviceId": "22222222-2222-4222-8222-222222222222",
        "userId": "33333333-3333-4333-8333-333333333333",
        "sessionId": "44444444-4444-4444-8444-444444444444",
        "sequence": 3,
        "detectedAt": "2026-01-15T10:00:00+00:00",
        "respondedAt": "2026-01-15T10:00:08+00:00",
        "response": response,
        "receivedAt": "2026-01-15T10:00:08+00:00",
    }


_MONGO_EVENT_ID = "11111111-1111-4111-8111-111111111111"
_MONGO_DEVICE_ID = "22222222-2222-4222-8222-222222222222"
_MONGO_USER_ID = "33333333-3333-4333-8333-333333333333"
_MONGO_SESSION_ID = "44444444-4444-4444-8444-444444444444"


# Fixture with the conceptual shape of a real .NET/Mongo document:
# PascalCase payload keys + lowercase auth `userId` + `_id`/`receivedAt`.
_MONGO_SUSPECTED = {
    "_id": _MONGO_EVENT_ID,
    "EventId": _MONGO_EVENT_ID,
    "DeviceId": _MONGO_DEVICE_ID,
    "UserId": None,
    "SessionId": _MONGO_SESSION_ID,
    "DetectedAt": "2026-01-15T10:00:00+00:00",
    "Sequence": 3,
    "State": "USER_VALIDATION",
    "Score": 0.62,
    "RulesVersion": "rules-v2",
    "Features": {
        "HeartRateMean": 81.0,
        "HeartRateMax": 88.0,
        "HeartRateSlopeBpmPerMinute": 1.5,
        "HeartRateDeltaFromBaseline": 8.0,
        "RmssdMillis": 40.0,
        "SdnnMillis": 55.0,
        "MovementMagnitudeMean": None,
        "MovementVariance": None,
        "ValidSampleRatio": 0.9,
        "LastSampleAgeSeconds": 0,
        "SampleCount": 40,
    },
    "Baseline": {
        "SampleCount": 100,
        "MeanHeartRate": 72.0,
        "HeartRateM2": 0.0,
        "UpdatedAtEpochMillis": 1736927400000,
    },
    "userId": _MONGO_USER_ID,
    "receivedAt": "2026-01-15T10:00:08+00:00",
}

_MONGO_DECISION = {
    "_id": _MONGO_EVENT_ID,
    "EventId": _MONGO_EVENT_ID,
    "DeviceId": _MONGO_DEVICE_ID,
    "UserId": None,
    "SessionId": _MONGO_SESSION_ID,
    "Sequence": 3,
    "DetectedAt": "2026-01-15T10:00:00+00:00",
    "RespondedAt": "2026-01-15T10:00:08+00:00",
    "Response": "SUPPORT_REQUESTED",
    "userId": _MONGO_USER_ID,
    "receivedAt": "2026-01-15T10:00:08+00:00",
}


def _camel_suspected_doc():
    """Same values as _MONGO_SUSPECTED but with camelCase keys."""
    return {
        "_id": _MONGO_EVENT_ID,
        "eventId": _MONGO_EVENT_ID,
        "deviceId": _MONGO_DEVICE_ID,
        "userId": _MONGO_USER_ID,
        "sessionId": _MONGO_SESSION_ID,
        "sequence": 3,
        "detectedAt": "2026-01-15T10:00:00+00:00",
        "state": "USER_VALIDATION",
        "score": 0.62,
        "rulesVersion": "rules-v2",
        "features": {
            "heartRateMean": 81.0,
            "heartRateMax": 88.0,
            "heartRateSlopeBpmPerMinute": 1.5,
            "heartRateDeltaFromBaseline": 8.0,
            "rmssdMillis": 40.0,
            "sdnnMillis": 55.0,
            "movementMagnitudeMean": None,
            "movementVariance": None,
            "validSampleRatio": 0.9,
            "lastSampleAgeSeconds": 0,
            "sampleCount": 40,
        },
        "baseline": {
            "sampleCount": 100,
            "meanHeartRate": 72.0,
            "heartRateM2": 0.0,
            "updatedAtEpochMillis": 1736927400000,
        },
        "receivedAt": "2026-01-15T10:00:08+00:00",
    }


def _snake_suspected_doc():
    """Same values as _MONGO_SUSPECTED but with snake_case keys."""
    return {
        "_id": _MONGO_EVENT_ID,
        "event_id": _MONGO_EVENT_ID,
        "device_id": _MONGO_DEVICE_ID,
        "user_id": _MONGO_USER_ID,
        "session_id": _MONGO_SESSION_ID,
        "sequence": 3,
        "detected_at": "2026-01-15T10:00:00+00:00",
        "state": "USER_VALIDATION",
        "score": 0.62,
        "rules_version": "rules-v2",
        "features": {
            "heart_rate_mean": 81.0,
            "heart_rate_max": 88.0,
            "heart_rate_slope_bpm_per_minute": 1.5,
            "heart_rate_delta_from_baseline": 8.0,
            "rmssd_millis": 40.0,
            "sdnn_millis": 55.0,
            "movement_magnitude_mean": None,
            "movement_variance": None,
            "valid_sample_ratio": 0.9,
            "last_sample_age_seconds": 0,
            "sample_count": 40,
        },
        "baseline": {
            "sample_count": 100,
            "mean_heart_rate": 72.0,
            "heart_rate_m2": 0.0,
            "updated_at_epoch_millis": 1736927400000,
        },
        "received_at": "2026-01-15T10:00:08+00:00",
    }


def _mongo_telemetry_batch():
    """Mongo-shaped telemetry covering [T-60s, T] with PascalCase keys."""
    detected_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    samples = []
    for i in range(31):
        samples.append(
            {
                "Timestamp": (detected_at - timedelta(seconds=60 - 2 * i)).isoformat(),
                "HeartRateBpm": 72.0 + i % 3,
                "IbiMs": [],
                "Accelerometer": None,
                "SkinTemperatureCelsius": 31.0,
                "AmbientTemperatureCelsius": None,
                "Quality": {
                    "HeartRate": "good",
                    "Ibi": "good",
                    "WearingState": "onBody",
                },
            }
        )
    return {
        "_id": "99999999-9999-4999-8999-999999999999",
        "BatchId": "99999999-9999-4999-8999-999999999999",
        "DeviceId": _MONGO_DEVICE_ID,
        "UserId": None,
        "SessionId": _MONGO_SESSION_ID,
        "StartedAt": (detected_at - timedelta(seconds=60)).isoformat(),
        "EndedAt": detected_at.isoformat(),
        "Sequence": 3,
        "Samples": samples,
        "userId": _MONGO_USER_ID,
        "receivedAt": detected_at.isoformat(),
    }


class TestGroundTruthContracts:
    """Contract normalization from backend Mongo documents."""

    def test_suspected_adapter_camelcase(self):
        event = SuspectedEventAdapter.from_backend_dict(_sample_suspected_doc())
        assert isinstance(event, SuspectedEvent)
        assert event.event_id is not None
        assert event.score == 0.81
        assert event.state == "USER_VALIDATION"
        assert event.rules_version == "rules-v1"
        assert event.features.heart_rate_mean == 88.5
        assert event.features.sample_count == 57
        assert event.baseline.mean_heart_rate == 70.0

    def test_suspected_adapter_ignores_mongo_metadata(self):
        event = SuspectedEventAdapter.from_backend_dict(_sample_suspected_doc())
        assert event.event_id is not None  # parsed; _id/receivedAt ignored silently

    def test_suspected_adapter_rejects_score_out_of_range(self):
        doc = _sample_suspected_doc()
        doc["score"] = 1.5
        with pytest.raises(ValueError):
            SuspectedEventAdapter.from_backend_dict(doc)

    def test_decision_adapter(self):
        decision = EventDecisionAdapter.from_backend_dict(_sample_decision_doc())
        assert isinstance(decision, EventDecision)
        assert decision.response == "SUPPORT_REQUESTED"
        assert decision.detected_at <= decision.responded_at

    def test_decision_adapter_rejects_responded_before_detected(self):
        doc = _sample_decision_doc()
        doc["respondedAt"] = "2026-01-15T09:59:00+00:00"
        with pytest.raises(ValueError):
            EventDecisionAdapter.from_backend_dict(doc)

    def test_telemetry_adapter_roundtrip(self, config):
        batch = create_generator(config).generate_batch()
        doc = batch_to_backend_dict(batch)
        parsed = TelemetryBatchAdapter.from_backend_dict(doc)
        assert isinstance(parsed, TelemetryBatch)
        assert parsed.batch_id == batch.batch_id
        assert len(parsed.samples) == len(batch.samples)


class TestLabelPolicy:
    """Derived label view; original response is always preserved."""

    def test_support_requested_target_1(self):
        result = apply_label_policy("SUPPORT_REQUESTED")
        assert result.target_support_requested == 1
        assert result.response_category == SUPPORT_REQUESTED
        assert result.response == "SUPPORT_REQUESTED"

    def test_activity_confirmed_target_0(self):
        result = apply_label_policy("ACTIVITY_CONFIRMED")
        assert result.target_support_requested == 0
        assert result.response_category == PHYSICAL_ACTIVITY

    def test_user_ok_target_0(self):
        result = apply_label_policy("USER_OK")
        assert result.target_support_requested == 0
        assert result.response_category == SELF_REPORTED_OK

    def test_case_insensitive(self):
        assert apply_label_policy("support_requested").target_support_requested == 1

    def test_original_response_preserved(self):
        assert apply_label_policy("USER_OK").response == "USER_OK"

    def test_invalid_response_raises(self):
        with pytest.raises(ValueError):
            apply_label_policy("BREATHING_HELPED")

    def test_only_primary_responses_accepted(self):
        for response in PRIMARY_RESPONSES:
            apply_label_policy(response)


class TestGroundTruthBuilder:
    """Dataset building from synthetic in-memory docs."""

    def test_build_from_synthetic_docs(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=30)
        builder = create_ground_truth_builder(config)
        dataset = builder.build(
            docs["telemetry_batches"],
            docs["suspected_events"],
            docs["event_decisions"],
        )
        assert len(dataset.X) > 0
        assert len(dataset.X) == len(dataset.y) == len(dataset.metadata)
        assert dataset.label_counts.get(1, 0) > 0
        assert dataset.label_counts.get(0, 0) > 0

    def test_x_contains_only_features(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=10)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        forbidden = set(EXCLUDED_METADATA_COLUMNS) | {
            "event_id",
            "user_id",
            "device_id",
            "session_id",
            "response",
            "response_category",
            "target_support_requested",
        }
        assert forbidden.isdisjoint(set(dataset.X.columns))

    def test_x_column_names_exact(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=10)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert set(dataset.X.columns) == EXPECTED_FEATURE_COLUMNS

    def test_detector_metadata_in_metadata_not_x(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=5)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        for col in EXCLUDED_METADATA_COLUMNS:
            assert col in dataset.metadata.columns
            assert col not in dataset.X.columns
        assert "detector_score" in dataset.metadata.columns
        assert dataset.metadata["detector_score"].notna().all()

    def test_original_response_preserved_in_metadata(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=10)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert set(dataset.metadata["response"]) <= set(PRIMARY_RESPONSES)

    def test_window_uses_only_decision_session(self, config, generator):
        docs = generator.generate_docs(n_events=2)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        # One row per decision, no cross-session mixing.
        sessions = dataset.metadata["session_id"].tolist()
        assert len(set(sessions)) == len(sessions)

    def test_decision_without_telemetry_is_dropped(self, config, generator):
        docs = generator.generate_docs(n_events=5)
        # Move one decision outside all telemetry timestamps.
        doc = docs["event_decisions"][0]
        doc["detectedAt"] = "2099-01-01T00:00:00+00:00"
        doc["respondedAt"] = "2099-01-01T00:00:08+00:00"
        # Keep the suspected event aligned (same detectedAt) so the exclusion
        # reaches the telemetry step instead of the event_mismatch step.
        docs["suspected_events"][0]["detectedAt"] = "2099-01-01T00:00:00+00:00"
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.dropped_no_telemetry == 1
        assert len(dataset.X) == 4

    def test_no_inf_in_features(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=10)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        numeric = dataset.X.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy()).any()

    def test_build_from_normalized_models(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=5)
        telemetry = [
            TelemetryBatchAdapter.from_backend_dict(d) for d in docs["telemetry_batches"]
        ]
        suspected = [
            SuspectedEventAdapter.from_backend_dict(d) for d in docs["suspected_events"]
        ]
        decisions = [
            EventDecisionAdapter.from_backend_dict(d) for d in docs["event_decisions"]
        ]
        builder = GroundTruthDatasetBuilder()
        dataset = builder.build(telemetry, suspected, decisions)
        assert len(dataset.X) == 5


class TestGroundTruthSmoke:
    """End-to-end smoke: synthetic docs -> dataset -> features/labels/metadata."""

    def test_full_smoke(self, config):
        cfg = dict(config)
        cfg["synthetic"] = dict(config["synthetic"])
        cfg["synthetic"]["anomaly_probability"] = 0.5

        generator = create_ground_truth_generator(cfg)
        docs = generator.generate_docs(n_events=30)
        builder = create_ground_truth_builder(cfg)
        dataset = builder.build(
            docs["telemetry_batches"],
            docs["suspected_events"],
            docs["event_decisions"],
        )

        assert len(dataset.X) == len(dataset.y) == len(dataset.metadata) > 0
        assert set(dataset.y.unique()).issubset({0, 1})
        assert dataset.label_counts.get(1, 0) > 0
        assert dataset.label_counts.get(0, 0) > 0

        # Exclusions guaranteed.
        for col in EXCLUDED_METADATA_COLUMNS:
            assert col not in dataset.X.columns

        # Parity check runs against the watch snapshot.
        parity = builder.parity_check(dataset)
        assert len(parity) > 0
        assert "diff_heart_rate_mean" in parity.columns

    def test_reproducible_docs(self, config):
        gen1 = create_ground_truth_generator(config)
        gen2 = create_ground_truth_generator(config)
        docs1 = gen1.generate_docs(n_events=5)
        docs2 = gen2.generate_docs(n_events=5)
        assert docs1["event_decisions"] == docs2["event_decisions"]
        assert docs1["telemetry_batches"] == docs2["telemetry_batches"]
        assert docs1["suspected_events"] == docs2["suspected_events"]

    def test_reproducible_dataset(self, config):
        generator = create_ground_truth_generator(config)
        docs = generator.generate_docs(n_events=10)
        d1 = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        d2 = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        pd.testing.assert_frame_equal(d1.X, d2.X)
        pd.testing.assert_series_equal(d1.y, d2.y)

    def test_dataset_save_roundtrip(self, config, tmp_path):
        docs = create_ground_truth_generator(config).generate_docs(n_events=5)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        dataset.save(str(tmp_path))
        assert (tmp_path / "X.csv").exists()
        assert (tmp_path / "y.csv").exists()
        assert (tmp_path / "metadata.csv").exists()

    def test_summary(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=5)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        summary = dataset.summary()
        assert summary["n_rows"] == len(dataset.X)
        assert summary["n_features"] == len(dataset.feature_names)
        assert set(summary["excluded_metadata_columns"]) == set(EXCLUDED_METADATA_COLUMNS)


class TestMongoShapedIngest:
    """Real .NET/Mongo documents (PascalCase payload + auth userId)."""

    def test_pascal_case_suspected_normalized(self):
        event = SuspectedEventAdapter.from_backend_dict(dict(_MONGO_SUSPECTED))
        assert event.event_id == UUID(_MONGO_EVENT_ID)
        assert event.user_id == UUID(_MONGO_USER_ID)
        assert event.sequence == 3
        assert event.score == 0.62
        assert event.state == "USER_VALIDATION"
        assert event.rules_version == "rules-v2"
        assert event.features.heart_rate_mean == 81.0
        assert event.features.sample_count == 40
        assert event.baseline.mean_heart_rate == 72.0
        assert event.baseline.sample_count == 100

    def test_pascal_case_decision_normalized(self):
        decision = EventDecisionAdapter.from_backend_dict(dict(_MONGO_DECISION))
        assert decision.event_id == UUID(_MONGO_EVENT_ID)
        assert decision.user_id == UUID(_MONGO_USER_ID)
        assert decision.response == "SUPPORT_REQUESTED"
        assert decision.detected_at <= decision.responded_at

    def test_pascal_case_telemetry_normalized(self):
        batch = TelemetryBatchAdapter.from_backend_dict(_mongo_telemetry_batch())
        assert batch.batch_id == UUID("99999999-9999-4999-8999-999999999999")
        assert batch.user_id == UUID(_MONGO_USER_ID)
        assert batch.session_id == UUID(_MONGO_SESSION_ID)
        assert len(batch.samples) == 31
        assert batch.samples[0].heart_rate_bpm == 72.0
        assert batch.samples[0].quality.heart_rate == SignalQuality.GOOD
        assert batch.samples[0].quality.ibi == SignalQuality.GOOD

    def test_three_key_spellings_equivalent(self):
        pascal = SuspectedEventAdapter.from_backend_dict(dict(_MONGO_SUSPECTED))
        camel = SuspectedEventAdapter.from_backend_dict(_camel_suspected_doc())
        snake = SuspectedEventAdapter.from_backend_dict(_snake_suspected_doc())
        assert pascal.model_dump() == camel.model_dump() == snake.model_dump()


class TestIdentityRule:
    """Canonical auth userId; conflicting UserId/userId excludes the doc."""

    def test_auth_user_id_is_canonical(self):
        doc = dict(_camel_suspected_doc())
        doc["UserId"] = None
        event = SuspectedEventAdapter.from_backend_dict(doc)
        assert event.user_id == UUID(_MONGO_USER_ID)

    def test_identity_mismatch_raises(self):
        doc = dict(_camel_suspected_doc())
        doc["UserId"] = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        with pytest.raises(IdentityMismatchError):
            SuspectedEventAdapter.from_backend_dict(doc)

    def test_build_excludes_identity_mismatch(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=3)
        docs["event_decisions"][0]["UserId"] = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.identity_mismatches == 1
        assert len(dataset.X) == 2


class TestDuplicateEvents:
    """Equivalent duplicates deduped; conflicting event_ids fully excluded."""

    def test_duplicate_suspected_identical_dedup(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        docs["suspected_events"].append(dict(docs["suspected_events"][0]))
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.duplicate_conflicts == 0
        assert dataset.metadata["has_suspected_event"].all()

    def test_duplicate_suspected_conflict_excluded(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        dup = dict(docs["suspected_events"][0])
        dup["score"] = 0.99
        docs["suspected_events"].append(dup)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.duplicate_conflicts == 1
        assert dataset.metadata["has_suspected_event"].tolist() == [False, True]

    def test_duplicate_decision_identical_dedup(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        docs["event_decisions"].append(dict(docs["event_decisions"][0]))
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.duplicate_conflicts == 0
        assert len(dataset.X) == 2

    def test_duplicate_decision_conflict_excluded(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        dup = dict(docs["event_decisions"][0])
        original = docs["event_decisions"][0]["response"]
        dup["response"] = "USER_OK" if original != "USER_OK" else "SUPPORT_REQUESTED"
        docs["event_decisions"].append(dup)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.duplicate_conflicts == 1
        assert len(dataset.X) == 1


class TestMongoShapedSmoke:
    """Mongo-shaped telemetry + suspected + decision -> GroundTruthDataset."""

    def test_build_from_mongo_shaped_docs(self):
        builder = GroundTruthDatasetBuilder()
        dataset = builder.build(
            [_mongo_telemetry_batch()],
            [dict(_MONGO_SUSPECTED)],
            [dict(_MONGO_DECISION)],
        )
        assert len(dataset.X) == 1
        assert len(dataset.X) == len(dataset.y) == len(dataset.metadata)
        assert dataset.identity_mismatches == 0
        assert dataset.duplicate_conflicts == 0
        assert dataset.metadata["user_id"].iloc[0] == _MONGO_USER_ID
        assert dataset.metadata["detector_score"].iloc[0] == 0.62
        assert dataset.metadata["response"].iloc[0] == "SUPPORT_REQUESTED"
        assert dataset.y.iloc[0] == 1
        assert set(dataset.X.columns) == EXPECTED_FEATURE_COLUMNS


class TestGroundTruthConfig:
    """Dataset quality policy must live in configuration, not hardcoded."""

    def test_min_hr_ratio_from_config(self, config):
        builder = create_ground_truth_builder(config)
        gt = config["ground_truth"]
        assert builder.config.window_size_seconds == gt["window_size_seconds"] == 60
        assert builder.config.min_samples_per_window == gt["min_samples_per_window"] == 10
        assert builder.config.min_hr_ratio == gt["min_hr_ratio"] == 0.3


class TestEventMismatch:
    """Same eventId is not enough: identities and detectedAt must also match."""

    def test_detected_at_mismatch_excluded(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        docs["suspected_events"][0]["detectedAt"] = "2099-01-01T00:00:00+00:00"
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.event_mismatches == 1
        assert len(dataset.X) == 1
        assert "event_mismatch" in dataset.exclusions["reason"].tolist()

    def test_session_device_mismatch_excluded(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        docs["suspected_events"][0]["sessionId"] = "55555555-5555-4555-8555-555555555555"
        docs["suspected_events"][0]["deviceId"] = "66666666-6666-4666-8666-666666666666"
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.event_mismatches == 1
        assert len(dataset.X) == 1

    def test_matching_event_kept(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        assert dataset.event_mismatches == 0
        assert len(dataset.X) == 2


class TestWindowBoundaries:
    """Permanent temporal invariants of the [T-60s, T] window."""

    def _decision_at(self, T):
        return EventDecision(
            event_id=UUID(_MONGO_EVENT_ID),
            device_id=UUID(_MONGO_DEVICE_ID),
            user_id=UUID(_MONGO_USER_ID),
            session_id=UUID(_MONGO_SESSION_ID),
            sequence=1,
            detected_at=T,
            responded_at=T + timedelta(seconds=8),
            response="SUPPORT_REQUESTED",
        )

    def _sample(self, ts):
        return TelemetrySample(
            timestamp=ts,
            heart_rate_bpm=72.0,
            ibi_ms=[],
            accelerometer=None,
            skin_temperature_celsius=31.0,
            ambient_temperature_celsius=None,
            quality=TelemetrySampleQuality(heart_rate="good", ibi="good", wearing_state="onBody"),
        )

    def _batch(self, uid, did, sid, bid, samples):
        return TelemetryBatch(
            batch_id=bid,
            device_id=did,
            user_id=uid,
            session_id=sid,
            started_at=samples[0].timestamp,
            ended_at=samples[-1].timestamp,
            sequence=1,
            samples=samples,
        )

    def _context(self):
        uid, did, sid = UUID(_MONGO_USER_ID), UUID(_MONGO_DEVICE_ID), UUID(_MONGO_SESSION_ID)
        T = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        return uid, did, sid, T

    def test_window_multi_batch_normal(self):
        uid, did, sid, T = self._context()
        bid_a, bid_b = uuid.uuid4(), uuid.uuid4()
        batch_a = self._batch(
            uid, did, sid, bid_a,
            [self._sample(T - timedelta(seconds=60 - i)) for i in range(30)],
        )
        batch_b = self._batch(
            uid, did, sid, bid_b,
            [self._sample(T - timedelta(seconds=30 - i)) for i in range(31)],
        )
        builder = GroundTruthDatasetBuilder()
        flat = builder._prep._flatten_batches([batch_a, batch_b])
        window = builder._select_window(flat, self._decision_at(T))
        assert len(window) == 61
        assert set(window["batch_id"]) == {str(bid_a), str(bid_b)}
        assert (window["timestamp"] >= T - timedelta(seconds=60)).all()
        assert (window["timestamp"] <= T).all()

    def test_window_multi_batch_inverted(self):
        uid, did, sid, T = self._context()
        bid_a, bid_b = uuid.uuid4(), uuid.uuid4()
        batch_a = self._batch(
            uid, did, sid, bid_a,
            [self._sample(T - timedelta(seconds=60 - i)) for i in range(30)],
        )
        batch_b = self._batch(
            uid, did, sid, bid_b,
            [self._sample(T - timedelta(seconds=30 - i)) for i in range(31)],
        )
        builder = GroundTruthDatasetBuilder()
        ds_normal = builder.build([batch_a, batch_b], [], [self._decision_at(T)])
        ds_inverted = builder.build([batch_b, batch_a], [], [self._decision_at(T)])
        pd.testing.assert_frame_equal(ds_normal.X, ds_inverted.X)
        pd.testing.assert_series_equal(ds_normal.y, ds_inverted.y)
        assert ds_normal.X["sample_count"].iloc[0] == 61

    def test_boundaries_t60_t_t1ms_excluded(self):
        uid, did, sid, T = self._context()
        samples = [
            self._sample(T - timedelta(seconds=60)),
            self._sample(T - timedelta(seconds=30)),
            self._sample(T),
            self._sample(T + timedelta(milliseconds=1)),
            self._sample(T + timedelta(seconds=30)),
        ]
        batch = self._batch(uid, did, sid, uuid.uuid4(), samples)
        builder = GroundTruthDatasetBuilder()
        flat = builder._prep._flatten_batches([batch])
        window = builder._select_window(flat, self._decision_at(T))
        assert set(window["timestamp"]) == {
            T - timedelta(seconds=60),
            T - timedelta(seconds=30),
            T,
        }
        assert int((window["timestamp"] > T).sum()) == 0
        assert int((window["timestamp"] == T).sum()) == 1
        assert int((window["timestamp"] == T - timedelta(seconds=60)).sum()) == 1


class TestExclusions:
    """Every exclusion carries an explicit reason; event_mismatch is observable."""

    def test_reasons_are_explicit(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=4)
        docs["event_decisions"][0]["UserId"] = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        docs["suspected_events"][1]["detectedAt"] = "2099-01-01T00:00:00+00:00"
        docs["event_decisions"][2]["detectedAt"] = "2099-01-01T00:00:00+00:00"
        docs["event_decisions"][2]["respondedAt"] = "2099-01-01T00:00:08+00:00"
        docs["suspected_events"][2]["detectedAt"] = "2099-01-01T00:00:00+00:00"
        docs["event_decisions"][3]["response"] = "BREATHING_HELPED"
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        reasons = set(dataset.exclusions["reason"])
        assert {
            "identity_mismatch",
            "event_mismatch",
            "missing_telemetry",
            "unsupported_response",
        } <= reasons
        assert reasons <= set(EXCLUSION_REASONS)
        assert dataset.summary()["n_exclusions"] == len(dataset.exclusions)
        assert dataset.summary()["event_mismatches"] == 1

    def test_duplicate_conflict_reasons(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=2)
        dup = dict(docs["suspected_events"][0])
        dup["score"] = 0.99
        docs["suspected_events"].append(dup)
        d2 = dict(docs["event_decisions"][0])
        original = docs["event_decisions"][0]["response"]
        d2["response"] = "USER_OK" if original != "USER_OK" else "SUPPORT_REQUESTED"
        docs["event_decisions"].append(d2)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        reasons = set(dataset.exclusions["reason"])
        assert {"duplicate_suspected_conflict", "duplicate_decision_conflict"} <= reasons
        assert dataset.summary()["duplicate_conflicts"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])