"""
Tests for dataset QA and feature parity (Phase 3-B).

Covers:
- feature parity: direct comparisons, derived baseline-delta check, and the
  fields that must NOT be compared (movement features).
- dataset QA: missingness, exclusions grouped by reason, single-class warning,
  IBI-entirely-missing warning, empty dataset, and input-order invariance.

Neither QA nor parity trains a model.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from anxietywatch_ml.config import load_config
from anxietywatch_ml.contracts.telemetry import (
    TelemetryBatch,
    TelemetrySample,
    TelemetrySampleQuality,
)
from anxietywatch_ml.ground_truth.builder import (
    GroundTruthDataset,
    GroundTruthDatasetBuilder,
    create_ground_truth_builder,
)
from anxietywatch_ml.ground_truth.contracts import (
    EventDecision,
    SuspectedEvent,
    SuspectedEventBaseline,
    SuspectedEventFeatures,
)
from anxietywatch_ml.ground_truth.synthetic import create_ground_truth_generator
from anxietywatch_ml.qa import (
    DIRECTLY_COMPARABLE,
    ML_ONLY,
    NOT_COMPARABLE,
    WATCH_ONLY,
    compute_dataset_qa,
    compute_feature_parity,
)


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("configs/base.yaml")


@pytest.fixture
def generator(config):
    """Create synthetic ground-truth generator."""
    return create_ground_truth_generator(config)


def _batch(
    user_id,
    device_id,
    session_id,
    t_end,
    n=60,
    hr=72.0,
    ibi=None,
    ts_step=1,
):
    """TelemetryBatch covering [t_end - n*step + step, t_end] at step seconds."""
    ts = [t_end - timedelta(seconds=ts_step * (n - 1 - k)) for k in range(n)]
    samples = []
    for t in ts:
        samples.append(
            TelemetrySample(
                timestamp=t,
                heart_rate_bpm=hr,
                ibi_ms=list(ibi) if ibi else [],
                accelerometer=None,
                skin_temperature_celsius=31.0,
                ambient_temperature_celsius=None,
                quality=TelemetrySampleQuality(
                    heart_rate="good", ibi="good", wearing_state="onBody"
                ),
            )
        )
    return TelemetryBatch(
        batch_id=uuid.uuid4(),
        device_id=device_id,
        user_id=user_id,
        session_id=session_id,
        started_at=ts[0],
        ended_at=ts[-1],
        sequence=1,
        samples=samples,
    )


def _decision(
    user_id,
    device_id,
    session_id,
    t_end,
    event_id,
    response="USER_OK",
):
    return EventDecision(
        event_id=event_id,
        device_id=device_id,
        user_id=user_id,
        session_id=session_id,
        sequence=1,
        detected_at=t_end,
        responded_at=t_end + timedelta(seconds=8),
        response=response,
    )


def _suspected(
    user_id,
    device_id,
    session_id,
    t_end,
    event_id,
    features_overrides=None,
    baseline_overrides=None,
):
    features = SuspectedEventFeatures(
        heart_rate_mean=72.0,
        heart_rate_max=72.0,
        heart_rate_slope_bpm_per_minute=0.0,
        heart_rate_delta_from_baseline=None,
        rmssd_millis=None,
        sdnn_millis=None,
        movement_magnitude_mean=None,
        movement_variance=None,
        valid_sample_ratio=1.0,
        last_sample_age_seconds=0,
        sample_count=60,
    )
    if features_overrides:
        features = features.model_copy(update=features_overrides)
    baseline = SuspectedEventBaseline(
        sample_count=100,
        mean_heart_rate=70.0,
        heart_rate_m2=120.0,
        updated_at_epoch_millis=0,
    )
    if baseline_overrides:
        baseline = baseline.model_copy(update=baseline_overrides)
    return SuspectedEvent(
        event_id=event_id,
        device_id=device_id,
        user_id=user_id,
        session_id=session_id,
        sequence=1,
        detected_at=t_end,
        state="USER_VALIDATION",
        score=0.8,
        rules_version="rules-v1",
        features=features,
        baseline=baseline,
    )


def _event_set(n_events=2, hr=72.0, ibi=None, response="USER_OK"):
    """Return (batches, suspected, decisions) for ``n_events`` matching events."""
    user = uuid.uuid4()
    device = uuid.uuid4()
    batches, suspected, decisions = [], [], []
    for k in range(n_events):
        t_end = datetime(2026, 1, 15, 10, 0, k, tzinfo=UTC)
        session = uuid.uuid4()
        event_id = uuid.uuid4()
        batches.append(_batch(user, device, session, t_end, hr=hr, ibi=ibi))
        suspected.append(_suspected(user, device, session, t_end, event_id))
        decisions.append(_decision(user, device, session, t_end, event_id, response))
    return batches, suspected, decisions


def _permuted(dataset, perm):
    """A copy of the dataset with its rows reordered by ``perm``."""
    return GroundTruthDataset(
        X=dataset.X.iloc[perm].reset_index(drop=True),
        y=dataset.y.iloc[perm].reset_index(drop=True),
        metadata=dataset.metadata.iloc[perm].reset_index(drop=True),
        feature_names=list(dataset.feature_names),
        excluded_metadata_columns=list(dataset.excluded_metadata_columns),
        label_counts=dict(dataset.label_counts),
        dropped_no_telemetry=dataset.dropped_no_telemetry,
        dropped_insufficient_data=dataset.dropped_insufficient_data,
        identity_mismatches=dataset.identity_mismatches,
        duplicate_conflicts=dataset.duplicate_conflicts,
        event_mismatches=dataset.event_mismatches,
        exclusions=dataset.exclusions,
    )


class TestFeatureParity:
    """Watch vs ML feature parity: mapping and measured differences."""

    def test_directly_comparable_mapping(self):
        assert DIRECTLY_COMPARABLE == {
            "heart_rate_mean": "hr_mean",
            "heart_rate_max": "hr_max",
            "heart_rate_slope_bpm_per_minute": "hr_slope_bpm_per_min",
            "rmssd_millis": "hrv_rmssd",
            "sdnn_millis": "hrv_sdnn",
            "valid_sample_ratio": "valid_sample_ratio",
            "sample_count": "sample_count",
        }

    def test_not_comparable_has_movement(self):
        assert "movement_magnitude_mean" in NOT_COMPARABLE
        assert "movement_variance" in NOT_COMPARABLE
        for reason in NOT_COMPARABLE.values():
            assert "cloud" in reason

    def test_match_when_equal(self):
        batches, suspected, decisions = _event_set(n_events=2)
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_feature_parity(dataset)
        assert not report.rows.empty
        compared = report.rows[report.rows["status"].isin(["match", "diverging"])]
        assert not compared.empty
        for _, row in compared.iterrows():
            if row["watch_field"] in (
                "heart_rate_mean",
                "heart_rate_max",
                "sample_count",
                "valid_sample_ratio",
            ):
                assert row["status"] == "match"
                assert row["diff"] == pytest.approx(0.0, abs=1e-9)

    def test_diverging_reported_with_diff(self):
        batches, suspected, decisions = _event_set(n_events=1)
        suspected[0] = suspected[0].model_copy(
            update={
                "features": suspected[0].features.model_copy(
                    update={"heart_rate_mean": 80.0}
                )
            }
        )
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_feature_parity(dataset)
        row = report.rows[
            (report.rows["watch_field"] == "heart_rate_mean")
            & (report.rows["row_index"] == 0)
        ].iloc[0]
        assert row["status"] == "diverging"
        assert row["diff"] == pytest.approx(8.0, abs=1e-9)
        summary = report.summary.set_index("watch_field")
        assert summary.loc["heart_rate_mean", "n_diverging"] == 1

    def test_derived_baseline_delta(self):
        batches, suspected, decisions = _event_set(n_events=1)
        suspected[0] = suspected[0].model_copy(
            update={
                "features": suspected[0].features.model_copy(
                    update={"heart_rate_delta_from_baseline": 2.0}
                )
            }
        )
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_feature_parity(dataset)
        assert not report.derived.empty
        row = report.derived.iloc[0]
        assert row["watch_field"] == "heart_rate_delta_from_baseline"
        assert row["ml_value"] == pytest.approx(2.0, abs=1e-9)  # 72.0 - 70.0
        assert row["status"] == "match"

    def test_ml_only_and_watch_only_listed(self):
        batches, suspected, decisions = _event_set(n_events=1)
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_feature_parity(dataset)
        assert set(report.ml_only) == set(ML_ONLY)
        assert set(report.watch_only) == set(WATCH_ONLY)

    def test_missing_snapshot_warning(self):
        batches, suspected, decisions = _event_set(n_events=1)
        dataset = GroundTruthDatasetBuilder().build(batches, [], decisions)
        report = compute_feature_parity(dataset)
        assert report.rows.empty
        assert any("no watch_features_snapshot" in w for w in report.warnings)

    def test_empty_dataset_parity(self):
        builder = GroundTruthDatasetBuilder()
        dataset = builder.build([], [], [])
        report = compute_feature_parity(dataset)
        assert report.rows.empty
        assert any("empty dataset" in w for w in report.warnings)

    def test_synthetic_smoke(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=20)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        report = compute_feature_parity(dataset)
        assert not report.rows.empty
        assert not report.summary.empty
        assert set(report.not_comparable) == set(NOT_COMPARABLE)


class TestDatasetQA:
    """Dataset QA: missingness, exclusions, warnings, robustness."""

    def test_feature_missingness_correct(self):
        # Event 0 has no IBI (hrv features NaN), event 1 has IBI.
        ibi = [700.0, 710.0, 690.0]
        no_ibi = []
        batches0, sus0, dec0 = _event_set(n_events=1, ibi=no_ibi)
        batches1, sus1, dec1 = _event_set(n_events=1, ibi=ibi)
        dataset = GroundTruthDatasetBuilder().build(
            batches0 + batches1, sus0 + sus1, dec0 + dec1
        )
        report = compute_dataset_qa(dataset)
        missing = report.feature_missingness.set_index("feature")
        assert missing.loc["hrv_rmssd", "n_missing"] == 1
        assert missing.loc["hrv_rmssd", "missing_ratio"] == pytest.approx(0.5)
        assert missing.loc["hr_mean", "missing_ratio"] == 0.0
        assert missing.loc["ibi_available", "missing_ratio"] == 0.0

    def test_exclusions_grouped_by_reason(self):
        user = uuid.uuid4()
        device = uuid.uuid4()
        session = uuid.uuid4()
        t_end = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        batches = [_batch(user, device, session, t_end)]
        decisions = [
            _decision(user, device, session, t_end, uuid.uuid4(), "USER_OK"),
            _decision(
                user, device, session, t_end, uuid.uuid4(), "BREATHING_HELPED"
            ),
            _decision(
                user,
                device,
                session,
                t_end + timedelta(hours=2),
                uuid.uuid4(),
                "USER_OK",
            ),
        ]
        dataset = GroundTruthDatasetBuilder().build(batches, [], decisions)
        report = compute_dataset_qa(dataset)
        grouped = report.exclusions_by_reason.set_index("reason")["count"].to_dict()
        assert grouped == {"unsupported_response": 1, "missing_telemetry": 1}

    def test_single_class_produces_warning(self):
        batches, suspected, decisions = _event_set(n_events=2, response="USER_OK")
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_dataset_qa(dataset)
        assert report.n_classes == 1
        assert any("single class" in w for w in report.warnings)

    def test_ibi_totally_missing_produces_warning(self):
        batches, suspected, decisions = _event_set(n_events=2, ibi=[])
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_dataset_qa(dataset)
        assert report.ibi_coverage["n_no_ibi"] == report.n_rows
        assert any("IBI entirely missing" in w for w in report.warnings)

    def test_empty_dataset_handled_controlled(self):
        builder = GroundTruthDatasetBuilder()
        dataset = builder.build([], [], [])
        report = compute_dataset_qa(dataset)
        assert report.n_rows == 0
        assert any("empty dataset" in w for w in report.warnings)
        assert any("small dataset" in w for w in report.warnings)

    def test_input_order_does_not_change_qa(self):
        batches, suspected, decisions = _event_set(n_events=4)
        dataset = GroundTruthDatasetBuilder().build(batches, suspected, decisions)
        report = compute_dataset_qa(dataset)
        perm = [2, 0, 3, 1]
        shuffled = _permuted(dataset, perm)
        re_report = compute_dataset_qa(shuffled)

        assert re_report.class_balance == report.class_balance
        assert re_report.users == report.users
        assert re_report.sessions == report.sessions
        assert re_report.devices == report.devices
        assert re_report.responses == report.responses
        assert re_report.samples_per_window == report.samples_per_window
        pd.testing.assert_frame_equal(
            re_report.feature_missingness, report.feature_missingness
        )
        pd.testing.assert_frame_equal(
            re_report.exclusions_by_reason, report.exclusions_by_reason
        )
        assert (
            re_report.temporal_coverage["span_seconds"]
            == report.temporal_coverage["span_seconds"]
        )

    def test_synthetic_smoke(self, config):
        docs = create_ground_truth_generator(config).generate_docs(n_events=30)
        dataset = create_ground_truth_builder(config).build(
            docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
        )
        report = compute_dataset_qa(dataset)
        assert report.n_rows == len(dataset.X)
        assert report.n_classes >= 2
        assert len(report.users) >= 1
        assert len(report.sessions) == report.n_rows
        assert report.class_balance.get("1", 0) > 0
        assert report.class_balance.get("0", 0) > 0