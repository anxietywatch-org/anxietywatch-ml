"""Tests for the event-anchored raw-window inference endpoint (007-B1).

Covers the /predict/window contract, API-key authentication, window
validation gates, determinism and — critically — training-serving config
parity between the serving path (EventWindowProcessor.from_bundle) and the
offline GroundTruthDatasetBuilder path.

The window contract is NEVER hardcoded in serving: it is derived from the
training-time config embedded in the serialized bundle. The regression tests
below prove that a NON-default window contract (90s / 20 samples / 0.4 HR
ratio) is honored consistently by BOTH paths, so serving cannot silently drift
back to a different (e.g. 60s / 10 / 0.3) contract.
"""

import copy
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from anxietywatch_ml.config import load_config
from anxietywatch_ml.ground_truth.builder import (
    GroundTruthBuilderConfig,
    create_ground_truth_builder,
)
from anxietywatch_ml.ground_truth.synthetic import create_ground_truth_generator
from anxietywatch_ml.serving import (
    FEATURE_SCHEMA,
    GroundTruthPredictor,
    PredictWindowRequest,
    create_app,
    train_demo_model,
)
from anxietywatch_ml.serving.predictor import PredictorError
from anxietywatch_ml.serving.window_inference import EventWindowProcessor
from anxietywatch_ml.training import load_ground_truth_bundle

TEST_API_KEY = "test-window-api-key-007-b1"
AUTH_HEADERS = {"X-Api-Key": TEST_API_KEY}

# A non-default window contract used to prove training-serving config parity.
CUSTOM_WINDOW_SIZE = 90.0
CUSTOM_MIN_SAMPLES = 20
CUSTOM_MIN_HR_RATIO = 0.4


@pytest.fixture(scope="module")
def config():
    return load_config("configs/base.yaml")


@pytest.fixture(scope="module")
def custom_config(config):
    cfg = copy.deepcopy(config)
    cfg["ground_truth"]["window_size_seconds"] = CUSTOM_WINDOW_SIZE
    cfg["ground_truth"]["min_samples_per_window"] = CUSTOM_MIN_SAMPLES
    cfg["ground_truth"]["min_hr_ratio"] = CUSTOM_MIN_HR_RATIO
    cfg["window"]["size_seconds"] = CUSTOM_WINDOW_SIZE
    return cfg


@pytest.fixture(scope="module")
def docs(config):
    generator = create_ground_truth_generator(config)
    return generator.generate_docs(n_events=5)


@pytest.fixture(scope="module")
def custom_docs(custom_config):
    generator = create_ground_truth_generator(custom_config)
    return generator.generate_docs(n_events=5)


@pytest.fixture(scope="module")
def dataset(config, docs):
    builder = create_ground_truth_builder(config)
    return builder.build(
        docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
    )


@pytest.fixture(scope="module")
def custom_dataset(custom_config, custom_docs):
    builder = create_ground_truth_builder(custom_config)
    return builder.build(
        custom_docs["telemetry_batches"],
        custom_docs["suspected_events"],
        custom_docs["event_decisions"],
    )


@pytest.fixture(scope="module")
def bundle_path(config, tmp_path_factory):
    out = tmp_path_factory.mktemp("window_serving") / "demo.pkl"
    train_demo_model(config, output_path=out)
    return out


@pytest.fixture(scope="module")
def custom_bundle_path(custom_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("window_custom") / "custom.pkl"
    train_demo_model(custom_config, output_path=out)
    return out


@pytest.fixture(scope="module")
def client(bundle_path):
    return TestClient(create_app(str(bundle_path), api_key=TEST_API_KEY))


def _window_payload(docs, index):
    """Plain camelCase JSON payload for a synthetic detector event.

    Deep-copies the samples so tests may mutate their payload without
    corrupting the shared module-scoped ``docs`` fixture.
    """
    suspected = docs["suspected_events"][index]
    batch = docs["telemetry_batches"][index]
    return {
        "eventId": suspected["eventId"],
        "deviceId": suspected["deviceId"],
        "sessionId": suspected["sessionId"],
        "detectedAt": suspected["detectedAt"],
        "userId": suspected.get("userId"),
        "samples": copy.deepcopy(batch["samples"]),
    }


def _in_window_samples(payload, window_size: float = 60.0):
    t_end = datetime.fromisoformat(payload["detectedAt"])
    t_start = t_end - timedelta(seconds=window_size)
    return [
        s
        for s in payload["samples"]
        if t_start <= datetime.fromisoformat(s["timestamp"]) <= t_end
    ]


def _hr_sample_at(iso_time, hr=70.0):
    return {
        "timestamp": iso_time,
        "heartRateBpm": hr,
        "ibiMs": [],
        "skinTemperatureCelsius": 33.0,
        "quality": {"heartRate": "good", "ibi": "good", "wearingState": "onBody"},
    }


def _handcrafted_request(detected_at, samples):
    return PredictWindowRequest(
        eventId="0d2a0d72-b0ff-4a0b-ba4f-6a8f2a0d3c1e",
        deviceId="22222222-2222-4222-8222-222222222222",
        sessionId="44444444-4444-4444-8444-444444444444",
        detectedAt=detected_at.isoformat(),
        samples=samples,
    )


class TestWindowConfigSource:
    def test_default_processor_matches_offline_default_builder(self, config):
        assert EventWindowProcessor().window_config == GroundTruthBuilderConfig()
        assert create_ground_truth_builder(config).config == GroundTruthBuilderConfig()

    def test_from_bundle_honors_non_default_training_config(
        self, custom_config, custom_bundle_path
    ):
        bundle = load_ground_truth_bundle(custom_bundle_path)
        processor = EventWindowProcessor.from_bundle(bundle)
        expected = GroundTruthBuilderConfig(
            window_size_seconds=CUSTOM_WINDOW_SIZE,
            min_samples_per_window=CUSTOM_MIN_SAMPLES,
            min_hr_ratio=CUSTOM_MIN_HR_RATIO,
        )
        assert processor.window_config == expected
        assert processor.window_config != GroundTruthBuilderConfig()
        assert create_ground_truth_builder(custom_config).config == expected

    def test_bundle_carries_training_config(self, custom_bundle_path):
        bundle = load_ground_truth_bundle(custom_bundle_path)
        gt = bundle.runtime_config["ground_truth"]
        assert gt["window_size_seconds"] == CUSTOM_WINDOW_SIZE
        assert gt["min_samples_per_window"] == CUSTOM_MIN_SAMPLES
        assert gt["min_hr_ratio"] == CUSTOM_MIN_HR_RATIO

    def test_custom_window_size_enforced_by_serving(self, custom_bundle_path):
        bundle = load_ground_truth_bundle(custom_bundle_path)
        processor = EventWindowProcessor.from_bundle(bundle)
        detected_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        # 30 samples covering [T-100, T-71]: inside a 90s window, outside a 60s one.
        samples = [
            _hr_sample_at((detected_at - timedelta(seconds=100 - i)).isoformat())
            for i in range(30)
        ]
        request = _handcrafted_request(detected_at, samples)
        features = processor.build_features(request)
        assert features["sample_count"] == pytest.approx(CUSTOM_WINDOW_SIZE - 70)
        # A processor on the DEFAULT contract would trim to [T-60, T] and find nothing.
        with pytest.raises(PredictorError, match="window"):
            EventWindowProcessor().build_features(request)

    def test_custom_min_samples_enforced_by_serving(self, custom_bundle_path):
        bundle = load_ground_truth_bundle(custom_bundle_path)
        processor = EventWindowProcessor.from_bundle(bundle)
        detected_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        # 15 samples in [T-60, T]: inside BOTH windows, passes the default 10,
        # fails the custom 20.
        samples = [
            _hr_sample_at((detected_at - timedelta(seconds=60 - i)).isoformat())
            for i in range(15)
        ]
        request = _handcrafted_request(detected_at, samples)
        with pytest.raises(PredictorError, match="insufficient window data"):
            processor.build_features(request)
        default_features = EventWindowProcessor().build_features(request)
        assert default_features["sample_count"] == pytest.approx(15)


class TestWindowAuthentication:
    def test_requires_api_key(self, client):
        assert client.post("/predict/window", json=_window_payload_from_docs()).status_code == 401

    def test_rejects_wrong_api_key(self, client):
        response = client.post(
            "/predict/window",
            json=_window_payload_from_docs(),
            headers={"X-Api-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert "wrong-key" not in response.text
        assert TEST_API_KEY not in response.text

    def test_health_still_unauthd(self, client):
        assert client.get("/health").status_code == 200

    def test_unconfigured_key_is_503_not_public(self, monkeypatch):
        monkeypatch.delenv("ANXIETYWATCH_API_KEY", raising=False)
        client = TestClient(create_app("nonexistent_bundle.pkl", api_key=None))
        response = client.post(
            "/predict/window", json=_window_payload_from_docs(), headers=AUTH_HEADERS
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "inference authentication is not configured"


def _window_payload_from_docs():
    """Build a syntactically valid body lazily (never reaches the handler)."""
    from anxietywatch_ml.config import load_config
    from anxietywatch_ml.ground_truth.synthetic import create_ground_truth_generator

    config = load_config("configs/base.yaml")
    docs = create_ground_truth_generator(config).generate_docs(n_events=1)
    return _window_payload(docs, 0)


class TestWindowEndpoint:
    def test_valid_window_returns_200(self, client, docs):
        response = client.post(
            "/predict/window", json=_window_payload(docs, 0), headers=AUTH_HEADERS
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["support_probability"] <= 1.0
        assert body["model_version"] == "0.1.0"
        assert body["target"] == "target_support_requested"
        assert body["prediction"] == (1 if body["support_probability"] >= body["threshold"] else 0)

    def test_threshold_comes_from_bundle_metadata(self, client, docs, bundle_path):
        bundle = load_ground_truth_bundle(bundle_path)
        expected = bundle.runtime_config["model"]["threshold"]
        body = client.post(
            "/predict/window", json=_window_payload(docs, 0), headers=AUTH_HEADERS
        ).json()
        assert body["threshold"] == pytest.approx(expected)
        assert body["threshold"] != 0.5 or expected == 0.5

    def test_snake_case_transport_accepted(self, client, docs):
        payload = _window_payload(docs, 0)
        snake = {
            "event_id": payload["eventId"],
            "device_id": payload["deviceId"],
            "session_id": payload["sessionId"],
            "detected_at": payload["detectedAt"],
            "user_id": payload.get("userId"),
            "samples": [
                {
                    "timestamp": s["timestamp"],
                    "heart_rate_bpm": s["heartRateBpm"],
                    "ibi_ms": s["ibiMs"],
                    "skin_temperature_celsius": s["skinTemperatureCelsius"],
                    "quality": {
                        "heart_rate": s["quality"]["heartRate"],
                        "ibi": s["quality"]["ibi"],
                        "wearing_state": s["quality"]["wearingState"],
                    },
                }
                for s in payload["samples"]
            ],
        }
        response = client.post("/predict/window", json=snake, headers=AUTH_HEADERS)
        assert response.status_code == 200

    def test_unknown_top_level_field_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        payload["batchId"] = "11111111-1111-4111-8111-111111111111"
        assert client.post("/predict/window", json=payload, headers=AUTH_HEADERS).status_code == 422

    def test_unknown_sample_field_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        payload["samples"][0]["accelerometer"] = None
        payload["samples"][0]["extraSensorField"] = 1.0
        assert client.post("/predict/window", json=payload, headers=AUTH_HEADERS).status_code == 422

    def test_missing_required_field_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        del payload["eventId"]
        assert client.post("/predict/window", json=payload, headers=AUTH_HEADERS).status_code == 422

    def test_empty_samples_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        payload["samples"] = []
        assert client.post("/predict/window", json=payload, headers=AUTH_HEADERS).status_code == 422

    def test_no_samples_in_window_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        before = detected_at - timedelta(seconds=180)
        payload["samples"] = [
            _hr_sample_at((before + timedelta(seconds=i)).isoformat())
            for i in range(20)
        ]
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert "window" in response.json()["detail"]

    def test_too_few_in_window_samples_rejected(self, client, docs):
        min_samples = GroundTruthBuilderConfig().min_samples_per_window
        payload = _window_payload(docs, 0)
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        payload["samples"] = [
            _hr_sample_at(
                (detected_at - timedelta(seconds=min_samples + 5 - i)).isoformat(),
                hr=70.0 + i,
            )
            for i in range(min_samples - 1)
        ]
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert str(min_samples) in response.json()["detail"]

    def test_insufficient_hr_coverage_rejected(self, client, docs):
        min_samples = GroundTruthBuilderConfig().min_samples_per_window
        payload = _window_payload(docs, 0)
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        payload["samples"] = [
            _hr_sample_at(
                (detected_at - timedelta(seconds=min_samples + 5 - i)).isoformat(),
                hr=70.0 + i if i < 3 else None,
            )
            for i in range(min_samples + 5)
        ]
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert "heart-rate coverage" in response.json()["detail"]

    def test_out_of_window_samples_ignored(self, client, docs):
        payload = _window_payload(docs, 0)
        in_window = _in_window_samples(payload)
        assert len(in_window) >= GroundTruthBuilderConfig().min_samples_per_window
        minimal = dict(payload)
        minimal["samples"] = in_window
        body_minimal = client.post("/predict/window", json=minimal, headers=AUTH_HEADERS).json()
        # Extras far outside [T-60s, T] (even a multi-batch volume >600 samples)
        # must be trimmed and produce the identical prediction.
        extra = []
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        base = detected_at - timedelta(minutes=30)
        for i in range(700):
            extra.append(_hr_sample_at((base + timedelta(seconds=i)).isoformat()))
        full = dict(payload)
        full["samples"] = in_window + extra
        body_full = client.post("/predict/window", json=full, headers=AUTH_HEADERS).json()
        assert body_minimal == body_full

    def test_out_of_order_samples_deterministic(self, client, docs):
        payload = _window_payload(docs, 0)
        ordered = client.post(
            "/predict/window", json=payload, headers=AUTH_HEADERS
        ).json()
        shuffled = dict(payload)
        shuffled["samples"] = list(reversed(payload["samples"]))
        reversed_body = client.post(
            "/predict/window", json=shuffled, headers=AUTH_HEADERS
        ).json()
        assert ordered == reversed_body

    def test_malformed_sample_422_without_internals(self, client, docs):
        payload = _window_payload(docs, 0)
        payload["samples"][0]["heartRateBpm"] = "not-a-number"
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 422
        assert "Traceback" not in response.text

    def test_no_model_503(self):
        client = TestClient(create_app("nonexistent_bundle.pkl", api_key=TEST_API_KEY))
        payload = _window_payload_from_docs()
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 503


class TestWindowParityWithGroundTruth:
    def test_serving_features_match_offline_builder(self, docs, dataset):
        processor = EventWindowProcessor()
        kept = set(dataset.metadata["event_id"].astype(str))
        compared = 0
        for index, suspected in enumerate(docs["suspected_events"]):
            event_id = str(suspected["eventId"])
            if event_id not in kept:
                continue
            row_idx = dataset.metadata.index[
                dataset.metadata["event_id"] == event_id
            ][0]
            request = PredictWindowRequest(**_window_payload(docs, index))
            serving = processor.build_features(request)
            offline = dataset.X.iloc[row_idx]
            for name in FEATURE_SCHEMA:
                s = serving[name]
                o = offline[name]
                if pd.isna(o):
                    assert s is None, (name, s, o)
                else:
                    assert s == pytest.approx(float(o), abs=1e-9), (name, s, o)
            compared += 1
        assert compared >= 3

    def test_custom_config_offline_serving_parity(
        self, custom_docs, custom_dataset, custom_bundle_path
    ):
        bundle = load_ground_truth_bundle(custom_bundle_path)
        processor = EventWindowProcessor.from_bundle(bundle)
        kept = set(custom_dataset.metadata["event_id"].astype(str))
        compared = 0
        for index, suspected in enumerate(custom_docs["suspected_events"]):
            event_id = str(suspected["eventId"])
            if event_id not in kept:
                continue
            row_idx = custom_dataset.metadata.index[
                custom_dataset.metadata["event_id"] == event_id
            ][0]
            request = PredictWindowRequest(**_window_payload(custom_docs, index))
            serving = processor.build_features(request)
            offline = custom_dataset.X.iloc[row_idx]
            for name in FEATURE_SCHEMA:
                s = serving[name]
                o = offline[name]
                if pd.isna(o):
                    assert s is None, (name, s, o)
                else:
                    assert s == pytest.approx(float(o), abs=1e-9), (name, s, o)
            compared += 1
        assert compared >= 3

    def test_window_prediction_uses_existing_predictor(self, docs, bundle_path, monkeypatch):
        predictor = GroundTruthPredictor.from_path(bundle_path)

        def boom(*args, **kwargs):
            raise AssertionError("fit() must never run during inference")

        monkeypatch.setattr(predictor.bundle.model, "fit", boom)
        processor = EventWindowProcessor.from_bundle(predictor.bundle, predictor=predictor)
        response = processor.predict(PredictWindowRequest(**_window_payload(docs, 0)))
        assert response.prediction in (0, 1)