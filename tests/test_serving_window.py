"""Tests for the event-anchored raw-window inference endpoint (007-B1).

Covers the /predict/window contract, API-key authentication, window
validation gates, determinism and — critically — feature parity between the
serving path (EventWindowProcessor) and the offline GroundTruthDatasetBuilder
path.
"""

import copy
from datetime import datetime, timedelta

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
    MIN_HR_RATIO,
    MIN_WINDOW_SAMPLES,
    WINDOW_SIZE_SECONDS,
    GroundTruthPredictor,
    PredictWindowRequest,
    create_app,
    train_demo_model,
)
from anxietywatch_ml.serving.window_inference import EventWindowProcessor
from anxietywatch_ml.training import load_ground_truth_bundle

TEST_API_KEY = "test-window-api-key-007-b1"
AUTH_HEADERS = {"X-Api-Key": TEST_API_KEY}


@pytest.fixture(scope="module")
def config():
    return load_config("configs/base.yaml")


@pytest.fixture(scope="module")
def docs(config):
    generator = create_ground_truth_generator(config)
    return generator.generate_docs(n_events=5)


@pytest.fixture(scope="module")
def dataset(config, docs):
    builder = create_ground_truth_builder(config)
    return builder.build(
        docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
    )


@pytest.fixture(scope="module")
def bundle_path(config, tmp_path_factory):
    out = tmp_path_factory.mktemp("window_serving") / "demo.pkl"
    train_demo_model(config, output_path=out)
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


def _in_window_samples(payload):
    t_end = datetime.fromisoformat(payload["detectedAt"])
    t_start = t_end - timedelta(seconds=WINDOW_SIZE_SECONDS)
    return [
        s
        for s in payload["samples"]
        if t_start <= datetime.fromisoformat(s["timestamp"]) <= t_end
    ]


class TestWindowConstants:
    def test_constants_match_ground_truth_builder(self):
        cfg = GroundTruthBuilderConfig()
        assert cfg.window_size_seconds == WINDOW_SIZE_SECONDS
        assert cfg.min_samples_per_window == MIN_WINDOW_SAMPLES
        assert cfg.min_hr_ratio == MIN_HR_RATIO


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
    """Placeholder replaced by a module-scoped fixture-free helper.

    Authentication tests only need *a* syntactically valid body; it never
    reaches the handler. Build one lazily from the shared docs.
    """
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
            {
                "timestamp": (before + timedelta(seconds=i)).isoformat(),
                "heartRateBpm": 70.0,
                "ibiMs": [],
                "skinTemperatureCelsius": 33.0,
                "quality": {"heartRate": "good", "ibi": "good", "wearingState": "onBody"},
            }
            for i in range(20)
        ]
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert "window" in response.json()["detail"]

    def test_too_few_in_window_samples_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        payload["samples"] = [
            {
                "timestamp": (
                    detected_at - timedelta(seconds=MIN_WINDOW_SAMPLES + 5 - i)
                ).isoformat(),
                "heartRateBpm": 70.0 + i,
                "ibiMs": [],
                "skinTemperatureCelsius": 33.0,
                "quality": {"heartRate": "good", "ibi": "good", "wearingState": "onBody"},
            }
            for i in range(MIN_WINDOW_SAMPLES - 1)
        ]
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert str(MIN_WINDOW_SAMPLES) in response.json()["detail"]

    def test_insufficient_hr_coverage_rejected(self, client, docs):
        payload = _window_payload(docs, 0)
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        samples = []
        for i in range(MIN_WINDOW_SAMPLES + 5):
            samples.append(
                {
                    "timestamp": (
                        detected_at - timedelta(seconds=MIN_WINDOW_SAMPLES + 5 - i)
                    ).isoformat(),
                    "heartRateBpm": 70.0 + i if i < 3 else None,
                    "ibiMs": [],
                    "skinTemperatureCelsius": 33.0,
                    "quality": {"heartRate": "good", "ibi": "good", "wearingState": "onBody"},
                }
            )
        payload["samples"] = samples
        response = client.post("/predict/window", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert "heart-rate coverage" in response.json()["detail"]

    def test_out_of_window_samples_ignored(self, client, docs):
        payload = _window_payload(docs, 0)
        in_window = _in_window_samples(payload)
        assert len(in_window) >= MIN_WINDOW_SAMPLES
        minimal = dict(payload)
        minimal["samples"] = in_window
        body_minimal = client.post("/predict/window", json=minimal, headers=AUTH_HEADERS).json()
        # Extras far outside [T-60s, T] (even a multi-batch volume >600 samples)
        # must be trimmed and produce the identical prediction.
        extra = []
        detected_at = datetime.fromisoformat(payload["detectedAt"])
        base = detected_at - timedelta(minutes=30)
        for i in range(700):
            extra.append(
                {
                    "timestamp": (base + timedelta(seconds=i)).isoformat(),
                    "heartRateBpm": 70.0,
                    "ibiMs": [],
                    "skinTemperatureCelsius": 33.0,
                    "quality": {"heartRate": "good", "ibi": "good", "wearingState": "onBody"},
                }
            )
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

    def test_window_prediction_uses_existing_predictor(self, docs, bundle_path, monkeypatch):
        predictor = GroundTruthPredictor.from_path(bundle_path)

        def boom(*args, **kwargs):
            raise AssertionError("fit() must never run during inference")

        monkeypatch.setattr(predictor.bundle.model, "fit", boom)
        processor = EventWindowProcessor(predictor)
        response = processor.predict(PredictWindowRequest(**_window_payload(docs, 0)))
        assert response.prediction in (0, 1)