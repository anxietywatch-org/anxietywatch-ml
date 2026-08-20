"""Tests for the prototype inference service (005-A / 006-A container-ready)."""

import json

import pytest
from fastapi.testclient import TestClient

from anxietywatch_ml.config import load_config
from anxietywatch_ml.pipelines.model_pipeline import transform_for_inference
from anxietywatch_ml.serving import (
    FEATURE_SCHEMA,
    GroundTruthPredictor,
    create_app,
    train_demo_model,
)
from anxietywatch_ml.serving.predictor import PredictorError
from anxietywatch_ml.training import load_ground_truth_bundle

VALID_FEATURES = {
    "hr_mean": 72.0,
    "hr_std": 4.2,
    "hr_min": 60.0,
    "hr_max": 90.0,
    "hr_slope_bpm_per_min": 0.3,
    "hrv_rmssd": 38.0,
    "hrv_sdnn": 41.0,
    "ibi_available": 1.0,
    "ibi_coverage_ratio": 0.85,
    "skin_temp_mean": 33.2,
    "quality_good_ratio": 0.9,
    "quality_fair_ratio": 0.1,
    "quality_poor_ratio": 0.0,
    "valid_sample_ratio": 0.95,
    "window_duration_seconds": 60.0,
    "sample_count": 61,
}


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    config = load_config("configs/base.yaml")
    out = tmp_path_factory.mktemp("serving") / "demo.pkl"
    train_demo_model(config, output_path=out)
    return out


@pytest.fixture(scope="module")
def predictor(bundle_path):
    return GroundTruthPredictor.from_path(bundle_path)


@pytest.fixture(scope="module")
def client(bundle_path):
    return TestClient(create_app(str(bundle_path)))


@pytest.fixture(scope="module")
def expected_metadata(bundle_path):
    bundle = load_ground_truth_bundle(bundle_path)
    return bundle.runtime_config["model"]


class TestHealth:
    def test_health_with_loaded_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "0.1.0"

    def test_missing_model_health_not_loaded(self):
        client = TestClient(create_app("nonexistent_bundle.pkl"))
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"

    def test_degraded_health_does_not_leak_paths_or_exceptions(self):
        client = TestClient(create_app("nonexistent_bundle.pkl"))
        response = client.get("/health")
        assert response.status_code == 503
        assert "nonexistent_bundle.pkl" not in response.text
        assert "Traceback" not in response.text

    def test_missing_model_predict_503(self):
        client = TestClient(create_app("nonexistent_bundle.pkl"))
        assert client.post("/predict", json=VALID_FEATURES).status_code == 503


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        assert client.post("/predict", json=VALID_FEATURES).status_code == 200

    def test_prediction_is_binary(self, client):
        body = client.post("/predict", json=VALID_FEATURES).json()
        assert body["prediction"] in (0, 1)

    def test_probability_in_range(self, client):
        body = client.post("/predict", json=VALID_FEATURES).json()
        assert 0.0 <= body["support_probability"] <= 1.0

    def test_threshold_correct(self, client, expected_metadata):
        body = client.post("/predict", json=VALID_FEATURES).json()
        assert body["threshold"] == expected_metadata["threshold"]
        assert body["threshold"] == pytest.approx(expected_metadata["threshold"])

    def test_model_version_correct(self, client):
        body = client.post("/predict", json=VALID_FEATURES).json()
        assert body["model_version"] == "0.1.0"
        assert body["target"] == "target_support_requested"

    def test_same_payload_same_prediction(self, client):
        first = client.post("/predict", json=VALID_FEATURES).json()
        second = client.post("/predict", json=VALID_FEATURES).json()
        assert first == second

    def test_nan_missing_sensor_supported(self, client):
        payload = dict(VALID_FEATURES)
        payload["hrv_rmssd"] = None
        payload["hrv_sdnn"] = None
        payload["ibi_available"] = None
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["prediction"] in (0, 1)

    def test_invalid_feature_422(self, client):
        payload = dict(VALID_FEATURES)
        payload["detector_score"] = 0.62
        assert client.post("/predict", json=payload).status_code == 422

    def test_non_numeric_value_422(self, client):
        payload = dict(VALID_FEATURES)
        payload["hr_mean"] = "not-a-number"
        assert client.post("/predict", json=payload).status_code == 422


class TestNonFiniteInputRejection:
    @pytest.mark.parametrize("feature", ["hr_mean", "hr_std", "hrv_rmssd"])
    def test_positive_infinity_rejected(self, client, feature):
        # Raw JSON with an `Infinity` literal: not strict JSON, but Python's
        # json.loads accepts it, so the server must reject it explicitly.
        payload = dict(VALID_FEATURES)
        payload[feature] = "Infinity"
        raw = json.dumps(payload).replace('"Infinity"', "Infinity")
        response = client.post(
            "/predict", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize("feature", ["hr_mean", "skin_temp_mean"])
    def test_negative_infinity_rejected(self, client, feature):
        payload = dict(VALID_FEATURES)
        payload[feature] = "-Infinity"
        raw = json.dumps(payload).replace('"-Infinity"', "-Infinity")
        response = client.post(
            "/predict", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400

    def test_nan_still_supported_as_semantic_missing(self, client):
        payload = dict(VALID_FEATURES)
        payload["hr_mean"] = "NaN"
        raw = json.dumps(payload).replace('"NaN"', "NaN")
        response = client.post(
            "/predict", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        assert response.json()["prediction"] in (0, 1)


class TestRequiredModelStartup:
    def test_missing_artifact_with_require_model_raises(self):
        with pytest.raises(RuntimeError):
            create_app("nonexistent_bundle.pkl", require_model=True)

    def test_corrupt_artifact_with_require_model_raises(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        bad.write_bytes(b"not a valid anxietywatch artifact")
        with pytest.raises(RuntimeError):
            create_app(str(bad), require_model=True)

    def test_valid_artifact_with_require_model_starts_ok(self, bundle_path):
        client = TestClient(create_app(str(bundle_path), require_model=True))
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_require_model_from_environment_flag(self, monkeypatch):
        monkeypatch.setenv("ANXIETYWATCH_REQUIRE_MODEL", "1")
        with pytest.raises(RuntimeError):
            create_app("nonexistent_bundle.pkl")

    def test_require_model_env_disabled_starts_degraded(self, monkeypatch):
        monkeypatch.setenv("ANXIETYWATCH_REQUIRE_MODEL", "false")
        client = TestClient(create_app("nonexistent_bundle.pkl"))
        assert client.get("/health").status_code == 503


class TestPredictorInternals:
    def test_detector_leakage_never_enters_x(self, predictor):
        payload = dict(VALID_FEATURES)
        X = predictor._build_frame(payload)
        assert set(X.columns) == set(predictor.feature_names)
        assert "detector_score" not in X.columns
        with pytest.raises(PredictorError):
            predictor._build_frame({**payload, "detector_score": 0.62})

    def test_missing_structural_feature_raises(self, predictor):
        payload = dict(VALID_FEATURES)
        payload.pop("hr_mean")
        with pytest.raises(PredictorError):
            predictor.predict(payload)

    def test_uses_transform_for_inference(self, predictor, monkeypatch):
        called = []

        def spy_transform(bundle, X):
            called.append(X)
            return transform_for_inference(bundle, X)

        monkeypatch.setattr(
            "anxietywatch_ml.serving.predictor.transform_for_inference", spy_transform
        )
        predictor.predict(dict(VALID_FEATURES))
        assert len(called) == 1

    def test_no_fit_during_inference(self, predictor, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("fit() must never run during inference")

        monkeypatch.setattr(predictor.bundle.model, "fit", boom)
        result = predictor.predict(dict(VALID_FEATURES))
        assert result.prediction in (0, 1)

    def test_threshold_comes_from_metadata_not_0_5(self, predictor, expected_metadata):
        assert predictor.threshold == expected_metadata["threshold"]
        assert predictor.threshold != 0.5 or expected_metadata["threshold"] == 0.5


class TestRoundTrip:
    def test_save_load_predict_roundtrip(self, predictor, bundle_path):
        assert predictor.bundle is not None
        bundle = load_ground_truth_bundle(bundle_path)
        reloaded = GroundTruthPredictor(bundle)
        first = predictor.predict(dict(VALID_FEATURES))
        second = reloaded.predict(dict(VALID_FEATURES))
        assert first == second

    def test_feature_schema_contract_matches_dataset(self):
        assert set(FEATURE_SCHEMA) == set(VALID_FEATURES)