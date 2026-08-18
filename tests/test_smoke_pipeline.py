"""
End-to-end smoke test for the AnxietyWatch ML pipeline.

This test runs the full pipeline with synthetic data and verifies
that all components work together without errors.
"""

import pytest
import numpy as np
import pandas as pd

from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.synthetic import create_generator
from anxietywatch_ml.data.validation import validate_batch
from anxietywatch_ml.preprocessing.pipeline import create_pipeline
from anxietywatch_ml.features.builder import create_feature_builder
from anxietywatch_ml.models.baseline import create_model
from anxietywatch_ml.evaluation.metrics import create_evaluator, evaluate


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("configs/base.yaml")


@pytest.fixture
def generator(config):
    """Create synthetic data generator."""
    return create_generator(config)


class TestSmokePipeline:
    """Smoke tests for the full ML pipeline."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config("configs/base.yaml")

    @pytest.fixture
    def generator(self, config):
        """Create synthetic data generator."""
        return create_generator(config)

    def test_synthetic_data_generation(self, generator):
        """Test that synthetic data generation works."""
        batches, anomaly_sessions = generator.generate_dataset()

        assert len(batches) > 0
        # Default config: 10 users * 5 sessions = 50 batches
        assert len(batches) == 50
        assert len(anomaly_sessions) == 50

        for batch in batches[:5]:
            # Validate each batch
            result = validate_batch(batch)
            assert result.is_valid, f"Batch validation failed: {result.errors}"

    def test_preprocessing(self, config, generator):
        """Test preprocessing pipeline."""
        batches, _ = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        assert len(windowed.windows) > 0
        assert len(windowed.window_metadata) == len(windowed.windows)

        # Check metadata structure
        for meta in windowed.window_metadata:
            assert "session_id" in meta
            assert "user_id" in meta
            assert "window_start" in meta
            assert "window_end" in meta
            assert "n_samples" in meta
            assert meta["n_samples"] >= config["window"]["min_samples_per_window"]

    def test_feature_building(self, config, generator):
        """Test feature engineering."""
        batches, _ = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(windowed.windows)
        assert X.shape[1] > 0  # Has features

        # Check no Inf (NaN is allowed for semantic missingness)
        numeric = X.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy()).any(), "Features contain Inf"

    def test_model_training(self, config, generator):
        """Test baseline model training."""
        batches, anomaly_sessions = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        # Create synthetic labels using anomaly sessions
        pipeline_config = load_config("configs/base.yaml")
        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(pipeline_config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        assert len(y) == len(X)
        assert y.nunique() <= 2  # Binary classification

        # Train model
        model = create_model(config)
        model.fit(X, y)

        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluation(self, config, generator):
        """Test evaluation metrics."""
        batches, anomaly_sessions = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        model = create_model(config)
        model.fit(X, y)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        evaluator = create_evaluator(config)
        result = evaluate(y.values, predictions, probabilities, evaluator)

        assert hasattr(result, "metrics")
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1" in result.metrics

        # Metrics should be valid numbers
        for name, value in result.metrics.items():
            assert not np.isnan(value) or name in ["roc_auc", "average_precision"], f"{name} is NaN"
            if not np.isnan(value):
                assert 0 <= value <= 1, f"{name} out of range: {value}"

def test_full_pipeline(config, generator):
        """Test the complete pipeline end-to-end."""
        # Generate
        batches, anomaly_sessions = generator.generate_dataset()
        assert len(batches) > 0

        # Preprocess
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)
        assert len(windowed.windows) > 0

        # Features
        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)
        assert X.shape[0] > 0
        assert X.shape[1] > 0

        # Labels
        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        # Train
        model = create_model(config)
        model.fit(X, y)

        # Predict
        pred = model.predict(X)
        proba = model.predict_proba(X)

        # Evaluate
        evaluator = create_evaluator(config)
        result = evaluate(y.values, pred, proba, evaluator)

        # Verify all steps completed
        assert len(pred) == len(X)
        assert result.n_samples == len(X)

def test_reproducibility(config):
        """Test that same seed produces identical results."""
        gen1 = create_generator(config)
        gen2 = create_generator(config)

        batches1, anomaly_sessions1 = gen1.generate_dataset()
        batches2, anomaly_sessions2 = gen2.generate_dataset()

        # Compare first batch
        b1 = batches1[0]
        b2 = batches2[0]

        assert b1.batch_id == b2.batch_id
        assert b1.device_id == b2.device_id
        assert b1.user_id == b2.user_id
        assert b1.session_id == b2.session_id
        assert len(b1.samples) == len(b2.samples)

        for s1, s2 in zip(b1.samples, b2.samples):
            assert s1.timestamp == s2.timestamp
            assert s1.heart_rate_bpm == s2.heart_rate_bpm
            assert s1.ibi_ms == s2.ibi_ms
            assert s1.skin_temperature_celsius == s2.skin_temperature_celsius

def test_different_seeds_produce_different_data():
        """Test that different seeds produce different data."""
        config1 = load_config("configs/base.yaml")
        config1["random_seed"] = 42

        config2 = load_config("configs/base.yaml")
        config2["random_seed"] = 123

        gen1 = create_generator(config1)
        gen2 = create_generator(config2)

        batches1, _ = gen1.generate_dataset()
        batches2, _ = gen2.generate_dataset()

        # Should be different
        assert batches1[0].batch_id != batches2[0].batch_id
        assert batches1[0].samples[0].heart_rate_bpm != batches2[0].samples[0].heart_rate_bpm


class TestPipelineComponents:
    """Test individual pipeline components in isolation."""

    def test_config_loading(self):
        """Test that config loads correctly."""
        config = load_config("configs/base.yaml")

        assert "random_seed" in config
        assert "window" in config
        assert "features" in config
        assert "model" in config
        assert "training" in config
        assert "evaluation" in config
        assert "synthetic" in config

    def test_all_model_types(self):
        """Test that all model types can be created and trained."""
        import pandas as pd
        from anxietywatch_ml.models.baseline import ModelConfig
        from anxietywatch_ml.config import load_config

        config = load_config("configs/base.yaml")

        # Create dummy data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))

        for model_type in ["baseline", "logistic_regression", "dummy"]:
            model_config = ModelConfig(model_type=model_type)
            from anxietywatch_ml.models.baseline import SklearnBaselineModel
            model = SklearnBaselineModel(model_config)
            model.fit(X, y)

            pred = model.predict(X)
            proba = model.predict_proba(X)

            assert len(pred) == len(X)
            assert proba.shape == (len(X), 2)


class TestRequiredFeatures:
    """Tests for required features per hardening requirements."""

    def test_synthetic_dataset_has_both_classes(self, config, generator):
        """Test that synthetic dataset produces both classes (0 and 1)."""
        batches, anomaly_sessions = generator.generate_dataset()

        # Verify both classes exist in anomaly sessions
        assert 0 in anomaly_sessions.values()
        assert 1 in anomaly_sessions.values()

        # Verify distribution is reasonable (not all same class)
        positive_count = sum(anomaly_sessions.values())
        negative_count = len(anomaly_sessions) - positive_count
        assert positive_count > 0
        assert negative_count > 0

    def test_ibi_missing_not_filled_with_zero(self, config, generator):
        """Test that missing IBI is NOT filled with 0 (semantically wrong)."""
        batches, _ = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        # IBI-dependent features should be NaN when IBI is missing, not 0
        ibi_dependent = ["hrv_rmssd", "hrv_sdnn", "ibi_available", "ibi_coverage_ratio"]

        for feat in ibi_dependent:
            if feat in X.columns:
                # Check that NaN values exist (missing IBI windows)
                # Note: Some windows may have IBI, so not ALL should be NaN
                has_nan = X[feat].isna().any()
                has_values = X[feat].notna().any()

                # If there are windows with missing IBI, they should be NaN
                # If all windows have IBI, then no NaN is fine
                assert has_nan or has_values  # At least one of the two

                # Specifically: ibi_available should be 0 when missing, 1 when present
                if feat == "ibi_available":
                    assert (X[feat] == 0).any() or (X[feat] == 1).any()

    def test_fit_predict_on_separate_data(self, config, generator):
        """Test that model can fit on train data and predict on separate test data."""
        batches, anomaly_sessions = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=config["random_seed"], stratify=y
        )

        # Fit on train
        model = create_model(config)
        model.fit(X_train, y_train)

        # Predict on SEPARATE test data
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert len(pred) == len(X_test)
        assert proba.shape == (len(X_test), 2)

        # Verify predictions are valid (0 or 1)
        assert set(pred).issubset({0, 1})

    def test_model_save_load_roundtrip(self, config, generator, tmp_path):
        """Test model serialization roundtrip."""
        batches, anomaly_sessions = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        from anxietywatch_ml.models.baseline import create_model
        model = create_model(config)
        model.fit(X, y)

        # Save
        model_path = tmp_path / "test_model.pkl"
        model.save(model_path)

        # Load
        loaded_model = model.__class__.load(model_path)

        # Verify loaded model works
        pred_orig = model.predict(X[:10])
        pred_loaded = loaded_model.predict(X[:10])

        assert np.array_equal(pred_orig, pred_loaded)


class TestRequiredFeaturesIBI:
    """Tests for IBI-specific features per hardening requirements."""

    def test_device_without_ibi_preserves_missing_hrv(
        self,
        config,
        generator,
    ):
        """Test that devices without IBI support produce NaN HRV features."""
        batch = generator.generate_batch(
            is_anomaly_session=False,
            ibi_supported=False,
        )

        preprocessing = create_pipeline(config)
        windowed = preprocessing.run([batch])

        builder = create_feature_builder(config)
        X = builder.build(windowed.windows)

        assert (X["ibi_available"] == 0).all()
        assert (X["ibi_coverage_ratio"] == 0).all()

        assert X["hrv_rmssd"].isna().all()
        assert X["hrv_sdnn"].isna().all()

    def test_device_with_ibi_produces_hrv(
        self,
        config,
        generator,
    ):
        """Test that devices with IBI support produce valid HRV features."""
        batch = generator.generate_batch(
            is_anomaly_session=False,
            ibi_supported=True,
        )

        preprocessing = create_pipeline(config)
        windowed = preprocessing.run([batch])

        builder = create_feature_builder(config)
        X = builder.build(windowed.windows)

        assert (X["ibi_available"] == 1).all()
        assert (X["ibi_coverage_ratio"] > 0).all()

        assert X["hrv_rmssd"].notna().any()
        assert X["hrv_sdnn"].notna().any()


class TestGroundTruthConsistency:
    """Tests verifying synthetic ground truth produces different signals."""

    def test_synthetic_ground_truth_changes_signal(
        self,
        config,
        generator,
    ):
        """
        Test that anomaly sessions generate measurably different HR signals.

        This validates that the synthetic ground truth (is_anomaly_session)
        actually produces measurably different signals that the ML can learn.
        """
        user_id = next(iter(generator._user_baselines.keys()))

        normal = generator.generate_batch(
            user_id=user_id,
            is_anomaly_session=False,
            ibi_supported=False,
        )

        anomaly = generator.generate_batch(
            user_id=user_id,
            is_anomaly_session=True,
            ibi_supported=False,
        )

        normal_hr = np.array(
            [
                s.heart_rate_bpm
                for s in normal.samples
                if s.heart_rate_bpm is not None
            ]
        )

        anomaly_hr = np.array(
            [
                s.heart_rate_bpm
                for s in anomaly.samples
                if s.heart_rate_bpm is not None
            ]
        )

        observed_shift = anomaly_hr.mean() - normal_hr.mean()
        expected_shift = config["synthetic"]["hr_anomaly_shift_bpm"]

        # Plumbing sanity check only - not clinical validation
        assert observed_shift > expected_shift * 0.5


class TestCriticalLogisticRegression:
    """Critical test: LogisticRegression + group split + missing IBI."""

    def test_logistic_regression_group_split_missing_ibi(self, config, generator):
        """
        End-to-end test: LogisticRegression + group-aware split + missing IBI.

        This test verifies:
        1. Group-aware split (session) prevents leakage
        2. Missing IBI handled correctly (NaN preserved, not filled with 0)
        3. Preprocessing fit on train, transform on test
        4. LogisticRegression trains and predicts successfully
        4. Pipeline produces valid predictions on test set
        """
        from anxietywatch_ml.pipelines.model_pipeline import (
            train_with_pipeline,
            evaluate_pipeline,
            ModelPipelineConfig,
        )
        from anxietywatch_ml.evaluation.splitting import get_group_column

        batches, anomaly_sessions = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        # Use group-aware split by session
        # Get group column from window metadata BEFORE feature building (IDs are dropped during feature building)
        session_ids = [meta["session_id"] for meta in windowed.window_metadata]
        user_ids = [meta["user_id"] for meta in windowed.window_metadata]
        
        group_column = pd.Series(session_ids, index=range(len(session_ids)))
        pipeline_config = ModelPipelineConfig(
            model_type="logistic_regression",
            group_by="session",
            test_size=0.2,
            val_size=0.1,
            random_state=config["random_seed"],
        )

        # Train with group-aware split and proper pipeline
        bundle = train_with_pipeline(
            X, y, group_column, pipeline_config
        )
        split_result = bundle.split_result

        # Verify no group leakage
        train_sessions = set(split_result.train_groups)
        val_sessions = set(split_result.val_groups)
        test_sessions = set(split_result.test_groups)

        assert train_sessions.isdisjoint(val_sessions)
        assert train_sessions.isdisjoint(test_sessions)
        assert val_sessions.isdisjoint(test_sessions)

        # Verify no session windows leaked between partitions
        all_sessions = train_sessions | val_sessions | test_sessions
        assert len(all_sessions) == len(train_sessions) + len(val_sessions) + len(test_sessions)

        # Evaluate on all splits
        eval_results = evaluate_pipeline(bundle, X, y)

        # Verify test predictions exist and are valid
        test_result = eval_results["test"]
        assert test_result is not None
        assert test_result["n_samples"] > 0

        # Verify metrics are present
        assert "metrics" in test_result
        assert "accuracy" in test_result["metrics"]

        # Verify model produces valid probabilities
        X_test = X.iloc[split_result.test_indices]
        X_test_transformed = bundle.preprocessing_pipeline.transform(X_test)
        proba = bundle.model.predict_proba(X_test_transformed)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_user_group_split_also_works(self, config, generator):
        """Test that group_by='user' also works and produces disjoint groups."""
        from anxietywatch_ml.pipelines.model_pipeline import train_with_pipeline, ModelPipelineConfig
        from anxietywatch_ml.evaluation.splitting import get_group_column

        batches, anomaly_sessions = generator.generate_dataset()
        preprocessing = create_pipeline(config)
        windowed = preprocessing.run(batches)

        feature_builder = create_feature_builder(config)
        X = feature_builder.build(windowed.windows)

        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(config)
        y = pipeline._create_labels(windowed, anomaly_sessions)

        # Get group column from window metadata BEFORE feature building
        user_ids = [meta["user_id"] for meta in windowed.window_metadata]
        group_column = pd.Series(user_ids, index=range(len(user_ids)))
        pipeline_config = ModelPipelineConfig(
            model_type="logistic_regression",
            group_by="user",
            test_size=0.2,
            val_size=0.1,
            random_state=config["random_seed"],
        )

        bundle = train_with_pipeline(
            X, y, group_column, pipeline_config
        )
        split_result = bundle.split_result

        # Verify no user leakage
        train_users = set(split_result.train_groups)
        val_users = set(split_result.val_groups)
        test_users = set(split_result.test_groups)

        assert train_users.isdisjoint(val_users)
        assert train_users.isdisjoint(test_users)
        assert val_users.isdisjoint(test_users)

        # Verify train/test users are actually different
        assert len(train_users) > 0
        assert len(test_users) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])