"""Integration tests for the reproducible AnxietyWatch ML training artifact."""

import pytest

from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.synthetic import create_generator
from anxietywatch_ml.pipelines.model_pipeline import load_trained_bundle
from anxietywatch_ml.pipelines.predict import PredictionPipeline
from anxietywatch_ml.pipelines.train import TrainingPipeline


@pytest.fixture
def config():
    return load_config("configs/base.yaml")


def test_training_pipeline_uses_group_aware_split(config):
    result = TrainingPipeline(config).run()
    split = result.bundle.split_result

    train_groups = set(split.train_groups)
    val_groups = set(split.val_groups)
    test_groups = set(split.test_groups)

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_trained_bundle_save_load_roundtrip(config, tmp_path):
    artifact_path = tmp_path / "bundle.pkl"
    result = TrainingPipeline(config).run(artifact_path)
    loaded = load_trained_bundle(artifact_path)

    assert loaded.config.group_by == result.bundle.config.group_by
    assert loaded.runtime_config["training"]["group_by"] == "session"
    assert loaded.preprocessing_pipeline is not None
    assert loaded.model is not None


def test_prediction_pipeline_reuses_saved_preprocessing(config, tmp_path):
    artifact_path = tmp_path / "bundle.pkl"
    TrainingPipeline(config).run(artifact_path)

    generator = create_generator(config)
    batches, _ = generator.generate_dataset()

    prediction_pipeline = PredictionPipeline(config, artifact_path)
    result = prediction_pipeline.run(batches)

    assert len(result.predictions) > 0
    assert set(result.predictions["prediction"].unique()).issubset({0, 1})
    assert prediction_pipeline.bundle.preprocessing_pipeline is not None
