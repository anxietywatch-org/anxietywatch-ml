"""
Tests for the ground-truth training & evaluation protocol (Phase 4).

Covers the final 004-A2 invariant set:

- readiness errors (empty, misaligned, single class, too few users);
- three models on the SAME group-by-user split (Dummy, LR class_weight=None,
  LR class_weight="balanced");
- balanced_accuracy / specificity / false_positive_rate are computed;
- threshold selected on val (fallback train), NEVER on test;
- LR "winner" decided on validation only, never on test;
- imputer statistics learned from TRAIN only; perturbing val/test does not
  change them;
- serialized artifact carries NO user/session/device/event IDs and still
  serves inference after load;
- artifact roundtrip keeps no dataset rows/raw telemetry.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from anxietywatch_ml.config import load_config
from anxietywatch_ml.evaluation.metrics import (
    EvaluationConfig,
    create_evaluator,
    evaluate,
    evaluate_with_threshold,
)
from anxietywatch_ml.ground_truth.builder import GroundTruthDataset, create_ground_truth_builder
from anxietywatch_ml.ground_truth.synthetic import create_ground_truth_generator
from anxietywatch_ml.pipelines.model_pipeline import (
    ModelInputImputer,
    save_trained_bundle,
    transform_for_inference,
)
from anxietywatch_ml.training import (
    DatasetReadinessError,
    assert_dataset_ready,
    check_dataset_ready,
    load_ground_truth_bundle,
    train_ground_truth,
)


@pytest.fixture(scope="module")
def config():
    """Load test configuration."""
    return load_config("configs/base.yaml")


def _build_synthetic(config, n_events=40):
    docs = create_ground_truth_generator(config).generate_docs(n_events=n_events)
    return create_ground_truth_builder(config).build(
        docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
    )


@pytest.fixture(scope="module")
def ds_and_result(config):
    """One shared build + full protocol run for the whole module."""
    ds = _build_synthetic(config)
    result = train_ground_truth(ds, config)
    return ds, result


def _empty_dataset():
    return GroundTruthDataset(
        X=pd.DataFrame(),
        y=pd.Series(dtype=int, name="target_support_requested"),
        metadata=pd.DataFrame(),
        feature_names=[],
        excluded_metadata_columns=[],
        label_counts={},
        dropped_no_telemetry=0,
        dropped_insufficient_data=0,
        identity_mismatches=0,
        duplicate_conflicts=0,
        event_mismatches=0,
        exclusions=pd.DataFrame(columns=["doc_id", "kind", "reason"]),
    )


def _misaligned_dataset():
    return GroundTruthDataset(
        X=pd.DataFrame(np.zeros((3, 2)), columns=["a", "b"]),
        y=pd.Series([0, 1, 0, 1], name="target_support_requested"),
        metadata=pd.DataFrame({"user_id": ["u1", "u2", "u3"]}),
        feature_names=["a", "b"],
        excluded_metadata_columns=[],
        label_counts={},
        dropped_no_telemetry=0,
        dropped_insufficient_data=0,
        identity_mismatches=0,
        duplicate_conflicts=0,
        event_mismatches=0,
        exclusions=pd.DataFrame(columns=["doc_id", "kind", "reason"]),
    )


def _iter_strings(obj):
    """Yield every str reachable from a pickle-able object (for ID scanning)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _iter_strings(item)
    elif isinstance(obj, np.ndarray):
        for item in obj.flatten():
            yield from _iter_strings(item)
    elif hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            yield from _iter_strings(value)


def _imputer_input(bundle, ds, indices):
    """Replicate the pipeline prefix feeding the imputer for the given rows."""
    prefix = bundle.preprocessing_pipeline.named_steps
    return prefix["nan_indicator"].transform(
        prefix["feature_selector"].transform(ds.X.iloc[indices])
    )


def _f1_at_threshold(bundle, ds, config, indices, threshold):
    Xt = transform_for_inference(bundle, ds.X.iloc[indices])
    proba = bundle.model.predict_proba(Xt)
    return evaluate_with_threshold(
        ds.y.iloc[indices].values, proba, threshold, create_evaluator(config)
    ).metrics["f1"]


class TestDatasetReadiness:
    """Errors are raised loudly before anything is trained."""

    def test_empty_dataset_raises_clear_error(self):
        with pytest.raises(DatasetReadinessError) as exc:
            assert_dataset_ready(_empty_dataset())
        assert "empty" in str(exc.value)

    def test_misaligned_labels_features_metadata_raises(self):
        with pytest.raises(DatasetReadinessError) as exc:
            assert_dataset_ready(_misaligned_dataset())
        assert "misaligned" in str(exc.value)

    def test_single_class_raises(self, config):
        dataset = _build_synthetic(config)
        dataset.y = pd.Series(np.zeros(len(dataset.y), dtype=int))
        with pytest.raises(DatasetReadinessError) as exc:
            assert_dataset_ready(dataset)
        assert "single class" in str(exc.value)

    def test_too_few_users_raises(self, config):
        dataset = _build_synthetic(config)
        dataset.metadata = dataset.metadata.assign(user_id="only-one-user")
        with pytest.raises(DatasetReadinessError) as exc:
            assert_dataset_ready(dataset)
        assert "distinct users" in str(exc.value)

    def test_report_lists_every_failing_check(self):
        report = check_dataset_ready(_empty_dataset())
        assert not report.ready
        assert len(report.errors) >= 2  # empty + too small
        assert all(report.checks[k] is False for k in ("not_empty", "enough_rows"))


class TestConfusionDerivedMetrics:
    """balanced_accuracy / specificity / false_positive_rate."""

    def test_default_config_includes_derived_metrics(self):
        assert set(EvaluationConfig().metrics) >= {
            "balanced_accuracy",
            "specificity",
            "false_positive_rate",
        }

    def test_evaluate_computes_derived_metrics_from_confusion_matrix(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        result = evaluate(y_true, y_pred, config=EvaluationConfig())
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        assert (tn, fp, fn, tp) == (3, 1, 1, 3)
        assert result.metrics["specificity"] == 3 / 4
        assert result.metrics["false_positive_rate"] == 1 / 4
        tpr = 3 / 4
        tnr = 3 / 4
        assert abs(result.metrics["balanced_accuracy"] - (tpr + tnr) / 2) < 1e-9

    def test_evaluate_with_threshold_includes_derived_metrics(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
                            [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
        result = evaluate_with_threshold(y_true, y_proba, 0.5, EvaluationConfig())
        assert "balanced_accuracy" in result.metrics
        assert "specificity" in result.metrics
        assert "false_positive_rate" in result.metrics

    def test_positive_only_availability(self):
        y_true = np.ones(4, dtype=int)
        y_pred = np.array([1, 0, 1, 0])
        result = evaluate(y_true, y_pred, config=EvaluationConfig())
        # TN + FP == 0: negative-side metrics do not exist.
        assert result.metrics_available["recall"] is True
        assert result.metrics_available["specificity"] is False
        assert np.isnan(result.metrics["specificity"])
        assert result.metrics_available["false_positive_rate"] is False
        assert np.isnan(result.metrics["false_positive_rate"])
        assert result.metrics_available["balanced_accuracy"] is False
        assert np.isnan(result.metrics["balanced_accuracy"])

    def test_negative_only_availability(self):
        y_true = np.zeros(4, dtype=int)
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate(y_true, y_pred, config=EvaluationConfig())
        # TP + FN == 0: positive-side recall does not exist.
        assert result.metrics_available["recall"] is False
        assert np.isnan(result.metrics["recall"])
        assert result.metrics_available["specificity"] is True
        assert result.metrics["specificity"] == 0.5
        assert result.metrics_available["false_positive_rate"] is True
        assert result.metrics["false_positive_rate"] == 0.5
        assert result.metrics_available["balanced_accuracy"] is False
        assert np.isnan(result.metrics["balanced_accuracy"])

    def test_single_class_no_artificial_balanced_accuracy(self):
        # All-negative, all-predicted-negative: no 0.0 / 0.25 fabrications.
        y_true = np.zeros(6, dtype=int)
        y_pred = np.zeros(6, dtype=int)
        result = evaluate(y_true, y_pred, config=EvaluationConfig())
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics_available["specificity"] is True
        assert result.metrics["specificity"] == 1.0
        assert result.metrics_available["false_positive_rate"] is True
        assert result.metrics["false_positive_rate"] == 0.0
        for name in ("recall", "f1", "balanced_accuracy"):
            assert result.metrics_available[name] is False
            assert np.isnan(result.metrics[name])


class TestSameSplitAndWeights:
    """The three models share the identical split; weights are as configured."""

    def test_three_models_share_same_split(self, ds_and_result):
        _, result = ds_and_result
        splits = [
            result.dummy_bundle.split_result,
            result.unweighted_bundle.split_result,
            result.balanced_bundle.split_result,
        ]
        for name in ("train", "val", "test"):
            for split in splits[1:]:
                assert list(split.train_indices) == list(splits[0].train_indices)
                assert list(split.val_indices) == list(splits[0].val_indices)
                assert list(split.test_indices) == list(splits[0].test_indices)

    def test_unweighted_uses_no_class_weight(self, ds_and_result):
        _, result = ds_and_result
        assert result.unweighted_bundle.model.config.model_type == "logistic_regression"
        assert result.unweighted_bundle.model.config.class_weight is None

    def test_balanced_uses_class_weight_balanced(self, ds_and_result):
        _, result = ds_and_result
        assert result.balanced_bundle.model.config.model_type == "logistic_regression"
        assert result.balanced_bundle.model.config.class_weight == "balanced"

    def test_user_split_disjoint(self, ds_and_result):
        _, result = ds_and_result
        assert result.user_intersections == {
            "train_vs_val": 0,
            "train_vs_test": 0,
            "val_vs_test": 0,
        }
        for name in ("train", "val", "test"):
            assert result.split_counts[name].n_users > 0

    def test_diagnostic_result_keeps_user_ids_and_indices(self, ds_and_result):
        _, result = ds_and_result
        # IDs and indices belong to the training diagnostic, not the artifact.
        assert len(result.split.train_groups) > 0
        assert len(result.split.test_groups) > 0
        assert len(result.split.train_indices) == result.split_counts["train"].n_rows
        assert len(result.split.test_indices) == result.split_counts["test"].n_rows


class TestMetricsReporting:
    """Every model reports the full metric set including derived ones."""

    def test_all_models_report_derived_metrics(self, ds_and_result):
        _, result = ds_and_result
        required = {"balanced_accuracy", "specificity", "false_positive_rate"}
        for metrics in (
            result.dummy_metrics["test"],
            result.unweighted_metrics["test"],
            result.balanced_metrics["test"],
            result.selected_test_at_threshold,
        ):
            assert required.issubset(metrics.metrics)
            for name in required:
                assert metrics.metrics[name] is not None
                assert 0.0 <= metrics.metrics[name] <= 1.0


class TestThresholdAndWinnerPolicy:
    """Threshold and winner are decided on validation, never on test."""

    def test_threshold_selected_on_val_not_test(self, ds_and_result):
        _, result = ds_and_result
        assert result.unweighted_threshold_source in ("val", "train")
        assert result.balanced_threshold_source in ("val", "train")
        assert result.unweighted_threshold_source == result.balanced_threshold_source
        assert result.unweighted_threshold > 0.0
        assert result.balanced_threshold > 0.0

    def test_test_does_not_affect_threshold(self, ds_and_result):
        _, result = ds_and_result
        for variant in ("unweighted", "balanced"):
            threshold = getattr(result, f"{variant}_threshold")
            at_threshold = getattr(result, f"{variant}_test_at_threshold")
            assert at_threshold.threshold == threshold

    def test_selected_variant_decided_on_validation_only(self, ds_and_result, config):
        ds, result = ds_and_result
        assert result.selection_source in ("val", "train")
        assert result.selection_metric == "f1"
        indices = getattr(result.split, f"{result.selection_source}_indices")
        f1_unw = _f1_at_threshold(
            result.unweighted_bundle, ds, config, indices, result.unweighted_threshold
        )
        f1_bal = _f1_at_threshold(
            result.balanced_bundle, ds, config, indices, result.balanced_threshold
        )
        if result.selected_variant == "balanced":
            assert f1_bal > f1_unw
        else:
            assert f1_unw >= f1_bal

    def test_selected_test_evaluation_matches_selected_variant(self, ds_and_result):
        _, result = ds_and_result
        selected = getattr(result, f"{result.selected_variant}_test_at_threshold")
        assert result.selected_test_at_threshold.threshold == selected.threshold
        assert (
            result.selected_test_at_threshold.n_samples
            == result.split_counts["test"].n_rows
        )


class TestImputationInvariants:
    """Imputer statistics come from TRAIN rows only."""

    def test_imputer_fit_train_only(self, ds_and_result):
        ds, result = ds_and_result
        bundle = result.balanced_bundle
        imp = bundle.preprocessing_pipeline.named_steps["model_input_imputer"]
        assert hasattr(imp, "fill_values_")
        X_train = _imputer_input(bundle, ds, result.split.train_indices)
        fresh = ModelInputImputer().fit(X_train)
        assert fresh.fill_values_ == imp.fill_values_

    def test_modifying_val_does_not_change_statistics(self, ds_and_result):
        ds, result = ds_and_result
        bundle = result.balanced_bundle
        imp = bundle.preprocessing_pipeline.named_steps["model_input_imputer"]
        X_train = _imputer_input(bundle, ds, result.split.train_indices)
        X_val_pert = _imputer_input(bundle, ds, result.split.val_indices).apply(
            lambda c: c.fillna(0) + 1e6
        )
        refit_train = ModelInputImputer().fit(X_train)
        refit_with_val = ModelInputImputer().fit(pd.concat([X_train, X_val_pert]))
        # Sensitivity: folding val in WOULD move the learned statistics.
        assert refit_with_val.fill_values_ != refit_train.fill_values_
        # Invariant: the artifact statistics are the train-only ones.
        assert refit_train.fill_values_ == imp.fill_values_

    def test_modifying_test_does_not_change_statistics(self, ds_and_result):
        ds, result = ds_and_result
        bundle = result.balanced_bundle
        imp = bundle.preprocessing_pipeline.named_steps["model_input_imputer"]
        X_train = _imputer_input(bundle, ds, result.split.train_indices)
        X_test_pert = _imputer_input(bundle, ds, result.split.test_indices).apply(
            lambda c: c.fillna(0) + 1e6
        )
        refit_train = ModelInputImputer().fit(X_train)
        refit_with_test = ModelInputImputer().fit(pd.concat([X_train, X_test_pert]))
        assert refit_with_test.fill_values_ != refit_train.fill_values_
        assert refit_train.fill_values_ == imp.fill_values_


class TestArtifactContents:
    """The artifact is a model bundle: no IDs, no dataset, still inferable."""

    def test_bundle_does_not_contain_metadata_or_raw_telemetry(self, ds_and_result):
        _, result = ds_and_result
        bundle = result.balanced_bundle
        for forbidden in ("X", "y", "metadata", "telemetry", "dataframe", "raw_telemetry"):
            assert not hasattr(bundle, forbidden)
        runtime = bundle.runtime_config
        assert not isinstance(runtime.get("metadata"), pd.DataFrame)
        assert not isinstance(runtime.get("telemetry_batches"), list)

    def test_artifact_roundtrip_keeps_no_dataset(self, ds_and_result, tmp_path):
        ds, result = ds_and_result
        out = tmp_path / "bundle.pkl"
        bundle = (
            result.balanced_bundle
            if result.selected_variant == "balanced"
            else result.unweighted_bundle
        )
        save_trained_bundle(bundle, out)
        loaded = load_ground_truth_bundle(out)
        for forbidden in ("X", "y", "metadata", "telemetry", "raw_telemetry"):
            assert not hasattr(loaded, forbidden)
        assert loaded.config.group_by == "user"

    def test_artifact_has_no_user_ids_or_row_indices(self, ds_and_result, tmp_path):
        ds, result = ds_and_result
        out = tmp_path / "bundle.pkl"
        bundle = (
            result.balanced_bundle
            if result.selected_variant == "balanced"
            else result.unweighted_bundle
        )
        save_trained_bundle(bundle, out)
        loaded = load_ground_truth_bundle(out)

        # split_result groups AND row indices are stripped on serialization.
        for name in ("train", "val", "test"):
            assert len(getattr(loaded.split_result, f"{name}_groups")) == 0
            assert len(getattr(loaded.split_result, f"{name}_indices")) == 0

        # No user/session/device/event identifier string anywhere in the object.
        user_ids = {str(u) for u in ds.metadata["user_id"].unique()}
        found = set(_iter_strings(loaded))
        assert user_ids.isdisjoint(found)

    def test_artifact_supports_inference_after_load(self, ds_and_result, tmp_path):
        ds, result = ds_and_result
        out = tmp_path / "bundle.pkl"
        bundle = (
            result.balanced_bundle
            if result.selected_variant == "balanced"
            else result.unweighted_bundle
        )
        save_trained_bundle(bundle, out)
        loaded = load_ground_truth_bundle(out)

        X_new = ds.X.iloc[result.split.test_indices]
        X_transformed = transform_for_inference(loaded, X_new)
        predictions = loaded.model.predict(X_transformed)
        assert len(predictions) == len(X_new)
        assert set(predictions).issubset({0, 1})
        proba = loaded.model.predict_proba(X_transformed)
        assert proba.shape == (len(X_new), 2)


class TestSyntheticExperiment:
    """End-to-end smoke on synthetic data (plumbing only, not performance)."""

    def test_protocol_smoke(self, ds_and_result):
        ds, result = ds_and_result
        assert ds.label_counts.get(0, 0) > 0 and ds.label_counts.get(1, 0) > 0
        assert result.dataset_readiness.ready
        assert result.feature_names == list(ds.X.columns)
        assert result.selected_variant in ("unweighted", "balanced")

        for name in ("train", "val", "test"):
            assert result.dummy_metrics[name] is not None
            assert result.unweighted_metrics[name] is not None
            assert result.balanced_metrics[name] is not None

        # Split counts are internally consistent.
        for name in ("train", "val", "test"):
            counts = result.split_counts[name]
            n0 = counts.class_counts.get("0", 0)
            n1 = counts.class_counts.get("1", 0)
            assert counts.n_rows == n0 + n1