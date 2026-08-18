"""Training & evaluation protocol for GroundTruthDataset (Phase 4).

Turns a built ground-truth dataset into a reproducible, leakage-free
train/val/test experiment with a DummyClassifier baseline, LogisticRegression
and an explicit threshold policy. See docs/training.md.

Does NOT deploy to Azure; produces the protocol and the artifact only.
"""

from anxietywatch_ml.training.protocol import (
    DEFAULT_MIN_ROWS,
    DEFAULT_MIN_USERS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET,
    GROUP_COLUMN,
    DatasetReadinessError,
    DatasetReadinessReport,
    GroundTruthTrainingResult,
    SplitCounts,
    assert_dataset_ready,
    check_dataset_ready,
    load_ground_truth_bundle,
    save_bundle_with_metadata,
    train_ground_truth,
)

__all__ = [
    "DEFAULT_MIN_ROWS",
    "DEFAULT_MIN_USERS",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_TARGET",
    "GROUP_COLUMN",
    "DatasetReadinessError",
    "DatasetReadinessReport",
    "GroundTruthTrainingResult",
    "SplitCounts",
    "assert_dataset_ready",
    "check_dataset_ready",
    "load_ground_truth_bundle",
    "save_bundle_with_metadata",
    "train_ground_truth",
]