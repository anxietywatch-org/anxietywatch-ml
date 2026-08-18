"""Reproducible generation of the prototype v0.1.0 demo model.

PROTOTYPE MODEL
NOT CLINICALLY VALIDATED
SYNTHETIC/DEMO TRAINING DATA

The model is trained on the in-memory synthetic GroundTruthDataset to validate
deployment/inference plumbing. Its metrics are NOT real performance. The
selected variant follows the existing validation-only selection of
``train_ground_truth``. No user IDs and no raw telemetry are persisted; the
artifact carries non-personal metadata (model_version, target, threshold,
threshold_source, feature_names) in ``runtime_config``.
"""

from pathlib import Path

from anxietywatch_ml.config import load_config
from anxietywatch_ml.ground_truth.builder import create_ground_truth_builder
from anxietywatch_ml.ground_truth.synthetic import create_ground_truth_generator
from anxietywatch_ml.training import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET,
    GroundTruthTrainingResult,
    train_ground_truth,
)

DEMO_N_EVENTS = 40


def train_demo_model(
    config: dict | None = None,
    output_path: Path | str | None = None,
    n_events: int = DEMO_N_EVENTS,
    model_version: str = DEFAULT_MODEL_VERSION,
    target: str = DEFAULT_TARGET,
) -> GroundTruthTrainingResult:
    """Train the prototype demo model and optionally persist the artifact."""
    cfg = config or load_config("configs/base.yaml")
    docs = create_ground_truth_generator(cfg).generate_docs(n_events=n_events)
    dataset = create_ground_truth_builder(cfg).build(
        docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
    )
    return train_ground_truth(
        dataset,
        cfg,
        output_path=output_path,
        model_version=model_version,
        target=target,
    )