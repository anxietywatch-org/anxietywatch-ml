"""Ground-truth dataset builder for AnxietyWatch ML.

Builds a supervised dataset (X, y, metadata) from durable backend documents
(telemetry_batches, suspected_events, event_decisions) WITHOUT training any model.

See docs/ground-truth.md for the exact target semantics and selection bias.
"""

from anxietywatch_ml.ground_truth.builder import (
    EXCLUDED_METADATA_COLUMNS,
    GroundTruthBuilderConfig,
    GroundTruthDataset,
    GroundTruthDatasetBuilder,
    create_ground_truth_builder,
)
from anxietywatch_ml.ground_truth.contracts import (
    EventDecision,
    EventDecisionAdapter,
    SuspectedEvent,
    SuspectedEventAdapter,
    SuspectedEventBaseline,
    SuspectedEventFeatures,
)
from anxietywatch_ml.ground_truth.label_policy import (
    PHYSICAL_ACTIVITY,
    PRIMARY_RESPONSES,
    SELF_REPORTED_OK,
    SUPPORT_REQUESTED,
    LabelPolicyResult,
    apply_label_policy,
)
from anxietywatch_ml.ground_truth.synthetic import (
    SyntheticGroundTruthGenerator,
    create_ground_truth_generator,
)

__all__ = [
    "EventDecision",
    "EventDecisionAdapter",
    "SuspectedEvent",
    "SuspectedEventAdapter",
    "SuspectedEventBaseline",
    "SuspectedEventFeatures",
    "EXCLUDED_METADATA_COLUMNS",
    "GroundTruthBuilderConfig",
    "GroundTruthDataset",
    "GroundTruthDatasetBuilder",
    "create_ground_truth_builder",
    "PHYSICAL_ACTIVITY",
    "PRIMARY_RESPONSES",
    "SELF_REPORTED_OK",
    "SUPPORT_REQUESTED",
    "LabelPolicyResult",
    "apply_label_policy",
    "SyntheticGroundTruthGenerator",
    "create_ground_truth_generator",
]