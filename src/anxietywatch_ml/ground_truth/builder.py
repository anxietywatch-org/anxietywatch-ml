"""Ground-truth dataset builder for AnxietyWatch ML.

Builds a supervised dataset (X, y, metadata) from durable backend documents:

- ``telemetry_batches``  raw telemetry (source of model features)
- ``suspected_events``   heuristic detections (detector metadata, audit only)
- ``event_decisions``    primary user responses (source of the label)

Target: "whether the user requested support after a heuristic detector event".
See docs/ground-truth.md for the exact semantics and the selection bias.

This module builds the DATASET ONLY. It does not train any model.

ML window: [T - window_size_seconds, T] where T = event.detected_at, restricted
to the same user_id/device_id/session_id as the decision.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from anxietywatch_ml.contracts.normalize import IdentityMismatchError
from anxietywatch_ml.contracts.telemetry import TelemetryBatch, TelemetryBatchAdapter
from anxietywatch_ml.features.builder import FeatureBuilder, create_feature_builder
from anxietywatch_ml.preprocessing.pipeline import PreprocessingPipeline

from .contracts import EventDecision, EventDecisionAdapter, SuspectedEvent, SuspectedEventAdapter
from .label_policy import LabelPolicyResult, apply_label_policy

logger = logging.getLogger(__name__)

# Detector metadata columns that are ALWAYS excluded from the model feature
# matrix (exclude_from_X=true). They are kept in ``metadata`` for audit and
# for parity checks against the Watch-computed features.
EXCLUDED_METADATA_COLUMNS: tuple[str, ...] = (
    "detector_score",
    "detector_state",
    "rules_version",
    "watch_features_snapshot",
    "watch_baseline_snapshot",
)

# Mapping from Watch snapshot fields to the corresponding ML-computed feature
# names used for the parity check.
PARITY_FIELD_MAP: dict[str, str] = {
    "heart_rate_mean": "hr_mean",
    "heart_rate_max": "hr_max",
    "heart_rate_slope_bpm_per_minute": "hr_slope_bpm_per_min",
    "rmssd_millis": "hrv_rmssd",
    "sdnn_millis": "hrv_sdnn",
    "valid_sample_ratio": "valid_sample_ratio",
    "sample_count": "sample_count",
}

# All exclusion reasons emitted by the builder. Every excluded item is
# recorded in ``GroundTruthDataset.exclusions`` with an explicit ``reason``.
EXCLUSION_REASONS: tuple[str, ...] = (
    "identity_mismatch",
    "duplicate_suspected_conflict",
    "duplicate_decision_conflict",
    "event_mismatch",
    "missing_decision",
    "unsupported_response",
    "missing_telemetry",
    "insufficient_samples",
    "invalid_document",
)


@dataclass
class GroundTruthBuilderConfig:
    """Configuration for the ground-truth dataset builder."""

    window_size_seconds: float = 60.0
    min_samples_per_window: int = 10
    min_hr_ratio: float = 0.3


@dataclass
class GroundTruthDataset:
    """Built dataset: feature matrix, derived labels and metadata.

    ``X`` contains ONLY features recomputed from raw telemetry. Detector
    metadata (score, state, rules_version, watch snapshots) lives in
    ``metadata`` and is excluded from ``X`` (exclude_from_X=true).
    """

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    feature_names: list[str]
    excluded_metadata_columns: list[str]
    label_counts: dict
    dropped_no_telemetry: int
    dropped_insufficient_data: int
    identity_mismatches: int
    duplicate_conflicts: int
    event_mismatches: int
    exclusions: pd.DataFrame

    def summary(self) -> dict:
        """Human-readable summary of the dataset."""
        return {
            "n_rows": len(self.X),
            "n_features": len(self.feature_names),
            "label_counts": self.label_counts,
            "dropped_no_telemetry": self.dropped_no_telemetry,
            "dropped_insufficient_data": self.dropped_insufficient_data,
            "identity_mismatches": self.identity_mismatches,
            "duplicate_conflicts": self.duplicate_conflicts,
            "event_mismatches": self.event_mismatches,
            "n_exclusions": len(self.exclusions),
            "excluded_metadata_columns": list(self.excluded_metadata_columns),
        }

    def save(self, output_dir) -> None:
        """Persist the dataset as X.csv, y.csv and metadata.csv."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.X.to_csv(output_dir / "X.csv", index=False)
        self.y.to_csv(output_dir / "y.csv", index=True)
        self.metadata.to_csv(output_dir / "metadata.csv", index=False)


class GroundTruthDatasetBuilder:
    """Builds the ground-truth dataset from backend documents."""

    def __init__(
        self,
        config: Optional[GroundTruthBuilderConfig] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.config = config or GroundTruthBuilderConfig()
        self.feature_builder = feature_builder or FeatureBuilder()
        self._prep = PreprocessingPipeline()

    def build(
        self,
        telemetry_batches: list,
        suspected_events: list,
        event_decisions: list,
    ) -> GroundTruthDataset:
        """Build the dataset.

        Inputs may be already-normalized models or backend Mongo-style dicts
        (PascalCase, camelCase or snake_case). Normalization happens via the
        contract adapters, which also enforce the canonical auth ``userId``
        and raise :class:`IdentityMismatchError` on conflicts.
        """
        batches, telemetry_excluded = self._normalize_batches(telemetry_batches)
        suspected_list, suspected_excluded = self._normalize_suspected(suspected_events)
        decisions_list, decision_excluded = self._normalize_decisions(event_decisions)

        suspected, suspected_conflicts = self._dedup_events(suspected_list)
        decisions, decision_conflicts = self._dedup_events(decisions_list)

        exclusions = telemetry_excluded + suspected_excluded + decision_excluded
        for event_id in suspected_conflicts:
            exclusions.append(
                {
                    "doc_id": str(event_id),
                    "kind": "suspected",
                    "reason": "duplicate_suspected_conflict",
                }
            )
        for event_id in decision_conflicts:
            exclusions.append(
                {
                    "doc_id": str(event_id),
                    "kind": "decision",
                    "reason": "duplicate_decision_conflict",
                }
            )

        identity_mismatches = sum(
            1 for r in exclusions if r["reason"] == "identity_mismatch"
        )
        duplicate_conflicts = len(suspected_conflicts) + len(decision_conflicts)

        # Suspected events without a decision produce no label and are excluded.
        for event_id in suspected:
            if event_id not in decisions:
                exclusions.append(
                    {
                        "doc_id": str(event_id),
                        "kind": "suspected",
                        "reason": "missing_decision",
                    }
                )

        flat = self._prep._flatten_batches(batches)
        logger.info(
            "Ground truth: %d decisions, %d suspected events, %d batches (%d samples)",
            len(decisions),
            len(suspected),
            len(batches),
            len(flat),
        )

        windows = []
        meta_rows = []
        event_mismatches = 0
        dropped_no_telemetry = 0
        dropped_insufficient_data = 0

        for decision in decisions.values():
            suspected_event = suspected.get(decision.event_id)
            if suspected_event is not None and not self._suspected_matches(
                decision, suspected_event
            ):
                event_mismatches += 1
                exclusions.append(
                    {
                        "doc_id": str(decision.event_id),
                        "kind": "decision",
                        "reason": "event_mismatch",
                    }
                )
                continue

            window = self._select_window(flat, decision)
            if window is None or len(window) == 0:
                dropped_no_telemetry += 1
                exclusions.append(
                    {
                        "doc_id": str(decision.event_id),
                        "kind": "decision",
                        "reason": "missing_telemetry",
                    }
                )
                continue
            if not self._window_ok(window):
                dropped_insufficient_data += 1
                exclusions.append(
                    {
                        "doc_id": str(decision.event_id),
                        "kind": "decision",
                        "reason": "insufficient_samples",
                    }
                )
                continue

            try:
                label = apply_label_policy(decision.response)
            except ValueError:
                exclusions.append(
                    {
                        "doc_id": str(decision.event_id),
                        "kind": "decision",
                        "reason": "unsupported_response",
                    }
                )
                continue

            windows.append(self._clean_window(window))
            meta_rows.append(self._build_metadata(decision, label, suspected_event))

        X = self.feature_builder.build(windows)
        y = pd.Series(
            [m["target_support_requested"] for m in meta_rows],
            name="target_support_requested",
            dtype=int,
        )
        metadata = pd.DataFrame(meta_rows)
        exclusions_df = pd.DataFrame(exclusions, columns=["doc_id", "kind", "reason"])

        dataset = GroundTruthDataset(
            X=X,
            y=y,
            metadata=metadata,
            feature_names=list(X.columns),
            excluded_metadata_columns=list(EXCLUDED_METADATA_COLUMNS),
            label_counts=y.value_counts().sort_index().to_dict(),
            dropped_no_telemetry=dropped_no_telemetry,
            dropped_insufficient_data=dropped_insufficient_data,
            identity_mismatches=identity_mismatches,
            duplicate_conflicts=duplicate_conflicts,
            event_mismatches=event_mismatches,
            exclusions=exclusions_df,
        )
        logger.info(
            "Built ground-truth dataset: %d rows x %d features; labels=%s",
            len(X),
            X.shape[1],
            dataset.label_counts,
        )
        return dataset

    def parity_check(self, dataset: GroundTruthDataset) -> pd.DataFrame:
        """Compare ML-computed features (X) with the Watch feature snapshot.

        The Watch snapshot is metadata only; it never enters X. This check
        quantifies the difference between both feature computations.
        """
        rows = []
        for i, (_, meta) in enumerate(dataset.metadata.iterrows()):
            snapshot = meta.get("watch_features_snapshot")
            if not isinstance(snapshot, dict):
                continue
            row = {"row_index": i, "event_id": meta["event_id"]}
            for watch_field, ml_field in PARITY_FIELD_MAP.items():
                watch_value = snapshot.get(watch_field)
                if ml_field in dataset.X.columns:
                    ml_value = dataset.X.iloc[i][ml_field]
                    if watch_value is not None and pd.notna(ml_value):
                        row[f"diff_{watch_field}"] = float(watch_value) - float(ml_value)
                    else:
                        row[f"diff_{watch_field}"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    def _select_window(self, flat: pd.DataFrame, decision: EventDecision) -> Optional[pd.DataFrame]:
        """Select the [T-60s, T] window for the decision's telemetry."""
        t_end = decision.detected_at
        t_start = t_end - timedelta(seconds=self.config.window_size_seconds)
        session_id = str(decision.session_id)
        device_id = str(decision.device_id)
        user_id = str(decision.user_id) if decision.user_id else None

        mask = (
            (flat["session_id"] == session_id)
            & (flat["device_id"] == device_id)
            & (flat["timestamp"] >= t_start)
            & (flat["timestamp"] <= t_end)
        )
        if user_id is not None:
            mask &= flat["user_id"] == user_id
        if not mask.any():
            return None
        return flat[mask].copy()

    def _window_ok(self, window: pd.DataFrame) -> bool:
        if len(window) < self.config.min_samples_per_window:
            return False
        hr_ratio = window["heart_rate_bpm"].notna().mean()
        return hr_ratio >= self.config.min_hr_ratio

    def _clean_window(self, window: pd.DataFrame) -> pd.DataFrame:
        """Apply the shared preprocessing steps (missing values, outliers).

        Delegates to the public :meth:`PreprocessingPipeline.clean_window` so
        training and serving consume one canonical implementation.
        """
        return self._prep.clean_window(window)

    def _build_metadata(
        self,
        decision: EventDecision,
        label: LabelPolicyResult,
        suspected_event: Optional[SuspectedEvent],
    ) -> dict:
        row = {
            "event_id": str(decision.event_id),
            "user_id": str(decision.user_id) if decision.user_id else None,
            "device_id": str(decision.device_id),
            "session_id": str(decision.session_id),
            "detected_at": decision.detected_at.isoformat(),
            "responded_at": decision.responded_at.isoformat(),
            "response": decision.response,  # original response, always preserved
            "response_category": label.response_category,
            "target_support_requested": label.target_support_requested,
            "has_suspected_event": suspected_event is not None,
        }
        for col in EXCLUDED_METADATA_COLUMNS:
            row[col] = None
        if suspected_event is not None:
            row["detector_score"] = suspected_event.score
            row["detector_state"] = suspected_event.state
            row["rules_version"] = suspected_event.rules_version
            row["watch_features_snapshot"] = suspected_event.features.model_dump()
            row["watch_baseline_snapshot"] = suspected_event.baseline.model_dump()
        return row

    @staticmethod
    def _excluded_row(item, kind: str, reason: str) -> dict:
        """Exclusion row for a document rejected during normalization."""
        return {"doc_id": GroundTruthDatasetBuilder._doc_id(item), "kind": kind, "reason": reason}

    @staticmethod
    def _normalize_batches(items: list) -> tuple[list[TelemetryBatch], list[dict]]:
        """Normalize telemetry docs; identity/invalid docs are excluded with a reason."""
        batches, excluded = [], []
        for item in items:
            if isinstance(item, TelemetryBatch):
                batches.append(item)
                continue
            try:
                batches.append(TelemetryBatchAdapter.from_backend_dict(item))
            except IdentityMismatchError:
                excluded.append(
                    GroundTruthDatasetBuilder._excluded_row(item, "telemetry", "identity_mismatch")
                )
            except ValidationError:
                excluded.append(
                    GroundTruthDatasetBuilder._excluded_row(item, "telemetry", "invalid_document")
                )
        return batches, excluded

    @staticmethod
    def _normalize_suspected(items: list) -> tuple[list[SuspectedEvent], list[dict]]:
        """Normalize suspected docs; identity/invalid docs are excluded with a reason."""
        events, excluded = [], []
        for item in items:
            if isinstance(item, SuspectedEvent):
                events.append(item)
                continue
            try:
                events.append(SuspectedEventAdapter.from_backend_dict(item))
            except IdentityMismatchError:
                excluded.append(
                    GroundTruthDatasetBuilder._excluded_row(item, "suspected", "identity_mismatch")
                )
            except ValidationError:
                excluded.append(
                    GroundTruthDatasetBuilder._excluded_row(item, "suspected", "invalid_document")
                )
        return events, excluded

    @staticmethod
    def _normalize_decisions(items: list) -> tuple[list[EventDecision], list[dict]]:
        """Normalize decision docs; identity/invalid docs are excluded with a reason."""
        decisions, excluded = [], []
        for item in items:
            if isinstance(item, EventDecision):
                decisions.append(item)
                continue
            try:
                decisions.append(EventDecisionAdapter.from_backend_dict(item))
            except IdentityMismatchError:
                excluded.append(
                    GroundTruthDatasetBuilder._excluded_row(item, "decision", "identity_mismatch")
                )
            except ValidationError:
                excluded.append(
                    GroundTruthDatasetBuilder._excluded_row(item, "decision", "invalid_document")
                )
        return decisions, excluded

    @staticmethod
    def _doc_id(data: dict):
        """Best-effort document id (eventId/batchId) for exclusion rows."""
        for key, value in data.items():
            flat = key.lower().replace("_", "")
            if flat in ("eventid", "batchid"):
                return str(value)
        return None

    @staticmethod
    def _dedup_events(items: list) -> tuple[dict, list]:
        """Group by event_id; equivalent duplicates are deduped, conflicts excluded.

        Returns (event_id -> canonical item, list of conflicting event_ids).
        A conflicting event_id is excluded entirely (never "last wins").
        """
        groups = {}
        for item in items:
            groups.setdefault(item.event_id, []).append(item)

        result = {}
        conflicts = []
        for event_id, group in groups.items():
            if len(group) == 1:
                result[event_id] = group[0]
                continue
            reference = group[0].model_dump(mode="json")
            if all(item.model_dump(mode="json") == reference for item in group[1:]):
                result[event_id] = group[0]
            else:
                conflicts.append(event_id)
        return result, conflicts

    @staticmethod
    def _suspected_matches(decision: EventDecision, suspected_event: SuspectedEvent) -> bool:
        """The suspected event must reference the exact same event as the decision.

        eventId alone is not enough: user/device/session/detectedAt must also
        agree, otherwise a recycled/corrupted eventId could attach a user
        response to the wrong physiological window.
        """
        if decision.device_id != suspected_event.device_id:
            return False
        if decision.session_id != suspected_event.session_id:
            return False
        if decision.detected_at != suspected_event.detected_at:
            return False
        if (
            decision.user_id is not None
            and suspected_event.user_id is not None
            and decision.user_id != suspected_event.user_id
        ):
            return False
        return True


def create_ground_truth_builder(config: dict) -> GroundTruthDatasetBuilder:
    """Factory function to create a builder from config."""
    window_cfg = config.get("window", {})
    gt_cfg = config.get("ground_truth", {})
    builder_config = GroundTruthBuilderConfig(
        window_size_seconds=gt_cfg.get(
            "window_size_seconds", window_cfg.get("size_seconds", 60)
        ),
        min_samples_per_window=gt_cfg.get(
            "min_samples_per_window", window_cfg.get("min_samples_per_window", 10)
        ),
        min_hr_ratio=gt_cfg.get("min_hr_ratio", 0.3),
    )
    return GroundTruthDatasetBuilder(builder_config, create_feature_builder(config))