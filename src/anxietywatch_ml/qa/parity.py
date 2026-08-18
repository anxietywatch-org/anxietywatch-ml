"""Feature parity: Watch (Kotlin) vs ML (Python) feature computation.

Two independent computations run over the SAME physiological window
[T - window_size_seconds, T] (T = event.detected_at):

    Watch/Kotlin  --DerivedFeatures-->  watch_features_snapshot (metadata only)
    ML/Python     --FeatureBuilder--->  X (model feature matrix)

This module MEASURES where they agree and where they diverge. It does NOT
force them to match. The watch snapshot is excluded from X (exclude_from_X);
it lives in ``metadata`` only for this audit.

Comparisons are grouped into:

- :data:`DIRECTLY_COMPARABLE`: watch field vs ML feature with the same
  mathematical definition (within a documented tolerance).
- :data:`DERIVED_COMPARABLE`: watch field that must be recomputed on the ML
  side from other features (e.g. ``heart_rate_delta_from_baseline``).
- :data:`NOT_COMPARABLE`: watch fields the cloud does not transport enough
  information to recompute in Python (e.g. movement features).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from anxietywatch_ml.ground_truth.builder import GroundTruthDataset

# Directly comparable pairs: watch snapshot field -> ML feature column. Both
# computations use the same definition on the same window; residual differences
# come from preprocessing, filtering and sampling details.
DIRECTLY_COMPARABLE: dict[str, str] = {
    "heart_rate_mean": "hr_mean",
    "heart_rate_max": "hr_max",
    "heart_rate_slope_bpm_per_minute": "hr_slope_bpm_per_min",
    "rmssd_millis": "hrv_rmssd",
    "sdnn_millis": "hrv_sdnn",
    "valid_sample_ratio": "valid_sample_ratio",
    "sample_count": "sample_count",
}

# Derived comparisons: the watch exposes a field that the ML side must rebuild
# from other features. Value = how to describe the recomputation.
DERIVED_COMPARABLE: dict[str, str] = {
    "heart_rate_delta_from_baseline": "hr_mean - baseline.mean_heart_rate",
}

# Watch fields that CANNOT be compared with ML: the cloud does not transport
# the raw signal needed to recompute them in Python. Reason is documented,
# divergence is not hidden.
NOT_COMPARABLE: dict[str, str] = {
    "movement_magnitude_mean": (
        "cloud does not transport accelerometer data to recompute the "
        "movement magnitude in Python"
    ),
    "movement_variance": (
        "cloud does not transport accelerometer data to recompute the "
        "movement variance in Python"
    ),
}

# Watch snapshot fields with no ML equivalent (no comparison attempted).
WATCH_ONLY: tuple[str, ...] = ("last_sample_age_seconds",)

# ML features with no Watch equivalent (no comparison attempted).
ML_ONLY: tuple[str, ...] = (
    "hr_std",
    "hr_min",
    "ibi_available",
    "ibi_coverage_ratio",
    "skin_temp_mean",
    "quality_good_ratio",
    "quality_fair_ratio",
    "quality_poor_ratio",
    "window_duration_seconds",
)

# Default match tolerance per directly-comparable watch field: (atol, rtol).
# A pair is classified as a "match" when |watch - ml| <= atol + rtol * |ml|.
# Tolerances only label the pair; the raw diff is always reported.
PARITY_TOLERANCES: dict[str, tuple[float, float]] = {
    "heart_rate_mean": (0.5, 0.02),
    "heart_rate_max": (0.5, 0.02),
    "heart_rate_slope_bpm_per_minute": (0.05, 0.05),
    "rmssd_millis": (1.0, 0.05),
    "sdnn_millis": (1.0, 0.05),
    "valid_sample_ratio": (0.01, 0.0),
    "sample_count": (0.0, 0.0),
}


@dataclass
class FeatureParityReport:
    """Measured differences between Watch and ML feature computations.

    ``rows``: long-format, one row per (event, comparable field) pair.
    ``summary``: one row per comparable field with match/divergence stats.
    ``derived``: rows for the derived checks (baseline delta).
    ``not_comparable``: watch fields with no ML recomputation available.
    ``ml_only`` / ``watch_only``: fields that exist on exactly one side.
    ``warnings``: structural caveats (missing snapshots, empty input).
    """

    rows: pd.DataFrame
    summary: pd.DataFrame
    derived: pd.DataFrame
    not_comparable: dict[str, str] = field(default_factory=dict)
    ml_only: tuple[str, ...] = ()
    watch_only: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Compact JSON-friendly view for smoke output."""
        summary = {}
        if not self.summary.empty:
            for _, r in self.summary.iterrows():
                summary[r["watch_field"]] = {
                    "n_pairs": int(r["n_pairs"]),
                    "n_both_present": int(r["n_both_present"]),
                    "n_match": int(r["n_match"]),
                    "n_diverging": int(r["n_diverging"]),
                    "n_watch_missing": int(r["n_watch_missing"]),
                    "n_ml_missing": int(r["n_ml_missing"]),
                    "mean_abs_diff": (
                        float(r["mean_abs_diff"]) if pd.notna(r["mean_abs_diff"]) else None
                    ),
                }
        return {
            "n_events": (
                int(self.rows["event_id"].nunique()) if not self.rows.empty else 0
            ),
            "summary": summary,
            "derived": (
                self.derived.to_dict(orient="records") if not self.derived.empty else []
            ),
            "not_comparable": dict(self.not_comparable),
            "ml_only": list(self.ml_only),
            "watch_only": list(self.watch_only),
            "warnings": list(self.warnings),
        }


def _close(watch_value: float, ml_value: float, atol: float, rtol: float) -> bool:
    """Pair classification tolerance: |watch - ml| <= atol + rtol * |ml|."""
    return abs(watch_value - ml_value) <= atol + rtol * abs(ml_value)


def _relative_diff(watch_value: float, ml_value: float) -> float | None:
    """Relative difference |watch - ml| / |ml| (None when ml == 0)."""
    if ml_value == 0:
        return None
    return abs(watch_value - ml_value) / abs(ml_value)


def compute_feature_parity(
    dataset: GroundTruthDataset,
    tolerances: dict[str, tuple[float, float]] | None = None,
) -> FeatureParityReport:
    """Measure Watch vs ML feature differences over a built dataset.

    Compares each row of ``dataset.X`` with the ``watch_features_snapshot`` in
    the corresponding ``dataset.metadata`` row. The watch snapshot never
    enters X; this audit only reads it from metadata.
    """
    tolerances = tolerances or PARITY_TOLERANCES
    rows: list[dict] = []
    derived: list[dict] = []
    n_no_snapshot = 0

    for i, meta in dataset.metadata.iterrows():
        snapshot = meta.get("watch_features_snapshot")
        if not isinstance(snapshot, dict):
            n_no_snapshot += 1
            continue
        if i >= len(dataset.X):
            continue

        event_id = meta["event_id"]
        ml_row = dataset.X.iloc[i]

        rows.extend(
            _compare_direct(
                snapshot, ml_row, i, event_id, dataset.X.columns, tolerances
            )
        )
        derived.extend(
            _compare_derived(snapshot, meta, ml_row, i, event_id, dataset.X.columns)
        )

    rows_df = pd.DataFrame(rows)
    derived_df = pd.DataFrame(derived)

    warnings: list[str] = []
    if dataset.metadata.empty:
        warnings.append("empty dataset: no metadata rows to compare")
    if n_no_snapshot:
        warnings.append(
            f"{n_no_snapshot} rows have no watch_features_snapshot; not compared"
        )

    summary = _build_summary(rows_df)
    return FeatureParityReport(
        rows=rows_df,
        summary=summary,
        derived=derived_df,
        not_comparable=dict(NOT_COMPARABLE),
        ml_only=ML_ONLY,
        watch_only=WATCH_ONLY,
        warnings=warnings,
    )


def _compare_direct(
    snapshot: dict,
    ml_row: pd.Series,
    i: int,
    event_id,
    columns,
    tolerances: dict[str, tuple[float, float]],
) -> list[dict]:
    """Compare the directly-comparable watch fields against the ML row."""
    out: list[dict] = []
    for watch_field, ml_field in DIRECTLY_COMPARABLE.items():
        watch_value = snapshot.get(watch_field)
        ml_value = ml_row[ml_field] if ml_field in columns else None

        status, diff, rel_diff = _pair_status(watch_value, ml_value, tolerances, watch_field)
        out.append(
            {
                "row_index": i,
                "event_id": event_id,
                "watch_field": watch_field,
                "ml_field": ml_field,
                "watch_value": watch_value,
                "ml_value": ml_value,
                "diff": diff,
                "rel_diff": rel_diff,
                "status": status,
            }
        )
    return out


def _compare_derived(
    snapshot: dict,
    meta: pd.Series,
    ml_row: pd.Series,
    i: int,
    event_id,
    columns,
) -> list[dict]:
    """Compare the derived checks (watch field rebuilt from other features)."""
    out: list[dict] = []
    for watch_field, expression in DERIVED_COMPARABLE.items():
        watch_value = snapshot.get(watch_field)
        ml_delta = _baseline_delta(meta, ml_row, columns)
        status, diff, rel_diff = _pair_status(
            watch_value, ml_delta, PARITY_TOLERANCES, watch_field, derived=True
        )
        out.append(
            {
                "row_index": i,
                "event_id": event_id,
                "watch_field": watch_field,
                "expression": expression,
                "watch_value": watch_value,
                "ml_value": ml_delta,
                "diff": diff,
                "rel_diff": rel_diff,
                "status": status,
            }
        )
    return out


def _baseline_delta(meta: pd.Series, ml_row: pd.Series, columns) -> float | None:
    """ML recomputation of heart_rate_delta_from_baseline: hr_mean - baseline."""
    hr_mean_field = DIRECTLY_COMPARABLE["heart_rate_mean"]
    baseline = meta.get("watch_baseline_snapshot")
    if not isinstance(baseline, dict) or hr_mean_field not in columns:
        return None
    ml_value_hr = ml_row[hr_mean_field]
    baseline_mean = baseline.get("mean_heart_rate")
    if ml_value_hr is None or pd.isna(ml_value_hr) or baseline_mean is None:
        return None
    return float(ml_value_hr) - float(baseline_mean)


def _pair_status(
    watch_value,
    ml_value,
    tolerances: dict[str, tuple[float, float]],
    watch_field: str,
    derived: bool = False,
) -> tuple[str, float | None, float | None]:
    """Classify a single pair and compute its diff/rel_diff."""
    if watch_value is None and (ml_value is None or pd.isna(ml_value)):
        return "both_missing", None, None
    if watch_value is None or pd.isna(watch_value):
        return "watch_missing", None, None
    if ml_value is None or pd.isna(ml_value):
        return "ml_missing", None, None

    watch_value = float(watch_value)
    ml_value = float(ml_value)
    diff = watch_value - ml_value
    rel_diff = _relative_diff(watch_value, ml_value)
    if derived:
        atol, rtol = 1.0, 0.05
    else:
        atol, rtol = tolerances.get(watch_field, (0.0, 0.0))
    status = "match" if _close(watch_value, ml_value, atol, rtol) else "diverging"
    return status, diff, rel_diff


def _build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-field match/divergence statistics from the long-format rows."""
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "watch_field",
                "ml_field",
                "n_pairs",
                "n_both_present",
                "n_match",
                "n_diverging",
                "n_watch_missing",
                "n_ml_missing",
                "mean_abs_diff",
                "median_abs_diff",
                "max_abs_diff",
                "mean_rel_diff",
            ]
        )
    records = []
    for (watch_field, ml_field), group in rows.groupby(
        ["watch_field", "ml_field"], sort=False
    ):
        both = group["diff"].notna()
        diffs = group.loc[both, "diff"]
        rel_diffs = group.loc[both, "rel_diff"]
        records.append(
            {
                "watch_field": watch_field,
                "ml_field": ml_field,
                "n_pairs": int(len(group)),
                "n_both_present": int(both.sum()),
                "n_match": int((group["status"] == "match").sum()),
                "n_diverging": int((group["status"] == "diverging").sum()),
                "n_watch_missing": int((group["status"] == "watch_missing").sum()),
                "n_ml_missing": int((group["status"] == "ml_missing").sum()),
                "mean_abs_diff": float(diffs.abs().mean()) if len(diffs) else np.nan,
                "median_abs_diff": (
                    float(diffs.abs().median()) if len(diffs) else np.nan
                ),
                "max_abs_diff": float(diffs.abs().max()) if len(diffs) else np.nan,
                "mean_rel_diff": (
                    float(rel_diffs.mean()) if len(rel_diffs) else np.nan
                ),
            }
        )
    return pd.DataFrame(records)