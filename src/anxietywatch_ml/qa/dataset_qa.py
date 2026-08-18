"""Dataset QA: measurable quality report for the ground-truth dataset.

Runs AFTER the dataset is built (003-A) and BEFORE any training (003-C).
It answers "what is in this dataset" without training a model:

- class balance
- users / sessions / devices
- responses
- feature missingness
- IBI coverage
- samples per window
- excluded events (grouped by reason)
- feature distributions
- temporal coverage

Structural problems are surfaced as ``warnings`` (single class, IBI entirely
missing, empty dataset, high feature missingness) instead of silent failure.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from anxietywatch_ml.ground_truth.builder import GroundTruthDataset

# Default fraction of a feature's rows that may be missing before the QA report
# flags it as a warning.
DEFAULT_MISSINGNESS_THRESHOLD: float = 0.5
# Default minimum number of rows before the dataset is flagged as too small.
DEFAULT_MIN_ROWS: int = 10


@dataclass
class DatasetQAReport:
    """Quality report for a built ground-truth dataset.

    All sections are derived from the dataset itself. The report is
    order-independent: shuffling the rows does not change the results.
    """

    n_rows: int
    n_features: int
    class_balance: dict
    n_classes: int
    users: dict
    sessions: dict
    devices: dict
    responses: dict
    response_categories: dict
    feature_missingness: pd.DataFrame
    ibi_coverage: dict
    samples_per_window: dict
    exclusions_by_reason: pd.DataFrame
    feature_distributions: pd.DataFrame
    temporal_coverage: dict
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Compact JSON-friendly view for smoke output."""
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "class_balance": self.class_balance,
            "n_classes": self.n_classes,
            "n_users": len(self.users),
            "n_sessions": len(self.sessions),
            "n_devices": len(self.devices),
            "responses": self.responses,
            "feature_missingness": self.feature_missingness.to_dict(orient="records"),
            "ibi_coverage": self.ibi_coverage,
            "samples_per_window": self.samples_per_window,
            "exclusions_by_reason": (
                self.exclusions_by_reason.to_dict(orient="records")
                if not self.exclusions_by_reason.empty
                else []
            ),
            "temporal_coverage": self.temporal_coverage,
            "warnings": list(self.warnings),
        }


def compute_dataset_qa(
    dataset: GroundTruthDataset,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> DatasetQAReport:
    """Compute the quality report for a built ground-truth dataset.

    The report is derived only from ``dataset.X``, ``dataset.y``,
    ``dataset.metadata`` and ``dataset.exclusions``; it does not train or fit
    anything and never mutates the input.
    """
    warnings: list[str] = []

    n_rows = len(dataset.X)
    if n_rows == 0:
        warnings.append(
            "empty dataset: no rows in X (all events were excluded)"
        )

    class_balance = _value_counts_dict(dataset.y)
    n_classes = len(class_balance)
    if n_rows > 0 and n_classes <= 1:
        warnings.append(
            "single class in y; model would have no signal to learn from"
        )

    users = _column_counts(dataset.metadata, "user_id")
    sessions = _column_counts(dataset.metadata, "session_id")
    devices = _column_counts(dataset.metadata, "device_id")
    responses = _column_counts(dataset.metadata, "response")
    response_categories = _column_counts(
        dataset.metadata, "response_category"
    )

    feature_missingness = _feature_missingness(dataset.X, missingness_threshold)
    flagged = feature_missingness.loc[
        feature_missingness["flagged"], "feature"
    ].tolist()
    for feature in flagged:
        warnings.append(
            f"high missingness in feature '{feature}' "
            f"(missing ratio above {missingness_threshold:.0%})"
        )

    ibi_coverage = _ibi_coverage(dataset)
    if n_rows > 0 and ibi_coverage["n_no_ibi"] == n_rows:
        warnings.append(
            "IBI entirely missing across the dataset; HRV features are NaN"
        )

    samples_per_window = _describe_dict(dataset.X.get("sample_count"))
    temporal_coverage = _temporal_coverage(dataset.metadata)

    if n_rows < min_rows:
        warnings.append(f"small dataset: only {n_rows} rows (min {min_rows})")

    exclusions_by_reason = _exclusions_by_reason(dataset.exclusions)
    feature_distributions = (
        dataset.X.describe() if not dataset.X.empty else pd.DataFrame()
    )

    return DatasetQAReport(
        n_rows=n_rows,
        n_features=dataset.X.shape[1] if not dataset.X.empty else 0,
        class_balance=class_balance,
        n_classes=n_classes,
        users=users,
        sessions=sessions,
        devices=devices,
        responses=responses,
        response_categories=response_categories,
        feature_missingness=feature_missingness,
        ibi_coverage=ibi_coverage,
        samples_per_window=samples_per_window,
        exclusions_by_reason=exclusions_by_reason,
        feature_distributions=feature_distributions,
        temporal_coverage=temporal_coverage,
        warnings=warnings,
    )


def _column_counts(metadata: pd.DataFrame, column: str) -> dict:
    """value_counts of a metadata column, tolerating an empty DataFrame."""
    if metadata.empty or column not in metadata.columns:
        return {}
    return _value_counts_dict(metadata[column])


def _value_counts_dict(series) -> dict:
    """value_counts as plain dict, tolerating None/empty input."""
    if series is None or len(series) == 0:
        return {}
    counts = series.value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _feature_missingness(
    X: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Per-feature missing count and ratio, flagged above ``threshold``."""
    if X.empty:
        return pd.DataFrame(columns=["feature", "n_missing", "missing_ratio", "flagged"])
    missing = X.isna().sum()
    return pd.DataFrame(
        {
            "feature": missing.index,
            "n_missing": missing.values.astype(int),
            "missing_ratio": (missing / len(X)).values,
            "flagged": (missing / len(X) > threshold).values,
        }
    ).reset_index(drop=True)


def _ibi_coverage(dataset: GroundTruthDataset) -> dict:
    """IBI availability and coverage over the dataset."""
    if dataset.X.empty:
        return {
            "n_no_ibi": 0,
            "ibi_available_mean": np.nan,
            "ibi_coverage_ratio_mean": np.nan,
        }
    if "ibi_available" in dataset.X.columns:
        n_no_ibi = int((dataset.X["ibi_available"] == 0).sum())
        ibi_available_mean = float(dataset.X["ibi_available"].mean())
    else:
        n_no_ibi = int(len(dataset.X))
        ibi_available_mean = 0.0
    if "ibi_coverage_ratio" in dataset.X.columns:
        ibi_coverage_mean = float(dataset.X["ibi_coverage_ratio"].mean())
    else:
        ibi_coverage_mean = np.nan
    return {
        "n_no_ibi": n_no_ibi,
        "ibi_available_mean": ibi_available_mean,
        "ibi_coverage_ratio_mean": ibi_coverage_mean,
    }


def _describe_dict(series) -> dict:
    """describe() as a plain dict, tolerating None/empty input."""
    if series is None or len(series) == 0:
        return {}
    desc = series.describe()
    return {k: (float(v) if pd.notna(v) else None) for k, v in desc.items()}


def _exclusions_by_reason(exclusions: pd.DataFrame) -> pd.DataFrame:
    """Excluded events grouped by reason, sorted by count descending."""
    if exclusions.empty or "reason" not in exclusions.columns:
        return pd.DataFrame(columns=["reason", "count"])
    counts = exclusions["reason"].value_counts()
    return pd.DataFrame({"reason": counts.index, "count": counts.values}).reset_index(
        drop=True
    )


def _temporal_coverage(metadata: pd.DataFrame) -> dict:
    """Temporal span of detected_at over the dataset."""
    if metadata.empty or "detected_at" not in metadata.columns:
        return {}
    detected = pd.to_datetime(metadata["detected_at"], errors="coerce").dropna()
    if detected.empty:
        return {}
    span = detected.max() - detected.min()
    return {
        "min": detected.min().isoformat(),
        "max": detected.max().isoformat(),
        "span_seconds": float(span.total_seconds()),
        "span_days": float(span.total_seconds() / 86400.0),
        "n_distinct_days": int(detected.dt.date.nunique()),
    }