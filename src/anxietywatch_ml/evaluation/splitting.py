"""
Group-aware data splitting for AnxietyWatch ML.

Prevents data leakage by ensuring all windows from the same session/user
belong to the same partition (train/val/test).
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


GroupBy = Literal["session", "user"]


@dataclass(frozen=True)
class SplitResult:
    """Result of a group-aware split."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: np.ndarray
    val_groups: np.ndarray
    test_groups: np.ndarray
    group_by: GroupBy


def group_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    group_by: GroupBy = "session",
) -> SplitResult:
    """
    Split data ensuring all samples from the same group stay together.

    Args:
        X: Feature matrix
        y: Labels
        groups: Group identifiers (session_id or user_id)
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
        group_by: "session" or "user" - column name in X to use for grouping

    Returns:
        SplitResult with indices and group assignments for each partition
    """
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must have same length")

    n_samples = len(X)
    indices = np.arange(n_samples)

    # First split: train+val vs test
    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    try:
        train_val_idx, test_idx = next(test_splitter.split(X, y, groups=groups))
    except StopIteration:
        raise ValueError("Could not perform group-aware split. Check group distribution.")

    train_val_groups = groups.iloc[train_val_idx]
    train_val_y = y.iloc[train_val_idx]

    # Second split: train vs val (from train+val)
    val_relative_size = val_size / (1 - test_size) if test_size < 1.0 else 0

    if val_relative_size > 0:
        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_relative_size,
            random_state=random_state,
        )
        try:
            train_idx_rel, val_idx_rel = next(
                val_splitter.split(
                    X.iloc[train_val_idx],
                    train_val_y,
                    groups=train_val_groups,
                )
            )
            # Convert relative indices back to absolute
            train_idx = train_val_idx[train_idx_rel]
            val_idx = train_val_idx[val_idx_rel]
        except StopIteration:
            raise ValueError("Could not perform validation split. Check group distribution.")
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    # Verify disjoint groups
    train_groups = groups.iloc[train_idx].unique()
    val_groups = groups.iloc[val_idx].unique() if len(val_idx) > 0 else np.array([])
    test_groups = groups.iloc[test_idx].unique()

    if not set(train_groups).isdisjoint(test_groups):
        raise ValueError("Train and test groups overlap!")
    if len(val_groups) > 0 and not set(val_groups).isdisjoint(train_groups):
        raise ValueError("Val and train groups overlap!")
    if len(val_groups) > 0 and not set(val_groups).isdisjoint(test_groups):
        raise ValueError("Val and test groups overlap!")

    return SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        group_by=group_by,
    )


def get_group_column(X: pd.DataFrame, group_by: GroupBy) -> pd.Series:
    """Extract group column from feature matrix."""
    col_map = {
        "session": "session_id",
        "user": "user_id",
    }
    col = col_map[group_by]
    if col not in X.columns:
        raise ValueError(f"Column '{col}' not found in feature matrix for group_by='{group_by}'")
    return X[col]


def print_split_summary(result: SplitResult, X: pd.DataFrame, y: pd.Series) -> None:
    """Print human-readable summary of split."""
    train_groups = set(result.train_groups)
    val_groups = set(result.val_groups)
    test_groups = set(result.test_groups)

    print(f"\nSplit Summary (group_by={result.group_by}):")
    print(f"  Train: {len(result.train_indices)} windows, {len(train_groups)} groups")
    print(f"  Val:   {len(result.val_indices)} windows, {len(val_groups)} groups")
    print(f"  Test:  {len(result.test_indices)} windows, {len(test_groups)} groups")

    if len(result.train_indices) > 0:
        train_y = y.iloc[result.train_indices]
        print(f"  Train class dist: {train_y.value_counts().to_dict()}")
    if len(result.val_indices) > 0:
        val_y = y.iloc[result.val_indices]
        print(f"  Val class dist:   {val_y.value_counts().to_dict()}")
    if len(result.test_indices) > 0:
        test_y = y.iloc[result.test_indices]
        print(f"  Test class dist:  {test_y.value_counts().to_dict()}")