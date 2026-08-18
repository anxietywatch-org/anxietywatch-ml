"""Dataset QA and Feature Parity for the AnxietyWatch ML dataset.

Two audit tools that run AFTER the ground-truth dataset is built (003-A) and
BEFORE any model training (003-C):

- :func:`compute_feature_parity` — measures where the Watch (Kotlin) feature
  computation and the ML (Python) FeatureBuilder agree and where they diverge.
- :func:`compute_dataset_qa` — quality report over the built dataset
  (balance, coverage, missingness, exclusions, warnings).

Neither trains a model.
"""

from anxietywatch_ml.qa.dataset_qa import (
    DEFAULT_MIN_ROWS,
    DEFAULT_MISSINGNESS_THRESHOLD,
    DatasetQAReport,
    compute_dataset_qa,
)
from anxietywatch_ml.qa.parity import (
    DERIVED_COMPARABLE,
    DIRECTLY_COMPARABLE,
    ML_ONLY,
    NOT_COMPARABLE,
    PARITY_TOLERANCES,
    WATCH_ONLY,
    FeatureParityReport,
    compute_feature_parity,
)

__all__ = [
    "DIRECTLY_COMPARABLE",
    "DERIVED_COMPARABLE",
    "ML_ONLY",
    "NOT_COMPARABLE",
    "PARITY_TOLERANCES",
    "WATCH_ONLY",
    "FeatureParityReport",
    "compute_feature_parity",
    "DEFAULT_MISSINGNESS_THRESHOLD",
    "DEFAULT_MIN_ROWS",
    "DatasetQAReport",
    "compute_dataset_qa",
]