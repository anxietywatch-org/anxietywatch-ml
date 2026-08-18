# Training & Evaluation Protocol (004-A)

This document defines how a built `GroundTruthDataset` enters supervised
training and evaluation without leakage and without fooling ourselves on small
datasets. The protocol lives in `src/anxietywatch_ml/training/`.

> ⚠️ Metrics computed on **synthetic/in-memory ground truth validate plumbing
> only**. They are NOT real performance numbers. Do not read a synthetic
> `accuracy = 0.9x` as product success.

## Target semantics

The supervised target is `target_support_requested`, derived from the user's
primary response (`event_decisions`):

```
P(SUPPORT_REQUESTED | WATCH_DETECTOR_TRIGGERED)
```

NOT `P(ANXIETY | ALL_TELEMETRY)`.

A label of `1` means: *given that the heuristic watch detector already fired a
validation prompt, the user asked for support*. A label of `0` means the user
confirmed the event or reported themselves OK. There is no clinical label and
no label for arbitrary non-triggered moments. See `docs/ground-truth.md` for
the full semantics.

## Group-by-user rationale

Train/val/test are split **by user** (`GroupShuffleSplit` on `user_id`), not
by row:

- A user's physiological profile, watch and response behavior are
  correlated across their sessions. Splitting by row would place rows of the
  same user in both train and test, letting the model "recognize the user"
  instead of the signal — leakage.
- The protocol verifies that the user sets of train/val/test are pairwise
  disjoint (`user_intersections` must be 0) and refuses to train if there are
  fewer than two distinct users.

## Class imbalance

`SUPPORT_REQUESTED` is expected to be the minority class. The protocol:

- reports per-split class counts explicitly (never hides the imbalance);
- trains **both** `class_weight=None` and `class_weight="balanced"`
  `LogisticRegression` variants on the same split, so the balancing decision
  is made on evidence (validation) and never assumed;
- does **not** rely on accuracy as the success metric (see below).

## Models compared (same split)

Three models are fitted on the **exact same** group-by-user split:

1. `DummyClassifier(strategy="prior")` — reference baseline. If real data
   shows the Logistic Regression does not beat the dummy on the relevant
   metrics, that is a useful scientific result, not an infrastructure failure;
2. `LogisticRegression(class_weight=None)` — unweighted;
3. `LogisticRegression(class_weight="balanced")` — balanced.

## Metrics

Default metrics (`evaluation.metrics`): `accuracy`, `precision`, `recall`,
`f1`, `roc_auc`, `average_precision`, plus the confusion-matrix derived
`balanced_accuracy`, `specificity` (TNR) and `false_positive_rate` (FPR).

- `accuracy` alone is misleading under imbalance (a majority-class dummy can
  score high accuracy with zero positive recall).
- `roc_auc` / `average_precision` measure ranking and are meaningful even when
  the positive class is small.
- `balanced_accuracy` averages TPR and TNR; `specificity`/`FPR` expose the
  false-alarm tradeoff explicitly.
- `precision`/`recall`/`f1`/`balanced_accuracy`/`specificity`/`FPR` are
  computed at a fixed decision threshold.

Availability: when a split lacks a class, the affected metric is undefined.
`metrics_available[name]` is the source of truth; the numeric value is `NaN`
(never a fabricated `0.0`). Positive-only split: `recall` available,
`specificity`/`FPR`/`balanced_accuracy` unavailable. Negative-only split:
`recall` unavailable, `specificity`/`FPR` available, `balanced_accuracy`
unavailable (it needs both TPR and TNR).

## Threshold and winner policy

- Each LR variant selects its decision threshold by maximizing F1 on the
  **validation** split (fallback: train if val has no usable classes).
- The selected threshold is applied to the **test** split and reported per
  variant as `*_test_at_threshold`.
- The LR "winner" (`selected_variant`) is chosen on **validation metrics
  only** (`selection_source`, `selection_metric="f1"`). **Test metrics never
  select a winner.** The selected variant's test evaluation is the final
  estimate.

> The test split must remain untouched during model and threshold selection.

The threshold and the winner are model artifact decisions, not dataset
properties; changing them after looking at test metrics would invalidate the
estimate.

## Selection bias

Same as the dataset (`docs/ground-truth.md`): every row is conditioned on the
watch heuristic detector having fired; the target is the user's reaction to a
trigger. Any downstream evaluation must respect that the model cannot detect
triggers in arbitrary telemetry, and that `rules_version` distribution shifts
change the population.

## Leakage guards

Enforced by `train_ground_truth` and checked by tests:

1. readiness checks refuse empty / misaligned / single-class / too-few-user
   datasets;
2. group-by-user split with verified disjoint user sets;
3. all learned preprocessing (`FeatureSelector`, `NaNIndicator`,
   `ModelInputImputer`, `StandardScaler`) is fitted on TRAIN only and applied
   to val/test through the serialized bundle; perturbing val/test rows does
   not change the learned statistics;
4. the dummy and both logistic regressions share the identical split;
5. threshold and winner selection never read the test split;
6. the serialized artifact strips `split_result` group identifiers AND row
   indices (no user/session/device/event IDs, no train/val/test indices in
   the `.pkl`); the full diagnostic split lives in
   `GroundTruthTrainingResult`.

## Artifact

`train_ground_truth(dataset, config, output_path=...)` persists the selected
bundle via `save_trained_bundle`, which strips group identifiers before
serialization. The artifact is a `TrainedModelBundle` containing:

- the fitted `preprocessing_pipeline` (train-fitted fill values, scaler);
- the fitted estimator (`LogisticRegression` of the selected variant);
- the `split_result` with `train_groups`/`val_groups`/`test_groups` AND
  `train_indices`/`val_indices`/`test_indices` emptied, `group_by="user"`;
- the `ModelPipelineConfig` and the runtime `config` dict.

The artifact contains **no user/session/device/event IDs, no train/val/test
row indices, no dataset rows, no raw telemetry and no metadata table** — it is
a model bundle, not a data export. User identities and the complete split
diagnostic may live in the training diagnostic result
(`GroundTruthTrainingResult`), never in the inference bundle. Load it with
`load_ground_truth_bundle`.

## Limitations

- Synthetic ground truth validates plumbing only.
- Single train/val/test split on small user counts is noisy; no cross
  validation is performed in 004-A.
- The target is self-reported behavior after a trigger, not a clinical
  measurement.
- Azure deployment is out of scope for 004-A.