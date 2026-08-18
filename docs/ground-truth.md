# Ground-Truth Dataset — Semantics and Pipeline

This document defines what the ground-truth dataset actually measures, the
selection bias it inherits, the label policy, and the exact pipeline used to
build it. The builder lives in `src/anxietywatch_ml/ground_truth/` and is
**dataset-only**: it never trains a model.

> ⚠️ **The target is NOT "did the user have anxiety".** It is:
>
> **"Did the user request support after a heuristic detector event?"**
>
> The dataset learns `P(SUPPORT_REQUESTED | WATCH_DETECTOR_TRIGGERED)`,
> **not** `P(ANXIETY | ALL_TELEMETRY)`.

---

## 1. Why This Target

The only durable human signal available is the primary decision the user made
in response to a watch prompt (`event_decisions`): `ACTIVITY_CONFIRMED`,
`USER_OK`, or `SUPPORT_REQUESTED`. There is no clinical label. The binary
target used for training is derived from that response:

| Original response | target_support_requested | response_category |
|---|---|---|
| `SUPPORT_REQUESTED` | `1` | `SUPPORT_REQUESTED` |
| `ACTIVITY_CONFIRMED` | `0` | `PHYSICAL_ACTIVITY` |
| `USER_OK` | `0` | `SELF_REPORTED_OK` |

- The **original response is always preserved** in `metadata.response`.
- `target_support_requested` is a **derived 0/1 view for training only** and
  never replaces the value stored in MongoDB.
- `SUPPORT_REQUESTED` is deliberately chosen over `USER_OK`/`ACTIVITY_CONFIRMED`
  because it is the only response that expresses a need for active support, is
  least confounded by everyday physical activity, and maps to the product
  action (breathing intervention / escalation).

### Target semantics

- `target_support_requested` is a **derived 0/1 training view**; it never
  replaces the original `response` stored in MongoDB.
- The model estimates `P(SUPPORT_REQUESTED | WATCH_DETECTOR_TRIGGERED)`, not
  `P(ANXIETY | TELEMETRY)` and not `P(USER_OK)`.
- Class balance is dataset-dependent (driven by the detector and
  `ground_truth.label_noise`). The QA report surfaces the observed balance;
  training must not assume it is fixed.

## 2. Selection Bias (Explicit)

A decision only exists **for events that triggered the watch heuristic
detector** (state reached `USER_VALIDATION`). Therefore:

- The dataset contains **no negatives from arbitrary moments** — every row is
  conditioned on the detector having fired.
- The model learns to predict the user's **reaction given a trigger**, not to
  find triggers in arbitrary telemetry.
- Distribution shifts in the heuristic rules (`rules_version`) change which
  events enter the dataset. This is auditable because `rules_version` is kept
  as metadata (excluded from features).

A second, more subtle bias: the ML features are recomputed from raw telemetry,
so any signal the cloud does not transport is invisible to the model even if
the Watch computed it on-device (see Feature Parity). This is a deliberate
information boundary, not a leak, but it means the model can never recover
e.g. movement-derived features.

This bias is a **feature of the product** (the prompt is only shown after a
trigger) and must be documented in any downstream evaluation.

## 3. Durable Sources (normalized contracts)

| Collection | Document | Normalized schema |
|---|---|---|
| `telemetry_batches` | camelCase `TelemetryBatchRequest` | `TelemetryBatch` (existing contract) |
| `suspected_events` | camelCase `SuspectedEventRequest` + `_id`/`receivedAt` | `SuspectedEvent` (new) |
| `event_decisions` | camelCase `EventDecisionRequest` + `_id`/`receivedAt` | `EventDecision` (new) |

Adapters (`SuspectedEventAdapter`, `EventDecisionAdapter`, and the existing
`TelemetryBatchAdapter`) normalize the Mongo documents (camelCase, extra
`_id`/`receivedAt`/auth `userId`) into snake_case Pydantic models with
`extra="ignore"`.

## 4. ML Window

- `T = event.detected_at` (present in both the suspected event and the
  decision).
- Window: **`[T - 60s, T]`** (`window.size_seconds`, default 60).
- Restriction: same `user_id`, `device_id`, `session_id` as the decision.
- Windows are dropped if below `min_samples_per_window` (10) or below
  `min_hr_ratio` (0.3) HR availability.

## 5. Pipeline

```
telemetry_batches ──▶ flatten samples ──▶ select [T-60s, T] ──▶ TelemetryBatch contract
                                                                   │
                                                                   ▼
                                              PreprocessingPipeline (missing values, outliers)
                                                                   │
                                                                   ▼
                                                     FeatureBuilder (windowed features) ──▶ X
event_decisions ──▶ response ──▶ label policy ──▶ y (target_support_requested)
suspected_events ──▶ detector metadata (audit only, NOT in X) ──▶ metadata
```

1. `TelemetryBatchAdapter.from_backend_dict` → `TelemetryBatch` (Pydantic contract).
2. Flatten batches to a DataFrame (one row per sample).
3. For each `EventDecision`, select the `[T-60s, T]` window restricted to the
   same user/device/session.
4. Apply the shared preprocessing steps (missing-value handling, outlier
   detection) to the window only — this also prevents any leakage from samples
   after `T`.
5. `FeatureBuilder` computes the window features → `X`. Features are derived
   **only from raw telemetry**.
6. `apply_label_policy(response)` → `y` and `response_category`.
7. Detector metadata (from `suspected_events`) goes to `metadata`, never to `X`.

## 6. Feature Exclusions (exclude_from_X=true)

The following detector artifacts are **never** model features. They are kept
in `metadata` for audit and parity checks:

| metadata column | source field | why excluded |
|---|---|---|
| `detector_score` | `suspected.score` | output of the detector being learned; would be leakage |
| `detector_state` | `suspected.state` | same |
| `rules_version` | `suspected.rulesVersion` | dataset-construction metadata; audit |
| `watch_features_snapshot` | `suspected.features` | watch-computed features; parity only |
| `watch_baseline_snapshot` | `suspected.baseline` | watch-computed baseline; parity only |

The model must produce its own features from raw telemetry. The watch snapshot
is used only by `GroundTruthDatasetBuilder.parity_check()` and by the richer
audit in `src/anxietywatch_ml/qa/parity.py` to quantify how the ML-computed
features differ from the on-watch computation.

## 7. Feature Parity

Two independent computations run over the same physiological window
`[T - 60s, T]`:

| Watch (Kotlin `DerivedFeatures`) | ML (Python `FeatureBuilder`) |
|---|---|
| `heartRateMean` | `hr_mean` |
| `heartRateMax` | `hr_max` |
| `heartRateSlopeBpmPerMinute` | `hr_slope_bpm_per_min` |
| `rmssdMillis` | `hrv_rmssd` |
| `sdnnMillis` | `hrv_sdnn` |
| `validSampleRatio` | `valid_sample_ratio` |
| `sampleCount` | `sample_count` |

`compute_feature_parity` (`src/anxietywatch_ml/qa/parity.py`) recomputes the ML
features from raw telemetry and measures the per-event difference
(`watch_value - ml_value`) plus per-field match/divergence statistics. It does
**not** force the two computations to agree; it reports where they do and where
they do not.

### Why Watch and Python can differ

- **Different preprocessing**: the Watch computes features on-device over its
  own live buffer; ML runs the shared `PreprocessingPipeline` (missing-value
  handling, outlier detection) on the windowed telemetry.
- **Different missingness handling**: the Watch decides validity at sampling
  time; ML decides from what the cloud received (e.g. `valid_sample_ratio`
  diverges when samples lack HR or carry `poor` quality).
- **Irregular sampling**: timestamps are uneven; each side buckets samples
  slightly differently.
- **IBI filtering**: ML keeps IBIs in `[250, 2000]` ms and requires >= 3 IBIs
  before computing `hrv_rmssd`/`hrv_sdnn`; the Watch applies its own filters.
  The synthetic generator stores `rmssdMillis`/`sdnnMillis` as `None`, so
  synthetic parity shows those fields as one-side missing.
- **Sensor availability**: the cloud does not receive accelerometer data, so
  `movementMagnitudeMean`/`movementVariance` are **not comparable** (see
  below).
- **Mathematical definition**: Watch `sampleCount` counts HR-present samples
  while ML `sample_count` counts all window samples; ML slope uses linear
  regression over minutes since window start.
- **Window differences (if any)**: the Watch snapshot is computed at detection
  time on the on-device buffer; the ML window is `[T - 60s, T]` over ingested
  batches. Any discrepancy surfaces in the parity report.

### Derived check

`heartRateDeltaFromBaseline` (Watch) is compared with the ML recomputation
`hr_mean - baseline.mean_heart_rate` using `watch_baseline_snapshot`. The
baseline snapshot may not be exactly the baseline the Watch used for its own
delta; that difference is itself part of the parity measurement.

### Fields never compared

- `movementMagnitudeMean`, `movementVariance`: the cloud does not transport
  the accelerometer data needed to recompute them in Python. They are listed
  as `NOT_COMPARABLE` with the documented reason rather than being faked as
  comparable.
- `lastSampleAgeSeconds` (Watch-only) and the ML-only features (`hr_std`,
  `hr_min`, `ibi_available`, `ibi_coverage_ratio`, `skin_temp_mean`, the
  quality ratios, `window_duration_seconds`) have no counterpart on the other
  side and are not compared.

Divergences are surfaced, not hidden. They inform whether
`watch_features_snapshot` can ever become model input — which today it cannot,
because it is detector-triggered metadata.

## 8. Output Artifacts

- `X.csv` — feature matrix (one row per decision, one column per feature).
- `y.csv` — derived `target_support_requested`.
- `metadata.csv` — identity, timestamps, original `response`,
  `response_category`, `target_support_requested`, and the excluded
  detector columns.

## 9. Usage

```python
from anxietywatch_ml.config import load_config
from anxietywatch_ml.ground_truth.builder import create_ground_truth_builder
from anxietywatch_ml.ground_truth.synthetic import create_ground_truth_generator

config = load_config("configs/base.yaml")
docs = create_ground_truth_generator(config).generate_docs(n_events=20)
dataset = create_ground_truth_builder(config).build(
    docs["telemetry_batches"], docs["suspected_events"], docs["event_decisions"]
)
dataset.save("data/ground_truth")
```

```bash
anxietywatch-ml build-dataset --output data/ground_truth --events 20
```

## 10. Dataset QA

`compute_dataset_qa` (`src/anxietywatch_ml/qa/dataset_qa.py`) produces a
quality report over the built dataset **before any training**. It covers:

- class balance and number of classes
- unique users / sessions / devices
- original responses and response categories
- per-feature missingness (`missing_ratio`, flagged above a threshold)
- IBI coverage (`ibi_available`, `ibi_coverage_ratio`, rows with no IBI)
- samples per window (`sample_count` distribution)
- excluded events grouped by `reason` (from `dataset.exclusions`)
- feature distributions (`describe()`)
- temporal coverage of `detected_at`

Structural problems are surfaced as warnings instead of failing silently:
single class in `y`, IBI entirely missing (HRV features NaN), empty dataset,
high feature missingness, and small datasets. The report is order-independent:
shuffling the input rows does not change any result.

## 11. Explicit Limitations

- Synthetic docs validate plumbing only; they are not representative of real
  user behavior.
- The label is user self-report, not a clinical measurement.
- No model is trained by this component (003-A is dataset-only).