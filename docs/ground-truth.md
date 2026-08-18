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
is used only by `GroundTruthDatasetBuilder.parity_check()` to quantify how the
ML-computed features differ from the on-watch computation.

## 7. Output Artifacts

- `X.csv` — feature matrix (one row per decision, one column per feature).
- `y.csv` — derived `target_support_requested`.
- `metadata.csv` — identity, timestamps, original `response`,
  `response_category`, `target_support_requested`, and the excluded
  detector columns.

## 8. Usage

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

## 9. Explicit Limitations

- Synthetic docs validate plumbing only; they are not representative of real
  user behavior.
- The label is user self-report, not a clinical measurement.
- No model is trained by this component (003-A is dataset-only).