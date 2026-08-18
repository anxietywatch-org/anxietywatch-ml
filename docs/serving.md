# Prototype Inference Service v0.1 (005-A)

FastAPI HTTP service that serves one trained GroundTruth bundle.

> **This prototype model is not clinically validated.**
> It is trained on SYNTHETIC/DEMO training data to validate deployment and
> inference plumbing. Its metrics are NOT real performance.
>
> **The current model predicts support-request behavior conditional on a Watch
> detector event.** `prediction=1` means "the model predicts SUPPORT_REQUESTED
> for an event that already passed the Watch detector". It does NOT mean "the
> user has anxiety".

## Scope

- HTTP: `GET /health`, `POST /predict`.
- The model is loaded ONCE at application startup.
- No online auto-learning, no Azure, no backend/Watch/Fog integration.

## Artifact

`train_ground_truth(output_path=...)` persists the validation-selected variant
as a `TrainedModelBundle` whose `runtime_config.model` carries non-personal
inference metadata:

| key                 | value                                    |
| ------------------- | ---------------------------------------- |
| `model_version`     | `0.1.0`                                  |
| `target`            | `target_support_requested`               |
| `threshold`         | decision threshold from training         |
| `threshold_source`  | split the threshold was selected on      |
| `feature_names`     | the 16 structural features               |

No user/session/device/event identifiers and no row indices are persisted.

## Request (`POST /predict`)

JSON object with the 16 structural features. Values may be `null` where the
pipeline supports semantic NaN. Extra/detector/identity keys are rejected with
`422` (`extra="forbid"`).

| feature                | unit / type     |
| ---------------------- | --------------- |
| `hr_mean`              | bpm             |
| `hr_std`               | bpm             |
| `hr_min`               | bpm             |
| `hr_max`               | bpm             |
| `hr_slope_bpm_per_min` | bpm/min         |
| `hrv_rmssd`            | ms              |
| `hrv_sdnn`             | ms              |
| `ibi_available`        | flag            |
| `ibi_coverage_ratio`   | ratio           |
| `skin_temp_mean`       | °C              |
| `quality_good_ratio`   | ratio           |
| `quality_fair_ratio`   | ratio           |
| `quality_poor_ratio`   | ratio           |
| `valid_sample_ratio`   | ratio           |
| `window_duration_seconds` | seconds      |
| `sample_count`         | count           |

Never accepted: `detector_score`, `detector_state`, `rules_version`,
`response`, `user_id`, `session_id`, `device_id`, `event_id`.

### Example

```json
{
  "hr_mean": 72.0,
  "hr_std": 4.2,
  "hr_min": 60.0,
  "hr_max": 90.0,
  "hr_slope_bpm_per_min": 0.3,
  "hrv_rmssd": 38.0,
  "hrv_sdnn": 41.0,
  "ibi_available": 1.0,
  "ibi_coverage_ratio": 0.85,
  "skin_temp_mean": 33.2,
  "quality_good_ratio": 0.9,
  "quality_fair_ratio": 0.1,
  "quality_poor_ratio": 0.0,
  "valid_sample_ratio": 0.95,
  "window_duration_seconds": 60.0,
  "sample_count": 61
}
```

## Response

```json
{
  "prediction": 1,
  "support_probability": 0.73,
  "threshold": 0.50,
  "model_version": "0.1.0",
  "target": "target_support_requested"
}
```

- `prediction`: `1` iff `support_probability >= threshold`.
- `support_probability`: positive-class probability from the trained model.
- `threshold`: the training metadata threshold. Never silently 0.5, never
  recomputed at inference time.
- `model_version` / `target`: from training metadata.

## Health (`GET /health`)

```json
{ "status": "ok", "model_loaded": true, "model_version": "0.1.0" }
```

If no valid artifact is available: `status="degraded"`, `model_loaded=false`,
and `/predict` returns `503`.

## Errors

| case                              | status |
| --------------------------------- | ------ |
| missing required feature          | 400/422 |
| extra/prohibited (leakage) key    | 422    |
| non-numeric value                 | 422    |
| no model loaded                   | 503    |
| internal inference failure        | 500 (no stack trace) |

## Local startup

```powershell
# 1. Train the demo artifact (synthetic data, prototype only)
python -c "from anxietywatch_ml.serving import train_demo_model; train_demo_model(output_path='models/prototype_v0.1.0.pkl')"

# 2. Run the service (artifacts are read once at startup)
python -m uvicorn anxietywatch_ml.serving.app:app --port 8000
```

Point a different artifact with `ANXIETYWATCH_MODEL_PATH` or
`create_app(model_path=...)`.