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

- HTTP: `GET /health`, `POST /predict`, `POST /predict/window`.
- The model is loaded ONCE at application startup.
- Inference endpoints require an API key (`X-Api-Key`). `/health` is exempt.
- The ML service owns windowing (`/predict/window`); the backend never sends
  pre-computed windows or features.
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
  "threshold": 0.63,
  "model_version": "0.1.0",
  "target": "target_support_requested"
}
```

- `prediction`: `1` iff `support_probability >= threshold`.
- `support_probability`: positive-class probability from the trained model.
- `threshold`: the training metadata threshold. Never silently 0.5, never
  recomputed at inference time. (The `0.63` above is illustrative; the actual
  value comes from the deployed bundle's training metadata.)
- `model_version` / `target`: from training metadata.

## Authentication

Inference endpoints (`POST /predict`, `POST /predict/window`) require the
header `X-Api-Key` set to `ANXIETYWATCH_API_KEY`. Comparison is constant-time
(`secrets.compare_digest`). The key is never logged, echoed, or returned.

| situation                                  | status | detail                                  |
| ------------------------------------------ | ------ | --------------------------------------- |
| key not configured (`ANXIETYWATCH_API_KEY` unset) | 503 | `inference authentication is not configured` |
| missing `X-Api-Key` header                 | 401    | `missing API key`                        |
| wrong `X-Api-Key` value                    | 401    | `invalid API key`                        |

Secure default: if no key is configured the inference endpoints refuse
everything with 503 — they never become public. Configure the key as an Azure
Container Apps secret (`ANXIETYWATCH_API_KEY`) before enabling traffic.
`GET /health` stays unauthenticated so probes work.

Example:

```bash
curl -X POST http://localhost:8000/predict/window \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: $ANXIETYWATCH_API_KEY" \
  -d @window.json
```

## Request (`POST /predict/window`)

Event-anchored **raw telemetry** window. The ML service owns windowing:
samples are flattened, sorted by timestamp, trimmed to
`[detectedAt - 60s, detectedAt]` (inclusive), validated, cleaned with the
canonical training preprocessing and fed through the same `FeatureBuilder`, so
the 16-feature vector is identical to the offline ground-truth path.

```json
{
  "eventId": "0d2a0d72-b0ff-4a0b-ba4f-6a8f2a0d3c1e",
  "deviceId": "22222222-2222-4222-8222-222222222222",
  "sessionId": "44444444-4444-4444-8444-444444444444",
  "detectedAt": "2026-01-15T10:00:00+00:00",
  "samples": [
    {
      "timestamp": "2026-01-15T09:59:00+00:00",
      "heartRateBpm": 88.5,
      "ibiMs": [812.4, 803.1, 818.9],
      "skinTemperatureCelsius": 33.1,
      "quality": { "heartRate": "good", "ibi": "good", "wearingState": "onBody" }
    }
  ]
}
```

- `eventId` / `deviceId` / `sessionId` / `detectedAt`: the detector event that
  anchors the window. Identifiers are correlation/parity metadata only; they are
  never part of the feature matrix.
- `userId`: optional — not required for correct windowing; omit unless identity
  scoping is genuinely needed.
- `samples`: raw samples covering the event period. May span several backend
  batches — there is intentionally **no `batchId`** here (backend batches are
  arbitrary 1–600 sample chunks, not windows). Samples outside
  `[detectedAt - 60s, detectedAt]` are ignored. Unknown keys are rejected
  (`extra="forbid"`, 422).
- Transport is camelCase; snake_case is also accepted (canonical normalization).

Data-quality gates (identical to the ground-truth dataset builder):

| gate                    | value                                  |
| ----------------------- | -------------------------------------- |
| window                 | `[detectedAt - 60s, detectedAt]` (incl.) |
| min in-window samples  | `10`                                   |
| min HR presence ratio  | `0.30`                                 |
| cleaning               | canonical missing-value + HR-outlier   |

Violations return `400` (with a `PredictorError` message), never a fabricated
prediction.

## Health (`GET /health`)

```json
{ "status": "ok", "model_loaded": true, "model_version": "0.1.0" }
```

- Model loaded: `200 OK` with the body above.
- No valid artifact: `503` with `{ "status": "degraded", "model_loaded": false,
  "model_version": "unknown" }`. The body is stable and never leaks filesystem
  paths or internal exception details.
- In production (`ANXIETYWATCH_REQUIRE_MODEL=true`) startup fails fast instead
  of serving a degraded process, so the degraded branch is effectively dev-only.
- Suitable for an Azure Container Apps HTTP probe: non-2xx when the model is
  not available for inference.

## Errors

| case                              | status |
| --------------------------------- | ------ |
| missing required feature          | 400/422 |
| extra/prohibited (leakage) key    | 422    |
| non-numeric value                 | 422    |
| non-finite (`Infinity`) value     | 400    |
| window: no samples in `[T-60s, T]`| 400    |
| window: < 10 in-window samples   | 400    |
| window: HR presence ratio < 0.30  | 400    |
| missing `X-Api-Key`               | 401    |
| invalid `X-Api-Key`               | 401    |
| auth not configured (no key set)  | 503    |
| no model loaded                   | 503    |
| internal inference failure        | 500 (no stack trace) |

## Configuration / environment variables

| variable                     | default                              | purpose                                      |
| ---------------------------- | ------------------------------------ | -------------------------------------------- |
| `ANXIETYWATCH_MODEL_PATH`    | `models/prototype_v0.1.0.pkl`        | path to the trained bundle (`.pkl`)          |
| `ANXIETYWATCH_REQUIRE_MODEL` | unset → `false` (dev)                | `true` ⇒ startup fails if artifact missing   |
| `ANXIETYWATCH_API_KEY`       | unset                                | inference auth (`X-Api-Key`); unset ⇒ 503    |
| `PORT`                       | `8000`                               | ASGI bind port (Container Apps sets `PORT`)  |

`create_app(model_path=..., require_model=..., api_key=...)` overrides all
three in tests/code.

## Deployment readiness — Docker / Azure Container Apps

> Azure resources are NOT provisioned by this repository. This section only
> documents how to run the service as a container.

**Build**

```bash
docker build -t anxietywatch-ml-api:dev .
```

The trained bundle is a **runtime input**, never baked into the image: supply
it with a volume mount (or an Azure Container Apps mounted secret).

**Run locally**

```bash
docker run --rm -p 8000:8000 \
  -v "${PWD}/models:/app/models" \
  -e ANXIETYWATCH_MODEL_PATH=/app/models/prototype_v0.1.0.pkl \
  -e ANXIETYWATCH_API_KEY="<set-your-key>" \
  anxietywatch-ml-api:dev
```

**Behavior**

- Container runs as a non-root user on `python:3.12-slim`.
- Binds `0.0.0.0:${PORT:-8000}`; Container Apps maps public HTTPS ingress to
  this target port.
- ASGI command: `uvicorn anxietywatch_ml.serving.app:app --host 0.0.0.0
  --port ${PORT:-8000} --workers 1`.
- With `ANXIETYWATCH_REQUIRE_MODEL=true` (default in the image), the process
  **exits at startup** if the configured artifact cannot be loaded — no
  degraded serving, no untrained fallback.
- Health probe: `GET /health` (200 ready / 503 not ready).
- Prediction endpoints: `POST /predict` and `POST /predict/window` (contracts
  above), both behind `X-Api-Key` (`ANXIETYWATCH_API_KEY`).
- CPU inference only; min replicas may be `0` (container starts fast, no GPU,
  no external services required).
- No Azure SDK dependencies; the FastAPI app is cloud-provider-neutral.

## GitHub Container Registry (GHCR) publishing

The inference image is published **by GitHub Actions** (never from a developer
workstation) to:

```
ghcr.io/anxietywatch-org/anxietywatch-ml-api
```

**Workflow:** `.github/workflows/publish-container.yml`

**Triggers**

- **Manual** (`workflow_dispatch`): run from the Actions UI selecting the
  `develop` ref to publish an explicit Azure candidate.
- **Push to `develop`**: publishes the integrated image automatically.
- **Pull requests**: only build the image to validate it; they **never** push
  to GHCR.

**Tags**

Every publish carries immutable Git-SHA tags:

- `ghcr.io/anxietywatch-org/anxietywatch-ml-api:<full-sha>` (40 chars)
- `ghcr.io/anxietywatch-org/anxietywatch-ml-api:<short-sha>` (12 chars)

A push to `develop` additionally publishes the moving tag:

- `ghcr.io/anxietywatch-org/anxietywatch-ml-api:develop`

Deployment does not rely on `latest`; Azure Container Apps must be pinned to an
immutable SHA tag. OCI labels include `org.opencontainers.image.source`
(repository) and `org.opencontainers.image.revision` (Git commit SHA).

**Authentication:** the repo-scoped `GITHUB_TOKEN` is used with
`docker/login-action`; no personal access token is created or required.

**The model artifact is NOT inside the image.** The published container
contains only the inference application and its runtime dependencies. The
trained bundle (`models/*.pkl`) is excluded by `.dockerignore` (see above) and
must be mounted separately at runtime. No credentials and no `.env` files are
included.

**Pull for local verification**

```bash
docker pull ghcr.io/anxietywatch-org/anxietywatch-ml-api:develop
docker run --rm -p 8000:8000 \
  -v "${PWD}/models:/app/models" \
  -e ANXIETYWATCH_MODEL_PATH=/app/models/prototype_v0.1.0.pkl \
  ghcr.io/anxietywatch-org/anxietywatch-ml-api:develop
```

> Pulling anonymously requires the GHCR package to be public; otherwise
> `docker login ghcr.io` with a GitHub token is needed first.

## Local startup

```powershell
# 1. Train the demo artifact (synthetic data, prototype only)
python -c "from anxietywatch_ml.serving import train_demo_model; train_demo_model(output_path='models/prototype_v0.1.0.pkl')"

# 2. Run the service (artifacts are read once at startup)
python -m uvicorn anxietywatch_ml.serving.app:app --port 8000
```

Point a different artifact with `ANXIETYWATCH_MODEL_PATH` or
`create_app(model_path=...)`.