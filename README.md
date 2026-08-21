# AnxietyWatch ML

**Machine Learning component for AnxietyWatch — Technical MVP Complete**

> **Current Status: Technical ML MVP — COMPLETE**
>
> This is **NOT** a clinical anxiety detector. The models in this repository are **infrastructure baselines** used solely to validate the ML plumbing (data contracts, preprocessing, feature engineering, training, prediction). They have no clinical validity.

## Overview

This repository contains the Machine Learning pipeline for AnxietyWatch. It serves canonical preprocessing/feature extraction and inference for event-anchored raw telemetry windows.

### Data Flow (IMPLEMENTED)

```
Galaxy Watch → Wear Data Layer → Mobile Fog Node → Backend API
                                                          ├──→ MongoDB
                                                          │      (telemetry persistence / query)
                                                          │
                                                          └──→ raw event window over HTTPS
                                                                     ↓
                                                               ML API
                                                                     ↓
                                                          canonical preprocessing
                                                                     ↓
                                                              features
                                                                     ↓
                                                               model
```

**ML NEVER reads Backend Mongo directly.** Backend owns persistence/query and sends the raw event window to ML over HTTPS.

### What This Pipeline DOES (IMPLEMENTED)

- ✅ Validates incoming telemetry against the internal ML contract
- ✅ Preprocesses and windows time-series data (event-anchored raw windows)
- ✅ Engineers features from available signals (HR, HRV, temperature, quality) — **ML-owned canonical preprocessing**
- ✅ Trains baseline models (DummyClassifier, LogisticRegression) on synthetic data
- ✅ Evaluates with standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Provides reproducible, configurable pipeline with CLI
- ✅ **Serves model v0.1.0 via FastAPI**
- ✅ **GET /health** — health check with `model_loaded=true`, `model_version=0.1.0`
- ✅ **POST /predict** — single-sample inference (authenticated)
- ✅ **POST /predict/window** — event-window inference (authenticated, X-Api-Key)
- ✅ **Canonical event-anchored raw telemetry serving** — ML owns preprocessing/features
- ✅ **Training-serving parity** — bundle config derives feature spec for both training and inference
- ✅ **API-key authentication** — X-Api-Key header, HTTPS-only
- ✅ **Docker** — multi-stage build, non-root user
- ✅ **GHCR** — published to `ghcr.io/anxietywatch-org/anxietywatch-ml-api`
- ✅ **Azure Container Apps** — production-ready deployment target
- ✅ **Azure Files artifact mount** — model bundle loaded from persistent storage
- ✅ **GitHub OIDC/CD** — secret-safe image-only deployments (no secret round-trip)
- ✅ **Backend multi-batch window retrieval** — `[detectedAt-60s, detectedAt]` across batches
- ✅ **Backend secure ML HTTP client** — typed `AddHttpClient`, timeout/retries/failure classification
- ✅ **Suspected-event inference orchestration** — B4 integration
- ✅ **EventInferences persistence** — `event_inferences` collection, `eventId` keyed
- ✅ **EventId idempotency** — duplicate suspected events never re-trigger ML
- ✅ **Decision linkage** — `/events/decision` with same `eventId` for supervised labels
- ✅ **Real Backend → Real Azure ML isolated E2E acceptance** — 2 telemetry batches → suspected event → Azure ML → EventInferenceResult → decision with same eventId

### What This Pipeline Does NOT Do (Yet)

- ❌ Detect anxiety clinically
- ❌ Use real patient data (only synthetic for training/validation)
- ❌ Monitor drift or retrain automatically
- ❌ Real-data-trained successor model
- ❌ Drift/performance monitoring
- ❌ Automatic online learning
- ❌ Durable inference reconciliation

### REJECTED ARCHITECTURE (Explicitly NOT Done)

> **ML does NOT connect directly to backend MongoDB.**
>
> That architecture was intentionally rejected. Current boundary:
> - **Backend owns persistence/query** → sends raw event window
> - **ML owns preprocessing/features/inference**

### Watch-Computed Features — NOT Model Input

Watch-computed `DerivedFeatures`/`Baseline` may be transmitted for **audit/parity** only. They are **NOT** the model's canonical inference features. ML calculates its own 16-feature vector from **RAW telemetry**.

## Installation

```bash
# Clone and navigate
cd anxietywatch-ml

# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

Configuration is in `configs/base.yaml`. Key sections:

- `random_seed`: Reproducibility seed (default: 42)
- `window`: Time-window size/stride for segmentation
- `features`: Feature toggles (HR, HRV, movement, temperature, quality)
- `model`: Model type and hyperparameters
- `training`: Train/val/test splits
- `evaluation`: Metrics and thresholds
- `synthetic`: Synthetic data generation parameters

Environment variables override YAML:
- `RANDOM_SEED`
- `WINDOW_SIZE_SECONDS`
- `WINDOW_STRIDE_SECONDS`
- `MODEL_TYPE`

## Usage

### CLI Commands

```bash
# Run full smoke test (generate → preprocess → features → train → predict)
anxietywatch-ml smoke

# Train baseline model on synthetic data
anxietywatch-ml train --output models/baseline.pkl

# Generate synthetic data for testing
anxietywatch-ml generate --output data/synthetic.csv --format csv

# Validate data against ML contract
anxietywatch-ml validate --data data/synthetic.csv

# Run prediction (requires trained model)
anxietywatch-ml predict --model models/baseline.pkl --output predictions.csv
```

### Python API

```python
from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.synthetic import create_generator
from anxietywatch_ml.pipelines.train import TrainingPipeline
from anxietywatch_ml.pipelines.predict import PredictionPipeline

# Load config
config = load_config("configs/base.yaml")

# Generate synthetic data
generator = create_generator(config)
batches = generator.generate_dataset()

# Train
pipeline = TrainingPipeline(config)
result = pipeline.run(model_output_path="models/baseline.pkl")

# Predict
predictor = PredictionPipeline(config, "models/baseline.pkl")
predictions = predictor.run(batches)
```

## Project Structure

```
anxietywatch-ml/
├── configs/
│   └── base.yaml           # Main configuration
├── docs/
│   ├── architecture.md     # Architecture decisions
│   └── data-discovery.md   # Real contract discovery
├── src/
│   └── anxietywatch_ml/
│       ├── __init__.py
│       ├── config.py       # Configuration loader
│       ├── cli.py          # CLI entry point
│       ├── contracts/
│       │   └── telemetry.py    # Internal ML telemetry schema (Pydantic)
│       ├── data/
│       │   ├── synthetic.py    # Synthetic data generator
│       │   └── validation.py   # Data validation utilities
│       ├── preprocessing/
│       │   └── pipeline.py     # Preprocessing & windowing
│       ├── features/
│       │   └── builder.py      # Feature engineering
│       ├── models/
│       │   └── baseline.py     # Baseline models (infrastructure only)
│       ├── evaluation/
│       │   └── metrics.py      # Evaluation metrics
│       └── pipelines/
│           ├── train.py        # Training pipeline
│           └── predict.py      # Prediction pipeline
├── tests/
│   ├── test_contracts.py       # Contract validation tests
│   ├── test_validation.py      # Data validation tests
│   └── test_smoke_pipeline.py  # End-to-end smoke tests
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Key Design Decisions

### 1. Adapter Pattern for Transport DTOs
The backend sends `TelemetryBatchRequest` (camelCase, .NET DTO). The ML pipeline uses `TelemetryBatch` (snake_case, Pydantic). An adapter (`TelemetryBatchAdapter`) handles the conversion. This decouples ML from transport changes.

### 2. Synthetic Data First
All development uses synthetic data that matches the **real backend contract**. The generator is configurable and reproducible (seed=42). Real data integration comes later.

### 3. Only Available Signals
Features are computed ONLY from signals actually present in the pipeline:
- ✅ Heart rate (bpm)
- ✅ IBI (partial - Samsung only)
- ✅ Skin temperature (partial)
- ❌ Raw accelerometer x/y/z (not transmitted)
- ❌ Ambient temperature (not captured)
- ❌ Wearing state (always "unknown")
- ❌ Baseline HR / derived features (computed on watch, not sent)

### 4. Infrastructure Baselines Only
Models are explicitly labeled:
```
BASELINE DE INFRAESTRUCTURA
NO MODELO CLÍNICO
NO MODELO MVP FINAL
```

### 5. Reproducibility
All randomness controlled by `random_seed` (Python, NumPy, scikit-learn). Same seed = identical results.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=anxietywatch_ml --cov-report=term-missing

# Run specific test file
pytest tests/test_smoke_pipeline.py -v
```

## Code Quality

```bash
# Lint with Ruff
ruff check .

# Format with Ruff
ruff format .
```

## Architecture

See `docs/architecture.md` for detailed architecture decisions including the planned separation:
- Data Layer → Feature Pipeline → Model → Inference → Product Logic

The ML component produces a **technical signal only**. It does NOT decide clinical actions or emergencies.

## Data Discovery

See `docs/data-discovery.md` for the complete analysis of real contracts found in:
- WearAnxietyWatch (watch app, contracts, sensors)
- Mobile Fog Node (enrichment, transport)
- anxietywatch-backend (API DTOs, validation, MongoDB)

---

## NEXT STAGE (Immediate)

1. **Real Wear/Fog contract correction** — Fix event routing (suspected/decision vs SOS), ensure telemetry-before-suspected ordering, durable outbox ACKs
2. **Real-device validation** — Test with actual Galaxy Watch sensors, verify 60s window coverage
3. **Real user-label collection** — Collect `ACTIVITY_CONFIRMED`, `USER_OK`, `SUPPORT_REQUESTED` decisions for supervised learning

## FUTURE MODEL EVOLUTION

4. Train next model from sufficient real labeled data
5. Compare candidate vs incumbent before deployment (shadow/eval)
6. Data/model drift and performance monitoring
7. Durable inference reconciliation (pending-marker + re-drive)

---

## ⚠️ Non-Clinical Disclaimer

**Model 0.1.0 is synthetic-data / academic MVP only.**

- `prediction = 1` does **NOT** mean: anxiety detected, panic attack, crisis, SOS required.
- It means the current model predicts greater propensity for **SUPPORT_REQUESTED**, conditioned on a detector-prompted suspected event.
- `prediction = 0` does **NOT** mean: no anxiety, safe, all clear.
- **No automatic SOS/caregiver action from ML prediction.** Product decisions belong to later work.
- Do **NOT** describe this model as clinically validated.

---

## License

MIT