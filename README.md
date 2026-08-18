# AnxietyWatch ML

**Machine Learning component for AnxietyWatch — MVP Bootstrap / Data Pipeline**

> ⚠️ **Current Status: ML MVP — bootstrap / data pipeline**
>
> This is **NOT** a clinical anxiety detector. The models in this repository are **infrastructure baselines** used solely to validate the ML plumbing (data contracts, preprocessing, feature engineering, training, prediction). They have no clinical validity.

## Overview

This repository contains the Machine Learning pipeline for AnxietyWatch. It consumes wearable telemetry (heart rate, IBI, accelerometer magnitude, skin temperature) from the AnxietyWatch backend and produces technical signals for downstream product logic.

### Data Flow

```
Galaxy Watch → Wear Data Layer → Mobile Fog Node → Backend API → MongoDB → ML Pipeline
                                                                ↓
                                              Validation → Preprocessing → Features → Model → Prediction
```

### What This Pipeline Does

- ✅ Validates incoming telemetry against the internal ML contract
- ✅ Preprocesses and windows time-series data
- ✅ Engineers features from available signals (HR, HRV, temperature, quality)
- ✅ Trains baseline models (DummyClassifier, LogisticRegression) on synthetic data
- ✅ Evaluates with standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Provides reproducible, configurable pipeline with CLI

### What This Pipeline Does NOT Do (Yet)

- ❌ Detect anxiety clinically
- ❌ Use real patient data (only synthetic for now)
- ❌ Deploy to production
- ❌ Monitor drift or retrain automatically
- ❌ Integrate with Azure/cloud infrastructure

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
- ⚠️ IBI (partial - Samsung only)
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

## Next Steps (Not Implemented)

1. **Real data adapter**: Connect to backend MongoDB `telemetry_batches` collection
2. **Feature enrichment**: Advocate for watch/fog to export baseline HR, derived features, detection scores
3. **Label acquisition**: Integrate user responses (ground truth) from watch
4. **Model iteration**: Replace baseline with proper model once real labels available
5. **Monitoring**: Add data drift detection, performance monitoring
6. **Deployment**: Containerize, add API endpoint, integrate with product

## License

MIT