# AnxietyWatch ML — Architecture Decisions

This document records the architectural decisions made during the MVP bootstrap phase.

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANXIETYWATCH ML COMPONENT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DATA LAYER          FEATURE PIPELINE        MODEL          INFERENCE      │
│   ───────────         ─────────────────       ──────         ──────────    │
│   ┌─────────┐         ┌───────────────┐      ┌────────┐     ┌──────────┐  │
│   │ Adapter │────────▶│ Preprocessing │─────▶│ Train  │────▶│ Predict  │  │
│   │ (DTO→   │         │ • Flatten     │      │ • Base │     │ • Load   │  │
│   │  Schema)│         │ • Window      │      │ • Eval │     │ • Score  │  │
│   └─────────┘         │ • Clean       │      └────────┘     └──────────┘  │
│        ▲              │               │              ▲            ▲         │
│        │              └───────────────┘              │            │         │
│        │                     ▲                       │            │         │
│        │              ┌─────┴─────┐                  │            │         │
│        │              │ Features  │                  │            │         │
│        │              │ • HR stats│                  │            │         │
│        │              │ • HRV     │                  │            │         │
│        │              │ • Quality │                  │            │         │
│        │              └───────────┘                  │            │         │
│        │                     │                       │            │         │
│   ┌────┴────┐        ┌──────┴──────┐       ┌────────┴────┐ ┌────┴────┐   │
│   │ Synthetic│       │  Config     │       │  Artifacts  │ │ Metrics │   │
│   │ Generator│       │  (YAML)     │       │  (pickle)   │ │ (JSON)  │   │
│   └─────────┘        └─────────────┘       └─────────────┘ └─────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

| Layer | Responsibility | Key Files |
|-------|---------------|-----------|
| **Data Layer** | Transport DTO → Internal schema, validation, synthetic generation | `contracts/telemetry.py`, `data/validation.py`, `data/synthetic.py` |
| **Feature Pipeline** | Preprocessing, windowing, feature engineering | `preprocessing/pipeline.py`, `features/builder.py` |
| **Model** | Training, evaluation, serialization | `models/baseline.py`, `evaluation/metrics.py` |
| **Inference** | Load model, predict on new data | `pipelines/predict.py` |
| **Product Logic** | (External) Consumes ML signal for UX decisions | — |

> **Critical**: The ML component produces a **technical signal only**. It does NOT decide clinical actions, trigger SOS, or make medical recommendations. Product logic consumes the signal and applies business rules.

---

## 2. Contract Strategy

### Problem
The backend API uses `TelemetryBatchRequest` (camelCase, .NET records, FluentValidation). The ML pipeline needs a clean, Pythonic schema with Pydantic validation.

### Solution: Adapter Pattern

```
Backend DTO (TelemetryBatchRequest)
        │
        ▼
TelemetryBatchAdapter.from_backend_dict()
        │
        ▼
Internal ML Schema (TelemetryBatch - Pydantic)
```

**Benefits:**
- Backend can evolve independently (add fields, change validation)
- ML pipeline has strict, documented contract
- Single point of transformation logic
- Easy to test adapter in isolation

### Internal Schema (TelemetryBatch)

```python
class TelemetryBatch(BaseModel):
    batch_id: UUID
    device_id: UUID
    user_id: UUID | None
    session_id: UUID
    started_at: datetime
    ended_at: datetime
    sequence: int
    samples: list[TelemetrySample]  # 1-600

class TelemetrySample(BaseModel):
    timestamp: datetime
    heart_rate_bpm: float | None
    ibi_ms: list[float]  # max 16, may be empty
    accelerometer: dict | None  # ALWAYS None currently
    skin_temperature_celsius: float | None
    ambient_temperature_celsius: float | None  # ALWAYS None currently
    quality: TelemetrySampleQuality  # heart_rate, ibi, wearing_state enums
```

---

## 3. Data Availability Reality

Based on inspection of WearAnxietyWatch, Fog Node, and Backend:

| Signal | Available? | Notes |
|--------|------------|-------|
| Heart rate (bpm) | ✅ Yes | Every HR sample |
| IBI (ms) | ⚠️ Partial | Only Samsung Health Sensor SDK (Galaxy Watch7) |
| Accelerometer (x,y,z) | ❌ No | Watch computes magnitude+variance only |
| Accelerometer magnitude | ⚠️ Partial | Not transmitted to cloud |
| Skin temperature | ⚠️ Partial | When sensor available |
| Ambient temperature | ❌ No | Not captured |
| Signal quality (0-1) | ✅ Yes | From sensor, mapped to enum in fog |
| Wearing state | ❌ No | Always "unknown" in current pipeline |
| Baseline HR (watch) | ❌ No | Computed on watch, not sent |
| Derived features (watch) | ❌ No | RMSSD, SDNN, slope computed on watch, not sent |
| Detection score (watch) | ❌ No | Rule-based detector score not transmitted |
| User response | ❌ No | Captured on watch, not sent to backend |
| Steps | ❌ No | Dropped at fog enricher |

**Implication**: ML features limited to what survives the pipeline. No raw accelerometer, no baseline, no watch-derived features.

---

## 4. Synthetic Data Strategy

### Why Synthetic First?
- Real data requires backend integration, privacy review, labeling
- Pipeline must be validated before real data arrives
- Contract testing requires known-good data

### Generator Design (`data/synthetic.py`)

```python
class SyntheticTelemetryGenerator:
    - Matches REAL backend contract exactly
    - Configurable via YAML (seed, users, sessions, signal params)
    - Reproducible: same seed = identical data
    - Clearly marked: `is_synthetic: True` in output
    - Generates physiological plausible signals with anomalies
```

### Synthetic Labels
Current: heuristic based on mean HR > 100 bpm in window.
Future: real user responses from watch (`UserResponse` enum).

---

## 5. Preprocessing Pipeline

### Steps (`preprocessing/pipeline.py`)

1. **Flatten** batches → DataFrame (one row per sample)
2. **Sort** by user/session/timestamp
3. **Handle missing**: Forward-fill HR (short gaps), temperature (long gaps)
4. **Outlier detection**: Rolling z-score on HR (configurable threshold)
5. **Window**: Fixed-size sliding windows per session (configurable size/stride)
6. **Filter**: Minimum samples, minimum HR availability per window

### Windowing
- Default: 60s windows, 30s stride
- Per-session (no cross-session windows)
- Metadata preserved (user_id, session_id, window bounds, sample count)

---

## 6. Feature Engineering

### Available Features (`features/builder.py`)

| Category | Features | Source |
|----------|----------|--------|
| HR Statistics | mean, std, min, max, slope (bpm/min) | `heart_rate_bpm` |
| HRV (if IBI) | RMSSD, SDNN, pNN50 | `ibi_ms` |
| Temperature | mean, std | `skin_temperature_celsius` |
| Quality | good/fair/poor ratios, valid sample ratio | `quality_*` |
| Temporal | duration, sample count, last sample age | `timestamp` |
| Movement | ❌ NaN | Not available |

### Missing Features (Future)
- Raw accelerometer features (require watch firmware change)
- Baseline-relative HR (require watch to send baseline)
- Watch-derived features (RMSSD, slope, delta) - already computed on watch!

---

## 7. Baseline Models

### Model Types (`models/baseline.py`)

| Type | Use Case |
|------|----------|
| `baseline` (DummyClassifier, strategy="prior") | Sanity check - predicts majority class |
| `dummy` (configurable strategy) | Testing pipeline with known behavior |
| `logistic_regression` | Simple interpretable baseline |

### Explicit Labeling
```python
# In code and logs:
BASELINE DE INFRAESTRUCTURA
NO MODELO CLÍNICO
NO MODELO MVP FINAL
```

### Serialization
- Pickle (Python-only, sufficient for MVP)
- Includes: pipeline (scaler + classifier), feature names, config
- Version tied to code via git, not model registry (yet)

---

## 8. Evaluation

### Metrics (`evaluation/metrics.py`)
- Accuracy, Precision, Recall, F1
- ROC-AUC, Average Precision (require probabilities)
- Confusion matrix
- Threshold optimization (F1-max)

### Splits
- Train / Val / Test (configurable, default 70/10/20)
- Stratified by label
- Fixed random seed for reproducibility

---

## 9. Configuration

### Single Source: `configs/base.yaml`

```yaml
random_seed: 42

window:
  size_seconds: 60
  stride_seconds: 30
  min_samples_per_window: 10

features:
  hr_mean: true
  hrv_rmssd: true
  # ... all feature toggles

model:
  type: baseline
  logistic_regression:
    C: 1.0
    # ...

training:
  test_size: 0.2
  val_size: 0.1

synthetic:
  n_users: 10
  # ... all generation params
```

### Environment Overrides
- `RANDOM_SEED`, `WINDOW_SIZE_SECONDS`, `MODEL_TYPE`, etc.
- Useful for CI, experiments, containerization

---

## 10. Reproducibility

All randomness controlled by `random_seed`:
- Python `random`
- NumPy `default_rng(seed)`
- scikit-learn `random_state`

```python
# In every component factory:
rng = np.random.default_rng(config["random_seed"])
```

**Verification**: `test_smoke_pipeline.py::TestSmokePipeline::test_reproducibility`

---

## 11. Testing Strategy

| Test Type | File | Coverage |
|-----------|------|----------|
| Contracts | `test_contracts.py` | Pydantic validation, adapter |
| Validation | `test_validation.py` | Data quality checks |
| Smoke | `test_smoke_pipeline.py` | Full pipeline E2E |

**Run**: `pytest` (all pass required for CI)

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Backend DTO changes break adapter | Medium | High | Adapter tests, versioned schema in contracts |
| IBI missing for non-Samsung | High | Medium | Features handle missing IBI gracefully (NaN → filled) |
| No ground truth labels | High | High | Synthetic labels for plumbing; advocate for user_response export |
| Watch-derived features not transmitted | High | High | Document in data-discovery; advocate for feature export |
| Concept drift in synthetic vs real | Medium | High | Monitor feature distributions when real data arrives |
| Clinical misuse of baseline | Low | Critical | Explicit labeling, no deployment path yet |

---

## 13. Future Evolution (Not Implemented)

### Phase 1: Real Data Integration
- MongoDB reader for `telemetry_batches`
- Incremental processing (watermarks)
- Data quality monitoring

### Phase 2: Feature Enrichment
- Advocate for watch/fog to export:
  - Baseline HR (mean, std)
  - Derived features (RMSSD, SDNN, slope, delta)
  - Detection score + monitoring state
  - User responses (ground truth)
  - Steps data

### Phase 3: Model Iteration
- Replace baseline with proper model (XGBoost, small NN)
- Cross-validation, hyperparameter tuning
- Model registry (MLflow or similar)

### Phase 4: Productionization
- Inference API (FastAPI)
- Container (Docker)
- Monitoring (drift, latency, performance)
- A/B testing framework

### Phase 5: Clinical Validation
- Prospective study design
- Regulatory pathway (if applicable)
- Explainability, fairness audits

---

## 14. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-17 | Adapter pattern for DTOs | Decouple ML from backend evolution |
| 2026-08-17 | Synthetic data only for MVP | Validate plumbing before real data access |
| 2026-08-17 | Only available signals for features | No phantom features; honest about limitations |
| 2026-08-17 | Baseline = DummyClassifier | Simplest possible model to test pipeline |
| 2026-08-17 | Pickle for model serialization | Python-only, zero dependencies, sufficient for MVP |
| 2026-08-17 | YAML config + env overrides | Human-readable, CI-friendly, no code changes |
| 2026-08-17 | Explicit "NOT CLINICAL" labeling | Prevent misuse, set expectations |
| 2026-08-17 | Reproducibility via single seed | Debugging, CI, experiment tracking |

---

## 15. Non-Goals (Explicit)

- ❌ Real-time / streaming inference
- ❌ Kubernetes / Azure deployment
- ❌ Automated retraining
- ❌ Drift detection
- ❌ Deep learning (TensorFlow/PyTorch)
- ❌ AutoML
- ❌ MLflow / model registry
- ❌ Dashboards
- ❌ Feature store
- ❌ Data versioning (DVC)

These are deferred until the baseline pipeline works with real data and labels.

---

## 16. Data Leakage Prevention

### Problem
Standard random splits (`train_test_split`) allow windows from the same session/user to appear in both train and test sets. This creates data leakage because:
- Consecutive windows from the same session are highly correlated
- User-specific patterns (baseline HR, device characteristics) leak into test evaluation
- Inflates performance metrics unrealistically

### Solution: Group-Aware Splitting (`evaluation/splitting.py`)

```python
def group_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,  # session_id or user_id
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    group_by: Literal["session", "user"] = "session",
) -> SplitResult:
    """Split ensuring all samples from the same group stay together."""
```

### Strategy
1. **Group by session_id (default)**: All windows from the same session go to the same split
2. **Group by user_id (optional)**: All sessions from the same user go to the same split

Uses `sklearn.model_selection.GroupShuffleSplit` which guarantees:
- No group appears in more than one partition
- Approximate size ratios maintained
- Reproducible with `random_state`

### Verification
Tests explicitly verify disjoint groups:
```python
assert train_sessions.isdisjoint(val_sessions)
assert train_sessions.isdisjoint(test_sessions)
assert val_sessions.isdisjoint(test_sessions)
```

### Usage in Pipeline
```python
# In train_with_pipeline():
split_result = group_aware_split(X, y, group_column, ...)
# Preprocessing fit on train only
preprocessing_pipeline.fit(X_train)
# Transform all splits with fitted preprocessor
X_train_transformed = preprocessing_pipeline.transform(X_train)
X_val_transformed = preprocessing_pipeline.transform(X_val)
X_test_transformed = preprocessing_pipeline.transform(X_test)
```

### Future Evaluation Strategy
Production evaluation will use multiple holdout strategies:
- **Session holdout**: Test on unseen sessions from train users
- **User holdout**: Test on completely unseen users (harder, more realistic)
- **Temporal holdout**: Test on future time periods (simulates deployment)

This prevents optimistic bias and ensures the model generalizes to truly unseen data.