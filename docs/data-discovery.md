# Data Discovery — AnxietyWatch Telemetry Contracts

This document summarizes the real data contracts found across the AnxietyWatch system (WearAnxietyWatch, mobile-app-in-background/Fog Node, anxietywatch-backend) as of the bootstrap inspection.

## Summary of Data Flow

```
Galaxy Watch (Wear OS)
    ↓ Wear Data Layer (DataClient / MessageClient)
Fog Node (Mobile App - React Native)
    ↓ HTTPS (enriched with user/device/session identity)
Backend API (.NET Core)
    ↓ MongoDB persistence
Machine Learning (this repository)
```

---

## 1. WearAnxietyWatch (Watch App)

### 1.1 Sensor Acquisition (packages/contracts, apps/wear)

| Field | Type | Source | Moment it appears | Available | Potential ML Use |
|-------|------|--------|-------------------|-----------|------------------|
| `timestamp` (ISO 8601) | string | `SensorReading.capturedAtEpochMillis` | Every sensor reading | ✅ AVAILABLE | Time-series alignment, windowing |
| `heartRateBpm` | number (nullable) | `SensorReading.HeartRate.bpm` | Heart rate samples | ✅ AVAILABLE | Primary physiological signal |
| `ibiMs` | number[] (max 16) | `SensorReading.HeartRate.ibiMillis` | IBI samples (Samsung SDK only) | ⚠️ PARTIAL | HRV features (RMSSD, SDNN) |
| `accelerometer` (x,y,z) | object/null | `SensorReading.Motion` | Motion samples | ⚠️ PARTIAL | Activity context, artifact rejection |
| `skinTemperatureCelsius` | number (nullable) | `SensorReading.SkinTemperature.celsius` | Skin temp samples | ⚠️ PARTIAL | Thermal context |
| `signalQuality` | number (0-1) | `SensorReading.HeartRate.signalQuality` | Heart rate samples | ✅ AVAILABLE | Sample weighting, quality filtering |
| `wearingState` | enum | Derived from Health Services availability | Availability events | ⚠️ PARTIAL | Data validity gate |
| `deviceCapabilities` | object | `DeviceCapabilities` | Startup / capability query | ✅ AVAILABLE | Sensor availability flags |

**Notes:**
- IBI (inter-beat intervals) only available on Samsung devices with Health Sensor SDK bundled (`SamsungSensorProvider`). On standard Wear OS Health Services, `ibiMillis` is `null`.
- Accelerometer data captured as `magnitudeG` and `variance` in `SensorReading.Motion`, not raw x/y/z triaxial. Raw triaxial not currently exported.
- `signalQuality` is a 0-1 float from the sensor; the fog enricher maps it to `good`/`fair`/`poor`/`unknown`.
- Baseline heart rate tracked locally on watch (`BaselineSnapshot` with Welford's online algorithm).

### 1.2 Watch → Phone Transport (Wear Data Layer)

**Telemetry Envelope** (`BackendEndpointContract.telemetryEnvelope`):
```json
{
  "schemaVersion": "wear-telemetry-records-v2",
  "targetEndpoint": "/fog/v1/telemetry",
  "transport": "WEAR_DATA_LAYER",
  "batchId": "uuid",
  "startedAt": "ISO8601",
  "endedAt": "ISO8601",
  "mobileEnrichmentRequired": ["userId", "deviceId", "sessionId", "sequence", "samples"],
  "records": [
    {
      "id": "uuid",
      "capturedAt": "ISO8601",
      "type": "heart_rate|motion|steps|skin_temperature|availability",
      "payload": { ... }
    }
  ]
}
```

**Record payloads by type:**
- `heart_rate`: `{ "bpm": number, "ibiMillis": number[]|null, "signalQuality": number, "source": string }`
- `motion`: `{ "magnitudeG": number, "variance": number, "source": string }`
- `steps`: `{ "dailyTotal": number, "source": string }`
- `skin_temperature`: `{ "celsius": number, "source": string }`
- `availability`: `{ "sensor": string, "status": string, "reason": string, "source": string }`

---

## 2. Mobile Fog Node (apps/mobile/src/fog)

### 2.1 Enrichment (`enricher.ts`)

The fog node **enriches** the watch envelope with identity and transforms to the backend DTO:

| Field Added | Source |
|-------------|--------|
| `userId` | Authenticated user (JWT) |
| `deviceId` | Persisted device identity (UUID) |
| `sessionId` | Persisted session identity (UUID) |
| `sequence` | Monotonic counter (persisted) |

**Transformation logic (`sampleFromRecord`):**
- Only `heart_rate`, `motion`, `skin_temperature` records become `TelemetrySample`
- `steps`, `availability`, unknown types → **dropped**
- `accelerometer` in output is **always null** (watch sends `magnitudeG`/`variance`, not x/y/z)
- `ambientTemperatureCelsius` → **always null** (not captured)
- `quality.wearingState` → **always "unknown"** (not derived on watch/fog)
- `quality.heartRate` / `quality.ibi` → mapped from `signalQuality` (0-1) to enum

### 2.2 Output to Backend (`TelemetryBatchPayload`)

```typescript
interface TelemetryBatchPayload {
  batchId: string;
  deviceId: string;
  userId: string;
  sessionId: string;
  startedAt: string;      // ISO8601
  endedAt: string;        // ISO8601
  sequence: number;
  samples: TelemetrySample[];
}

interface TelemetrySample {
  timestamp: string;
  heartRateBpm: number | null;
  ibiMs: number[];
  accelerometer: { x: number; y: number; z: number } | null;  // ALWAYS NULL
  skinTemperatureCelsius: number | null;
  ambientTemperatureCelsius: number | null;                    // ALWAYS NULL
  quality: {
    heartRate: 'good'|'fair'|'poor'|'unknown';
    ibi: 'good'|'fair'|'poor'|'unknown';
    wearingState: 'onBody'|'offBody'|'unknown';                // ALWAYS "unknown"
  };
}
```

---

## 3. Backend API (anxietywatch-backend)

### 3.1 Endpoint Contracts

**POST `/api/v1/telemetry/batch`** → `TelemetryBatchRequest`

```csharp
public sealed record TelemetryBatchRequest(
    Guid BatchId,
    Guid DeviceId,
    Guid? UserId,
    Guid SessionId,
    DateTimeOffset StartedAt,
    DateTimeOffset EndedAt,
    long Sequence,
    IReadOnlyList<TelemetrySampleRequest> Samples);

public sealed record TelemetrySampleRequest(
    DateTimeOffset Timestamp,
    double? HeartRateBpm,
    IReadOnlyList<double> IbiMs,
    AccelerometerRequest? Accelerometer,          // ALWAYS NULL from fog
    double? SkinTemperatureCelsius,
    double? AmbientTemperatureCelsius,            // ALWAYS NULL from fog
    TelemetryQualityRequest Quality);

public sealed record TelemetryQualityRequest(
    string HeartRate,      // "good"|"fair"|"poor"|"unknown"
    string Ibi,            // "good"|"fair"|"poor"|"unknown"
    string WearingState);  // "onBody"|"offBody"|"unknown" → ALWAYS "unknown"

public sealed record AccelerometerRequest(double X, double Y, double Z); // NEVER POPULATED
```

**Validation rules (FluentValidation):**
- `BatchId`, `DeviceId`, `SessionId` required
- `Sequence` ≥ 0
- `Samples` count: 1–600
- `EndedAt` ≥ `StartedAt`
- Per sample: `IbiMs` ≤ 16 items, all > 0
- `HeartRateBpm` > 0 when present
- Quality enums must match allowed values

### 3.2 Persistence (MongoDB)

Collection: `telemetry_batches`
- Document = serialized `TelemetryBatchRequest` + `_id` (batchId) + `userId` + `receivedAt`
- No additional enrichment or derived fields stored

---

## 4. Consolidated Field Availability for ML

| Campo | Tipo | Fuente original | Momento | Disponibilidad | Uso potencial ML |
|-------|------|-----------------|---------|----------------|------------------|
| `timestamp` | datetime | Watch `capturedAtEpochMillis` | Cada muestra | ✅ AVAILABLE | Index temporal, ventanas |
| `heartRateBpm` | float | Watch `HeartRate.bpm` | HR samples | ✅ AVAILABLE | Señal principal |
| `ibiMs` | float[] | Watch `HeartRate.ibiMillis` | IBI samples (Samsung) | ⚠️ PARTIAL | HRV (RMSSD, SDNN) |
| `accelerometer_x/y/z` | float | — | — | ❌ NOT FOUND | Contexto de actividad |
| `accelerometer_magnitude` | float | Watch `Motion.magnitudeG` | Motion samples | ⚠️ PARTIAL | Proxy de movimiento |
| `accelerometer_variance` | float | Watch `Motion.variance` | Motion samples | ⚠️ PARTIAL | Proxy de movimiento |
| `skinTemperatureCelsius` | float | Watch `SkinTemperature.celsius` | Temp samples | ⚠️ PARTIAL | Contexto térmico |
| `ambientTemperatureCelsius` | float | — | — | ❌ NOT FOUND | — |
| `signalQuality` | float (0-1) | Watch `HeartRate.signalQuality` | HR samples | ✅ AVAILABLE | Peso de muestra, filtrado |
| `quality_heartRate` | enum | Fog mapping of signalQuality | Cada muestra | ✅ AVAILABLE | Gate de calidad |
| `quality_ibi` | enum | Fog mapping of signalQuality | Cada muestra | ⚠️ PARTIAL | Gate de calidad IBI |
| `quality_wearingState` | enum | — | — | ❌ NOT FOUND | — (siempre "unknown") |
| `deviceId` | UUID | Fog persisted identity | Por batch | ✅ AVAILABLE | Agrupación por dispositivo |
| `userId` | UUID | Auth (JWT) | Por batch | ✅ AVAILABLE | Agrupación por usuario |
| `sessionId` | UUID | Fog persisted session | Por batch | ✅ AVAILABLE | Agrupación por sesión |
| `sequence` | int64 | Fog monotonic counter | Por batch | ✅ AVAILABLE | Orden, detección gaps |
| `batchId` | UUID | Watch UUID | Por batch | ✅ AVAILABLE | Trazabilidad |
| `steps` | int64 | Watch `Steps.dailyTotal` | Steps samples | ⚠️ PARTIAL | Nivel de actividad diario |
| `baseline_hr_mean` | float | Watch `BaselineSnapshot` | Local en reloj | ❌ NOT FOUND (no enviado) | Referencia personalizada |
| `baseline_hr_std` | float | Watch `BaselineSnapshot` | Local en reloj | ❌ NOT FOUND (no enviado) | Referencia personalizada |
| `derived_features` | object | Watch `FeatureExtractor` | Local en reloj | ❌ NOT FOUND (no enviado) | Features precomputadas |
| `detection_score` | float | Watch `PreliminaryDetector` | Local en reloj | ❌ NOT FOUND (no enviado) | Etiqueta débil / target |
| `monitoring_state` | enum | Watch `MonitoringState` | Local en reloj | ❌ NOT FOUND (no enviado) | Estado del detector |
| `user_response` | enum | Watch `UserResponse` | Interacción usuario | ❌ NOT FOUND (no enviado) | Ground truth débil |
| `sos_events` | object | Watch/Backend `SosTrigger` | Eventos SOS | ✅ AVAILABLE (separate) | Etiquetas de crisis |

---

## 5. Fields Missing / Not Yet Available for ML

| Signal | Why Missing | Feasibility |
|--------|-------------|-------------|
| Raw triaxial accelerometer (x,y,z) | Watch only computes magnitude/variance; not transmitted | Medium (requires watch firmware change) |
| `wearingState` (onBody/offBody) | Not derived on watch; Health Services availability not mapped | High (can derive from availability events) |
| `ambientTemperatureCelsius` | No sensor / not exposed | Low |
| Baseline HR (mean, std) | Computed locally on watch, not sent to cloud | High (add to batch enrichment) |
| Derived features (RMSSD, SDNN, slope, delta from baseline) | Computed locally on watch, not sent to cloud | High (add to batch enrichment) |
| Watch-level detection score / state | Local only, not transmitted | High (add to telemetry or separate stream) |
| User response to prompts (ground truth) | Local only, not transmitted | High (add to event stream) |
| ECG / SpO2 / EDA | Hardware not available on current target (Galaxy Watch7) | Low (future hardware) |
| Sleep stages | Not captured | Medium (Health Services sleep API) |

---

## 6. Inconsistencies / Risks Identified

1. **IBI availability fragmented**: Only Samsung devices with bundled SDK provide IBI. Standard Wear OS Health Services does not. ML pipeline must handle missing IBI gracefully.

2. **Accelerometer not triaxial**: Watch sends magnitude + variance only. No raw x/y/z. Activity classification limited.

3. **`wearingState` always "unknown"**: Fog enricher hardcodes this. Could be derived from `Availability` records (off-body detection).

4. **`ambientTemperatureCelsius` always null**: Not captured anywhere in pipeline.

5. **Baseline & derived features not exported**: Valuable personalized features computed on-watch never reach ML. Consider adding to telemetry batch or separate feature stream.

6. **Watch-level detection score not exported**: The on-watch rule-based detector produces a score and state (`NORMAL` → `OBSERVING` → `USER_VALIDATION` → `INTERVENTION`). This is a valuable weak label / target proxy not available in backend.

7. **User responses not exported**: `UserResponse` enum (ACTIVITY_CONFIRMED, USER_OK, SUPPORT_REQUESTED, etc.) captured on watch but not sent to backend. This is the closest thing to ground truth.

8. **Steps data dropped at fog**: `steps` records are filtered out in `enricher.ts` and never reach backend.

9. **Sequence gaps**: Fog maintains a monotonic `sequence` per session. Gaps indicate missed batches — useful for data quality monitoring.

10. **Schema versioning**: Watch uses `wear-telemetry-records-v2`. Backend validates via FluentValidation but no explicit schema version check. Drift risk if watch updates without backend coordination.

---

## 7. Recommended ML Contract (Internal)

Based on what is **actually available** in the backend `telemetry_batches` collection, the ML pipeline should expect:

```python
# Minimal viable fields (guaranteed present)
- batch_id: UUID
- user_id: UUID
- device_id: UUID
- session_id: UUID
- sequence: int
- started_at: datetime
- ended_at: datetime
- samples: List[{
    timestamp: datetime,
    heart_rate_bpm: float | None,
    ibi_ms: List[float],           # may be empty
    accelerometer: None,           # always None currently
    skin_temperature_celsius: float | None,
    ambient_temperature_celsius: None,  # always None
    quality: {
        heart_rate: "good"|"fair"|"poor"|"unknown",
        ibi: "good"|"fair"|"poor"|"unknown",
        wearing_state: "unknown"   # always "unknown"
    }
}]

# Fields NOT available (do not assume):
- accelerometer_x/y/z
- wearing_state (onBody/offBody)
- ambient_temperature
- baseline_hr_mean/std
- derived_features (RMSSD, SDNN, slope, delta)
- detection_score
- monitoring_state
- user_response
- steps
```

---

## 8. Next Steps for ML Pipeline

1. **Adapter layer**: Transform backend `TelemetryBatchRequest` → internal ML schema (handles nulls, drops always-null fields, renames).
2. **Synthetic generator**: Match the above contract exactly (including nulls for unavailable fields).
3. **Feature engineering**: Compute windowed features (HR mean, HRV via IBI if present, movement magnitude mean, temperature, quality ratios) from raw samples.
4. **Baseline model**: Rule-based or simple LogisticRegression on windowed features + synthetic labels.
5. **Future enrichment**: Advocate for watch/fog/backend changes to export baseline, derived features, detection score, user responses, and steps.