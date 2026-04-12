# Live API Contracts

This document tracks the current API contracts across all three services (Frontend, Rust, Python).

## Frontend → Rust (port 3001)

### `POST /predict`
**Request:**
```json
{
  "age": 72,
  "heart_rate": 145,
  "resp_rate": 38,
  "spo2": 78,
  "temp_f": 101.2,
  "systolic_bp": 72,
  "pain_scale": 0,
  "chief_complaint": "Unresponsive, found on floor",
  "image_base64": "base64-encoded-string-or-null"
}
```

**Response:**
```json
{
  "predicted_esi": 1,
  "esi_label": "ESI 1 (Resuscitation)",
  "probabilities": [0.89, 0.06, 0.02, 0.02, 0.01],
  "confidence": {
    "top_probability": 0.89,
    "margin": 0.83,
    "entropy": 0.45,
    "is_uncertain": false,
    "confidence_label": "High"
  },
  "feature_vector": [72.0, 145.0, 38.0, 78.0, 101.2, 72.0, 0.0, -1.23, ...],
  "shap": {
    "base_value": -0.5,
    "features": [{"name": "spo2", "value": 78, "shap_value": 1.2}, ...],
    "predicted_class": 0,
    "prediction_label": "ESI 1 (Resuscitation)"
  },
  "audit_id": "uuid-string"
}
```

### `POST /next-steps`
**Request:**
Same as `/predict`

**Response:**
```json
{
  "recommendation": "## IMMEDIATE TRIAGE ACTIONS\n- Place patient in resuscitation bay...",
  "similar_cases": [
    {"complaint": "Crushing chest pain", "target_esi": 2, "similarity": 0.93, "heart_rate": 125, "spo2": 93}
  ]
}
```

### `POST /batch-predict`
**Request:**
```json
{
  "patients": [
    {...patient1...},
    {...patient2...}
  ]
}
```

**Response:**
```json
{
  "total_patients": 2,
  "patients": [
    {"index": 0, "chief_complaint": "...", "prediction": {...}},
    {"index": 1, "chief_complaint": "...", "prediction": {...}}
  ],
  "summary": {
    "esi_1_count": 1,
    "esi_2_count": 0,
    "esi_3_count": 1,
    "esi_4_count": 0,
    "esi_5_count": 0,
    "uncertain_count": 0
  }
}
```

### `GET /health`
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "python_service_url": "http://localhost:8000"
}
```

---

## Frontend → Python (port 8000)

### `POST /rag-stream`
**Request:**
```json
{
  "complaint": "string",
  "vitals": {
    "age": 45,
    "heart_rate": 80,
    "resp_rate": 16,
    "spo2": 98,
    "temp_f": 98.6,
    "systolic_bp": 120,
    "pain_scale": 0
  },
  "predicted_esi": 3
}
```

**Response:** SSE stream with:
- `data: {"similar_cases": [...]}`
- `data: {"token": "..."}`
- `data: [DONE]`

---

## Rust → Python (internal)

### `POST /embed`
**Request:**
```json
{
  "complaint": "string",
  "image_base64": "base64-string-or-null"
}
```

**Response:**
```json
{
  "text_features": [0.1, 0.2, ...],  // 10 floats
  "image_features": [0.0, 0.0, ...]  // 5 floats
}
```

### `POST /rag`
**Request:** Same as `/rag-stream`

**Response:**
```json
{
  "recommendation": "string",
  "similar_cases": [
    {
      "complaint": "string",
      "target_esi": 2,
      "similarity": 0.93,
      "text_similarity": 0.91,
      "vitals_similarity": 0.85,
      "source": "both",
      "flag_high_risk": 1,
      "heart_rate": 125,
      "spo2": 93
    }
  ]
}
```

### `POST /shap`
**Request:**
```json
{
  "feature_vector": [72.0, 145.0, ...],  // 22 floats
  "predicted_class": 0
}
```

**Response:**
```json
{
  "base_value": -0.5,
  "features": [
    {"name": "spo2", "value": 78, "shap_value": 1.2},
    ...
  ],
  "predicted_class": 0,
  "prediction_label": "ESI 1 (Resuscitation)"
}
```

### `GET /health`
**Response:**
```json
{
  "status": "ok",
  "bert_loaded": true,
  "resnet_loaded": true,
  "shap_loaded": true,
  "rag_available": true,
  "hybrid_retrieval": true,
  "chroma_patients": 1197,
  "device": "cuda",
  "startup_complete": true
}
```

### `GET /ready`
**Response:**
```json
{
  "ready": true,
  "python_ready": true,
  "gemini_configured": true,
  "rust_backend": "ok",
  "rust_model_loaded": true
}
```

---

## Field Naming Conventions

| Concept | Frontend | Rust | Python |
|---------|----------|------|--------|
| Image data | `image_base64` | `image_base64` | `image_base64` |
| Chief complaint | `chief_complaint` | `chief_complaint` | `complaint` |
| Vital signs | snake_case | snake_case | snake_case |

---

## Contract Drift Notes

- `next_steps.rs` uses heuristic ESI fallback when model unavailable (not full prediction path)
- `/next-steps` does not receive predicted ESI from frontend — determines internally
- ✅ FIXED: SimilarCase Rust model now includes text_similarity, vitals_similarity, source, flag_high_risk