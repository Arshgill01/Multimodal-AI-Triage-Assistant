# Live API Contracts

## Frontend API base URLs

- Rust API: `NEXT_PUBLIC_RUST_API` or `http://localhost:3001`
- Python API: `NEXT_PUBLIC_PYTHON_API` or `http://localhost:8000`

## Rust endpoints

### GET `/health`

Returns backend liveness / readiness.

### POST `/predict`

Used by the single-patient UI.

Request shape:

```json
{
  "age": 45,
  "heart_rate": 80,
  "resp_rate": 16,
  "spo2": 98,
  "temp_f": 98.6,
  "systolic_bp": 120,
  "pain_scale": 0,
  "chief_complaint": "example complaint",
  "image_base64": "optional-base64-without-data-uri-prefix"
}
```
