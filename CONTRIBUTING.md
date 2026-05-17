# Contributing to Multimodal AI Triage Assistant

Thanks for jumping in. This project has three moving parts — a Python preprocessing service, a Rust inference backend, and a Next.js frontend. This guide will get you running locally.

## Required tools

- **Rust toolchain** — The backend uses Rust edition 2024 with Cargo. Install via rustup.
- **Python 3.10+** — The sidecar runs on FastAPI with PyTorch, HuggingFace, and ChromaDB.
- **Node.js 18+ / npm** — The frontend is Next.js 16 with Turbopack.

If you're on Windows, the Rust backend needs a C++ linker (either from Visual Studio Build Tools or llvm). The Python sidecar and frontend work fine cross-platform.

## One-time setup

### 1. Python sidecar (port 8000)

```bash
cd preprocessing_service  # or project root, depending on your layout
pip install -r requirements.txt
```

The heavy dependencies are torch, transformers, torchvision, chromadb, and google-generativeai. On a fresh install this can take a few minutes.

Then start it:

```bash
uvicorn preprocessing_service:app --host 0.0.0.0 --port 8000
```

The service loads ClinicalBERT, ResNet-50, ChromaDB, and optionally a LightGBM model for SHAP at startup. You'll see log output for each component. If any model file is missing, the service logs a warning and that particular endpoint won't be available.

### 2. Rust backend (port 3001)

```bash
cd backend
cargo run
```

This starts the Axum server. It reads these environment variables (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `TRIAGE_MODEL_PATH` | `../triage_multimodal_model.txt` | LightGBM model |
| `TRIAGE_PYTHON_URL` | `http://localhost:8000` | Python service URL |
| `TRIAGE_AUDIT_DB` | `../triage_audit.db` | SQLite audit trail |

If the model file isn't found, the backend starts in **degraded mode** — health endpoints still work but `/predict` and `/batch-predict` will return 503. The audit trail still functions in degraded mode.

### 3. Frontend (port 3000)

```bash
cd frontend
npm install
npm run dev
```

These environment variables control API routing:

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_RUST_API` | `http://localhost:3001` | Rust backend |
| `NEXT_PUBLIC_PYTHON_API` | `http://localhost:8000` | Python service |

### 4. Smoke test the full stack

```bash
./smoke.sh
```

This checks that all three services are up, verifies cross-service connectivity, makes a test prediction, and runs a batch prediction with two dummy patients. It handles degraded mode gracefully — if the model isn't loaded, it reports a warning instead of failing.

## Project structure

```
├── preprocessing_service.py   # FastAPI — embedding, SHAP, RAG
├── backend/                   # Rust Axum server
│   └── src/
│       ├── main.rs            # Entry point, router setup
│       ├── models.rs          # Request/response types, SBAR formatting
│       ├── state.rs           # AppState, LightGBM FFI, audit DB
│       └── routes/
│           ├── predict.rs     # /predict endpoint
│           ├── batch.rs       # /batch-predict for MCI
│           ├── health.rs      # /health
│           ├── next_steps.rs  # /next-steps (fallback heuristic)
│           └── audit.rs       # /audit-log, /audit/override
├── frontend/                  # Next.js 16 app
│   └── src/
│       ├── components/        # UI components
│       └── lib/               # Store, config, API types
├── clinical_rag_engine.py     # ChromaDB + Gemini RAG pipeline
├── build_final_dataset.py     # Dataset construction
├── benchmark_baseline.py      # Baseline benchmark
└── smoke.sh                   # Integration smoke test
```

## Environment variables reference

### Python service

- `TRIAGE_DATA_DIR` — Path to triage dataset (default: auto-detects)
- `GEMINI_API_KEY` — Required for RAG features (Gemini 2.5 Flash)

### Rust backend

- `TRIAGE_MODEL_PATH` — Path to LightGBM .txt model (see above)
- `TRIAGE_PYTHON_URL` — URL of the Python sidecar
- `TRIAGE_AUDIT_DB` — Path to SQLite audit database

### Frontend

- `NEXT_PUBLIC_RUST_API` — Rust backend URL
- `NEXT_PUBLIC_PYTHON_API` — Python service URL

## What works in degraded mode

Not everyone will have every model file or a Gemini API key. Here's what still works:

| Component | Full mode | Degraded mode |
|---|---|---|
| Health checks | All pass | All pass |
| Single prediction | Full pipeline | 503 if no model |
| Batch prediction | With audit trail | 503 if no model |
| SHAP explainability | Real-time values | None returned |
| RAG / clinical guidance | ChromaDB + Gemini | Placeholder text |
| Audit trail | Read/write | Read/write |
| Trust Console | Full | Full |

## Making changes

- The Rust backend is the primary inference engine. Changes to prediction logic go here.
- The Python service handles ML preprocessing that doesn't have Rust equivalents (BERT, ResNet, SHAP).
- Frontend types are in `frontend/src/lib/api-types.ts`. If you add fields to a Rust response, update them there too.
- The audit database is SQLite. Schema changes need matching migrations in `state.rs`.

## Running tests

```bash
# Rust
cd backend && cargo test

# Python
python -m pytest tests/

# Frontend
cd frontend && npm run build  # type-checks everything
```

The smoke test (`./smoke.sh`) is the best integration-level verification. It exercises all three services end-to-end.
