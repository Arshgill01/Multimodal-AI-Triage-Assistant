# Contributing to Frostbyte Multimodal AI Triage

Welcome! We appreciate your interest in contributing to the Frostbyte Triage system. This guide covers how to set up the three core services (Python, Rust, Frontend) for local development.

## Prerequisites

Ensure you have the following installed:
- **Rust toolchain** (1.70+, via `rustup`)
- **Python 3.10+**
- **Node.js** (v20+) & **npm**

## Environment Variables

The system relies on several environment variables. You can export these in your shell before running the services.

### Rust Backend
- `TRIAGE_MODEL_PATH`: Path to the LightGBM model file (default: `../triage_multimodal_model.txt`).
- `TRIAGE_PYTHON_URL`: URL to the Python sidecar (default: `http://localhost:8000`).
- `TRIAGE_AUDIT_DB`: Path for the SQLite audit trail database (default: `../triage_audit.db`).

### Python Sidecar
- `TRIAGE_DATA_DIR`: Directory containing CSV/NPY artifacts for RAG (default: `.`).
- `GEMINI_API_KEY`: *(Optional)* API key for Google Gemini 2.5 Flash used in clinical recommendations.

### Next.js Frontend
- `NEXT_PUBLIC_RUST_API`: URL to the Rust inference backend (default: `http://localhost:3001`).
- `NEXT_PUBLIC_PYTHON_API`: URL to the Python preprocessing service (default: `http://localhost:8000`).

## Local Development Setup

While you can start all services at once using `./startup.sh`, it is usually better to run them individually in separate terminal windows for development so you can monitor their logs and utilize hot-reloading.

### 1. Python Preprocessing Service (Port 8000)
Handles ML preprocessing (ClinicalBERT, ResNet-50) and the RAG pipeline.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server with auto-reload
uvicorn preprocessing_service:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Rust Inference Backend (Port 3001)
Handles API routing, audit logging, and LightGBM inference.

```bash
cd backend

# Build and run
cargo run
```

### 3. Next.js Frontend (Port 3000)
The clinical Head-Up Display (HUD).

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

## Testing Your Setup

To verify all services are running and communicating correctly, execute the smoke test script from the repository root:

```bash
./smoke.sh
```
This script checks Python service health, Rust backend health, frontend availability, and Rust → Python cross-service connectivity.

## Degraded Modes & Optional Features

The system is designed to degrade gracefully if certain ML artifacts are missing, reducing friction when you only need to work on specific parts of the codebase:
- **Missing LightGBM Model (`triage_multimodal_model.txt`)**: The Rust backend will start in *Degraded Mode*. The `/predict` endpoint will return an HTTP 503, but `/next-steps` will fall back to a rule-based physiological heuristic to estimate the ESI level.
- **Missing RAG Data (`triage_master_multimodal.csv`, `clinicalbert_embeddings_768d.npy`)**: The Python service will start up, but the `/rag` and `/rag-stream` hybrid retrieval endpoints will be unavailable.
- **Missing `GEMINI_API_KEY`**: The RAG engine's retrieval pipeline (ChromaDB) will still fetch and rank similar historical patients, but the Gemini text generation phase will return a placeholder fallback message.
