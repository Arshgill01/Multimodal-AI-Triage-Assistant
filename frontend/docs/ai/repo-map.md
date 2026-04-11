# Repo Map

## Top-level structure

- `frontend/`
  - Next.js product UI
  - main page: `frontend/src/app/page.tsx`
  - state: `frontend/src/lib/store.ts`
  - API base URLs: `frontend/src/lib/config.ts`
  - key product panes:
    - `TelemetryPane.tsx`
    - `AICorePane.tsx`
    - `RagIntelligencePane.tsx`
    - `MCIMode.tsx`
    - `OverrideModal.tsx`
    - `ShapWaterfall.tsx`

- `backend/`
  - Rust / Axum backend
  - entrypoint: `backend/src/main.rs`
  - shared state: `backend/src/state.rs`
  - DTOs: `backend/src/models.rs`
  - routes:
    - `health.rs`
    - `predict.rs`
    - `next_steps.rs`
    - `batch.rs`
    - `audit.rs`

- `preprocessing_service.py`
  - Python sidecar
  - loads BERT, ResNet, SHAP, ChromaDB, Gemini
  - exposes `/embed`, `/shap`, `/rag`, `/rag-stream`

- offline data / model pipeline
  - `dataset.py`
  - `build_final_dataset.py`
  - `text_embeddings.py`
  - `vision_embeddings.py`
  - `late_fusion.py`
  - `train_tabular.py`
  - `clinical_rag_engine.py`
  - `pytorch_fusion_model.py`

## Critical live paths

### Single-patient prediction

1. `TelemetryPane.tsx` posts patient JSON to Rust `/predict`
2. Rust route `backend/src/routes/predict.rs`
3. Rust calls Python `/embed`
4. Rust runs LightGBM FFI
5. Rust calls Python `/shap`
6. Rust returns prediction + confidence + SHAP + audit id
7. frontend updates ESI, confidence badge, SHAP chart

### Streaming RAG

1. frontend enters `rag` phase
2. `RagIntelligencePane.tsx` posts to Python `/rag-stream`
3. SSE returns similar cases first, then markdown tokens
4. frontend renders similar-case cards + streamed markdown

### MCI / batch mode

1. `MCIMode.tsx` generates synthetic patient batch
2. posts to Rust `/batch-predict`
3. Rust loops prediction path per patient
4. frontend renders triage table, latency, throughput, ESI distribution

### Clinician override

1. `OverrideModal.tsx` posts override to Rust `/audit/override`
2. Rust updates SQLite audit log
3. frontend updates displayed ESI after submission

## Important note

The README does not fully reflect the current live product.
When in doubt, trust code first, then update docs.
