# System Architecture

## Runtime components

### 1. Frontend

Next.js app presenting a "Frostbyte Obsidian HUD" triage interface.
Responsibilities:

- collect patient vitals, complaint, optional image
- trigger single-patient inference
- trigger MCI batch triage
- render confidence, SHAP, similar cases, streamed action plan
- allow clinician override logging

### 2. Rust backend

Axum server responsible for:

- `/health`
- `/predict`
- `/next-steps`
- `/batch-predict`
- `/audit-log`
- `/audit/override`

Responsibilities:

- orchestrate live inference
- call Python preprocessing service
- run LightGBM via FFI
- compute confidence metrics
- write audit trail to SQLite

### 3. Python sidecar

FastAPI service responsible for:

- `/embed`
- `/shap`
- `/rag`
- `/rag-stream`

Responsibilities:

- ClinicalBERT embeddings
- ResNet image embeddings
- PCA transforms
- SHAP explanations
- ChromaDB retrieval
- Gemini generation / streaming

## Offline pipeline

1. `dataset.py`
   - generate synthetic balanced ESI data

2. `build_final_dataset.py`
   - merge synthetic + MIMIC-style data
   - map image assets

3. `text_embeddings.py`
   - ClinicalBERT 768-d
   - PCA to 10 features

4. `vision_embeddings.py`
   - ResNet 2048-d
   - PCA to 5 features

5. `late_fusion.py`
   - train LightGBM on 22 features
   - save model artifacts
   - generate SHAP artifacts

6. `clinical_rag_engine.py`
   - stronger hybrid retrieval prototype / standalone path

7. `pytorch_fusion_model.py`
   - research prototype, not the deployed production path

## Key architecture truth

This repo contains both:

- the live deployed path
- stronger but partially disconnected prototype logic

A task is only complete when the live path is updated, not when a standalone script is improved in isolation.
