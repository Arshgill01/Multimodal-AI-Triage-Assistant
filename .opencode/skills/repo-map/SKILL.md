# Skill: Repo Map

Use this skill when a task needs repo grounding before edits.

## What matters

This repo has three live runtime layers:
- Next frontend
- Rust backend
- Python sidecar

And a separate offline pipeline:
- data generation
- embedding generation
- LightGBM training
- standalone RAG experiments
- research neural model

## First files to inspect

- `frontend/src/app/page.tsx`
- `frontend/src/lib/store.ts`
- `frontend/src/lib/config.ts`
- `backend/src/routes/predict.rs`
- `backend/src/routes/batch.rs`
- `backend/src/routes/audit.rs`
- `preprocessing_service.py`

## Warning

Do not assume the README fully reflects current live behavior.
Trust code, then update docs.
