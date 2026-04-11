# Skill: Triage Runtime

Use this skill when touching live runtime behavior.

## Live single-patient path

1. frontend sends patient JSON to Rust `/predict`
2. Rust calls Python `/embed`
3. Rust runs LightGBM FFI
4. Rust calls Python `/shap`
5. frontend separately streams RAG from Python `/rag-stream`

## Live batch path

1. frontend MCI mode sends a batch to Rust `/batch-predict`
2. Rust runs repeated prediction flow
3. frontend renders sorted triage table

## Live trust path

1. frontend shows confidence + SHAP
2. clinician can submit override
3. Rust writes override to SQLite audit trail

## Common pitfalls

- changing field names without updating all layers
- improving a standalone script but not the live sidecar
- forgetting docs when contract or behavior changes
