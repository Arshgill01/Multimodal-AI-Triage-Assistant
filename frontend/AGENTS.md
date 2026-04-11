# AGENTS.md

This repository is a full-stack multimodal emergency triage product.

Current live stack:

- `frontend/` = Next.js HUD-style product UI
- `backend/` = Rust / Axum inference and audit backend
- `preprocessing_service.py` = Python / FastAPI embedding + SHAP + RAG sidecar
- offline training/data scripts live at repo root

## Working rules

1. Treat `docs/ai/*.md` as the working source of truth for repo understanding.
2. The README is partially stale. If README and code conflict, trust the code and then update docs.
3. Do not make broad refactors unless the task explicitly requires it.
4. Preserve the current "Obsidian HUD" frontend style unless the task explicitly asks for visual redesign.
5. Do not rename endpoints or request fields casually. This repo has cross-stack coupling.
6. Before changing behavior, inspect the full live path:
   - frontend component
   - frontend config / store
   - Rust route
   - Python endpoint
   - supporting docs
7. Never claim a feature is "live" just because a standalone script exists. Confirm the live runtime path is wired.
8. Never claim tests, builds, or checks passed unless they were actually run.
9. Keep changes small, explicit, and easy to review.
10. Always end with:
    - changed files
    - commands run
    - results
    - open risks / follow-ups

## Repo-specific truths

- Live inference requests use `image_base64`, not `image_path`.
- The frontend already supports:
  - single-patient triage
  - streaming RAG
  - SHAP display
  - clinician override
  - MCI / batch mode
- The strongest hybrid RAG logic exists in `clinical_rag_engine.py`, but the live Python service is simpler right now.
- `backend/src/routes/next_steps.rs` currently uses heuristic ESI fallback instead of the full prediction path.
- Be careful with stale naming around model artifacts and image directories.

## Model behavior assumptions

Assume the active model may be fast but not deeply reliable.
That means:

- restate the task before editing
- list touched files before editing
- prefer bounded changes
- avoid hidden assumptions
- update docs when you change behavior

## Delivery style

Be concrete.
Use file paths.
Use exact endpoint names.
Use exact request / response fields.
Do not speak in vague abstractions when the code already defines the contract.
EOF
