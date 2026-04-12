# Open Risks

## 1. README is stale relative to code
Risk:
- judges or collaborators follow the wrong mental model

Fix:
- ✅ FIXED: README rewritten to match live product (single-patient triage, hybrid RAG, SHAP, override flow, MCI mode)
- ✅ FIXED: Demo walkthrough section added with sample scenarios

## 2. Live RAG is weaker than standalone RAG script
Risk:
- best retrieval idea is not actually what judges experience

Fix:
- ✅ FIXED: hybrid retrieval now live in `/rag` and `/rag-stream` endpoints
- Now returns: text_similarity, vitals_similarity, source, flag_high_risk

## 3. `next_steps.rs` uses heuristic ESI fallback
Risk:
- recommendation path may diverge from actual model prediction path

Fix:
- Documented in README: `/next-steps` uses vitals-based heuristic when model unavailable (fast fallback for RAG)
- The heuristic mirrors critical thresholds (SpO2 < 85%, SBP < 80 mmHg → ESI 1)
- Full model prediction runs in `/predict` endpoint first, then passes ESI to RAG in production flow
- Status: ACKNOWLEDGED — not blocking for demo, documented in README

## 4. Model artifact naming mismatch
Risk:
- startup failure or confusion around `.txt` artifact naming

Fix:
- ✅ FIXED: standardized to `triage_multimodal_model.txt` in docs and startup scripts

## 5. Image directory mismatch
Risk:
- image-mapping logic and docs may disagree

Fix:
- standardize directory names and docs

## 6. Cross-stack contract fragility
Risk:
- a field rename can silently break frontend, Rust, or Python

Fix:
- update all layers together
- update `docs/ai/live-api-contracts.md` with every contract change

## 7. Demo reliability risk
Risk:
- even a strong system loses if setup is brittle

Fix:
- ✅ FIXED: Added startup.sh, shutdown.sh, smoke.sh for one-command startup and verification
- Added enhanced Python /health and /ready endpoints with detailed status
- Improved frontend error messages in TelemetryPane with actionable guidance
