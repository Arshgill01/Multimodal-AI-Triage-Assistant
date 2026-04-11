# Open Risks

## 1. README is stale relative to code
Risk:
- judges or collaborators follow the wrong mental model

Fix:
- update README after behavior / architecture changes stabilize

## 2. Live RAG is weaker than standalone RAG script
Risk:
- best retrieval idea is not actually what judges experience

Fix:
- port hybrid retrieval logic into `preprocessing_service.py`

## 3. `next_steps.rs` uses heuristic ESI fallback
Risk:
- recommendation path may diverge from actual model prediction path

Fix:
- either unify flows or clearly document the current compromise

## 4. Model artifact naming mismatch
Risk:
- startup failure or confusion around `.txt` artifact naming

Fix:
- standardize artifact name and docs

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
- prioritize startup sanity, smoke-check commands, and clear failure states
