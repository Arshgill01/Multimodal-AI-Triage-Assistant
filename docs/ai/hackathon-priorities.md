# Hackathon Priorities

Goal: maximize odds of a first-place-quality submission.

## Judge-facing priorities

The project should score strongly on:
- innovation / creativity
- technical implementation
- practical impact
- design / usability
- presentation / clarity

## Tier 1: must-finish improvements

### 1. Make the strongest RAG logic live
The standalone hybrid retrieval logic is stronger than the current live path.
Priority: wire the stronger retrieval into the live Python service.

### 2. Remove path mismatches
Unify stale docs, endpoint assumptions, artifact names, and image-directory assumptions.

### 3. Strengthen demo reliability
The best demo loses if it breaks.
Priority:
- clear startup docs
- predictable local run flow
- improved error states
- clean sample walkthrough

### 4. Sharpen project story
Make the repo and pitch clearly show:
- why multimodal matters
- why explainability matters
- why override matters
- why MCI mode matters

## Tier 2: strong bonus upgrades

### 5. Unify next-steps with real prediction
Avoid heuristic divergence if possible.

### 6. Improve audit / trust story
Potential upgrades:
- better audit log viewer
- uncertainty surfacing
- explicit human override rationale visibility

### 7. Improve MCI storytelling
Potential upgrades:
- batch summary cards
- more triage-action sorting
- more obvious operational value

## Tier 3: only do if time remains

- research-model deployment work
- broad retraining experiments
- major redesigns
- speculative refactors with weak demo payoff

## Execution rule

Prefer upgrades that are:
- visible to judges
- easy to demo
- easy to explain
- grounded in current code
