# Frontend Flow

## Product modes

### Single Patient
Main HUD flow for a single live triage decision.

Primary pieces:
- `TelemetryPane.tsx`
- `AICorePane.tsx`
- `RagIntelligencePane.tsx`

### MCI Mode
Mass casualty simulation flow.

Primary piece:
- `MCIMode.tsx`

## Store shape

Core state lives in `frontend/src/lib/store.ts`.

Important state:
- `currentEsi`
- `analysisPhase`
- `isMciMode`
- `patientData`
- `prediction`
- `ragStream`
- `similarCases`
- `batchResults`

## UX sequence

### Single-patient sequence
1. user edits vitals / complaint / image
2. UI enters phased sequence:
   - extracting
   - routing
   - inferring
   - explainability
   - rag
   - complete
3. frontend calls Rust `/predict`
4. frontend displays:
   - ESI hero
   - confidence badge
   - SHAP chart
5. frontend separately streams RAG from Python `/rag-stream`
6. clinician may override AI decision

## Important UI constraints

- keep the current Obsidian HUD style
- do not replace the product with generic dashboard UI
- preserve visual meaning of ESI colors
- preserve MCI mode as a distinct, high-impact demo surface
- preserve override flow as a human-in-the-loop trust feature

## Product-story strengths already present

- high-drama single-patient hero moment
- real-time explainability
- streaming intelligence panel
- human override
- batch triage demo mode

These are hackathon strengths. Protect them.
