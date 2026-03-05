# %% [markdown]
# # 🧊 Frostbyte — Clinical RAG Engine
#
# **Objective:** Build a Retrieval-Augmented Generation pipeline that finds
# historically similar patients and generates evidence-grounded clinical
# next-step recommendations using Google Gemini.
#
# **Prerequisite:** Run the text embedding cell first (we need the raw 768-d
# ClinicalBERT embeddings saved as `clinicalbert_embeddings_768d.npy`).
#
# **Expected runtime:** ~1 minute for vector store build + ~5s per RAG query.

# %% — Cell 1: Install Dependencies
# ============================================================

# !pip install -q chromadb google-generativeai transformers

# %% — Cell 2: Imports & Config
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Mount Google Drive (uncomment on Colab)
# from google.colab import drive
# drive.mount('/content/drive')

# Auto-detect Colab vs local. Override: export FROSTBYTE_DATA_DIR="/your/path"
def _resolve_base_dir():
    env = os.environ.get("FROSTBYTE_DATA_DIR")
    if env:
        return env
    if os.path.exists("/content/drive/MyDrive/frostbyte"):
        return "/content/drive/MyDrive/frostbyte"  # Colab
    return "."  # Local

BASE_DIR = _resolve_base_dir()


# %% — Cell 3: Re-extract Raw 768-d ClinicalBERT Embeddings
# ============================================================
#
# The main pipeline only saved PCA-compressed 10-d features.
# For accurate vector retrieval, we need the FULL 768-d embeddings.
# This cell re-runs the extraction and saves a .npy file.
#
# If you already have `clinicalbert_embeddings_768d.npy` on Drive,
# SKIP this cell — Cell 4 will load it directly.
# ============================================================

import torch
from transformers import AutoModel, AutoTokenizer

df = pd.read_csv(os.path.join(BASE_DIR, "triage_master_multimodal.csv"))

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
print(f"Loading {MODEL_NAME} for raw embedding extraction…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device).eval()

complaints = df["chief_complaint"].fillna("Unknown").tolist()
BATCH_SIZE = 16
all_embeddings = []

print(f"Extracting 768-d embeddings for {len(complaints)} complaints…")
with torch.no_grad():
    for i in range(0, len(complaints), BATCH_SIZE):
        batch = complaints[i : i + BATCH_SIZE]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs = bert_model(**tokens)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls)
        if (i // BATCH_SIZE) % 25 == 0:
            print(f"  {min(i + BATCH_SIZE, len(complaints))}/{len(complaints)}")

embeddings_768d = np.vstack(all_embeddings)
npy_path = os.path.join(BASE_DIR, "clinicalbert_embeddings_768d.npy")
np.save(npy_path, embeddings_768d)
print(f"✅ Saved raw embeddings: {embeddings_768d.shape} → {npy_path}")


# %% — Cell 4: Build ChromaDB Vector Store
# ============================================================

import chromadb

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "triage_master_multimodal.csv"))
embeddings_768d = np.load(os.path.join(BASE_DIR, "clinicalbert_embeddings_768d.npy"))

print(f"Building ChromaDB vector store from {len(df)} patients…")

# Create in-memory ChromaDB instance
chroma_client = chromadb.Client()

# Delete collection if it exists (for re-runs)
try:
    chroma_client.delete_collection("triage_patients")
except:
    pass

collection = chroma_client.create_collection(
    name="triage_patients",
    metadata={"hnsw:space": "cosine"},  # Cosine similarity for text embeddings
)

# Prepare metadata for each patient
tabular_cols = [
    "age",
    "heart_rate",
    "resp_rate",
    "spo2",
    "temp_f",
    "systolic_bp",
    "pain_scale",
]

documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    # Document = the chief complaint text
    documents.append(str(row["chief_complaint"]))

    # Metadata = vitals + ESI (for context retrieval)
    meta = {
        "patient_id": str(row["patient_id"]),
        "age": int(row["age"]),
        "heart_rate": int(row["heart_rate"]),
        "resp_rate": int(row["resp_rate"]),
        "spo2": int(row["spo2"]),
        "temp_f": float(row["temp_f"]),
        "systolic_bp": int(row["systolic_bp"]),
        "pain_scale": int(row["pain_scale"]),
        "target_esi": int(row["target_esi"]),
        "flag_high_risk": int(row["flag_high_risk"]),
    }
    metadatas.append(meta)
    ids.append(f"patient_{idx}")

# Add all patients to the collection (in batches for ChromaDB)
CHROMA_BATCH = 100
for i in range(0, len(documents), CHROMA_BATCH):
    end = min(i + CHROMA_BATCH, len(documents))
    collection.add(
        documents=documents[i:end],
        embeddings=embeddings_768d[i:end].tolist(),
        metadatas=metadatas[i:end],
        ids=ids[i:end],
    )

print(f"✅ ChromaDB populated: {collection.count()} patients indexed")

# Precompute vitals normalization stats (mean/std) for reranking
VITALS_KEYS = ["heart_rate", "resp_rate", "spo2", "temp_f", "systolic_bp", "pain_scale"]
VITALS_STATS = {
    col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
    for col in VITALS_KEYS
}
print("📊 Vitals normalization stats precomputed for hybrid retrieval")


# %% — Cell 5: Dual-Path Hybrid Retrieval Function
# ============================================================
#
# Path A: Text retrieval — ClinicalBERT cosine similarity (broad fetch)
# Path B: Vitals/ESI retrieval — ChromaDB metadata filter by ESI ±1,
#          ranked purely by physiological similarity
#
# Both pools are merged, deduplicated, scored, and diversity-filtered.
# This ensures that even when text embeddings fail (e.g., "Unresponsive"
# maps to "Fever and cough"), the vitals path independently retrieves
# physiologically relevant patients.
# ============================================================

# Tuning knobs
ALPHA = 0.5          # Text vs vitals weight in combined score (0.5 = equal)
ESI_BOOST = 0.15     # +15% bonus when candidate ESI within ±1 of predicted
TEXT_POOL_SIZE = 50   # Candidates fetched via text path
VITALS_POOL_SIZE = 200 # Candidates fetched via vitals/ESI path


def _compute_vitals_similarity(query_vitals, candidate_meta):
    """
    Compute normalized similarity between query vitals and a historical patient.
    Uses z-score normalized Euclidean distance → converted to a 0-1 similarity.
    """
    dist_sq = 0.0
    for key in VITALS_KEYS:
        q_val = float(query_vitals.get(key, 0))
        c_val = float(candidate_meta.get(key, 0))
        std = VITALS_STATS[key]["std"]
        mean = VITALS_STATS[key]["mean"]
        if std > 0:
            q_z = (q_val - mean) / std
            c_z = (c_val - mean) / std
            dist_sq += (q_z - c_z) ** 2
    # Convert distance → similarity in [0, 1]
    dist = dist_sq ** 0.5
    return 1.0 / (1.0 + dist)


def _fetch_vitals_candidates(query_vitals, predicted_esi, pool_size):
    """
    Path B: Retrieve candidates by ESI metadata filter (±1), then rank
    purely by vitals similarity. This path is completely independent of
    text embeddings — it anchors on physiology, not language.
    """
    # Build ESI filter: predicted ±1, clamped to [1, 5]
    esi_targets = [
        esi for esi in range(max(1, predicted_esi - 1), min(5, predicted_esi + 1) + 1)
    ]

    # ChromaDB where filter for ESI range
    if len(esi_targets) == 1:
        where_filter = {"target_esi": esi_targets[0]}
    else:
        where_filter = {"target_esi": {"$in": esi_targets}}

    # Get all patients in this ESI range (up to pool_size)
    try:
        results = collection.get(
            where=where_filter,
            limit=pool_size,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    # Build candidate list and rank by vitals similarity
    candidates = []
    for i in range(len(results["ids"])):
        patient = results["metadatas"][i].copy()
        patient["complaint"] = results["documents"][i]
        patient["text_similarity"] = 0.0  # No text score for this path
        patient["vitals_similarity"] = _compute_vitals_similarity(query_vitals, patient)
        patient["_source"] = "vitals"
        candidates.append(patient)

    # Sort by vitals similarity (descending)
    candidates.sort(key=lambda x: x["vitals_similarity"], reverse=True)
    return candidates[:pool_size]


def retrieve_similar_patients(complaint_text, query_vitals=None, predicted_esi=None, k=5):
    """
    Dual-path hybrid retrieval:
      Path A — Text: Broad ClinicalBERT cosine similarity fetch
      Path B — Vitals/ESI: Metadata-filtered retrieval ranked by physiology

    Both pools are merged, deduplicated by patient_id, scored with a
    combined metric, ESI-boosted, and diversity-filtered.

    Falls back to pure text retrieval if query_vitals is None.
    """
    # Embed the query complaint
    with torch.no_grad():
        tokens = tokenizer(
            [complaint_text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs = bert_model(**tokens)
        query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    # ── Path A: Text retrieval ─────────────────────────────────
    text_k = TEXT_POOL_SIZE if query_vitals is not None else k
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=text_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates_by_id = {}  # Deduplicate by patient_id
    for i in range(len(results["ids"][0])):
        pid = results["ids"][0][i]
        patient = results["metadatas"][0][i].copy()
        patient["complaint"] = results["documents"][0][i]
        patient["text_similarity"] = 1 - results["distances"][0][i]
        patient["vitals_similarity"] = 0.0
        patient["_source"] = "text"
        patient["_id"] = pid
        candidates_by_id[pid] = patient

    # If no vitals provided, return pure text results (backward compat)
    if query_vitals is None:
        text_only = list(candidates_by_id.values())[:k]
        for c in text_only:
            c["similarity"] = c["text_similarity"]
        return text_only

    # ── Path B: Vitals/ESI retrieval ───────────────────────────
    if predicted_esi is not None:
        vitals_candidates = _fetch_vitals_candidates(
            query_vitals, predicted_esi, VITALS_POOL_SIZE
        )
        for vc in vitals_candidates:
            pid = vc.get("patient_id", "")
            vc_id = f"patient_vitals_{pid}"
            if vc_id not in candidates_by_id and f"patient_{pid}" not in candidates_by_id:
                # New candidate from vitals path
                vc["_id"] = vc_id
                candidates_by_id[vc_id] = vc
            else:
                # Already have this patient from text path — upgrade to "both"
                existing_key = vc_id if vc_id in candidates_by_id else f"patient_{pid}"
                if existing_key in candidates_by_id:
                    candidates_by_id[existing_key]["vitals_similarity"] = vc["vitals_similarity"]
                    candidates_by_id[existing_key]["_source"] = "both"

    # ── Scoring: Path-aware combined score ─────────────────────
    #
    # The scoring adapts based on which path(s) found the candidate:
    #   "both" → full blend:  α * text_sim + (1-α) * vitals_sim  (best case)
    #   "text" → blend after computing vitals_sim on the fly
    #   "vitals" → pure vitals_sim (no text penalty for missing text match)
    #
    # This prevents vitals-only candidates from being crushed by the
    # α * 0.0 penalty when ClinicalBERT put them in a different cluster.
    # ───────────────────────────────────────────────────────────
    all_candidates = list(candidates_by_id.values())

    for c in all_candidates:
        source = c["_source"]

        # Ensure vitals_sim is computed for text-path candidates
        if source == "text" and c["vitals_similarity"] == 0.0:
            c["vitals_similarity"] = _compute_vitals_similarity(query_vitals, c)

        # Path-aware scoring
        if source == "vitals":
            # Vitals-only: use vitals_sim directly (no text penalty)
            combined = c["vitals_similarity"]
        else:
            # "text" or "both": full blend
            combined = ALPHA * c["text_similarity"] + (1 - ALPHA) * c["vitals_similarity"]

        # ESI acuity boost
        if predicted_esi is not None:
            candidate_esi = int(c.get("target_esi", 3))
            if abs(candidate_esi - predicted_esi) <= 1:
                combined *= (1 + ESI_BOOST)

        c["similarity"] = combined

    # Sort by combined score (descending)
    all_candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # ── Diversity filter: deduplicate by complaint ─────────────
    seen_complaints = set()
    diverse_results = []
    overflow = []

    for c in all_candidates:
        complaint_key = c["complaint"].strip().lower()
        if complaint_key not in seen_complaints:
            seen_complaints.add(complaint_key)
            diverse_results.append(c)
        else:
            overflow.append(c)

        if len(diverse_results) >= k:
            break

    # If we couldn't fill k diverse results, pad from overflow
    while len(diverse_results) < k and overflow:
        diverse_results.append(overflow.pop(0))

    return diverse_results[:k]


# Quick test — hybrid retrieval with vitals
print("Testing hybrid retrieval for 'Severe chest pain'…")
test_vitals = {"heart_rate": 120, "resp_rate": 24, "spo2": 91, "temp_f": 98.6, "systolic_bp": 180, "pain_scale": 9}
test_results = retrieve_similar_patients("Severe chest pain", query_vitals=test_vitals, predicted_esi=2, k=3)
for i, p in enumerate(test_results):
    print(
        f"  {i + 1}. [{p['complaint']}] ESI={p['target_esi']} HR={p['heart_rate']} SpO2={p['spo2']} "
        f"TextSim={p['text_similarity']:.3f} VitalsSim={p.get('vitals_similarity', 0):.3f} Combined={p['similarity']:.3f}"
    )


# %% — Cell 6: RAG Generation with Gemini
# ============================================================

import google.generativeai as genai

# ============================================================
# 🔑 SET YOUR GEMINI API KEY HERE
#    Get a free key at: https://aistudio.google.com/apikey
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
# Get a free key at: https://aistudio.google.com/apikey
# Set it via: export GEMINI_API_KEY="your-key-here"  (or Colab Secrets)
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def build_rag_context(patient_vitals, similar_patients):
    """
    Build a grounded clinical context from retrieved similar patients.
    The LLM can ONLY reference data from these patients — no free generation.
    """
    context_lines = []
    for i, p in enumerate(similar_patients, 1):
        context_lines.append(
            f"Historical Patient {i}:\n"
            f'  Complaint: "{p["complaint"]}"\n'
            f"  Vitals: HR={p['heart_rate']}, RR={p['resp_rate']}, "
            f"SpO2={p['spo2']}%, Temp={p['temp_f']}°F, "
            f"SBP={p['systolic_bp']}, Pain={p['pain_scale']}/10\n"
            f"  ESI Level: {p['target_esi']} | High Risk: {'Yes' if p['flag_high_risk'] else 'No'}\n"
        )

    return "\n".join(context_lines)


def generate_clinical_recommendation(complaint, vitals, predicted_esi, k=5):
    """
    Full RAG pipeline:
      1. Retrieve K similar historical patients
      2. Build grounded context
      3. Generate clinical next-step recommendations via Gemini

    Args:
        complaint: Chief complaint text (e.g., "Crushing chest pain")
        vitals: Dict with keys: age, heart_rate, resp_rate, spo2, temp_f, systolic_bp, pain_scale
        predicted_esi: The model's ESI prediction (1-5)
        k: Number of similar patients to retrieve
    """
    # Step 1: Retrieve (hybrid — text + vitals + ESI boost + diversity)
    similar = retrieve_similar_patients(
        complaint, query_vitals=vitals, predicted_esi=predicted_esi, k=k
    )

    # Step 2: Build context
    historical_context = build_rag_context(vitals, similar)

    # Step 3: Generate with guardrails
    prompt = f"""You are a clinical decision support assistant for emergency department triage.
You provide evidence-grounded suggestions based ONLY on similar historical patient cases.

CRITICAL SAFETY RULES:
- NEVER prescribe specific medications or dosages
- NEVER make definitive diagnoses
- ALWAYS recommend physician confirmation
- ONLY suggest actions that align with the similar cases provided
- This is a DECISION SUPPORT tool, not a diagnostic system

SIMILAR HISTORICAL CASES FROM OUR DATABASE:
{historical_context}

NEW PATIENT PRESENTING NOW:
  Chief Complaint: "{complaint}"
  Age: {vitals["age"]}
  Vitals: HR={vitals["heart_rate"]}, RR={vitals["resp_rate"]}, SpO2={vitals["spo2"]}%, 
          Temp={vitals["temp_f"]}°F, SBP={vitals["systolic_bp"]}, Pain={vitals["pain_scale"]}/10
  AI Triage Prediction: ESI {predicted_esi}

Based ONLY on the similar historical cases above, provide:
1. IMMEDIATE TRIAGE ACTIONS (what should happen in the next 5 minutes)
2. RECOMMENDED ASSESSMENTS (tests/evaluations to consider)
3. CLINICAL REASONING (why these actions, based on the similar cases)

Keep your response concise and actionable. Format with clear headers."""

    response = gemini_model.generate_content(prompt)
    return response.text, similar


# %% — Cell 7: Demo — Run RAG on Sample Patients
# ============================================================

print("=" * 60)
print("  🧊 FROSTBYTE CLINICAL RAG ENGINE — LIVE DEMO")
print("=" * 60)

# Demo patients spanning the ESI spectrum
demo_patients = [
    {
        "complaint": "Unresponsive, found on floor",
        "vitals": {
            "age": 72,
            "heart_rate": 145,
            "resp_rate": 38,
            "spo2": 78,
            "temp_f": 101.2,
            "systolic_bp": 72,
            "pain_scale": 0,
        },
        "predicted_esi": 1,
    },
    {
        "complaint": "Crushing chest pain radiating to left arm",
        "vitals": {
            "age": 55,
            "heart_rate": 118,
            "resp_rate": 24,
            "spo2": 92,
            "temp_f": 98.6,
            "systolic_bp": 185,
            "pain_scale": 9,
        },
        "predicted_esi": 2,
    },
    {
        "complaint": "Twisted ankle while running, moderate swelling",
        "vitals": {
            "age": 28,
            "heart_rate": 78,
            "resp_rate": 14,
            "spo2": 99,
            "temp_f": 98.2,
            "systolic_bp": 118,
            "pain_scale": 4,
        },
        "predicted_esi": 4,
    },
]

for i, patient in enumerate(demo_patients):
    print(f"\n{'─' * 60}")
    print(f'  PATIENT {i + 1}: "{patient["complaint"]}"')
    print(f"  Predicted ESI: {patient['predicted_esi']}")
    print(f"{'─' * 60}\n")

    recommendation, similar_cases = generate_clinical_recommendation(
        complaint=patient["complaint"],
        vitals=patient["vitals"],
        predicted_esi=patient["predicted_esi"],
        k=5,
    )

    print("📋 RETRIEVED SIMILAR CASES (Hybrid: Text + Vitals + ESI Boost):")
    for j, case in enumerate(similar_cases, 1):
        txt = case.get('text_similarity', case['similarity'])
        vit = case.get('vitals_similarity', 0)
        print(
            f"   {j}. [{case['complaint']}] ESI={case['target_esi']} "
            f"(combined={case['similarity']:.3f} | text={txt:.3f} vitals={vit:.3f})"
        )

    print(f"\n🤖 AI RECOMMENDATION:\n")
    print(recommendation)
    print()

print("=" * 60)
print("  🎉 RAG ENGINE DEMO COMPLETE")
print("=" * 60)


# %% — Cell 8: Save RAG Artifacts
# ============================================================

# Save embeddings path reference
print("RAG Engine Artifacts:")
print(f"  • ChromaDB: in-memory ({collection.count()} vectors)")
print(
    f"  • Raw embeddings: {os.path.join(BASE_DIR, 'clinicalbert_embeddings_768d.npy')}"
)
print(f"  • Model: Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)")
print(f"  • LLM: Gemini 2.5 Flash")
print(f"\n💡 For production: persist ChromaDB to disk with chromadb.PersistentClient()")
