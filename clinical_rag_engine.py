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

BASE_DIR = "/content/drive/MyDrive/frostbyte"
# For local testing:
# BASE_DIR = "."


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


# %% — Cell 5: Retrieval Function
# ============================================================


def retrieve_similar_patients(complaint_text, k=5):
    """
    Given a chief complaint, find the K most similar historical patients
    using cosine similarity on ClinicalBERT embeddings.
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

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    similar_patients = []
    for i in range(len(results["ids"][0])):
        patient = results["metadatas"][0][i].copy()
        patient["complaint"] = results["documents"][0][i]
        patient["similarity"] = (
            1 - results["distances"][0][i]
        )  # Cosine sim = 1 - distance
        similar_patients.append(patient)

    return similar_patients


# Quick test
print("Testing retrieval for 'Severe chest pain'…")
test_results = retrieve_similar_patients("Severe chest pain", k=3)
for i, p in enumerate(test_results):
    print(
        f"  {i + 1}. [{p['complaint']}] ESI={p['target_esi']} HR={p['heart_rate']} SpO2={p['spo2']} Sim={p['similarity']:.4f}"
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
    # Step 1: Retrieve
    similar = retrieve_similar_patients(complaint, k=k)

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

    print("📋 RETRIEVED SIMILAR CASES:")
    for j, case in enumerate(similar_cases, 1):
        print(
            f"   {j}. [{case['complaint']}] ESI={case['target_esi']} (sim={case['similarity']:.3f})"
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
