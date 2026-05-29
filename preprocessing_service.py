"""
🏥 Multimodal Triage — Python Preprocessing Microservice

FastAPI sidecar for the Rust backend. Handles ML preprocessing that
requires HuggingFace models (no Rust equivalents exist):

  POST /embed      → ClinicalBERT text embeddings + ResNet-50 image features
  POST /shap       → Real-time per-patient SHAP explainability values
  POST /rag        → ChromaDB retrieval + Gemini generation
  GET  /rag-stream → Server-Sent Events streaming for Gemini RAG output

Run:  uvicorn preprocessing_service:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import warnings


import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

warnings.filterwarnings("ignore")

# Fix Windows console encoding for Unicode emoji logging
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Auto-detect paths ─────────────────────────────────────────
def _resolve_base_dir():
    env = os.environ.get("TRIAGE_DATA_DIR")
    if env:
        return env
    if os.path.exists("/content/drive/MyDrive/triage_data"):
        return "/content/drive/MyDrive/triage_data"
    return "."


BASE_DIR = _resolve_base_dir()

# ── FastAPI App ──────────────────────────────────────────────

app = FastAPI(
    title="Triage Preprocessing Service",
    description="ClinicalBERT + ResNet-50 embedding extraction and RAG pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state (loaded once at startup) ──────────────

bert_model = None
bert_tokenizer = None
text_pca = None
resnet_model = None
image_pca = None
image_preprocess = None
device = None

# SHAP + LightGBM
lgb_model = None
shap_explainer = None

# RAG components
chroma_collection = None
gemini_model = None
VITALS_STATS = None

# Hybrid retrieval tuning
ALPHA = 0.5
ESI_BOOST = 0.15
TEXT_POOL_SIZE = 50
VITALS_POOL_SIZE = 200


# ── Pydantic Models ──────────────────────────────────────────


class EmbedRequest(BaseModel):
    complaint: str
    image_base64: Optional[str] = None


class EmbedResponse(BaseModel):
    text_features: List[float]
    image_features: List[float]


class PatientVitals(BaseModel):
    age: float
    heart_rate: float
    resp_rate: float
    spo2: float
    temp_f: float
    systolic_bp: float
    pain_scale: float


class RagRequest(BaseModel):
    complaint: str
    vitals: PatientVitals
    predicted_esi: int


class SimilarCase(BaseModel):
    complaint: str
    target_esi: int
    similarity: float
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    text_similarity: Optional[float] = None
    vitals_similarity: Optional[float] = None
    source: Optional[str] = None
    flag_high_risk: Optional[int] = None


class RagResponse(BaseModel):
    recommendation: str
    similar_cases: List[SimilarCase]


class ShapRequest(BaseModel):
    feature_vector: List[float]
    predicted_class: int


class ShapFeature(BaseModel):
    name: str
    value: float
    shap_value: float


class ShapResponse(BaseModel):
    base_value: float
    features: List[ShapFeature]
    predicted_class: int
    prediction_label: str


# ── Startup: Load all models ─────────────────────────────────


@app.on_event("startup")
async def load_models():
    global bert_model, bert_tokenizer, text_pca, device
    global resnet_model, image_pca, image_preprocess
    global chroma_collection, gemini_model
    global lgb_model, shap_explainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- ClinicalBERT ----
    print("Loading ClinicalBERT...")
    from transformers import AutoModel, AutoTokenizer

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print("✅ ClinicalBERT loaded")

    # ---- PCA for Text (refit from dataset if available) ----
    try:
        from sklearn.decomposition import PCA
        import pandas as pd

        csv_path = os.path.join(BASE_DIR, "triage_master_multimodal.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            complaints = df["chief_complaint"].fillna("Unknown").tolist()

            # Re-extract embeddings for PCA fitting
            print("Fitting text PCA from dataset...")
            all_embs = []
            BATCH_SIZE = 32
            with torch.no_grad():
                for i in range(0, len(complaints), BATCH_SIZE):
                    batch = complaints[i : i + BATCH_SIZE]
                    tokens = bert_tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=64,
                        return_tensors="pt",
                    ).to(device)
                    outputs = bert_model(**tokens)
                    cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embs.append(cls)
            emb_matrix = np.vstack(all_embs)
            text_pca = PCA(n_components=10, random_state=42)
            text_pca.fit(emb_matrix)
            print(
                f"✅ Text PCA fitted (variance: {text_pca.explained_variance_ratio_.sum():.4f})"
            )
        else:
            print(
                f"⚠️  Dataset not found at {csv_path}. Text PCA unavailable — will return raw CLS features."
            )
    except Exception as e:
        print(f"⚠️  Text PCA setup failed: {e}")

    # ---- ResNet-50 ----
    print("Loading ResNet-50...")
    import torchvision.models as models
    import torchvision.transforms as transforms

    weights = models.ResNet50_Weights.DEFAULT
    resnet_model = models.resnet50(weights=weights)
    resnet_model.fc = torch.nn.Identity()
    resnet_model = resnet_model.to(device).eval()

    image_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("✅ ResNet-50 loaded")

    # ---- PCA for Images (refit from dataset if available) ----
    try:
        from sklearn.decomposition import PCA as PCAI
        from PIL import Image

        csv_path = os.path.join(BASE_DIR, "triage_master_multimodal.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            real_embs = []
            for path in df["image_path"]:
                if path != "None" and not pd.isna(path):
                    try:
                        full_path = os.path.join(BASE_DIR, path)
                        img = Image.open(full_path).convert("RGB")
                        img_tensor = image_preprocess(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = resnet_model(img_tensor).cpu().numpy().flatten()
                        real_embs.append(emb)
                    except Exception:
                        pass
            if len(real_embs) >= 5:
                image_pca = PCAI(n_components=5, random_state=42)
                image_pca.fit(np.array(real_embs))
                print(f"✅ Image PCA fitted on {len(real_embs)} real images")
            else:
                print(f"⚠️  Only {len(real_embs)} real images found. Image PCA skipped.")
    except Exception as e:
        print(f"⚠️  Image PCA setup failed: {e}")

    # ---- ChromaDB + Gemini (for RAG) ----
    try:
        import chromadb

        npy_path = os.path.join(BASE_DIR, "clinicalbert_embeddings_768d.npy")
        csv_path = os.path.join(BASE_DIR, "triage_master_multimodal.csv")

        if os.path.exists(npy_path) and os.path.exists(csv_path):
            embeddings_768d = np.load(npy_path)
            df = pd.read_csv(csv_path)

            chroma_client = chromadb.Client()
            try:
                chroma_client.delete_collection("triage_patients")
            except Exception:
                pass

            chroma_collection = chroma_client.create_collection(
                name="triage_patients",
                metadata={"hnsw:space": "cosine"},
            )

            tabular_cols = [
                "age",
                "heart_rate",
                "resp_rate",
                "spo2",
                "temp_f",
                "systolic_bp",
                "pain_scale",
            ]
            BATCH = 100
            for i in range(0, len(df), BATCH):
                end = min(i + BATCH, len(df))
                docs = [
                    str(row["chief_complaint"]) for _, row in df.iloc[i:end].iterrows()
                ]
                metas = []
                for _, row in df.iloc[i:end].iterrows():
                    metas.append(
                        {
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
                    )
                ids = [f"patient_{j}" for j in range(i, end)]
                chroma_collection.add(
                    documents=docs,
                    embeddings=embeddings_768d[i:end].tolist(),
                    metadatas=metas,
                    ids=ids,
                )
            print(f"✅ ChromaDB populated: {chroma_collection.count()} patients")

            global VITALS_STATS
            VITALS_KEYS = [
                "heart_rate",
                "resp_rate",
                "spo2",
                "temp_f",
                "systolic_bp",
                "pain_scale",
            ]
            VITALS_STATS = {
                col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
                for col in VITALS_KEYS
            }
            print("📊 Vitals normalization stats precomputed for hybrid retrieval")
        else:
            print("⚠️  RAG data not found. /rag endpoint will be unavailable.")

        # Gemini
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            print("✅ Gemini configured")
        else:
            print("⚠️  GEMINI_API_KEY not set. /rag endpoint will return placeholder.")

    except Exception as e:
        print(f"⚠️  RAG setup failed: {e}")

    # ---- LightGBM + SHAP (for real-time explainability) ----
    try:
        import joblib
        import pandas as pd

        try:
            # Fix for older shap versions with newer pandas
            if not hasattr(pd.core.strings, "StringMethods"):
                pd.core.strings.StringMethods = pd.core.strings.accessor.StringMethods
        except Exception:
            pass
        import shap

        model_path_txt = os.path.join(BASE_DIR, "triage_multimodal_model.txt")
        model_path_pkl = os.path.join(BASE_DIR, "triage_multimodal_model.pkl")

        if os.path.exists(model_path_txt):
            import lightgbm as lgb

            lgb_model = lgb.Booster(model_file=model_path_txt)
            shap_explainer = shap.TreeExplainer(lgb_model)
            print("✅ LightGBM + SHAP explainer loaded (from .txt)")
        elif os.path.exists(model_path_pkl):
            lgb_model = joblib.load(model_path_pkl)
            shap_explainer = shap.TreeExplainer(lgb_model)
            print("✅ LightGBM + SHAP explainer loaded (from .pkl)")
        else:
            print("⚠️  No LightGBM model found. /shap endpoint unavailable.")
    except Exception as e:
        print(f"⚠️  SHAP setup failed: {e}")

    print("\n🧊 Preprocessing service ready!")


# ── Hybrid Retrieval Helpers ───────────────────────────────────────


def _compute_vitals_similarity(query_vitals: dict, candidate_meta: dict) -> float:
    """Compute normalized similarity between query vitals and a historical patient."""
    if VITALS_STATS is None:
        return 0.0
    dist_sq = 0.0
    for key in VITALS_STATS:
        q_val = float(query_vitals.get(key, 0))
        c_val = float(candidate_meta.get(key, 0))
        std = VITALS_STATS[key]["std"]
        mean = VITALS_STATS[key]["mean"]
        if std > 0:
            q_z = (q_val - mean) / std
            c_z = (c_val - mean) / std
            dist_sq += (q_z - c_z) ** 2
    dist = dist_sq**0.5
    return 1.0 / (1.0 + dist)


def _fetch_vitals_candidates(
    query_vitals: dict, predicted_esi: int, pool_size: int
) -> list:
    """Path B: Retrieve candidates by ESI metadata filter (±1), ranked by vitals similarity."""
    if chroma_collection is None or VITALS_STATS is None:
        return []

    esi_targets = [
        esi for esi in range(max(1, predicted_esi - 1), min(5, predicted_esi + 1) + 1)
    ]

    if len(esi_targets) == 1:
        where_filter = {"target_esi": esi_targets[0]}
    else:
        where_filter = {"target_esi": {"$in": esi_targets}}

    try:
        results = chroma_collection.get(
            where=where_filter,
            limit=pool_size,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    candidates = []
    for i in range(len(results["ids"])):
        patient = results["metadatas"][i].copy()
        patient["complaint"] = results["documents"][i]
        patient["text_similarity"] = 0.0
        patient["vitals_similarity"] = _compute_vitals_similarity(query_vitals, patient)
        patient["_source"] = "vitals"
        candidates.append(patient)

    candidates.sort(key=lambda x: x["vitals_similarity"], reverse=True)
    return candidates[:pool_size]


def retrieve_similar_patients_hybrid(
    complaint_text: str, query_vitals: dict, predicted_esi: int, k: int = 5
) -> list:
    """
    Dual-path hybrid retrieval:
      Path A — Text: Broad ClinicalBERT cosine similarity fetch
      Path B — Vitals/ESI: Metadata-filtered retrieval ranked by physiology

    Returns list of dicts with: complaint, target_esi, similarity, text_similarity,
    vitals_similarity, source, flag_high_risk, heart_rate, spo2, resp_rate, etc.
    """
    if bert_model is None or chroma_collection is None:
        return []

    if VITALS_STATS is None:
        text_k = k
        results = chroma_collection.query(
            query_embeddings=[query_vitals.get("_query_emb", [0] * 768)],
            n_results=text_k,
            include=["documents", "metadatas", "distances"],
        )
        similar_cases = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            complaint = results["documents"][0][i]
            similarity = 1 - results["distances"][0][i]
            similar_cases.append(
                {
                    "complaint": complaint,
                    "target_esi": meta["target_esi"],
                    "similarity": round(similarity, 4),
                    "text_similarity": round(similarity, 4),
                    "vitals_similarity": 0.0,
                    "source": "text",
                    "flag_high_risk": meta.get("flag_high_risk", 0),
                    "heart_rate": meta.get("heart_rate"),
                    "spo2": meta.get("spo2"),
                }
            )
        return similar_cases[:k]

    if bert_tokenizer is None:
        return []

    with torch.no_grad():
        tokens = bert_tokenizer(
            [complaint_text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs = bert_model(**tokens)
        query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    candidates_by_id = {}

    text_k = TEXT_POOL_SIZE if query_vitals else k
    try:
        results = chroma_collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=text_k,
            include=["documents", "metadatas", "distances"],
        )

        for i in range(len(results["ids"][0])):
            pid = results["ids"][0][i]
            patient = results["metadatas"][0][i].copy()
            patient["complaint"] = results["documents"][0][i]
            patient["text_similarity"] = 1 - results["distances"][0][i]
            patient["vitals_similarity"] = 0.0
            patient["_source"] = "text"
            patient["_id"] = pid
            candidates_by_id[pid] = patient
    except Exception:
        pass

    if predicted_esi is not None:
        vitals_candidates = _fetch_vitals_candidates(
            query_vitals, predicted_esi, VITALS_POOL_SIZE
        )
        for vc in vitals_candidates:
            pid = vc.get("patient_id", "")
            vc_id = f"patient_vitals_{pid}"
            if (
                vc_id not in candidates_by_id
                and f"patient_{pid}" not in candidates_by_id
            ):
                vc["_id"] = vc_id
                candidates_by_id[vc_id] = vc
            else:
                existing_key = vc_id if vc_id in candidates_by_id else f"patient_{pid}"
                if existing_key in candidates_by_id:
                    candidates_by_id[existing_key]["vitals_similarity"] = vc[
                        "vitals_similarity"
                    ]
                    candidates_by_id[existing_key]["_source"] = "both"

    all_candidates = list(candidates_by_id.values())

    for c in all_candidates:
        source = c.get("_source", "text")

        if source == "text" and c.get("vitals_similarity", 0.0) == 0.0:
            c["vitals_similarity"] = _compute_vitals_similarity(query_vitals, c)

        if source == "vitals":
            combined = c["vitals_similarity"]
        else:
            combined = (
                ALPHA * c["text_similarity"] + (1 - ALPHA) * c["vitals_similarity"]
            )

        if predicted_esi is not None:
            candidate_esi = int(c.get("target_esi", 3))
            if abs(candidate_esi - predicted_esi) <= 1:
                combined *= 1 + ESI_BOOST

        c["similarity"] = combined

    all_candidates.sort(key=lambda x: x["similarity"], reverse=True)

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

    while len(diverse_results) < k and overflow:
        diverse_results.append(overflow.pop(0))

    formatted = []
    for c in diverse_results[:k]:
        formatted.append(
            {
                "complaint": c["complaint"],
                "target_esi": c["target_esi"],
                "similarity": round(c["similarity"], 4),
                "text_similarity": round(c.get("text_similarity", 0.0), 4),
                "vitals_similarity": round(c.get("vitals_similarity", 0.0), 4),
                "source": c.get("_source", "text"),
                "flag_high_risk": c.get("flag_high_risk", 0),
                "heart_rate": c.get("heart_rate"),
                "spo2": c.get("spo2"),
            }
        )

    return formatted


# ── Endpoints ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    chroma_count = 0
    if chroma_collection is not None:
        try:
            chroma_count = chroma_collection.count()
        except Exception:
            pass

    return {
        "status": "ok",
        "bert_loaded": bert_model is not None,
        "resnet_loaded": resnet_model is not None,
        "shap_loaded": shap_explainer is not None,
        "rag_available": chroma_collection is not None and gemini_model is not None,
        "hybrid_retrieval": VITALS_STATS is not None,
        "chroma_patients": chroma_count,
        "device": str(device) if device else "unknown",
        "startup_complete": True,
    }


@app.get("/ready")
async def ready():
    """
    Readiness probe with connectivity checks.
    Returns detailed status for demo verification.
    """
    import httpx

    rust_status = "unknown"
    rust_healthy = False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            rust_resp = await client.get("http://localhost:3001/health")
            if rust_resp.status_code == 200:
                rust_data = rust_resp.json()
                rust_status = rust_data.get("status", "unknown")
                rust_healthy = rust_data.get("model_loaded", False)
    except Exception as e:
        rust_status = f"unreachable: {str(e)[:50]}"

    overall_ready = (
        bert_model is not None
        and chroma_collection is not None
        and gemini_model is not None
    )

    return {
        "ready": overall_ready,
        "python_ready": bert_model is not None and chroma_collection is not None,
        "gemini_configured": gemini_model is not None,
        "rust_backend": rust_status,
        "rust_model_loaded": rust_healthy,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """Extract ClinicalBERT text features + ResNet-50 image features."""
    if bert_model is None:
        raise HTTPException(status_code=503, detail="ClinicalBERT not loaded")

    # ── Text embedding ──
    with torch.no_grad():
        tokens = bert_tokenizer(
            [req.complaint],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs = bert_model(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    # Apply PCA if available
    if text_pca is not None:
        text_features = text_pca.transform(cls_emb.reshape(1, -1)).flatten().tolist()
    else:
        # Fallback: return first 10 components of raw embedding
        text_features = cls_emb[:10].tolist()

    # ── Image embedding ──
    image_features = [0.0] * 5  # Default: no image

    if req.image_base64 and resnet_model is not None:
        try:
            import base64
            from io import BytesIO
            from PIL import Image

            image_bytes = base64.b64decode(req.image_base64)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_tensor = image_preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = resnet_model(img_tensor).cpu().numpy().flatten()

            if image_pca is not None:
                image_features = (
                    image_pca.transform(emb.reshape(1, -1)).flatten().tolist()
                )
            else:
                image_features = emb[:5].tolist()
        except Exception as e:
            print(f"⚠️  Image processing failed for base64 input: {e}")

    return EmbedResponse(text_features=text_features, image_features=image_features)


@app.post("/rag", response_model=RagResponse)
async def rag(req: RagRequest):
    """RAG pipeline: retrieve similar patients + generate Gemini recommendation."""
    if chroma_collection is None:
        raise HTTPException(
            status_code=503, detail="ChromaDB not initialized. Ensure data files exist."
        )

    query_vitals = {
        "heart_rate": req.vitals.heart_rate,
        "resp_rate": req.vitals.resp_rate,
        "spo2": req.vitals.spo2,
        "temp_f": req.vitals.temp_f,
        "systolic_bp": req.vitals.systolic_bp,
        "pain_scale": req.vitals.pain_scale,
    }

    similar_results = retrieve_similar_patients_hybrid(
        complaint_text=req.complaint,
        query_vitals=query_vitals,
        predicted_esi=req.predicted_esi,
        k=5,
    )

    similar_cases = []
    context_lines = []
    for r in similar_results:
        similar_cases.append(
            SimilarCase(
                complaint=r["complaint"],
                target_esi=r["target_esi"],
                similarity=r["similarity"],
                text_similarity=r.get("text_similarity"),
                vitals_similarity=r.get("vitals_similarity"),
                source=r.get("source"),
                flag_high_risk=r.get("flag_high_risk"),
                heart_rate=r.get("heart_rate"),
                spo2=r.get("spo2"),
            )
        )

        context_lines.append(
            f"Historical Patient:\n"
            f'  Complaint: "{r["complaint"]}"\n'
            f"  Vitals: HR={r.get('heart_rate')}, SpO2={r.get('spo2')}%\n"
            f"  ESI Level: {r['target_esi']} | "
            f"High Risk: {'Yes' if r.get('flag_high_risk') else 'No'}\n"
        )

    # ── Generate with Gemini ──
    if gemini_model is not None and similar_cases:
        historical_context = "\n".join(context_lines)
        v = req.vitals
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
  Chief Complaint: "{req.complaint}"
  Age: {v.age}
  Vitals: HR={v.heart_rate}, RR={v.resp_rate}, SpO2={v.spo2}%,
          Temp={v.temp_f}°F, SBP={v.systolic_bp}, Pain={v.pain_scale}/10
  AI Triage Prediction: ESI {req.predicted_esi}

Based ONLY on the similar historical cases above, provide:
1. IMMEDIATE TRIAGE ACTIONS (what should happen in the next 5 minutes)
2. RECOMMENDED ASSESSMENTS (tests/evaluations to consider)
3. CLINICAL REASONING (why these actions, based on the similar cases)

Keep your response concise and actionable. Format with clear headers."""

        try:
            response = gemini_model.generate_content(prompt)
            recommendation = response.text
        except Exception as e:
            recommendation = f"⚠️ Gemini generation failed: {str(e)}. Please review similar cases manually."
    elif not similar_cases:
        recommendation = (
            "⚠️ Could not retrieve similar cases. Check that ChromaDB is populated."
        )
    else:
        recommendation = (
            "⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable.\n"
            "Similar historical cases have been retrieved — review them manually for clinical guidance."
        )

    return RagResponse(recommendation=recommendation, similar_cases=similar_cases)


@app.post("/shap", response_model=ShapResponse)
async def shap_explain(req: ShapRequest):
    """Compute real-time SHAP values for a single patient's 22-feature vector."""
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded.")

    feature_names = [
        "age",
        "heart_rate",
        "resp_rate",
        "spo2",
        "temp_f",
        "systolic_bp",
        "pain_scale",
        "text_feat_0",
        "text_feat_1",
        "text_feat_2",
        "text_feat_3",
        "text_feat_4",
        "text_feat_5",
        "text_feat_6",
        "text_feat_7",
        "text_feat_8",
        "text_feat_9",
        "img_feat_0",
        "img_feat_1",
        "img_feat_2",
        "img_feat_3",
        "img_feat_4",
    ]

    esi_labels = [
        "ESI 1 (Resuscitation)",
        "ESI 2 (Emergent)",
        "ESI 3 (Urgent)",
        "ESI 4 (Less Urgent)",
        "ESI 5 (Non-Urgent)",
    ]

    features = np.array(req.feature_vector).reshape(1, -1)
    shap_values = shap_explainer.shap_values(features)

    # SHAP output: list of arrays (one per class) or 3D array
    cls = req.predicted_class  # 0-indexed
    if isinstance(shap_values, list):
        sv = shap_values[cls][0]  # (n_features,)
        base = float(shap_explainer.expected_value[cls])
    else:
        sv = shap_values[0, :, cls]
        base = float(shap_explainer.expected_value[cls])

    shap_features = []
    for i, name in enumerate(feature_names):
        shap_features.append(
            ShapFeature(
                name=name,
                value=round(float(features[0, i]), 4),
                shap_value=round(float(sv[i]), 6),
            )
        )

    # Sort by absolute SHAP value (most impactful first)
    shap_features.sort(key=lambda f: abs(f.shap_value), reverse=True)

    return ShapResponse(
        base_value=round(base, 6),
        features=shap_features,
        predicted_class=cls,
        prediction_label=esi_labels[cls] if cls < len(esi_labels) else "Unknown",
    )


@app.post("/rag-stream")
async def rag_stream(req: RagRequest):
    """SSE streaming endpoint — streams Gemini RAG response token by token."""
    if chroma_collection is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized.")
    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini not configured.")

    query_vitals = {
        "heart_rate": req.vitals.heart_rate,
        "resp_rate": req.vitals.resp_rate,
        "spo2": req.vitals.spo2,
        "temp_f": req.vitals.temp_f,
        "systolic_bp": req.vitals.systolic_bp,
        "pain_scale": req.vitals.pain_scale,
    }

    similar_results = retrieve_similar_patients_hybrid(
        complaint_text=req.complaint,
        query_vitals=query_vitals,
        predicted_esi=req.predicted_esi,
        k=5,
    )

    context_lines = []
    similar_cases_json = []
    for r in similar_results:
        similar_cases_json.append(
            {
                "complaint": r["complaint"],
                "target_esi": r["target_esi"],
                "similarity": r["similarity"],
                "text_similarity": r.get("text_similarity"),
                "vitals_similarity": r.get("vitals_similarity"),
                "source": r.get("source"),
                "flag_high_risk": r.get("flag_high_risk"),
            }
        )
        context_lines.append(
            f"Historical Patient:\n"
            f'  Complaint: "{r["complaint"]}"\n'
            f"  Vitals: HR={r.get('heart_rate')}, SpO2={r.get('spo2')}%\n"
            f"  ESI Level: {r['target_esi']} | "
            f"High Risk: {'Yes' if r.get('flag_high_risk') else 'No'}\n"
        )

    v = req.vitals
    prompt = f"""You are a clinical decision support assistant for emergency department triage.
You provide evidence-grounded suggestions based ONLY on similar historical patient cases.

CRITICAL SAFETY RULES:
- NEVER prescribe specific medications or dosages
- NEVER make definitive diagnoses
- ALWAYS recommend physician confirmation

SIMILAR HISTORICAL CASES FROM OUR DATABASE:
{chr(10).join(context_lines)}

NEW PATIENT PRESENTING NOW:
  Chief Complaint: "{req.complaint}"
  Age: {v.age}
  Vitals: HR={v.heart_rate}, RR={v.resp_rate}, SpO2={v.spo2}%,
          Temp={v.temp_f}°F, SBP={v.systolic_bp}, Pain={v.pain_scale}/10
  AI Triage Prediction: ESI {req.predicted_esi}

Based ONLY on the similar historical cases above, provide:
1. IMMEDIATE TRIAGE ACTIONS 
2. RECOMMENDED ASSESSMENTS
3. CLINICAL REASONING 

STRICT FORMATTING RULES:
- NEVER write long paragraphs.
- ALWAYS use bulleted lists (`-`) for every point.
- Keep bullet points to 1-2 sentences maximum.
- Use bold text (`**`) to emphasize key medical terms.
- Use horizontal rules (`---`) between the three main sections.
Keep your response concise, structured, and highly readable."""

    import json

    async def event_generator():
        yield f"data: {json.dumps({'similar_cases': similar_cases_json})}\n\n"

        try:
            response = gemini_model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
        except Exception as e:
            err_msg = f"\n\n**Error:** {str(e)}"
            yield f"data: {json.dumps({'token': err_msg})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
