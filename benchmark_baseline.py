"""
🧊 Frostbyte — Python-Only Inference Baseline Server

A pure-Python Flask server that performs the same LightGBM inference
as the Rust backend, used for latency benchmarking comparison.

Run: python benchmark_baseline.py
Serves on: http://localhost:5001/predict
"""

import os
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import torch
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────

def _resolve_base_dir():
    env = os.environ.get("FROSTBYTE_DATA_DIR")
    if env:
        return env
    return "."

BASE_DIR = _resolve_base_dir()

app = Flask(__name__)

# ── Load models at startup ────────────────────────────────────

print("Loading ClinicalBERT...")
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).eval()
device = torch.device("cpu")
bert_model = bert_model.to(device)
print("✅ ClinicalBERT loaded")

# Load PCA
from sklearn.decomposition import PCA

text_pca = None
csv_path = os.path.join(BASE_DIR, "triage_master_multimodal.csv")
if os.path.exists(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    complaints = df["chief_complaint"].fillna("Unknown").tolist()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(complaints), 32):
            batch = complaints[i : i + 32]
            tokens = tokenizer(batch, padding=True, truncation=True,
                             max_length=64, return_tensors="pt").to(device)
            outputs = bert_model(**tokens)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls)
    emb_matrix = np.vstack(all_embs)
    text_pca = PCA(n_components=10, random_state=42)
    text_pca.fit(emb_matrix)
    print("✅ Text PCA fitted")

# Load LightGBM
model_path_txt = os.path.join(BASE_DIR, "triage_multimodal_model(1).txt")
model_path_pkl = os.path.join(BASE_DIR, "triage_multimodal_model.pkl")

lgb_model = None
if os.path.exists(model_path_pkl):
    lgb_model = joblib.load(model_path_pkl)
    print("✅ LightGBM loaded (.pkl)")
elif os.path.exists(model_path_txt):
    lgb_model = lgb.Booster(model_file=model_path_txt)
    print("✅ LightGBM loaded (.txt)")
else:
    print("⚠️  No model found")

ESI_LABELS = [
    "ESI 1 (Resuscitation)", "ESI 2 (Emergent)", "ESI 3 (Urgent)",
    "ESI 4 (Less Urgent)", "ESI 5 (Non-Urgent)",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": lgb_model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single endpoint: takes patient JSON, extracts embeddings,
    runs LightGBM inference, returns ESI — all in Python.
    """
    data = request.json
    t_start = time.perf_counter()

    # ── Tabular features ──
    tabular = [
        data["age"], data["heart_rate"], data["resp_rate"],
        data["spo2"], data["temp_f"], data["systolic_bp"], data["pain_scale"],
    ]

    # ── ClinicalBERT embedding + PCA ──
    with torch.no_grad():
        tokens = tokenizer(
            [data["chief_complaint"]], padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        ).to(device)
        outputs = bert_model(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    if text_pca is not None:
        text_features = text_pca.transform(cls_emb.reshape(1, -1)).flatten().tolist()
    else:
        text_features = cls_emb[:10].tolist()

    # ── Image features (zeros — consistent with Rust pipeline) ──
    image_features = [0.0] * 5

    # ── Assemble 22-feature vector ──
    feature_vector = tabular + text_features + image_features

    # ── LightGBM inference ──
    features_arr = np.array(feature_vector).reshape(1, -1)

    if hasattr(lgb_model, 'predict_proba'):
        probabilities = lgb_model.predict_proba(features_arr)[0].tolist()
    else:
        probabilities = lgb_model.predict(features_arr)[0].tolist()

    predicted_class = int(np.argmax(probabilities))
    predicted_esi = predicted_class + 1

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    return jsonify({
        "predicted_esi": predicted_esi,
        "esi_label": ESI_LABELS[predicted_class],
        "probabilities": probabilities,
        "latency_ms": round(latency_ms, 3),
    })


if __name__ == "__main__":
    print("\n🐍 Python baseline server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
