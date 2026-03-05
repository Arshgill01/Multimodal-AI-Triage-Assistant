# %% [markdown]
# # 🧊 Frostbyte — ClinicalBERT Text Embeddings Pipeline
#
# **Objective:** Extract semantic embeddings from `chief_complaint` using
# Bio_ClinicalBERT, compress via PCA to 10 features, and save an enriched CSV
# for the downstream late-fusion meta-model.
#
# **Expected runtime on Colab T4:** ~2-3 minutes for 1,197 rows.

# %% — Cell 1: Environment & Google Drive Mount
# ============================================================
# Install missing packages (Colab already has torch & sklearn).
# Run this cell FIRST after opening the notebook.
# ============================================================

# !pip install -q transformers

# Mount Google Drive — uncomment the two lines below when running on Colab.
# from google.colab import drive
# drive.mount('/content/drive')

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 📁 DATA DIRECTORY — auto-detects Colab vs local environment.
#    Override with: export FROSTBYTE_DATA_DIR="/your/path"
# ============================================================
def _resolve_base_dir():
    env = os.environ.get("FROSTBYTE_DATA_DIR")
    if env:
        return env
    if os.path.exists("/content/drive/MyDrive/frostbyte"):
        return "/content/drive/MyDrive/frostbyte"  # Colab
    return "."  # Local

BASE_DIR = _resolve_base_dir()

print("✅ Cell 1 complete — environment ready.")

# %% — Cell 2: Load Data & Sanity Checks
# ============================================================

CSV_PATH = os.path.join(BASE_DIR, "triage_dataset_final.csv")

df = pd.read_csv(CSV_PATH)

# --- Quick sanity checks ---
print(f"Loaded {len(df)} rows  ×  {len(df.columns)} columns")
print(f"\nESI Distribution:\n{df['target_esi'].value_counts().sort_index()}")
print(f"\nRows with images: {(df['image_path'] != 'None').sum()}")
print(f"Unique complaints: {df['chief_complaint'].nunique()}")

# Peek at a few rows
df[["patient_id", "chief_complaint", "target_esi", "image_path"]].head(10)

# %% — Cell 3: ClinicalBERT Text Embeddings + PCA
# ============================================================
#
# Pipeline:
#   1. Tokenize chief_complaint with Bio_ClinicalBERT tokenizer
#   2. Forward-pass in batches of 16 → extract [CLS] embeddings (768-d)
#   3. PCA  768 → 10  to avoid overpowering the 7 tabular vitals
#
# Memory note: With batch_size=16 and max_length=64, peak GPU
# memory is ~1.2 GB — well within the T4's 16 GB budget.
# ============================================================

import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

# ---- 3a. Load model & tokenizer ----
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

print(f"Loading {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print(f"Model loaded on {device}")

# ---- 3b. Extract [CLS] embeddings in batches ----
complaints = df["chief_complaint"].fillna("Unknown").tolist()

BATCH_SIZE = 16
all_embeddings = []

print(f"Extracting embeddings for {len(complaints)} complaints …")

with torch.no_grad():
    for i in range(0, len(complaints), BATCH_SIZE):
        batch_texts = complaints[i : i + BATCH_SIZE]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        outputs = model(**tokens)

        # [CLS] token is at position 0 of the last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

        if (i // BATCH_SIZE) % 20 == 0:
            print(
                f"  Processed {min(i + BATCH_SIZE, len(complaints))}/{len(complaints)}"
            )

embeddings_matrix = np.vstack(all_embeddings)
print(f"\nRaw embedding matrix shape: {embeddings_matrix.shape}")  # (1197, 768)

# ---- 3c. PCA: 768 → 10 principal components ----
N_COMPONENTS = 10

pca = PCA(n_components=N_COMPONENTS, random_state=42)
text_pca = pca.fit_transform(embeddings_matrix)

explained = pca.explained_variance_ratio_
print(f"PCA explained variance per component: {np.round(explained, 4)}")
print(f"Total explained variance:             {explained.sum():.4f}")

# ---- 3d. Append to DataFrame ----
text_cols = [f"text_feat_{i}" for i in range(N_COMPONENTS)]
for idx, col in enumerate(text_cols):
    df[col] = text_pca[:, idx]

print(f"\n✅ Added {N_COMPONENTS} text feature columns to DataFrame.")
print(f"New shape: {df.shape}")

# Quick peek at the new columns
df[["chief_complaint"] + text_cols].head()

# %% — Cell 4: Save Enriched DataFrame & Verify
# ============================================================

OUT_PATH = os.path.join(BASE_DIR, "triage_with_text_features.csv")
df.to_csv(OUT_PATH, index=False)
print(f"💾 Saved enriched CSV to: {OUT_PATH}")

# ---- Verification checks ----
print("\n--- Verification ---")
print(f"Output rows:      {len(df)}")
print(f"Output columns:   {len(df.columns)}")
print(f"Text feat NaNs:   {df[text_cols].isna().sum().sum()}")
print(
    f"Text feat range:  [{df[text_cols].min().min():.4f}, {df[text_cols].max().max():.4f}]"
)
print(f"PCA variance sum: {explained.sum():.4f}")

# Final column listing
print(f"\nAll columns:\n{list(df.columns)}")

print("\n🎉 Text embedding pipeline complete!")
print("Next step → run the Vision (ResNet-50) notebook to add image features.")
