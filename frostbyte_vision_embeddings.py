# %% [markdown]
# # 🧊 Frostbyte — ResNet-50 Vision Embeddings Pipeline
#
# **Objective:** Extract visual feature vectors from mapped Kaggle burn/wound
# images using a pre-trained ResNet-50 (transfer learning), compress via PCA
# to 5 features, and save an enriched CSV for the late-fusion meta-model.
#
# **Prerequisite:** Run `frostbyte_text_embeddings.py` first to produce
# `triage_with_text_features.csv`.
#
# **Expected runtime on Colab T4:** ~20 seconds for 1,197 rows (28 real images).

# %% — Cell 1: Environment & Config
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Mount Google Drive (uncomment on Colab)
# from google.colab import drive
# drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/frostbyte"
# For local testing:
# BASE_DIR = "."

# %% — Cell 2: Load Text-Enriched Dataset
# ============================================================

print("1. Loading text-enriched dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "triage_with_text_features.csv"))
print(f"   Loaded {len(df)} rows × {len(df.columns)} columns")
print(f"   Rows with images: {(df['image_path'] != 'None').sum()}")

# %% — Cell 3: ResNet-50 Feature Extraction + PCA
# ============================================================
#
# Architecture:
#   1. Load pre-trained ResNet-50 from torchvision
#   2. Replace final FC layer with Identity → raw 2048-d embeddings
#   3. Forward-pass each of the 28 mapped images through ResNet
#   4. Zero-pad the remaining ~1,169 rows (missing modality handling)
#   5. PCA fitted on ONLY the 28 real embeddings → 5 components
#      (avoids mean-centering artifacts from the zero-padded rows)
#
# Memory note: Single-image inference, peak GPU ~1 GB.
# ============================================================

print("2. Initializing ResNet-50 (Transfer Learning Mode)...")
weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
resnet.fc = torch.nn.Identity()  # Strip classifier → raw 2048-d embeddings
resnet.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Standard ImageNet preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"3. Extracting Visual Embeddings on {device}...")
image_embeddings = []
has_image_mask = []  # Track which rows have real images
images_found = 0

for path in tqdm(df["image_path"], desc="Processing images"):
    if path == "None" or pd.isna(path):
        image_embeddings.append(np.zeros(2048))
        has_image_mask.append(False)
    else:
        try:
            full_path = os.path.join(BASE_DIR, path)
            img = Image.open(full_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = resnet(img_tensor).cpu().numpy().flatten()
            image_embeddings.append(emb)
            has_image_mask.append(True)
            images_found += 1
        except Exception as e:
            print(f"\n⚠️ Could not load {path}: {e}")
            image_embeddings.append(np.zeros(2048))
            has_image_mask.append(False)

image_embeddings = np.array(image_embeddings)
has_image_mask = np.array(has_image_mask)
print(f"\nExtracted features for {images_found} real images out of {len(df)} rows.")

# ---- PCA: 2048 → 5 (fit ONLY on real images, not zeros) ----
print("4. Compressing visual features via PCA (fit on real images only)...")
N_IMG_COMPONENTS = 5

pca_img = PCA(n_components=N_IMG_COMPONENTS, random_state=42)
pca_img.fit(image_embeddings[has_image_mask])  # Fit on real images only

# Initialize all image features as zeros
img_pca = np.zeros((len(df), N_IMG_COMPONENTS))
# Transform only the real-image rows
img_pca[has_image_mask] = pca_img.transform(image_embeddings[has_image_mask])

img_cols = [f"img_feat_{i}" for i in range(N_IMG_COMPONENTS)]
for i, col in enumerate(img_cols):
    df[col] = img_pca[:, i]

print(f"PCA explained variance (on real images): {np.round(pca_img.explained_variance_ratio_, 4)}")
print(f"Total explained variance: {pca_img.explained_variance_ratio_.sum():.4f}")

# %% — Cell 4: Save & Verify
# ============================================================

OUT_PATH = os.path.join(BASE_DIR, "triage_master_multimodal.csv")
df.to_csv(OUT_PATH, index=False)

print(f"\n--- VISION PIPELINE COMPLETE ---")
print(f"💾 Saved to: {OUT_PATH}")
print(f"Final shape: {df.shape}")
print(f"Zero-image rows (should be {len(df) - images_found}): {(~has_image_mask).sum()}")
print(f"\nFeatures: Vitals(7) + ClinicalBERT(10) + ResNet50(5) = 22 input features")
print("Next step → run frostbyte_late_fusion.py for the meta-model + SHAP.")
