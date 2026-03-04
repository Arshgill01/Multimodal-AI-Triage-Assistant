# %% [markdown]
# # 🧊 Frostbyte — Late-Fusion Meta-Model + SHAP Explainability
#
# **Objective:** Train a LightGBM classifier on the concatenated multimodal
# feature space (7 tabular + 10 text + 5 image = 22 features) and generate
# SHAP explainability plots for the judges.
#
# **Prerequisite:** Run `frostbyte_vision_embeddings.py` first to produce
# `triage_master_multimodal.csv`.
#
# **Expected runtime on Colab T4:** ~30 seconds (training) + ~30 seconds (SHAP).

# %% — Cell 1: Environment & Config
# ============================================================

import os
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Mount Google Drive (uncomment on Colab)
# from google.colab import drive
# drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/frostbyte"
# For local testing:
# BASE_DIR = "."

# %% — Cell 2: Load Multimodal Dataset
# ============================================================

print("1. Loading multimodal dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "triage_master_multimodal.csv"))
print(f"   Loaded {len(df)} rows × {len(df.columns)} columns")

# %% — Cell 3: Late-Fusion Meta-Model (LightGBM)
# ============================================================
#
# The "brain" of the system: a single gradient-boosted tree classifier
# that ingests all three modalities as a flat feature vector.
#
# Feature budget:
#   - Tabular vitals:          7  (age, HR, RR, SpO2, temp, SBP, pain)
#   - ClinicalBERT PCA:       10  (text_feat_0 … text_feat_9)
#   - ResNet-50 PCA:            5  (img_feat_0 … img_feat_4)
#   - Total:                   22
#
# Target: ESI 1-5 (zero-indexed as 0-4 for LightGBM)
# ============================================================

tabular_features = [
    "age", "heart_rate", "resp_rate", "spo2",
    "temp_f", "systolic_bp", "pain_scale",
]
text_features = [f"text_feat_{i}" for i in range(10)]
img_features = [f"img_feat_{i}" for i in range(5)]

ALL_FEATURES = tabular_features + text_features + img_features
print(f"Feature space: {len(tabular_features)} tabular + {len(text_features)} text + {len(img_features)} image = {len(ALL_FEATURES)} total")

X = df[ALL_FEATURES]
y = df["target_esi"] - 1  # Zero-index: ESI 1→0, ESI 2→1, ..., ESI 5→4

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"Train ESI distribution:\n{y_train.value_counts().sort_index()}\n")

# Train LightGBM
print("2. Training Late-Fusion Meta-Model (LightGBM)...")
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

ESI_NAMES = [
    "ESI 1 (Resuscitation)",
    "ESI 2 (Emergent)",
    "ESI 3 (Urgent)",
    "ESI 4 (Less Urgent)",
    "ESI 5 (Non-Urgent)",
]

print(f"\n{'='*50}")
print(f"  MULTIMODAL FUSION ACCURACY: {acc:.4f}")
print(f"{'='*50}\n")
print(classification_report(y_test, y_pred, target_names=ESI_NAMES))

# %% — Cell 4: SHAP Explainability Engine
# ============================================================
#
# Three visualizations for the hackathon pitch:
#   1. Global bar chart — proves which features matter most
#   2. ESI-1 beeswarm  — shows feature directionality for critical patients
#   3. Local waterfall — single-patient "why ESI 1?" explanation
# ============================================================

print("3. Generating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ---- Detect SHAP output format (varies by version) ----
if isinstance(shap_values, list):
    # Old SHAP: list of (n_samples, n_features) arrays, one per class
    sv_esi1 = shap_values[0]
    sv_esi1_patient = lambda idx: shap_values[0][idx]
    base_val_esi1 = explainer.expected_value[0]
else:
    # New SHAP: 3D array (n_samples, n_features, n_classes)
    sv_esi1 = shap_values[:, :, 0]
    sv_esi1_patient = lambda idx: shap_values[idx, :, 0]
    base_val_esi1 = explainer.expected_value[0]

print(f"   SHAP output shape: {np.array(shap_values).shape}")

# ---- Plot 1: Global Feature Importance ----
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False,
                  feature_names=ALL_FEATURES)
plt.title("Global Feature Importance — Multimodal Triage Model", fontsize=14)
plt.tight_layout()
p1 = os.path.join(BASE_DIR, "shap_global_importance.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {p1}")

# ---- Plot 2: ESI-1 Beeswarm ----
plt.figure(figsize=(12, 8))
shap.summary_plot(sv_esi1, X_test, show=False, feature_names=ALL_FEATURES)
plt.title("SHAP Beeswarm — ESI 1 (Resuscitation) Class", fontsize=14)
plt.tight_layout()
p2 = os.path.join(BASE_DIR, "shap_esi1_beeswarm.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {p2}")

# ---- Plot 3: Local Waterfall for a Critical Patient ----
esi1_indices = np.where(y_pred == 0)[0]
if len(esi1_indices) > 0:
    critical_idx = esi1_indices[0]
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv_esi1_patient(critical_idx),
            base_values=base_val_esi1,
            data=X_test.iloc[critical_idx],
            feature_names=ALL_FEATURES,
        ),
        show=False,
    )
    plt.title(f"Patient #{critical_idx} — Why ESI 1 (Resuscitation)?", fontsize=13)
    plt.tight_layout()
    p3 = os.path.join(BASE_DIR, "shap_critical_patient.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved: {p3}")
else:
    print("⚠️ No ESI 1 predictions found in test set.")

# %% — Cell 5: Save Model Artifacts
# ============================================================

model_path = os.path.join(BASE_DIR, "triage_multimodal_model.pkl")
joblib.dump(model, model_path)
print(f"\n💾 Model (pickle): {model_path}")

lgb_path = os.path.join(BASE_DIR, "triage_multimodal_model.txt")
model.booster_.save_model(lgb_path)
print(f"💾 Model (LightGBM native): {lgb_path}")

print("\n🎉 LATE-FUSION PIPELINE COMPLETE!")
print("Artifacts: model (.pkl + .txt) + 3 SHAP plots (.png)")
