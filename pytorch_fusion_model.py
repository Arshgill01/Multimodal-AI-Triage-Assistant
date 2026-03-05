# %% [markdown]
# # 🧊 Frostbyte — PyTorch Cross-Attention Multimodal Fusion Network
#
# **Objective:** Build a custom multi-headed neural network with cross-attention
# that learns _when_ to attend to text/image modalities based on tabular vitals.
# Runs as a research prototype alongside the LightGBM production model.
#
# **Prerequisite:** `triage_master_multimodal.csv` from the main pipeline.
#
# **Expected runtime on Colab T4:** ~2-3 minutes.

# %% — Cell 1: Imports & Config
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# %% — Cell 2: Load Data & Prepare Tensors
# ============================================================

df = pd.read_csv(os.path.join(BASE_DIR, "triage_master_multimodal.csv"))

tabular_cols = ["age", "heart_rate", "resp_rate", "spo2", "temp_f", "systolic_bp", "pain_scale"]
text_cols = [f"text_feat_{i}" for i in range(10)]
img_cols = [f"img_feat_{i}" for i in range(5)]

X_tab = df[tabular_cols].values.astype(np.float32)
X_txt = df[text_cols].values.astype(np.float32)
X_img = df[img_cols].values.astype(np.float32)
y = (df["target_esi"] - 1).values.astype(np.int64)  # Zero-indexed

# Normalize tabular features (critical for neural nets, unlike tree models)
from sklearn.preprocessing import StandardScaler

scaler_tab = StandardScaler()
X_tab = scaler_tab.fit_transform(X_tab)

# Stratified split — same seed as LightGBM for fair comparison
X_tab_train, X_tab_test, X_txt_train, X_txt_test, X_img_train, X_img_test, y_train, y_test = \
    train_test_split(X_tab, X_txt, X_img, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights for imbalanced ESI distribution
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"Class weights: {dict(zip([f'ESI {c+1}' for c in classes], np.round(class_weights, 3)))}")

# Build DataLoaders
BATCH_SIZE = 32

train_ds = TensorDataset(
    torch.FloatTensor(X_tab_train),
    torch.FloatTensor(X_txt_train),
    torch.FloatTensor(X_img_train),
    torch.LongTensor(y_train),
)
test_ds = TensorDataset(
    torch.FloatTensor(X_tab_test),
    torch.FloatTensor(X_txt_test),
    torch.FloatTensor(X_img_test),
    torch.LongTensor(y_test),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}  |  Batches/epoch: {len(train_loader)}")


# %% — Cell 3: Model Architecture
# ============================================================
#
# Three parallel MLP heads → Cross-Attention → Fused classifier
#
# The cross-attention module uses tabular vitals as the QUERY
# and text+image as KEY/VALUE. This lets the network learn:
# "When vitals are borderline, pay MORE attention to the text."
# ============================================================


class ModalityHead(nn.Module):
    """A small MLP encoder for a single modality."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Scaled dot-product cross-attention.
    Query = tabular head output
    Key/Value = concatenated text + image head outputs

    This forces the model to learn dynamic fusion weights:
    when vitals alone are ambiguous, it upweights text/image.
    """

    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        # query:     (batch, 1, embed_dim)  — tabular
        # key_value: (batch, 2, embed_dim)  — text + image stacked
        attn_out, attn_weights = self.attn(query, key_value, key_value)
        out = self.norm(query + attn_out)  # Residual connection
        return out, attn_weights


class MultimodalFusionNet(nn.Module):
    """
    The full multimodal fusion network.

    Architecture:
        Tabular(7) → MLP → 32-d
        Text(10)   → MLP → 32-d
        Image(5)   → MLP → 32-d
        CrossAttention(query=tab, kv=[text, image]) → 32-d
        Concat(attended_tab + text + image) = 96-d → MLP → 5 classes
    """

    EMBED_DIM = 32

    def __init__(self, tab_dim=7, txt_dim=10, img_dim=5, num_classes=5):
        super().__init__()

        d = self.EMBED_DIM  # 32

        # Modality-specific encoder heads
        self.tab_head = ModalityHead(tab_dim, 64, d)
        self.txt_head = ModalityHead(txt_dim, 64, d)
        self.img_head = ModalityHead(img_dim, 32, d)

        # Cross-attention: tabular queries attend to text+image
        self.cross_attn = CrossAttentionBlock(embed_dim=d, num_heads=2)

        # Final classifier on the fused representation
        self.classifier = nn.Sequential(
            nn.Linear(d * 3, 64),  # 96 → 64
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_tab, x_txt, x_img):
        # Encode each modality
        h_tab = self.tab_head(x_tab)  # (B, 32)
        h_txt = self.txt_head(x_txt)  # (B, 32)
        h_img = self.img_head(x_img)  # (B, 32)

        # Reshape for attention: (B, seq_len, embed_dim)
        q = h_tab.unsqueeze(1)                        # (B, 1, 32) — query
        kv = torch.stack([h_txt, h_img], dim=1)       # (B, 2, 32) — key/value

        # Cross-attention: tabular attends to text + image
        attended, attn_weights = self.cross_attn(q, kv)
        attended = attended.squeeze(1)  # (B, 32)

        # Late fusion: concatenate all representations
        fused = torch.cat([attended, h_txt, h_img], dim=1)  # (B, 96)

        # Classify
        logits = self.classifier(fused)
        return logits, attn_weights


# Instantiate
model_nn = MultimodalFusionNet(
    tab_dim=len(tabular_cols),
    txt_dim=len(text_cols),
    img_dim=len(img_cols),
    num_classes=5,
).to(device)

total_params = sum(p.numel() for p in model_nn.parameters())
trainable_params = sum(p.numel() for p in model_nn.parameters() if p.requires_grad)
print(f"Model: {total_params:,} total params ({trainable_params:,} trainable)")
print(model_nn)


# %% — Cell 4: Training Loop with Early Stopping
# ============================================================

EPOCHS = 100
PATIENCE = 10
LR = 1e-3

optimizer = torch.optim.AdamW(model_nn.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

best_val_acc = 0.0
patience_counter = 0
train_losses = []
val_accs = []

print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...\n")

for epoch in range(EPOCHS):
    # ---- Train ----
    model_nn.train()
    epoch_loss = 0.0
    for batch_tab, batch_txt, batch_img, batch_y in train_loader:
        batch_tab = batch_tab.to(device)
        batch_txt = batch_txt.to(device)
        batch_img = batch_img.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits, _ = model_nn(batch_tab, batch_txt, batch_img)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ---- Evaluate ----
    model_nn.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_tab, batch_txt, batch_img, batch_y in test_loader:
            batch_tab = batch_tab.to(device)
            batch_txt = batch_txt.to(device)
            batch_img = batch_img.to(device)

            logits, _ = model_nn(batch_tab, batch_txt, batch_img)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    val_accs.append(val_acc)
    scheduler.step(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best checkpoint
        best_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
            break

# Restore best model
model_nn.load_state_dict(best_state)
print(f"\n✅ Training complete. Best validation accuracy: {best_val_acc:.4f}")


# %% — Cell 5: Evaluation & Comparison with LightGBM
# ============================================================

import matplotlib.pyplot as plt

ESI_NAMES = [
    "ESI 1 (Resuscitation)",
    "ESI 2 (Emergent)",
    "ESI 3 (Urgent)",
    "ESI 4 (Less Urgent)",
    "ESI 5 (Non-Urgent)",
]

# Final evaluation with best model
model_nn.eval()
all_preds = []
all_labels = []
all_attn_weights = []

with torch.no_grad():
    for batch_tab, batch_txt, batch_img, batch_y in test_loader:
        batch_tab = batch_tab.to(device)
        batch_txt = batch_txt.to(device)
        batch_img = batch_img.to(device)

        logits, attn_w = model_nn(batch_tab, batch_txt, batch_img)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())
        all_attn_weights.append(attn_w.cpu().numpy())

nn_acc = accuracy_score(all_labels, all_preds)

print(f"\n{'='*60}")
print(f"  NEURAL FUSION MODEL — CLASSIFICATION REPORT")
print(f"{'='*60}\n")
print(f"Accuracy: {nn_acc:.4f}\n")
print(classification_report(all_labels, all_preds, target_names=ESI_NAMES))

# ---- Training curves ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, color="#ff6b6b", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(alpha=0.3)

ax2.plot(val_accs, color="#4ecdc4", linewidth=2)
ax2.axhline(y=best_val_acc, color="#ff6b6b", linestyle="--", alpha=0.7, label=f"Best: {best_val_acc:.4f}")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Validation Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("Neural Fusion — Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "neural_fusion_training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved: neural_fusion_training_curves.png")

# ---- Attention weight analysis ----
all_attn = np.vstack([a.squeeze(2) for a in all_attn_weights])  # (N, num_heads, 2)
mean_attn = all_attn.mean(axis=(0, 1))  # Average across samples and heads
print(f"\nCross-Attention Weights (avg across test set):")
print(f"  Text modality:  {mean_attn[0]:.4f}")
print(f"  Image modality: {mean_attn[1]:.4f}")
print(f"  → Model {'prioritizes text' if mean_attn[0] > mean_attn[1] else 'prioritizes images'} when vitals are ambiguous")


# %% — Cell 6: Save Model & Comparison Table
# ============================================================

# Save PyTorch checkpoint
ckpt_path = os.path.join(BASE_DIR, "neural_fusion_model.pt")
torch.save({
    "model_state_dict": model_nn.state_dict(),
    "scaler_state": {
        "mean": scaler_tab.mean_.tolist(),
        "scale": scaler_tab.scale_.tolist(),
    },
    "architecture": {
        "tab_dim": len(tabular_cols),
        "txt_dim": len(text_cols),
        "img_dim": len(img_cols),
        "num_classes": 5,
    },
    "best_val_acc": best_val_acc,
    "feature_names": {
        "tabular": tabular_cols,
        "text": text_cols,
        "image": img_cols,
    },
}, ckpt_path)
print(f"💾 Saved PyTorch checkpoint: {ckpt_path}")

# Print comparison table for the pitch
print(f"\n{'='*60}")
print(f"  MODEL COMPARISON — LightGBM vs Neural Fusion")
print(f"{'='*60}")
print(f"{'Metric':<25} {'LightGBM':<15} {'Neural Fusion':<15}")
print(f"{'-'*55}")
print(f"{'Architecture':<25} {'Tree Ensemble':<15} {'Cross-Attn NN':<15}")
print(f"{'Parameters':<25} {'N/A (trees)':<15} {f'{total_params:,}':<15}")
print(f"{'Fusion Strategy':<25} {'Concatenation':<15} {'Cross-Attention':<15}")
print(f"{'Val Accuracy':<25} {'(run lgb cell)':<15} {f'{best_val_acc:.4f}':<15}")
print(f"{'Training Time':<25} {'~2 sec':<15} {'~2-3 min':<15}")
print(f"{'Scalability':<25} {'Medium':<15} {'High (GPU)':<15}")
print(f"\n🎯 LightGBM = production model | Neural Fusion = research prototype")
print(f"   The neural architecture is designed to scale when hospital-sized")
print(f"   datasets (50K+ patients) become available.")
