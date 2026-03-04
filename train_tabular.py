import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load the Data
# (Currently using synthetic, will swap to the harmonized MIMIC dataset later)
df = pd.read_csv("triage_dataset_final.csv")

# 2. Feature Selection (Tabular only for now)
tabular_features = [
    "age",
    "heart_rate",
    "resp_rate",
    "spo2",
    "temp_f",
    "systolic_bp",
    "pain_scale",
]

X = df[tabular_features]
# LightGBM expects classes to start at 0. ESI is 1-5, so we subtract 1.
y = df["target_esi"] - 1

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize and Train the LightGBM Classifier
# These hyperparameters are tuned to prevent overfitting on small/medium clinical datasets
print("Training Tabular Meta-Model (LightGBM)...")
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    class_weight="balanced",  # Crucial for real-world ESI imbalance later
    random_state=42,
)

model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(
    classification_report(
        y_test, y_pred, target_names=["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"]
    )
)

# 6. SHAP Explainability Engine (The Hackathon Flex)
print("\nGenerating SHAP Values...")
# TreeExplainer is highly optimized for LightGBM
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- VISUALIZATION OUTPUTS FOR THE DEMO ---

# 1. Global Feature Importance (Saves to file)
# This proves to the judges that the model learned actual medicine (e.g., Heart Rate matters most)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("shap_global_importance.png", bbox_inches="tight")
print("Saved Global SHAP plot to 'shap_global_importance.png'")

# 2. Local Patient Explanation (The "Live Triage" Demo)
# Let's pick a critical patient (e.g., someone the model predicted as ESI 1)
critical_patient_idx = np.where(y_pred == 0)[0][0]  # Find first ESI 1 prediction

# Generate a Waterfall plot for this specific patient
# (Note: For multi-class, shap_values is a list. Index 0 is the ESI 1 class)
plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0][critical_patient_idx],
        base_values=explainer.expected_value[0],
        data=X_test.iloc[critical_patient_idx],
        feature_names=tabular_features,
    ),
    show=False,
)
plt.savefig("shap_critical_patient.png", bbox_inches="tight")
print(
    f"Saved Local SHAP explanation for Patient {critical_patient_idx} to 'shap_critical_patient.png'"
)
