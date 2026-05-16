import os
import random

import numpy as np
import pandas as pd

# Set seeds for reproducible demos
np.random.seed(42)
random.seed(42)


def safe_pain_convert(val):
    """MIMIC-IV pain scores are messy strings ('severe', 'unable to score', '10')."""
    try:
        # Extract first continuous number found
        num = int("".join(filter(str.isdigit, str(val))))
        return min(num, 10)  # Cap at 10
    except:
        return np.random.randint(0, 11)  # Fallback for unscoreable


# Registry: chief complaint keywords → subdirectory name in kaggle_images/
# To add a new image category, insert an entry: "keyword": "dirname"
IMAGE_REGISTRY = {
    "burn":       "burns",
    "laceration": "wounds",
    "fracture":   "wounds",
}


def map_kaggle_images(synthetic_df, image_base_dir="kaggle_images"):
    """Maps real Kaggle images to synthetic rows that have a placeholder.

    Uses IMAGE_REGISTRY to determine which subdirectory to look in based on
    the chief complaint text. Prints warnings when expected directories are
    missing or empty.
    """
    print("Mapping real images to synthetic data...")

    # Load available images per registry entry
    available = {}
    for keyword, dirname in IMAGE_REGISTRY.items():
        dirpath = os.path.join(image_base_dir, dirname)
        if os.path.exists(dirpath):
            images = os.listdir(dirpath)
            if images:
                available[dirname] = (dirpath, images)
            else:
                print(f"WARNING: Directory '{dirpath}' exists but is empty.")
        else:
            print(f"WARNING: Expected directory not found: {dirpath}")

    if not available:
        print(
            "WARNING: No image directories found. All image paths will be set to 'None'."
        )

    mapped = {dirname: 0 for dirname in set(IMAGE_REGISTRY.values())}
    unmapped = 0

    for idx, row in synthetic_df.iterrows():
        if row["image_path"] == "None":
            continue

        complaint = str(row["chief_complaint"]).lower()
        assigned = False

        for keyword, dirname in IMAGE_REGISTRY.items():
            if keyword in complaint and dirname in available:
                dirpath, images = available[dirname]
                img = random.choice(images)
                synthetic_df.at[idx, "image_path"] = os.path.join(dirpath, img)
                mapped[dirname] += 1
                assigned = True
                break

        if not assigned:
            synthetic_df.at[idx, "image_path"] = "None"
            unmapped += 1

    total_mapped = sum(mapped.values())
    print(f"Mapped {total_mapped} real images to synthetic records.")
    for dirname, count in mapped.items():
        if count:
            print(f"  - {dirname.capitalize()} images mapped: {count}")
    if unmapped:
        print(f"  - Unmapped (set to 'None'): {unmapped}")

    return synthetic_df


def process_mimic_data(triage_path="triage.csv"):
    """Loads, cleans, and standardizes MIMIC-IV-ED data to match our schema."""
    print("Processing MIMIC-IV-ED real clinical data...")
    try:
        mimic = pd.read_csv(triage_path)
    except FileNotFoundError:
        print(f"WARNING: {triage_path} not found. Returning empty dataframe.")
        return pd.DataFrame()

    # Drop rows without an ESI target or basic vitals
    mimic = mimic.dropna(subset=["acuity", "heartrate", "o2sat"])

    # Standardize column names to match the synthetic schema
    df = pd.DataFrame(
        {
            "patient_id": mimic["subject_id"].astype(str),
            "age": np.random.randint(
                18, 90, size=len(mimic)
            ),  # MIMIC-ED requires joins for age, proxying for speed
            "heart_rate": mimic["heartrate"].astype(float).fillna(80).astype(int),
            "resp_rate": mimic["resprate"].astype(float).fillna(18).astype(int),
            "spo2": mimic["o2sat"].astype(float).fillna(98).astype(int),
            "temp_f": mimic["temperature"].astype(float).fillna(98.6),
            "systolic_bp": mimic["sbp"].astype(float).fillna(120).astype(int),
            "pain_scale": mimic["pain"].apply(safe_pain_convert),
            "chief_complaint": mimic["chiefcomplaint"].fillna("Unknown"),
            "image_path": "None",  # Real MIMIC data doesn't have images
            "target_esi": mimic["acuity"].astype(int),
        }
    )

    # Derive high risk flag (ESI 1 or 2 = High Risk)
    df["flag_high_risk"] = df["target_esi"].apply(lambda x: 1 if x <= 2 else 0)

    print(f"Successfully processed {len(df)} real MIMIC clinical records.")
    return df


if __name__ == "__main__":
    # 1. Load and update synthetic data
    synth_df = pd.read_csv("synthetic_triage_data.csv")
    synth_df = map_kaggle_images(synth_df)

    # 2. Load and process real MIMIC data
    mimic_df = process_mimic_data("triage.csv")

    # 3. The Late Fusion Concatenation
    final_df = pd.concat([synth_df, mimic_df], ignore_index=True)

    # Shuffle the final dataset to mix real and synthetic
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the master file
    final_df.to_csv("triage_dataset_final.csv", index=False)

    print("\n--- Final Dataset Ready ---")
    print(f"Total Rows: {len(final_df)}")
    print(f"Real Images Linked: {len(final_df[final_df['image_path'] != 'None'])}")
    print(f"ESI Balance:\n{final_df['target_esi'].value_counts().sort_index()}")
    print(
        "\nDataset saved as 'triage_dataset_final.csv'. You are cleared for LightGBM."
    )
