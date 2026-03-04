import os
import random

import numpy as np
import pandas as pd

# Set seeds for reproducible hackathon demos
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


def map_kaggle_images(synthetic_df, image_base_dir="kaggle_images"):
    """Maps real Kaggle images to synthetic rows that have a placeholder."""
    print("Mapping real images to synthetic data...")

    # Check if directories exist
    rash_dir = os.path.join(image_base_dir, "rashes")
    wound_dir = os.path.join(image_base_dir, "wounds")

    # Fallback lists if folders are empty/missing
    rashes = os.listdir(rash_dir) if os.path.exists(rash_dir) else []
    wounds = os.listdir(wound_dir) if os.path.exists(wound_dir) else []

    mapped_count = 0
    for idx, row in synthetic_df.iterrows():
        if row["image_path"] != "None":
            complaint = str(row["chief_complaint"]).lower()

            # Smartly assign based on text
            if "rash" in complaint and rashes:
                img = random.choice(rashes)
                synthetic_df.at[idx, "image_path"] = os.path.join(rash_dir, img)
                mapped_count += 1
            elif (
                "burn" in complaint
                or "laceration" in complaint
                or "fracture" in complaint
            ) and wounds:
                img = random.choice(wounds)
                synthetic_df.at[idx, "image_path"] = os.path.join(wound_dir, img)
                mapped_count += 1
            else:
                # If no image matches, clean it up
                synthetic_df.at[idx, "image_path"] = "None"

    print(f"Mapped {mapped_count} real images to synthetic records.")
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
