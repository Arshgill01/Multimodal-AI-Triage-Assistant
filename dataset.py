import random
import uuid

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Clinical profiles for realistic text and vital correlations
esi_profiles = {
    1: {
        "complaints": [
            "Unresponsive",
            "Cardiac arrest",
            "Severe respiratory distress",
            "Gunshot wound to chest",
            "Massive hemorrhage",
        ],
        "hr_range": (130, 180),
        "rr_range": (30, 45),
        "spo2_range": (75, 89),
        "sys_bp_range": (60, 90),
        "high_risk_prob": 1.0,
    },
    2: {
        "complaints": [
            "Crushing chest pain radiating to jaw",
            "Stroke symptoms, left side weakness",
            "Severe abdominal pain",
            "Suicidal ideation",
        ],
        "hr_range": (110, 140),
        "rr_range": (20, 30),
        "spo2_range": (90, 94),
        "sys_bp_range": (160, 200),
        "high_risk_prob": 0.8,
    },
    3: {
        "complaints": [
            "Moderate abdominal pain with nausea",
            "Fever and cough for 3 days",
            "Closed leg fracture",
            "Dehydration",
        ],
        "hr_range": (80, 110),
        "rr_range": (16, 22),
        "spo2_range": (94, 98),
        "sys_bp_range": (110, 140),
        "high_risk_prob": 0.1,
    },
    4: {
        "complaints": [
            "Sprained ankle",
            "Laceration needing stitches",
            "Minor burn",
            "Urinary tract infection symptoms",
        ],
        "hr_range": (60, 100),
        "rr_range": (12, 18),
        "spo2_range": (97, 100),
        "sys_bp_range": (100, 130),
        "high_risk_prob": 0.0,
    },
    5: {
        "complaints": [
            "Medication refill",
            "Poison ivy rash",
            "Cold symptoms, no fever",
            "Suture removal",
        ],
        "hr_range": (60, 90),
        "rr_range": (12, 16),
        "spo2_range": (98, 100),
        "sys_bp_range": (110, 120),
        "high_risk_prob": 0.0,
    },
}


def generate_triage_data(num_samples=1000):
    data = []

    # Force a balanced dataset for easier training during the hackathon
    samples_per_esi = num_samples // 5

    for esi_level in range(1, 6):
        profile = esi_profiles[esi_level]

        for _ in range(samples_per_esi):
            # Demographics
            patient_id = str(uuid.uuid4())[:8]
            age = np.random.randint(18, 90)

            # Vitals with noise
            hr = np.random.randint(*profile["hr_range"])
            rr = np.random.randint(*profile["rr_range"])
            spo2 = np.random.randint(*profile["spo2_range"])
            sys_bp = np.random.randint(*profile["sys_bp_range"])
            temp = round(
                (
                    np.random.uniform(97.0, 103.5)
                    if esi_level <= 3
                    else np.random.uniform(97.5, 99.5)
                ),
                1,
            )
            pain = (
                np.random.randint(7, 11) if esi_level <= 3 else np.random.randint(0, 6)
            )

            # Text & Vision
            complaint = random.choice(profile["complaints"])

            # Simulate a 20% chance the patient has a visible issue (rash, wound)
            has_image = random.random() < 0.2
            image_path = f"images/synthetic_{patient_id}.jpg" if has_image else "None"

            # Target
            high_risk = 1 if random.random() < profile["high_risk_prob"] else 0

            data.append(
                [
                    patient_id,
                    age,
                    hr,
                    rr,
                    spo2,
                    temp,
                    sys_bp,
                    pain,
                    complaint,
                    image_path,
                    esi_level,
                    high_risk,
                ]
            )

    # Shuffle the dataset
    random.shuffle(data)

    columns = [
        "patient_id",
        "age",
        "heart_rate",
        "resp_rate",
        "spo2",
        "temp_f",
        "systolic_bp",
        "pain_scale",
        "chief_complaint",
        "image_path",
        "target_esi",
        "flag_high_risk",
    ]

    df = pd.DataFrame(data, columns=columns)
    return df


# Generate and save
triage_df = generate_triage_data(1000)
triage_df.to_csv("synthetic_triage_data.csv", index=False)
print(f"Generated {len(triage_df)} records successfully. Data shape: {triage_df.shape}")
