"""SBAR clinical handoff formatting for triage predictions.

SBAR (Situation-Background-Assessment-Recommendation) is the standard
structured communication framework used in emergency departments and
critical care settings for patient handoffs.
"""


def format_sbar(
    age: float,
    chief_complaint: str,
    vitals: list[float],
    predicted_esi: int,
    esi_label: str,
    confidence_label: str,
    top_probability: float,
    is_uncertain: bool,
) -> dict:
    """Format a prediction into SBAR clinical handoff structure.

    Returns a dict with keys: situation, background, assessment, recommendation.
    """
    hr = vitals[0] if len(vitals) > 0 else 0.0
    rr = vitals[1] if len(vitals) > 1 else 0.0
    spo2 = vitals[2] if len(vitals) > 2 else 0.0
    temp = vitals[3] if len(vitals) > 3 else 0.0
    sbp = vitals[4] if len(vitals) > 4 else 0.0
    pain = vitals[6] if len(vitals) > 6 else 0.0

    situation = f"{int(age)} y/o patient presenting with \"{chief_complaint}\"."

    background = (
        f"HR {int(hr)} | RR {int(rr)} | SpO2 {int(spo2)}% | "
        f"Temp {temp}°F | SBP {int(sbp)} | Pain {int(pain)}/10"
    )

    uncertainty_warning = ""
    if is_uncertain:
        uncertainty_warning = "⚠️ LOW CONFIDENCE — Manual review strongly recommended."

    assessment = (
        f"AI predicts {esi_label} ({predicted_esi}) — "
        f"Confidence: {top_probability * 100:.1f}% ({confidence_label})"
    )
    if uncertainty_warning:
        assessment += f"\n{uncertainty_warning}"

    recommendation = _esi_triage_recommendation(predicted_esi)

    return {
        "situation": situation,
        "background": background,
        "assessment": assessment,
        "recommendation": recommendation,
    }


def _esi_triage_recommendation(esi: int) -> str:
    recommendations = {
        1: (
            "Immediate resuscitation team activation. "
            "Continuous monitoring. Prepare intubation/CPR. "
            "Transfer to critical care bed."
        ),
        2: (
            "High-priority workup. Bed within 10 minutes. "
            "Establish IV access. Order relevant labs/imaging. "
            "Reevaluate frequently."
        ),
        3: (
            "Urgent evaluation. Bed within 30 minutes. "
            "Initiate diagnostic workup per protocol. "
            "Reassess within 60 minutes."
        ),
        4: (
            "Routine evaluation. Consider fast-track pathway. "
            "Reassess before discharge for clinical changes."
        ),
        5: (
            "Non-urgent management. May be suitable for "
            "primary care referral. Provide return precautions."
        ),
    }
    return recommendations.get(
        esi, "Unable to determine triage level. Manual evaluation required."
    )
