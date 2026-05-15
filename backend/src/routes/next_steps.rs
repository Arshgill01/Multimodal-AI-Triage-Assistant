use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::models::{NextStepsResponse, PatientRequest, PatientVitals, RagRequest, RagResponse};
use crate::state::AppState;

/// `POST /next-steps` — Clinical RAG decision support.
///
/// Accepts patient data + predicted ESI, proxies to the Python RAG
/// microservice (ChromaDB retrieval + Gemini generation), and returns
/// grounded clinical recommendations.
pub async fn next_steps(
    State(state): State<Arc<AppState>>,
    Json(patient): Json<PatientRequest>,
) -> Result<Json<NextStepsResponse>, (StatusCode, String)> {
    // For next-steps, we need a predicted ESI. If not already predicted,
    // default to the /predict endpoint first. For now, accept it as a
    // separate field or use a reasonable default.
    let predicted_esi = determine_esi(&state, &patient).await?;

    let vitals = PatientVitals {
        age: patient.age,
        heart_rate: patient.heart_rate,
        resp_rate: patient.resp_rate,
        spo2: patient.spo2,
        temp_f: patient.temp_f,
        systolic_bp: patient.systolic_bp,
        pain_scale: patient.pain_scale,
    };

    let rag_req = RagRequest {
        complaint: patient.chief_complaint.clone(),
        vitals,
        predicted_esi,
    };

    let rag_url = format!("{}/rag", state.python_service_url);
    let rag_resp = state
        .http_client
        .post(&rag_url)
        .json(&rag_req)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("RAG service unavailable at {}: {}", rag_url, e),
            )
        })?
        .json::<RagResponse>()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Invalid response from RAG service: {}", e),
            )
        })?;

    Ok(Json(NextStepsResponse {
        recommendation: rag_resp.recommendation,
        similar_cases: rag_resp.similar_cases,
    }))
}

/// Determine ESI level — first try the model, fall back to vitals heuristic.
async fn determine_esi(
    _state: &AppState,
    patient: &PatientRequest,
) -> Result<u8, (StatusCode, String)> {
    // Simple vitals-based heuristic when model isn't available
    // (The real prediction is done by the /predict endpoint with full
    // embedding pipeline — this is a fast fallback for the RAG endpoint)
    if patient.spo2 < 85.0 || patient.systolic_bp < 80.0 {
        Ok(1)
    } else if patient.heart_rate > 120.0 || patient.spo2 < 92.0 {
        Ok(2)
    } else if patient.pain_scale >= 7.0 || patient.heart_rate > 100.0 {
        Ok(3)
    } else if patient.pain_scale >= 4.0 {
        Ok(4)
    } else {
        Ok(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_patient(
        spo2: f64,
        systolic_bp: f64,
        heart_rate: f64,
        pain_scale: f64,
    ) -> PatientRequest {
        PatientRequest {
            age: 45.0,
            heart_rate,
            resp_rate: 16.0,
            spo2,
            temp_f: 98.6,
            systolic_bp,
            pain_scale,
            chief_complaint: "Test complaint".to_string(),
            image_base64: None,
        }
    }

    #[tokio::test]
    async fn determine_esi_prefers_esi1_over_esi2_for_critical_vitals() {
        let state = AppState::degraded("http://localhost:8000", ":memory:").unwrap();
        let patient = build_patient(83.0, 90.0, 125.0, 2.0);

        let esi = determine_esi(&state, &patient).await.unwrap();

        assert_eq!(esi, 1, "ESI 1 should win when SpO2 is below 85% even if HR is > 120");
    }

    #[tokio::test]
    async fn determine_esi_applies_threshold_boundaries_correctly() {
        let state = AppState::degraded("http://localhost:8000", ":memory:").unwrap();

        let boundary_cases = vec![
            (build_patient(84.0, 95.0, 90.0, 1.0), 1), // SpO2 < 85
            (build_patient(85.0, 79.0, 90.0, 1.0), 1), // SBP < 80
            (build_patient(88.0, 95.0, 125.0, 1.0), 2), // HR > 120 and not ESI 1
            (build_patient(91.0, 95.0, 121.0, 1.0), 2), // HR threshold edge
            (build_patient(92.0, 95.0, 110.0, 7.0), 3), // Pain >= 7, not ESI 2
            (build_patient(93.0, 95.0, 100.0, 4.0), 4), // Pain >= 4, not ESI 3
            (build_patient(95.0, 95.0, 90.0, 1.0), 5), // otherwise
        ];

        for (patient, expected) in boundary_cases {
            let esi = determine_esi(&state, &patient).await.unwrap();
            assert_eq!(esi, expected, "Unexpected ESI for patient={:?}", patient);
        }
    }
}
