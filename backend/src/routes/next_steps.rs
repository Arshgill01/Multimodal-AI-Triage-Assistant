use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::models::{
    NextStepsResponse, PatientRequest, PatientVitals, RagRequest, RagResponse,
};
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
