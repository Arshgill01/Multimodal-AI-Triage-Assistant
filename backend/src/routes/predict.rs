use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::models::{
    EmbedRequest, EmbedResponse, PatientRequest, PredictResponse, ESI_LABELS,
};
use crate::state::AppState;

/// `POST /predict` — Full triage inference pipeline.
///
/// Flow:
///   1. Extract 7 tabular vitals from the request JSON
///   2. Call Python `/embed` for ClinicalBERT (10) + ResNet-50 (5) features
///   3. Concatenate into a 22-feature vector
///   4. Run LightGBM inference via FFI
///   5. Return ESI prediction + class probabilities
pub async fn predict(
    State(state): State<Arc<AppState>>,
    Json(patient): Json<PatientRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    // ── Step 1: Tabular features (7) ──────────────────────────
    let tabular_features = vec![
        patient.age,
        patient.heart_rate,
        patient.resp_rate,
        patient.spo2,
        patient.temp_f,
        patient.systolic_bp,
        patient.pain_scale,
    ];

    // ── Step 2: Get text + image embeddings from Python ───────
    let embed_req = EmbedRequest {
        complaint: patient.chief_complaint.clone(),
        image_path: patient.image_path.clone(),
    };

    let embed_url = format!("{}/embed", state.python_service_url);
    let embed_resp = state
        .http_client
        .post(&embed_url)
        .json(&embed_req)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Python preprocessing service unavailable at {}: {}",
                    embed_url, e
                ),
            )
        })?
        .json::<EmbedResponse>()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Invalid response from Python service: {}", e),
            )
        })?;

    // ── Step 3: Assemble 22-feature vector ────────────────────
    let mut feature_vector: Vec<f64> = Vec::with_capacity(22);
    feature_vector.extend_from_slice(&tabular_features);       // 7 vitals
    feature_vector.extend_from_slice(&embed_resp.text_features);  // 10 text
    feature_vector.extend_from_slice(&embed_resp.image_features); // 5 image

    if feature_vector.len() != 22 {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "Feature vector has {} elements, expected 22. Text={}, Image={}",
                feature_vector.len(),
                embed_resp.text_features.len(),
                embed_resp.image_features.len()
            ),
        ));
    }

    // ── Step 4: LightGBM inference via FFI ────────────────────
    let booster_mutex = state.booster.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "LightGBM model not loaded. Set FROSTBYTE_MODEL_PATH.".to_string(),
        )
    })?;

    let booster_guard = booster_mutex.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Model lock poisoned: {}", e),
        )
    })?;

    // Access the inner Booster through the SendBooster wrapper
    let booster = &booster_guard.0;

    // predict_with_params returns Vec<f64> — for multiclass, this is
    // [p_class0, p_class1, ..., p_class4] flattened.
    let raw_preds = booster
        .predict_with_params(
            &feature_vector,
            22, // n_features
            true,
            "num_threads=1",
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("LightGBM prediction failed: {:?}", e),
            )
        })?;

    // ── Step 5: Parse output ──────────────────────────────────
    // LightGBM multiclass returns probabilities for each class
    let probabilities: Vec<f64> = raw_preds.iter().map(|&v| v).collect();

    let predicted_class = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(2); // Default to ESI 3 (moderate) on failure

    let predicted_esi = (predicted_class + 1) as u8; // Convert back to 1-indexed

    let esi_label = ESI_LABELS
        .get(predicted_class)
        .unwrap_or(&"Unknown")
        .to_string();

    Ok(Json(PredictResponse {
        predicted_esi,
        esi_label,
        probabilities,
        feature_vector,
    }))
}
