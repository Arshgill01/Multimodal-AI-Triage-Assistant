use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::models::{
    EmbedRequest, EmbedResponse, PatientRequest, PredictResponse,
    ShapExplanation, ShapRequest, ESI_LABELS,
};
use crate::state::AppState;

/// `POST /predict` — Full triage inference pipeline.
///
/// Flow:
///   1. Extract 7 tabular vitals from the request JSON
///   2. Call Python `/embed` for ClinicalBERT (10) + ResNet-50 (5) features
///   3. Concatenate into a 22-feature vector
///   4. Run LightGBM inference via FFI
///   5. Call Python `/shap` for real-time explainability
///   6. Return ESI prediction + probabilities + SHAP values
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
    // Scope the MutexGuard so it's dropped before any .await calls
    let raw_preds = {
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

        let booster = &booster_guard.0;

        booster
            .predict_with_params(
                &feature_vector,
                22,
                true,
                "num_threads=1",
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("LightGBM prediction failed: {:?}", e),
                )
            })?
    }; // MutexGuard dropped here — safe to .await below

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

    // ── Step 6: Real-time SHAP explainability ─────────────────
    let shap = fetch_shap_values(
        &state,
        &feature_vector,
        predicted_class as u8,
    )
    .await;

    Ok(Json(PredictResponse {
        predicted_esi,
        esi_label,
        probabilities,
        feature_vector,
        shap,
    }))
}

/// Fetch SHAP values from the Python service. Returns None on any failure
/// (graceful degradation — prediction still works without SHAP).
async fn fetch_shap_values(
    state: &AppState,
    feature_vector: &[f64],
    predicted_class: u8,
) -> Option<ShapExplanation> {
    let shap_url = format!("{}/shap", state.python_service_url);
    let shap_req = ShapRequest {
        feature_vector: feature_vector.to_vec(),
        predicted_class,
    };

    let resp = state
        .http_client
        .post(&shap_url)
        .json(&shap_req)
        .send()
        .await
        .ok()?;

    if !resp.status().is_success() {
        tracing::warn!("SHAP service returned {}", resp.status());
        return None;
    }

    resp.json::<ShapExplanation>().await.ok()
}

