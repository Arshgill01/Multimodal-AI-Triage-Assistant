use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::models::{
    BatchPatientResult, BatchPredictRequest, BatchPredictResponse, BatchSummary, ConfidenceMetrics,
    EmbedRequest, EmbedResponse, PatientRequest, PredictResponse, ShapExplanation, ShapRequest,
    ESI_LABELS,
};
use crate::state::AppState;

/// `POST /batch-predict` — Mass casualty triage simulation.
///
/// Accepts an array of patients, runs the full predict pipeline for each,
/// and returns results sorted by severity (ESI 1 first, then by confidence).
pub async fn batch_predict(
    State(state): State<Arc<AppState>>,
    Json(batch): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, (StatusCode, String)> {
    if batch.patients.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No patients provided".to_string()));
    }

    if batch.patients.len() > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Maximum 100 patients per batch".to_string(),
        ));
    }

    let mut results: Vec<BatchPatientResult> = Vec::with_capacity(batch.patients.len());

    for (idx, patient) in batch.patients.iter().enumerate() {
        match predict_single(&state, idx, patient).await {
            Ok(prediction) => {
                results.push(BatchPatientResult {
                    index: idx,
                    chief_complaint: patient.chief_complaint.clone(),
                    prediction,
                });
            }
            Err(e) => {
                tracing::warn!("Batch patient {} failed: {}", idx, e.1);
                // Continue with remaining patients
            }
        }
    }

    // Sort by severity: ESI 1 first, then by descending confidence
    results.sort_by(|a, b| {
        a.prediction
            .predicted_esi
            .cmp(&b.prediction.predicted_esi)
            .then(
                b.prediction
                    .confidence
                    .top_probability
                    .partial_cmp(&a.prediction.confidence.top_probability)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Build summary
    let mut summary = BatchSummary {
        esi_1_count: 0,
        esi_2_count: 0,
        esi_3_count: 0,
        esi_4_count: 0,
        esi_5_count: 0,
        uncertain_count: 0,
    };

    for r in &results {
        match r.prediction.predicted_esi {
            1 => summary.esi_1_count += 1,
            2 => summary.esi_2_count += 1,
            3 => summary.esi_3_count += 1,
            4 => summary.esi_4_count += 1,
            5 => summary.esi_5_count += 1,
            _ => {}
        }
        if r.prediction.confidence.is_uncertain {
            summary.uncertain_count += 1;
        }
    }

    Ok(Json(BatchPredictResponse {
        total_patients: results.len(),
        patients: results,
        summary,
    }))
}

/// Run the predict pipeline for a single patient (reuses the same logic as /predict).
async fn predict_single(
    state: &AppState,
    idx: usize,
    patient: &PatientRequest,
) -> Result<PredictResponse, (StatusCode, String)> {
    let tabular_features = vec![
        patient.age,
        patient.heart_rate,
        patient.resp_rate,
        patient.spo2,
        patient.temp_f,
        patient.systolic_bp,
        patient.pain_scale,
    ];

    let embed_req = EmbedRequest {
        complaint: patient.chief_complaint.clone(),
        image_base64: patient.image_base64.clone(),
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
                format!("Embed service unavailable: {}", e),
            )
        })?
        .json::<EmbedResponse>()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Invalid embed response: {}", e),
            )
        })?;

    let mut feature_vector: Vec<f64> = Vec::with_capacity(22);
    feature_vector.extend_from_slice(&tabular_features);
    feature_vector.extend_from_slice(&embed_resp.text_features);
    feature_vector.extend_from_slice(&embed_resp.image_features);

    let raw_preds = {
        let booster_mutex = state.booster.as_ref().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Model not loaded".to_string(),
            )
        })?;
        let guard = booster_mutex.lock().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Lock poisoned: {}", e),
            )
        })?;
        guard
            .0
            .predict_with_params(&feature_vector, 22, true, "num_threads=1")
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Prediction failed: {:?}", e),
                )
            })?
    };

    let probabilities: Vec<f64> = raw_preds.to_vec();
    let predicted_class = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(2);
    let predicted_esi = (predicted_class + 1) as u8;
    let esi_label = ESI_LABELS
        .get(predicted_class)
        .unwrap_or(&"Unknown")
        .to_string();

    // Fetch SHAP (best-effort)
    let shap = fetch_shap_batch(state, &feature_vector, predicted_class as u8).await;

    let confidence = ConfidenceMetrics::from_probabilities(&probabilities);

    // Simple hash from complaint + vitals for patient deduplication
    let patient_hash = format!(
        "{:x}",
        md5_hash(&format!(
            "{}|{}|{}|{}",
            patient.chief_complaint, patient.age, patient.heart_rate, patient.spo2
        ))
    );

    let top_shap_drivers: Vec<String> = shap
        .as_ref()
        .map(|s| {
            s.features
                .iter()
                .take(3)
                .map(|f| format!("{}={:.4}", f.name, f.shap_value))
                .collect()
        })
        .unwrap_or_default();

    let audit_id = match state.log_decision(
        &patient_hash,
        &patient.chief_complaint,
        predicted_esi,
        confidence.top_probability,
        confidence.is_uncertain,
        &top_shap_drivers,
    ) {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!("Batch audit log failed for patient {}: {}", idx, e);
            patient_hash.clone()
        }
    };

    Ok(PredictResponse {
        predicted_esi,
        esi_label,
        probabilities,
        confidence,
        feature_vector,
        shap,
        audit_id,
    })
}

async fn fetch_shap_batch(
    state: &AppState,
    feature_vector: &[f64],
    predicted_class: u8,
) -> Option<ShapExplanation> {
    let url = format!("{}/shap", state.python_service_url);
    let req = ShapRequest {
        feature_vector: feature_vector.to_vec(),
        predicted_class,
    };
    let resp = state.http_client.post(&url).json(&req).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<ShapExplanation>().await.ok()
}

/// Simple hash function (FNV-like) for patient deduplication.
fn md5_hash(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
