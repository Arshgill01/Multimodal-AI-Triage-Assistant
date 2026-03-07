use serde::{Deserialize, Serialize};

// ─── Inbound Request ─────────────────────────────────────────

/// Patient data submitted for triage prediction.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PatientRequest {
    pub age: f64,
    pub heart_rate: f64,
    pub resp_rate: f64,
    pub spo2: f64,
    pub temp_f: f64,
    pub systolic_bp: f64,
    pub pain_scale: f64,
    pub chief_complaint: String,
    #[serde(default)]
    pub image_path: Option<String>,
}

// ─── /predict Response ───────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub predicted_esi: u8,
    pub esi_label: String,
    pub probabilities: Vec<f64>,
    pub feature_vector: Vec<f64>,
    /// Real-time SHAP explainability (None if SHAP service unavailable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shap: Option<ShapExplanation>,
}

// ─── /next-steps Response ────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct NextStepsResponse {
    pub recommendation: String,
    pub similar_cases: Vec<SimilarCase>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SimilarCase {
    pub complaint: String,
    pub target_esi: u8,
    pub similarity: f64,
    #[serde(default)]
    pub heart_rate: Option<f64>,
    #[serde(default)]
    pub spo2: Option<f64>,
}

// ─── Python Microservice DTOs ────────────────────────────────

/// Request sent to `POST /embed` on the Python service.
#[derive(Debug, Serialize)]
pub struct EmbedRequest {
    pub complaint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_path: Option<String>,
}

/// Response from `POST /embed` on the Python service.
#[derive(Debug, Deserialize)]
pub struct EmbedResponse {
    pub text_features: Vec<f64>,
    pub image_features: Vec<f64>,
}

/// Request sent to `POST /shap` on the Python service.
#[derive(Debug, Serialize)]
pub struct ShapRequest {
    pub feature_vector: Vec<f64>,
    pub predicted_class: u8,
}

/// Per-feature SHAP contribution.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ShapFeature {
    pub name: String,
    pub value: f64,
    pub shap_value: f64,
}

/// Full SHAP explanation for a single prediction.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ShapExplanation {
    pub base_value: f64,
    pub features: Vec<ShapFeature>,
    pub predicted_class: u8,
    pub prediction_label: String,
}

/// Request sent to `POST /rag` on the Python service.
#[derive(Debug, Serialize)]
pub struct RagRequest {
    pub complaint: String,
    pub vitals: PatientVitals,
    pub predicted_esi: u8,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PatientVitals {
    pub age: f64,
    pub heart_rate: f64,
    pub resp_rate: f64,
    pub spo2: f64,
    pub temp_f: f64,
    pub systolic_bp: f64,
    pub pain_scale: f64,
}

/// Response from `POST /rag` on the Python service.
#[derive(Debug, Deserialize)]
pub struct RagResponse {
    pub recommendation: String,
    pub similar_cases: Vec<SimilarCase>,
}

// ─── Health Check ────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub python_service_url: String,
}

// ─── ESI label mapping ──────────────────────────────────────

pub const ESI_LABELS: [&str; 5] = [
    "ESI 1 (Resuscitation)",
    "ESI 2 (Emergent)",
    "ESI 3 (Urgent)",
    "ESI 4 (Less Urgent)",
    "ESI 5 (Non-Urgent)",
];
