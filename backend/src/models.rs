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

#[derive(Debug, Serialize, Clone)]
pub struct PredictResponse {
    pub predicted_esi: u8,
    pub esi_label: String,
    pub probabilities: Vec<f64>,
    pub confidence: ConfidenceMetrics,
    pub feature_vector: Vec<f64>,
    /// Real-time SHAP explainability (None if SHAP service unavailable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shap: Option<ShapExplanation>,
}

/// Confidence / uncertainty quantification derived from class probabilities.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfidenceMetrics {
    /// Probability of the predicted class (0.0–1.0)
    pub top_probability: f64,
    /// Margin between top-1 and top-2 class probabilities
    pub margin: f64,
    /// Shannon entropy across all classes (higher = more uncertain)
    pub entropy: f64,
    /// Clinical flag: true if margin < 0.15 (model is uncertain)
    pub is_uncertain: bool,
    /// Human-readable confidence label
    pub confidence_label: String,
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

// ─── Batch Triage ────────────────────────────────────────────

/// Batch of patients for mass triage simulation.
#[derive(Debug, Deserialize)]
pub struct BatchPredictRequest {
    pub patients: Vec<PatientRequest>,
}

/// Single patient result within a batch response.
#[derive(Debug, Serialize, Clone)]
pub struct BatchPatientResult {
    pub index: usize,
    pub chief_complaint: String,
    #[serde(flatten)]
    pub prediction: PredictResponse,
}

/// Batch triage response: patients sorted by severity (ESI 1 first).
#[derive(Debug, Serialize)]
pub struct BatchPredictResponse {
    pub total_patients: usize,
    pub patients: Vec<BatchPatientResult>,
    pub summary: BatchSummary,
}

#[derive(Debug, Serialize)]
pub struct BatchSummary {
    pub esi_1_count: usize,
    pub esi_2_count: usize,
    pub esi_3_count: usize,
    pub esi_4_count: usize,
    pub esi_5_count: usize,
    pub uncertain_count: usize,
}

// ─── Audit Trail ─────────────────────────────────────────────

/// A logged triage decision.
#[derive(Debug, Serialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: String,
    pub patient_hash: String,
    pub chief_complaint: String,
    pub predicted_esi: u8,
    pub confidence: f64,
    pub is_uncertain: bool,
    pub top_shap_drivers: Vec<String>,
    pub overridden: bool,
    pub override_esi: Option<u8>,
}

/// Request to override an AI triage decision.
#[derive(Debug, Deserialize)]
pub struct AuditOverrideRequest {
    pub audit_id: String,
    pub override_esi: u8,
    pub reason: String,
}

/// Query parameters for audit log retrieval.
#[derive(Debug, Deserialize)]
pub struct AuditQueryParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub esi_filter: Option<u8>,
    pub uncertain_only: Option<bool>,
}

fn default_limit() -> usize { 50 }

#[derive(Debug, Serialize)]
pub struct AuditLogResponse {
    pub total: usize,
    pub entries: Vec<AuditEntry>,
}

// ─── Confidence computation ──────────────────────────────────

impl ConfidenceMetrics {
    /// Compute confidence metrics from a probability vector.
    pub fn from_probabilities(probs: &[f64]) -> Self {
        let mut sorted = probs.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let top = sorted.get(0).copied().unwrap_or(0.0);
        let second = sorted.get(1).copied().unwrap_or(0.0);
        let margin = top - second;

        // Shannon entropy (bits)
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.log2())
            .sum();

        let is_uncertain = margin < 0.15;

        let confidence_label = if top > 0.95 {
            "Very High".to_string()
        } else if top > 0.80 {
            "High".to_string()
        } else if top > 0.60 {
            "Moderate".to_string()
        } else if top > 0.40 {
            "Low".to_string()
        } else {
            "Very Low — Recommend Manual Review".to_string()
        };

        Self {
            top_probability: (top * 10000.0).round() / 10000.0,
            margin: (margin * 10000.0).round() / 10000.0,
            entropy: (entropy * 10000.0).round() / 10000.0,
            is_uncertain,
            confidence_label,
        }
    }
}
