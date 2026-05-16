export interface ConfidenceMetrics {
    top_probability: number;
    margin: number;
    entropy: number;
    is_uncertain: boolean;
    confidence_label: string;
}

export interface ShapFeature {
    name: string;
    value: number;
    shap_value: number;
}

export interface ShapExplanation {
    base_value: number;
    features: ShapFeature[];
    predicted_class: number;
    prediction_label: string;
}

export interface PredictResponse {
    predicted_esi: number;
    esi_label: string;
    probabilities: number[];
    confidence: ConfidenceMetrics;
    feature_vector: number[];
    shap?: ShapExplanation;
    audit_id: string;
}

export interface BatchPatientResult {
    index: number;
    chief_complaint: string;
    predicted_esi: number;
    esi_label: string;
    probabilities: number[];
    confidence: ConfidenceMetrics;
    feature_vector: number[];
    shap?: ShapExplanation;
    audit_id: string;
}

export interface BatchSummary {
    esi_1_count: number;
    esi_2_count: number;
    esi_3_count: number;
    esi_4_count: number;
    esi_5_count: number;
    uncertain_count: number;
}

export interface BatchPredictResponse {
    total_patients: number;
    patients: BatchPatientResult[];
    summary: BatchSummary;
}

export interface AuditEntry {
    id: string;
    timestamp: string;
    patient_hash: string;
    chief_complaint: string;
    predicted_esi: number;
    confidence: number;
    is_uncertain: boolean;
    top_shap_drivers: string[];
    overridden: boolean;
    override_esi: number | null;
    override_reason?: string;
}

export interface AuditSummary {
    total_cases: number;
    uncertain_count: number;
    override_count: number;
    esi_distribution: Array<{ esi: number; count: number }>;
}

export interface SimilarCaseEvidence {
    complaint: string;
    target_esi: number;
    similarity: number;
    text_similarity?: number;
    vitals_similarity?: number;
    source?: "text" | "vitals" | "both";
    flag_high_risk?: number;
    heart_rate?: number;
    spo2?: number;
}
