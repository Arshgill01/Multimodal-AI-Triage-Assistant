use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};

use crate::models::{
    AuditEntry, AuditLogResponse, AuditOverrideRequest, AuditQueryParams,
    AuditSummary, EsiCount,
};
use crate::state::AppState;

/// `GET /audit-log` — Retrieve recent triage decisions from the audit trail.
///
/// Supports filtering by ESI level and uncertain-only predictions.
pub async fn audit_log(
    State(state): State<Arc<AppState>>,
    Query(params): Query<AuditQueryParams>,
) -> Result<Json<AuditLogResponse>, (StatusCode, String)> {
    let db = state.audit_db.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("DB lock error: {}", e),
        )
    })?;

    let mut query = String::from(
        "SELECT id, timestamp, patient_hash, chief_complaint, predicted_esi, \
         confidence, is_uncertain, top_shap_drivers, overridden, override_esi \
         FROM audit_log WHERE 1=1",
    );
    let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(esi) = params.esi_filter {
        query.push_str(" AND predicted_esi = ?");
        sql_params.push(Box::new(esi));
    }

    if params.uncertain_only.unwrap_or(false) {
        query.push_str(" AND is_uncertain = 1");
    }

    if params.overridden_only.unwrap_or(false) {
        query.push_str(" AND overridden = 1");
    }

    query.push_str(" ORDER BY timestamp DESC LIMIT ?");
    sql_params.push(Box::new(params.limit as i64));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = sql_params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = db.prepare(&query).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("SQL prepare error: {}", e),
        )
    })?;

    let entries: Vec<AuditEntry> = stmt
        .query_map(param_refs.as_slice(), |row| {
            let shap_str: String = row.get(7)?;
            let top_shap_drivers: Vec<String> = shap_str
                .split('|')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();

            Ok(AuditEntry {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                patient_hash: row.get(2)?,
                chief_complaint: row.get(3)?,
                predicted_esi: row.get(4)?,
                confidence: row.get(5)?,
                is_uncertain: row.get(6)?,
                top_shap_drivers,
                overridden: row.get(8)?,
                override_esi: row.get(9)?,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("SQL query error: {}", e),
            )
        })?
        .filter_map(|r| r.ok())
        .collect();

    let total = entries.len();
    Ok(Json(AuditLogResponse { total, entries }))
}

/// `POST /audit/override` — Record a clinician's override of an AI triage decision.
pub async fn audit_override(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AuditOverrideRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let db = state.audit_db.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("DB lock error: {}", e),
        )
    })?;

    let updated = db
        .execute(
            "UPDATE audit_log SET overridden = 1, override_esi = ?, override_reason = ? WHERE id = ?",
            rusqlite::params![req.override_esi, req.reason, req.audit_id],
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("SQL update error: {}", e),
            )
        })?;

    if updated == 0 {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Audit entry '{}' not found", req.audit_id),
        ));
    }

    tracing::info!(
        "📋 Clinician override: {} → ESI {} (reason: {})",
        req.audit_id,
        req.override_esi,
        req.reason
    );

    Ok(Json(serde_json::json!({
        "status": "overridden",
        "audit_id": req.audit_id,
        "new_esi": req.override_esi,
    })))
}

/// `GET /audit-summary` — Get trust metrics for the audit trail.
pub async fn audit_summary(
    State(state): State<Arc<AppState>>,
) -> Result<Json<AuditSummary>, (StatusCode, String)> {
    let db = state.audit_db.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("DB lock error: {}", e),
        )
    })?;

    let total_cases: usize = db
        .query_row("SELECT COUNT(*) FROM audit_log", [], |row| row.get(0))
        .unwrap_or(0);

    let uncertain_count: usize = db
        .query_row("SELECT COUNT(*) FROM audit_log WHERE is_uncertain = 1", [], |row| row.get(0))
        .unwrap_or(0);

    let override_count: usize = db
        .query_row("SELECT COUNT(*) FROM audit_log WHERE overridden = 1", [], |row| row.get(0))
        .unwrap_or(0);

    let mut stmt = db
        .prepare("SELECT predicted_esi, COUNT(*) FROM audit_log GROUP BY predicted_esi")
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("SQL prepare error: {}", e),
            )
        })?;

    let esi_distribution: Vec<EsiCount> = stmt
        .query_map([], |row| {
            Ok(EsiCount {
                esi: row.get(0)?,
                count: row.get(1)?,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("SQL query error: {}", e),
            )
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Json(AuditSummary {
        total_cases,
        uncertain_count,
        override_count,
        esi_distribution,
    }))
}
