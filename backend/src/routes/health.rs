use std::sync::Arc;

use axum::{extract::State, Json};

use crate::models::HealthResponse;
use crate::state::AppState;

/// `GET /health` — Simple liveness + model readiness check.
pub async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: state.is_model_loaded(),
        python_service_url: state.python_service_url.clone(),
    })
}
