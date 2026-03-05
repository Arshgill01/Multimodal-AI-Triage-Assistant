pub mod health;
pub mod next_steps;
pub mod predict;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};

use crate::state::AppState;

/// Mount all routes onto the Axum router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(health::health_check))
        .route("/predict", post(predict::predict))
        .route("/next-steps", post(next_steps::next_steps))
}
