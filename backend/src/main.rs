// 🧊 Frostbyte — Rust Inference Backend
//
// High-performance Axum server that:
//   1. Loads the LightGBM multimodal model via FFI
//   2. Accepts patient JSON on /predict
//   3. Calls Python microservice for BERT/ResNet preprocessing
//   4. Runs native LightGBM inference in Rust memory
//   5. Returns ESI prediction + probabilities

mod models;
mod routes;
mod state;

use std::sync::Arc;

use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::state::AppState;

#[tokio::main]
async fn main() {
    // Initialize structured logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "frostbyte_backend=info,tower_http=info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load LightGBM model
    let model_path = std::env::var("FROSTBYTE_MODEL_PATH")
        .unwrap_or_else(|_| "../triage_multimodal_model.txt".to_string());

    let python_service_url = std::env::var("FROSTBYTE_PYTHON_URL")
        .unwrap_or_else(|_| "http://localhost:8000".to_string());

    tracing::info!("Loading LightGBM model from: {}", model_path);

    let state = match AppState::new(&model_path, &python_service_url) {
        Ok(s) => {
            tracing::info!("✅ LightGBM model loaded successfully");
            Arc::new(s)
        }
        Err(e) => {
            tracing::warn!("⚠️  Could not load LightGBM model: {}. Starting in degraded mode.", e);
            Arc::new(AppState::degraded(&python_service_url))
        }
    };

    // CORS — allow React frontend (typically port 3000)
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .merge(routes::router())
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:3001";
    tracing::info!("🧊 Frostbyte backend listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
