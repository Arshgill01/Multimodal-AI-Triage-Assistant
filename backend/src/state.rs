use lightgbm3::Booster;
use std::sync::Mutex;

/// Wrapper around LightGBM `Booster` that opts into Send + Sync.
///
/// SAFETY: LightGBM's C API is thread-safe for prediction when called
/// behind a Mutex. The raw `*mut c_void` inside Booster points to a
/// C++ BoosterHandle which is safe to move between threads as long as
/// concurrent access is serialized (which Mutex guarantees).
pub struct SendBooster(pub Booster);

// SAFETY: Booster is guarded by Mutex — only one thread accesses at a time.
unsafe impl Send for SendBooster {}
unsafe impl Sync for SendBooster {}

/// Shared application state, stored in an `Arc` and passed to all handlers.
pub struct AppState {
    /// LightGBM model loaded via FFI from `triage_multimodal_model.txt`.
    /// Wrapped in Mutex for thread-safe prediction.
    /// None if model couldn't be loaded (degraded mode).
    pub booster: Option<Mutex<SendBooster>>,

    /// URL of the Python preprocessing microservice (default: http://localhost:8000).
    pub python_service_url: String,

    /// HTTP client for calling the Python service.
    pub http_client: reqwest::Client,
}

impl AppState {
    /// Create state with a loaded LightGBM model.
    pub fn new(model_path: &str, python_service_url: &str) -> Result<Self, String> {
        let booster = Booster::from_file(model_path)
            .map_err(|e| format!("Failed to load LightGBM model: {:?}", e))?;

        Ok(Self {
            booster: Some(Mutex::new(SendBooster(booster))),
            python_service_url: python_service_url.to_string(),
            http_client: reqwest::Client::new(),
        })
    }

    /// Create state without a model (for development/testing).
    pub fn degraded(python_service_url: &str) -> Self {
        Self {
            booster: None,
            python_service_url: python_service_url.to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Check if the model is loaded and ready for inference.
    pub fn is_model_loaded(&self) -> bool {
        self.booster.is_some()
    }
}
