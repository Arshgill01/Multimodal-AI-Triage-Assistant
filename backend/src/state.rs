use lightgbm3::Booster;
use rusqlite::Connection;
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

    /// SQLite connection for the clinical audit trail.
    pub audit_db: Mutex<Connection>,
}

impl AppState {
    /// Create state with a loaded LightGBM model.
    pub fn new(model_path: &str, python_service_url: &str, db_path: &str) -> Result<Self, String> {
        let booster = Booster::from_file(model_path)
            .map_err(|e| format!("Failed to load LightGBM model: {:?}", e))?;

        let db = Self::init_audit_db(db_path)?;

        Ok(Self {
            booster: Some(Mutex::new(SendBooster(booster))),
            python_service_url: python_service_url.to_string(),
            http_client: reqwest::Client::new(),
            audit_db: Mutex::new(db),
        })
    }

    /// Create state without a model (for development/testing).
    pub fn degraded(python_service_url: &str, db_path: &str) -> Result<Self, String> {
        let db = Self::init_audit_db(db_path)?;

        Ok(Self {
            booster: None,
            python_service_url: python_service_url.to_string(),
            http_client: reqwest::Client::new(),
            audit_db: Mutex::new(db),
        })
    }

    /// Check if the model is loaded and ready for inference.
    pub fn is_model_loaded(&self) -> bool {
        self.booster.is_some()
    }

    /// Initialize the SQLite audit trail database.
    fn init_audit_db(db_path: &str) -> Result<Connection, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open audit DB: {}", e))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS audit_log (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                patient_hash    TEXT NOT NULL,
                chief_complaint TEXT NOT NULL,
                predicted_esi   INTEGER NOT NULL,
                confidence      REAL NOT NULL,
                is_uncertain    INTEGER NOT NULL DEFAULT 0,
                top_shap_drivers TEXT NOT NULL DEFAULT '',
                overridden      INTEGER NOT NULL DEFAULT 0,
                override_esi    INTEGER,
                override_reason TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_esi ON audit_log(predicted_esi);",
        )
        .map_err(|e| format!("Failed to initialize audit tables: {}", e))?;

        Ok(conn)
    }

    /// Log a triage decision to the audit trail.
    pub fn log_decision(
        &self,
        patient_hash: &str,
        chief_complaint: &str,
        predicted_esi: u8,
        confidence: f64,
        is_uncertain: bool,
        top_shap_drivers: &[String],
    ) -> Result<String, String> {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().to_rfc3339();
        let shap_str = top_shap_drivers.join("|");

        let db = self.audit_db.lock().map_err(|e| format!("DB lock: {}", e))?;
        db.execute(
            "INSERT INTO audit_log (id, timestamp, patient_hash, chief_complaint, \
             predicted_esi, confidence, is_uncertain, top_shap_drivers) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                id,
                timestamp,
                patient_hash,
                chief_complaint,
                predicted_esi,
                confidence,
                is_uncertain as i32,
                shap_str,
            ],
        )
        .map_err(|e| format!("Failed to log decision: {}", e))?;

        Ok(id)
    }
}
