use std::sync::Arc;

use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::get};
use tokio::sync::Mutex;

use crate::config::{ApiKeyConfig, HttpConfig};
use crate::llm::{EmbedModel, OrchestratorModel, RerankModel};
use crate::profile::VaultProfile;
use crate::serve::RecentWrites;
use crate::store::Store;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApiState {
    pub store: Arc<Mutex<Store>>,
    pub embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    pub vault_path: Arc<std::path::PathBuf>,
    pub profile: Arc<Option<VaultProfile>>,
    pub orchestrator: Option<Arc<Mutex<Box<dyn OrchestratorModel + Send>>>>,
    pub reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>>,
    pub http_config: Arc<HttpConfig>,
    pub no_auth: bool,
    pub recent_writes: RecentWrites,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}

impl ApiError {
    pub fn unauthorized(msg: &str) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.to_string(),
        }
    }
    pub fn forbidden(msg: &str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.to_string(),
        }
    }
    pub fn bad_request(msg: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.to_string(),
        }
    }
    pub fn not_found(msg: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.to_string(),
        }
    }
    pub fn internal(msg: &str) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

/// Validate API key from Authorization header. Returns the matching key config.
pub fn validate_api_key<'a>(key: &str, config: &'a HttpConfig) -> Option<&'a ApiKeyConfig> {
    config.api_keys.iter().find(|k| k.key == key)
}

/// Check if a permission level allows the requested operation.
pub fn check_permission(permission: &str, is_write: bool) -> bool {
    if !is_write {
        return true;
    }
    permission == "write"
}

/// Extract and validate auth from request headers.
pub fn authorize(
    headers: &axum::http::HeaderMap,
    state: &ApiState,
    is_write: bool,
) -> Result<(), ApiError> {
    if state.no_auth {
        return Ok(());
    }
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError::unauthorized("Missing Authorization header"))?;
    let key = auth
        .strip_prefix("Bearer ")
        .ok_or_else(|| ApiError::unauthorized("Authorization must use Bearer scheme"))?;
    let key_config = validate_api_key(key, &state.http_config)
        .ok_or_else(|| ApiError::unauthorized("Invalid API key"))?;
    if !check_permission(&key_config.permissions, is_write) {
        return Err(ApiError::forbidden(
            "Insufficient permissions: write access required",
        ));
    }
    Ok(())
}

/// Generate a new API key with `eg_` prefix + 32 hex chars.
pub fn generate_api_key() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let hex: String = (0..32)
        .map(|_| format!("{:x}", rng.random_range(0..16u8)))
        .collect();
    format!("eg_{hex}")
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all API endpoints.
pub fn build_router(state: ApiState) -> Router {
    Router::new()
        // Placeholder -- handlers will be added in Tasks 3-5
        .route("/api/health-check", get(health_check))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "ok"
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_http_config() -> HttpConfig {
        HttpConfig {
            enabled: true,
            port: 3000,
            host: "127.0.0.1".to_string(),
            rate_limit: 0,
            cors_origins: vec![],
            api_keys: vec![
                ApiKeyConfig {
                    key: "eg_readkey".into(),
                    name: "reader".into(),
                    permissions: "read".into(),
                },
                ApiKeyConfig {
                    key: "eg_writekey".into(),
                    name: "writer".into(),
                    permissions: "write".into(),
                },
            ],
        }
    }

    #[test]
    fn test_validate_api_key_valid() {
        let config = test_http_config();
        let result = validate_api_key("eg_readkey", &config);
        assert!(result.is_some());
        assert_eq!(result.unwrap().permissions, "read");
    }

    #[test]
    fn test_validate_api_key_invalid() {
        let config = test_http_config();
        assert!(validate_api_key("eg_badkey", &config).is_none());
    }

    #[test]
    fn test_generate_api_key_format() {
        let key = generate_api_key();
        assert!(key.starts_with("eg_"));
        assert_eq!(key.len(), 35); // "eg_" + 32 hex chars
    }

    #[test]
    fn test_check_permission_read_on_read() {
        assert!(check_permission("read", false));
    }

    #[test]
    fn test_check_permission_read_on_write() {
        assert!(!check_permission("read", true));
    }

    #[test]
    fn test_check_permission_write_on_write() {
        assert!(check_permission("write", true));
    }

    #[test]
    fn test_check_permission_write_on_read() {
        assert!(check_permission("write", false));
    }
}
