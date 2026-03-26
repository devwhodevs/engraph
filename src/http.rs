use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::{get, post}};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::config::{ApiKeyConfig, HttpConfig};
use crate::context::{self, ContextParams};
use crate::health;
use crate::llm::{EmbedModel, OrchestratorModel, RerankModel};
use crate::profile::VaultProfile;
use crate::search;
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
// Request body / query structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SearchBody {
    query: String,
    top_n: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ReadSectionQuery {
    file: String,
    heading: String,
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    folder: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    limit: Option<usize>,
    created_by: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContextBody {
    topic: String,
    budget: Option<usize>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all API endpoints.
pub fn build_router(state: ApiState) -> Router {
    Router::new()
        .route("/api/health-check", get(health_check))
        .route("/api/search", post(handle_search))
        .route("/api/read/{*file}", get(handle_read))
        .route("/api/read-section", get(handle_read_section))
        .route("/api/list", get(handle_list))
        .route("/api/vault-map", get(handle_vault_map))
        .route("/api/who/{name}", get(handle_who))
        .route("/api/project/{name}", get(handle_project))
        .route("/api/context", post(handle_context))
        .route("/api/health", get(handle_health))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "ok"
}

// ---------------------------------------------------------------------------
// Read endpoint handlers
// ---------------------------------------------------------------------------

async fn handle_search(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<SearchBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let top_n = body.top_n.unwrap_or(10);
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;

    let mut orch_guard = match &state.orchestrator {
        Some(o) => Some(o.lock().await),
        None => None,
    };
    let mut rerank_guard = match &state.reranker {
        Some(r) => Some(r.lock().await),
        None => None,
    };

    let mut config = search::SearchConfig {
        orchestrator: orch_guard
            .as_mut()
            .map(|g| g.as_mut() as &mut dyn OrchestratorModel),
        reranker: rerank_guard
            .as_mut()
            .map(|g| g.as_mut() as &mut dyn RerankModel),
        store: &store,
        rerank_candidates: 30,
    };

    let output = search::search_with_intelligence(&body.query, top_n, &mut *embedder, &mut config)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(output.results)))
}

async fn handle_read(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(file): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let note = context::context_read(&ctx, &file)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(note)))
}

async fn handle_read_section(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<ReadSectionQuery>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let result = context::read_section(&store, &state.vault_path, &params.file, &params.heading)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_list(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<ListQuery>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let limit = params.limit.unwrap_or(20);
    let items = context::context_list(
        &ctx,
        params.folder.as_deref(),
        &params.tags,
        params.created_by.as_deref(),
        limit,
    )
    .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(items)))
}

async fn handle_vault_map(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let map = context::vault_map(&ctx)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(map)))
}

async fn handle_who(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let person = context::context_who(&ctx, &name)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(person)))
}

async fn handle_project(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let proj = context::context_project(&ctx, &name)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(proj)))
}

async fn handle_context(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<ContextBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let budget = body.budget.unwrap_or(32000);
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let bundle = context::context_topic_with_search(&ctx, &body.topic, budget, &mut *embedder)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(bundle)))
}

async fn handle_health(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let profile_ref = state.profile.as_ref().as_ref();
    let config = health::HealthConfig {
        daily_folder: profile_ref.and_then(|p| p.structure.folders.daily.clone()),
        inbox_folder: profile_ref.and_then(|p| p.structure.folders.inbox.clone()),
    };
    let report = health::generate_health_report(&store, &config)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(report)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::SystemTime;

    use axum::body::Body;
    use tower::ServiceExt;

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

    /// Dummy embedder that returns zero vectors. Only used for constructing
    /// `ApiState` in tests that don't exercise search/context endpoints.
    struct DummyEmbedder;
    impl crate::llm::EmbedModel for DummyEmbedder {
        fn embed_batch(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
        }
        fn token_count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn dim(&self) -> usize {
            384
        }
    }

    fn test_api_state() -> ApiState {
        let store = Store::open_memory().expect("in-memory store");
        ApiState {
            store: Arc::new(Mutex::new(store)),
            embedder: Arc::new(Mutex::new(Box::new(DummyEmbedder) as Box<dyn EmbedModel + Send>)),
            vault_path: Arc::new(PathBuf::from("/tmp/test-vault")),
            profile: Arc::new(None),
            orchestrator: None,
            reranker: None,
            http_config: Arc::new(test_http_config()),
            no_auth: false,
            recent_writes: Arc::new(Mutex::new(HashMap::<PathBuf, SystemTime>::new())),
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

    // -----------------------------------------------------------------------
    // Integration tests using axum oneshot
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_vault_map_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_vault_map_invalid_key() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer eg_badkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_vault_map_authorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_authorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_search_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_list_authorized_empty() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/list")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_no_auth_mode_skips_check() {
        let mut state = test_api_state();
        state.no_auth = true;
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
