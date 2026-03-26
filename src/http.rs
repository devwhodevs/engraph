use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, Method};
use axum::{
    Json, Router,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Deserialize;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::config::{ApiKeyConfig, HttpConfig};
use crate::context::{self, ContextParams};
use crate::health;
use crate::llm::{EmbedModel, OrchestratorModel, RerankModel};
use crate::profile::VaultProfile;
use crate::search;
use crate::serve::RecentWrites;
use crate::store::Store;
use crate::writer::{
    self, AppendInput, CreateNoteInput, DeleteMode, EditFrontmatterInput, EditInput, EditMode,
    FrontmatterOp, RewriteInput, UpdateMetadataInput,
};

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
    pub rate_limiter: Arc<RateLimiter>,
}

// ---------------------------------------------------------------------------
// Rate limiter (in-memory token bucket)
// ---------------------------------------------------------------------------

pub struct RateLimiter {
    buckets: std::sync::Mutex<HashMap<String, RateBucket>>,
    limit: u32, // requests per minute, 0 = unlimited
}

struct RateBucket {
    tokens: u32,
    last_refill: Instant,
}

impl RateLimiter {
    pub fn new(limit: u32) -> Self {
        Self {
            buckets: std::sync::Mutex::new(HashMap::new()),
            limit,
        }
    }

    /// Check if a request is allowed. Returns Ok(()) or Err with retry-after seconds.
    pub fn check(&self, key: &str) -> Result<(), u64> {
        if self.limit == 0 {
            return Ok(());
        }
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets.entry(key.to_string()).or_insert(RateBucket {
            tokens: self.limit,
            last_refill: Instant::now(),
        });
        // Refill tokens based on elapsed time
        let elapsed = bucket.last_refill.elapsed().as_secs_f64();
        let refill = (elapsed * self.limit as f64 / 60.0) as u32;
        if refill > 0 {
            bucket.tokens = (bucket.tokens + refill).min(self.limit);
            bucket.last_refill = Instant::now();
        }
        if bucket.tokens > 0 {
            bucket.tokens -= 1;
            Ok(())
        } else {
            let retry_after = (60.0 / self.limit as f64).ceil() as u64;
            Err(retry_after)
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub headers: Vec<(String, String)>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "error": self.message });
        let mut response = (self.status, Json(body)).into_response();
        for (name, value) in &self.headers {
            if let (Ok(n), Ok(v)) = (
                axum::http::header::HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                response.headers_mut().insert(n, v);
            }
        }
        response
    }
}

impl ApiError {
    pub fn unauthorized(msg: &str) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn forbidden(msg: &str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn bad_request(msg: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn not_found(msg: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn internal(msg: &str) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn rate_limited(retry_after: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: format!("Rate limit exceeded. Retry after {retry_after}s"),
            headers: vec![("retry-after".to_string(), retry_after.to_string())],
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

/// Extract and validate auth from request headers, then check rate limit.
pub fn authorize(
    headers: &axum::http::HeaderMap,
    state: &ApiState,
    is_write: bool,
) -> Result<(), ApiError> {
    if state.no_auth {
        state
            .rate_limiter
            .check("no_auth")
            .map_err(ApiError::rate_limited)?;
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
    state
        .rate_limiter
        .check(key)
        .map_err(ApiError::rate_limited)?;
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

// -- Write request bodies --

#[derive(Debug, Deserialize)]
struct CreateBody {
    content: String,
    filename: Option<String>,
    type_hint: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    folder: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AppendBody {
    file: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditBody {
    file: String,
    heading: String,
    content: String,
    mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RewriteBody {
    file: String,
    content: String,
    preserve_frontmatter: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct EditFrontmatterBody {
    file: String,
    operations: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct MoveBody {
    file: String,
    new_folder: String,
}

#[derive(Debug, Deserialize)]
struct ArchiveBody {
    file: String,
}

#[derive(Debug, Deserialize)]
struct UnarchiveBody {
    file: String,
}

#[derive(Debug, Deserialize)]
struct UpdateMetadataBody {
    file: String,
    tags: Option<Vec<String>>,
    aliases: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct DeleteBody {
    file: String,
    mode: Option<String>,
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

fn cors_layer(origins: &[String]) -> CorsLayer {
    if origins.is_empty() {
        return CorsLayer::new();
    }
    if origins.iter().any(|o| o == "*") {
        return CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);
    }
    let origins: Vec<HeaderValue> = origins.iter().filter_map(|o| o.parse().ok()).collect();
    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
            axum::http::header::ACCEPT,
        ])
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all API endpoints.
pub fn build_router(state: ApiState) -> Router {
    let cors = cors_layer(&state.http_config.cors_origins);
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
        // Write endpoints
        .route("/api/create", post(handle_create))
        .route("/api/append", post(handle_append))
        .route("/api/edit", post(handle_edit))
        .route("/api/rewrite", post(handle_rewrite))
        .route("/api/edit-frontmatter", post(handle_edit_frontmatter))
        .route("/api/move", post(handle_move))
        .route("/api/archive", post(handle_archive))
        .route("/api/unarchive", post(handle_unarchive))
        .route("/api/update-metadata", post(handle_update_metadata))
        .route("/api/delete", post(handle_delete))
        // Migration endpoints
        .route("/api/migrate/preview", post(handle_migrate_preview))
        .route("/api/migrate/apply", post(handle_migrate_apply))
        .route("/api/migrate/undo", post(handle_migrate_undo))
        // OpenAPI / ChatGPT plugin discovery (no auth required)
        .route("/openapi.json", get(handle_openapi))
        .route("/.well-known/ai-plugin.json", get(handle_plugin_manifest))
        .layer(cors)
        .with_state(state)
}

async fn health_check() -> &'static str {
    "ok"
}

async fn handle_openapi(State(state): State<ApiState>) -> impl IntoResponse {
    let default_url = format!(
        "http://{}:{}",
        state.http_config.host, state.http_config.port
    );
    let server_url = state
        .http_config
        .plugin
        .public_url
        .as_deref()
        .unwrap_or(&default_url);
    let spec = crate::openapi::build_openapi_spec(server_url);
    Json(spec)
}

async fn handle_plugin_manifest(State(state): State<ApiState>) -> impl IntoResponse {
    let default_url = format!(
        "http://{}:{}",
        state.http_config.host, state.http_config.port
    );
    let server_url = state
        .http_config
        .plugin
        .public_url
        .as_deref()
        .unwrap_or(&default_url);
    let manifest = crate::openapi::build_plugin_manifest(&state.http_config, server_url);
    Json(manifest)
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
    let note =
        context::context_read(&ctx, &file).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
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
    let map = context::vault_map(&ctx).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
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
    let person =
        context::context_who(&ctx, &name).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
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
    let proj =
        context::context_project(&ctx, &name).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
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
// Write helpers
// ---------------------------------------------------------------------------

/// Record a write to the recent-writes map so the file watcher skips re-indexing.
async fn record_write(recent_writes: &RecentWrites, path: &std::path::Path) {
    if let Ok(meta) = std::fs::metadata(path)
        && let Ok(mtime) = meta.modified()
    {
        recent_writes.lock().await.insert(path.to_path_buf(), mtime);
    }
}

/// Parse a JSON operations array into `Vec<FrontmatterOp>`.
fn parse_frontmatter_ops(operations: &[serde_json::Value]) -> Result<Vec<FrontmatterOp>, ApiError> {
    let mut ops = Vec::with_capacity(operations.len());
    for op_val in operations {
        let op_str = op_val.get("op").and_then(|v| v.as_str()).ok_or_else(|| {
            ApiError::bad_request("each operation must have an \"op\" string field")
        })?;
        match op_str {
            "set" => {
                let key = op_val.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                    ApiError::bad_request("\"set\" operation requires a \"key\" field")
                })?;
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ApiError::bad_request("\"set\" operation requires a \"value\" field")
                    })?;
                ops.push(FrontmatterOp::Set(key.to_string(), value.to_string()));
            }
            "remove" => {
                let key = op_val.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                    ApiError::bad_request("\"remove\" operation requires a \"key\" field")
                })?;
                ops.push(FrontmatterOp::Remove(key.to_string()));
            }
            "add_tag" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ApiError::bad_request("\"add_tag\" operation requires a \"value\" field")
                    })?;
                ops.push(FrontmatterOp::AddTag(value.to_string()));
            }
            "remove_tag" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ApiError::bad_request("\"remove_tag\" operation requires a \"value\" field")
                    })?;
                ops.push(FrontmatterOp::RemoveTag(value.to_string()));
            }
            "add_alias" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ApiError::bad_request("\"add_alias\" operation requires a \"value\" field")
                    })?;
                ops.push(FrontmatterOp::AddAlias(value.to_string()));
            }
            "remove_alias" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "\"remove_alias\" operation requires a \"value\" field",
                        )
                    })?;
                ops.push(FrontmatterOp::RemoveAlias(value.to_string()));
            }
            unknown => {
                return Err(ApiError::bad_request(&format!(
                    "unknown frontmatter operation: \"{unknown}\""
                )));
            }
        }
    }
    Ok(ops)
}

// ---------------------------------------------------------------------------
// Write endpoint handlers
// ---------------------------------------------------------------------------

async fn handle_create(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<CreateBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;
    let input = CreateNoteInput {
        content: body.content,
        filename: body.filename,
        type_hint: body.type_hint,
        tags: body.tags,
        folder: body.folder,
        created_by: "http-api".into(),
    };
    let result = writer::create_note(
        input,
        &store,
        &mut *embedder,
        &state.vault_path,
        state.profile.as_ref().as_ref(),
    )
    .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_append(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<AppendBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;
    let input = AppendInput {
        file: body.file,
        content: body.content,
        modified_by: "http-api".into(),
    };
    let result = writer::append_to_note(input, &store, &mut *embedder, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_edit(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<EditBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let mode = match body.mode.as_deref().unwrap_or("append") {
        "replace" => EditMode::Replace,
        "prepend" => EditMode::Prepend,
        _ => EditMode::Append,
    };
    let input = EditInput {
        file: body.file,
        heading: body.heading,
        content: body.content,
        mode,
        modified_by: "http-api".into(),
    };
    let result = writer::edit_note(&store, &state.vault_path, &input, None)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_rewrite(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<RewriteBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let input = RewriteInput {
        file: body.file,
        content: body.content,
        preserve_frontmatter: body.preserve_frontmatter.unwrap_or(true),
        modified_by: "http-api".into(),
    };
    let result = writer::rewrite_note(&store, &state.vault_path, &input)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_edit_frontmatter(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<EditFrontmatterBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let ops = parse_frontmatter_ops(&body.operations)?;
    let store = state.store.lock().await;
    let input = EditFrontmatterInput {
        file: body.file,
        operations: ops,
        modified_by: "http-api".into(),
    };
    let result = writer::edit_frontmatter(&store, &state.vault_path, &input)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_move(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<MoveBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let result = writer::move_note(&body.file, &body.new_folder, &store, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_archive(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<ArchiveBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let result = writer::archive_note(
        &body.file,
        &store,
        &state.vault_path,
        state.profile.as_ref().as_ref(),
    )
    .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_unarchive(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<UnarchiveBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;
    let result = writer::unarchive_note(&body.file, &store, &mut *embedder, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_update_metadata(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<UpdateMetadataBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let input = UpdateMetadataInput {
        file: body.file,
        tags: body.tags,
        aliases: body.aliases,
        modified_by: "http-api".into(),
    };
    let result = writer::update_metadata(input, &store, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

// ---------------------------------------------------------------------------
// Migration endpoint handlers
// ---------------------------------------------------------------------------

async fn handle_migrate_preview(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let profile_ref = state.profile.as_ref().as_ref();
    let preview = crate::migrate::generate_preview(&store, &state.vault_path, profile_ref)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::to_value(&preview).unwrap()))
}

#[derive(Deserialize)]
struct MigrateApplyBody {
    preview: serde_json::Value,
}

async fn handle_migrate_apply(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<MigrateApplyBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let preview: crate::migrate::MigrationPreview = serde_json::from_value(body.preview)
        .map_err(|e| ApiError::bad_request(&format!("Invalid preview: {e}")))?;
    let result = crate::migrate::apply_preview(&preview, &store, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::to_value(&result).unwrap()))
}

async fn handle_migrate_undo(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let result = crate::migrate::undo_last(&store, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::to_value(&result).unwrap()))
}

async fn handle_delete(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<DeleteBody>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let mode = match body.mode.as_deref().unwrap_or("soft") {
        "hard" => DeleteMode::Hard,
        _ => DeleteMode::Soft,
    };
    let archive_folder = state
        .profile
        .as_ref()
        .as_ref()
        .and_then(|p| p.structure.folders.archive.as_deref())
        .unwrap_or("04-Archive");
    writer::delete_note(&store, &state.vault_path, &body.file, mode, archive_folder)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!({
        "deleted": body.file,
        "mode": body.mode.as_deref().unwrap_or("soft"),
    })))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
            plugin: crate::config::PluginConfig::default(),
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
        let config = test_http_config();
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit));
        ApiState {
            store: Arc::new(Mutex::new(store)),
            embedder: Arc::new(Mutex::new(
                Box::new(DummyEmbedder) as Box<dyn EmbedModel + Send>
            )),
            vault_path: Arc::new(PathBuf::from("/tmp/test-vault")),
            profile: Arc::new(None),
            orchestrator: None,
            reranker: None,
            http_config: Arc::new(config),
            no_auth: false,
            recent_writes: Arc::new(Mutex::new(HashMap::<PathBuf, SystemTime>::new())),
            rate_limiter,
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

    // -----------------------------------------------------------------------
    // Write endpoint permission tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_write_endpoint_read_key_rejected() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/create")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::from(r##"{"content":"# Test"}"##))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_write_endpoint_write_key_accepted() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/edit")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_writekey")
                    .body(Body::from(
                        r#"{"file":"nonexistent","heading":"Test","content":"new"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Should be 500 (file not found via store) but NOT 403
        assert_ne!(response.status(), StatusCode::FORBIDDEN);
    }

    // -----------------------------------------------------------------------
    // Rate limiter unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let limiter = RateLimiter::new(5);
        for _ in 0..5 {
            assert!(limiter.check("key1").is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_rejects_over_limit() {
        let limiter = RateLimiter::new(2);
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_err());
    }

    #[test]
    fn test_rate_limiter_unlimited() {
        let limiter = RateLimiter::new(0);
        for _ in 0..1000 {
            assert!(limiter.check("key1").is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_separate_keys() {
        let limiter = RateLimiter::new(1);
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key2").is_ok()); // different key, separate bucket
        assert!(limiter.check("key1").is_err()); // key1 exhausted
    }

    #[tokio::test]
    async fn test_rate_limit_returns_429() {
        let mut state = test_api_state();
        state.rate_limiter = Arc::new(RateLimiter::new(1));
        let app = build_router(state);
        // First request passes (consumes the single token)
        let response = app
            .clone()
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
        // Second request gets 429
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
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().get("retry-after").is_some());
    }

    // -----------------------------------------------------------------------
    // OpenAPI / Plugin manifest tests (no auth required)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_openapi_no_auth_required() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_plugin_manifest_no_auth_required() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/.well-known/ai-plugin.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
