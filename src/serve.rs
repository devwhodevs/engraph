use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::schemars;
use rmcp::schemars::JsonSchema;
use rmcp::{ErrorData as McpError, ServiceExt, tool, tool_handler, tool_router};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::context::{self, ContextParams};
use crate::llm::{EmbedModel, OrchestratorModel, RerankModel};
use crate::profile::VaultProfile;
use crate::search;
use crate::store::Store;
use crate::writer::FrontmatterOp;

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// The search query.
    pub query: String,
    /// Number of results (default 10).
    pub top_n: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadParams {
    /// File path, basename, or #docid.
    pub file: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListParams {
    /// Filter to folder path prefix.
    pub folder: Option<String>,
    /// Filter to notes with all listed tags.
    pub tags: Option<Vec<String>>,
    /// Filter to notes created by a specific agent.
    pub created_by: Option<String>,
    /// Maximum results (default 20).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WhoParams {
    /// Person name (matches filename in People folder).
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProjectParams {
    /// Project name (matches filename).
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ContextToolParams {
    /// Search query for the topic.
    pub topic: String,
    /// Character budget (default 32000).
    pub budget: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateParams {
    /// Note content (markdown body).
    pub content: String,
    /// Optional filename (without .md). Auto-generated if omitted.
    pub filename: Option<String>,
    /// Type hint for placement: "person", "daily", "meeting", "decision".
    pub type_hint: Option<String>,
    /// Proposed tags (auto-resolved against registry).
    pub tags: Option<Vec<String>>,
    /// Explicit folder path (skips placement engine).
    pub folder: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AppendParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// Content to append to the note.
    pub content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateMetadataParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// New tags (replaces existing).
    pub tags: Option<Vec<String>>,
    /// New aliases.
    pub aliases: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MoveNoteParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// New folder path (relative to vault root).
    pub new_folder: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArchiveParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnarchiveParams {
    /// Archived note path (e.g., "04-Archive/01-Projects/note.md").
    pub file: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadSectionParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// Section heading to read (case-insensitive).
    pub heading: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HealthParams {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MigratePreviewParams {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MigrateApplyParams {
    /// Migration preview JSON (from migrate_preview).
    pub preview: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MigrateUndoParams {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// Section heading to edit (case-insensitive).
    pub heading: String,
    /// Content to add/replace in the section.
    pub content: String,
    /// Edit mode: "replace", "prepend", or "append" (default: "append").
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RewriteParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// New body content (replaces everything below frontmatter).
    pub content: String,
    /// Whether to preserve existing frontmatter (default: true).
    pub preserve_frontmatter: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditFrontmatterParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// Operations to apply. Array of objects like {"op": "add_tag", "value": "rust"} or {"op": "set", "key": "status", "value": "done"} or {"op": "remove", "key": "status"} or {"op": "remove_tag", "value": "old"}.
    pub operations: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteParams {
    /// Target note: file path, basename, or #docid.
    pub file: String,
    /// Delete mode: "soft" (archive, default) or "hard" (permanent).
    pub mode: Option<String>,
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Map of recently-written file paths to their mtime.
/// Used to tell the watcher "I just wrote this file, skip re-indexing it."
pub type RecentWrites = Arc<Mutex<HashMap<PathBuf, SystemTime>>>;

#[derive(Clone)]
pub struct EngraphServer {
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    vault_path: Arc<PathBuf>,
    profile: Arc<Option<VaultProfile>>,
    tool_router: ToolRouter<Self>,
    /// Query expansion orchestrator (None when intelligence is disabled or failed to load).
    orchestrator: Option<Arc<Mutex<Box<dyn OrchestratorModel + Send>>>>,
    /// Result reranker (None when intelligence is disabled or failed to load).
    reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>>,
    /// Tracks files recently written by MCP tools so the watcher can skip re-indexing them.
    recent_writes: RecentWrites,
    /// When true, write/edit/delete MCP tools return an error instead of executing.
    read_only: bool,
}

fn read_only_err() -> McpError {
    McpError::new(
        rmcp::model::ErrorCode::INVALID_REQUEST,
        "Write operations disabled in read-only mode. Start server without --read-only to enable writes.".to_string(),
        None::<serde_json::Value>,
    )
}

fn mcp_err(e: &anyhow::Error) -> McpError {
    McpError::new(
        rmcp::model::ErrorCode::INTERNAL_ERROR,
        format!("{e:#}"),
        None::<serde_json::Value>,
    )
}

fn to_json_result<T: serde::Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let json = serde_json::to_string_pretty(value).map_err(|e| {
        McpError::new(
            rmcp::model::ErrorCode::INTERNAL_ERROR,
            e.to_string(),
            None::<serde_json::Value>,
        )
    })?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

/// Record a recently-written file path + mtime so the watcher can skip re-indexing it.
async fn record_write(recent_writes: &RecentWrites, path: &Path) {
    if let Ok(meta) = std::fs::metadata(path)
        && let Ok(mtime) = meta.modified()
    {
        recent_writes.lock().await.insert(path.to_path_buf(), mtime);
    }
}

/// Parse a JSON operations array into `Vec<FrontmatterOp>`.
fn parse_frontmatter_ops(operations: &[serde_json::Value]) -> Result<Vec<FrontmatterOp>, McpError> {
    let mut ops = Vec::with_capacity(operations.len());
    for op_val in operations {
        let op_str = op_val.get("op").and_then(|v| v.as_str()).ok_or_else(|| {
            McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "each operation must have an \"op\" string field",
                None::<serde_json::Value>,
            )
        })?;
        match op_str {
            "set" => {
                let key = op_val.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                    McpError::new(
                        rmcp::model::ErrorCode::INVALID_PARAMS,
                        "\"set\" operation requires a \"key\" field",
                        None::<serde_json::Value>,
                    )
                })?;
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        McpError::new(
                            rmcp::model::ErrorCode::INVALID_PARAMS,
                            "\"set\" operation requires a \"value\" field",
                            None::<serde_json::Value>,
                        )
                    })?;
                ops.push(FrontmatterOp::Set(key.to_string(), value.to_string()));
            }
            "remove" => {
                let key = op_val.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                    McpError::new(
                        rmcp::model::ErrorCode::INVALID_PARAMS,
                        "\"remove\" operation requires a \"key\" field",
                        None::<serde_json::Value>,
                    )
                })?;
                ops.push(FrontmatterOp::Remove(key.to_string()));
            }
            "add_tag" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        McpError::new(
                            rmcp::model::ErrorCode::INVALID_PARAMS,
                            "\"add_tag\" operation requires a \"value\" field",
                            None::<serde_json::Value>,
                        )
                    })?;
                ops.push(FrontmatterOp::AddTag(value.to_string()));
            }
            "remove_tag" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        McpError::new(
                            rmcp::model::ErrorCode::INVALID_PARAMS,
                            "\"remove_tag\" operation requires a \"value\" field",
                            None::<serde_json::Value>,
                        )
                    })?;
                ops.push(FrontmatterOp::RemoveTag(value.to_string()));
            }
            "add_alias" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        McpError::new(
                            rmcp::model::ErrorCode::INVALID_PARAMS,
                            "\"add_alias\" operation requires a \"value\" field",
                            None::<serde_json::Value>,
                        )
                    })?;
                ops.push(FrontmatterOp::AddAlias(value.to_string()));
            }
            "remove_alias" => {
                let value = op_val
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        McpError::new(
                            rmcp::model::ErrorCode::INVALID_PARAMS,
                            "\"remove_alias\" operation requires a \"value\" field",
                            None::<serde_json::Value>,
                        )
                    })?;
                ops.push(FrontmatterOp::RemoveAlias(value.to_string()));
            }
            unknown => {
                return Err(McpError::new(
                    rmcp::model::ErrorCode::INVALID_PARAMS,
                    format!("unknown frontmatter operation: \"{unknown}\""),
                    None::<serde_json::Value>,
                ));
            }
        }
    }
    Ok(ops)
}

#[tool_router]
impl EngraphServer {
    #[tool(
        name = "search",
        description = "Semantic + keyword hybrid search across the vault. Returns ranked results with file paths, scores, headings, and snippets."
    )]
    async fn search(&self, params: Parameters<SearchParams>) -> Result<CallToolResult, McpError> {
        let top_n = params.0.top_n.unwrap_or(10);
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;

        // Lock orchestrator and reranker if available for intelligence-enhanced search.
        let mut orch_guard = match &self.orchestrator {
            Some(o) => Some(o.lock().await),
            None => None,
        };
        let mut rerank_guard = match &self.reranker {
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

        let output =
            search::search_with_intelligence(&params.0.query, top_n, &mut *embedder, &mut config)
                .map_err(|e| mcp_err(&e))?;
        to_json_result(&output.results)
    }

    #[tool(
        name = "read",
        description = "Read a note's full content with metadata, tags, and graph edges. Accepts file path, basename, or #docid."
    )]
    async fn read(&self, params: Parameters<ReadParams>) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let note = context::context_read(&ctx, &params.0.file).map_err(|e| mcp_err(&e))?;
        to_json_result(&note)
    }

    #[tool(
        name = "list",
        description = "List notes filtered by folder prefix and/or tags. Returns paths, docids, tags, and edge counts."
    )]
    async fn list(&self, params: Parameters<ListParams>) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let tags = params.0.tags.unwrap_or_default();
        let limit = params.0.limit.unwrap_or(20);
        let items = context::context_list(
            &ctx,
            params.0.folder.as_deref(),
            &tags,
            params.0.created_by.as_deref(),
            limit,
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&items)
    }

    #[tool(
        name = "vault_map",
        description = "Vault structure overview: folders, tags, file counts, recent files. Use to orient before deeper queries."
    )]
    async fn vault_map(&self) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let map = context::vault_map(&ctx).map_err(|e| mcp_err(&e))?;
        to_json_result(&map)
    }

    #[tool(
        name = "who",
        description = "Person context bundle: their note, mentions across the vault, and graph connections."
    )]
    async fn who(&self, params: Parameters<WhoParams>) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let person = context::context_who(&ctx, &params.0.name).map_err(|e| mcp_err(&e))?;
        to_json_result(&person)
    }

    #[tool(
        name = "project",
        description = "Project context bundle: project note, child notes, active tasks, team members, and recent daily mentions."
    )]
    async fn project(&self, params: Parameters<ProjectParams>) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let proj = context::context_project(&ctx, &params.0.name).map_err(|e| mcp_err(&e))?;
        to_json_result(&proj)
    }

    #[tool(
        name = "context",
        description = "Rich topic context with search-driven section selection and character budget trimming. Returns the most relevant note sections for a topic."
    )]
    async fn context(
        &self,
        params: Parameters<ContextToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let budget = params.0.budget.unwrap_or(32000);
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let bundle =
            context::context_topic_with_search(&ctx, &params.0.topic, budget, &mut *embedder)
                .map_err(|e| mcp_err(&e))?;
        to_json_result(&bundle)
    }

    #[tool(
        name = "create",
        description = "Create a new note with automatic tag resolution, link discovery, and folder placement. Returns the created file's path, docid, and what was auto-resolved."
    )]
    async fn create(&self, params: Parameters<CreateParams>) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let input = crate::writer::CreateNoteInput {
            content: params.0.content,
            filename: params.0.filename,
            type_hint: params.0.type_hint,
            tags: params.0.tags.unwrap_or_default(),
            folder: params.0.folder,
            created_by: "claude-code".into(),
        };
        let result = crate::writer::create_note(
            input,
            &store,
            &mut *embedder,
            &self.vault_path,
            self.profile.as_ref().as_ref(),
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "append",
        description = "Append content to an existing note. Safe: only adds content, never overwrites. Detects conflicts via mtime checking."
    )]
    async fn append(&self, params: Parameters<AppendParams>) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let input = crate::writer::AppendInput {
            file: params.0.file,
            content: params.0.content,
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::append_to_note(input, &store, &mut *embedder, &self.vault_path)
            .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "update_metadata",
        description = "Update a note's tags or aliases. Uses mtime conflict detection."
    )]
    async fn update_metadata(
        &self,
        params: Parameters<UpdateMetadataParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let input = crate::writer::UpdateMetadataInput {
            file: params.0.file,
            tags: params.0.tags,
            aliases: params.0.aliases,
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::update_metadata(input, &store, &self.vault_path)
            .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "move_note",
        description = "Move a note to a different folder. Updates the index path."
    )]
    async fn move_note(
        &self,
        params: Parameters<MoveNoteParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let result = crate::writer::move_note(
            &params.0.file,
            &params.0.new_folder,
            &store,
            &self.vault_path,
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "archive",
        description = "Archive a note: moves it to the archive folder, removes from search index. The note is preserved on disk but invisible to search/context. Use unarchive to restore."
    )]
    async fn archive(&self, params: Parameters<ArchiveParams>) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let result = crate::writer::archive_note(
            &params.0.file,
            &store,
            &self.vault_path,
            self.profile.as_ref().as_ref(),
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "unarchive",
        description = "Restore an archived note to its original location and re-index it for search."
    )]
    async fn unarchive(
        &self,
        params: Parameters<UnarchiveParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let result =
            crate::writer::unarchive_note(&params.0.file, &store, &mut *embedder, &self.vault_path)
                .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "read_section",
        description = "Read a specific heading section from a note. Returns content from that heading to the next same-level heading."
    )]
    async fn read_section(
        &self,
        params: Parameters<ReadSectionParams>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let result =
            context::read_section(&store, &self.vault_path, &params.0.file, &params.0.heading)
                .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "health",
        description = "Vault health report: orphans, broken links, stale notes, tag hygiene, index freshness."
    )]
    async fn health(&self, _params: Parameters<HealthParams>) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let profile_ref = self.profile.as_ref().as_ref();
        let config = crate::health::HealthConfig {
            daily_folder: profile_ref.and_then(|p| p.structure.folders.daily.clone()),
            inbox_folder: profile_ref.and_then(|p| p.structure.folders.inbox.clone()),
        };
        let report =
            crate::health::generate_health_report(&store, &config).map_err(|e| mcp_err(&e))?;
        to_json_result(&report)
    }

    #[tool(
        name = "edit",
        description = "Edit a specific section of a note. Supports replace, prepend, or append modes. Targets sections by heading name."
    )]
    async fn edit(&self, params: Parameters<EditParams>) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let mode = match params.0.mode.as_deref().unwrap_or("append") {
            "replace" => crate::writer::EditMode::Replace,
            "prepend" => crate::writer::EditMode::Prepend,
            _ => crate::writer::EditMode::Append,
        };
        let input = crate::writer::EditInput {
            file: params.0.file,
            heading: params.0.heading,
            content: params.0.content,
            mode,
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::edit_note(&store, &self.vault_path, &input, None)
            .map_err(|e| mcp_err(&e))?;
        // Record write so the watcher skips re-indexing
        let full_path = self.vault_path.join(&result.path);
        record_write(&self.recent_writes, &full_path).await;
        to_json_result(&result)
    }

    #[tool(
        name = "rewrite",
        description = "Replace the entire body of a note. Optionally preserves existing frontmatter. Use for major content overhauls."
    )]
    async fn rewrite(&self, params: Parameters<RewriteParams>) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let input = crate::writer::RewriteInput {
            file: params.0.file,
            content: params.0.content,
            preserve_frontmatter: params.0.preserve_frontmatter.unwrap_or(true),
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::rewrite_note(&store, &self.vault_path, &input)
            .map_err(|e| mcp_err(&e))?;
        let full_path = self.vault_path.join(&result.path);
        record_write(&self.recent_writes, &full_path).await;
        to_json_result(&result)
    }

    #[tool(
        name = "edit_frontmatter",
        description = "Edit frontmatter fields with granular operations: set/remove properties, add/remove tags, add/remove aliases."
    )]
    async fn edit_frontmatter(
        &self,
        params: Parameters<EditFrontmatterParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let ops = parse_frontmatter_ops(&params.0.operations)?;
        let store = self.store.lock().await;
        let input = crate::writer::EditFrontmatterInput {
            file: params.0.file,
            operations: ops,
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::edit_frontmatter(&store, &self.vault_path, &input)
            .map_err(|e| mcp_err(&e))?;
        let full_path = self.vault_path.join(&result.path);
        record_write(&self.recent_writes, &full_path).await;
        to_json_result(&result)
    }

    #[tool(
        name = "migrate_preview",
        description = "Generate PARA migration preview. Classifies all notes into Projects/Areas/Resources/Archive and returns proposed moves with confidence scores."
    )]
    async fn migrate_preview(
        &self,
        _params: Parameters<MigratePreviewParams>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let profile_ref = self.profile.as_ref().as_ref();
        let preview = crate::migrate::generate_preview(&store, &self.vault_path, profile_ref)
            .map_err(|e| mcp_err(&e))?;
        to_json_result(&preview)
    }

    #[tool(
        name = "migrate_apply",
        description = "Apply a PARA migration preview. Moves files to their classified PARA locations. Reversible via migrate_undo."
    )]
    async fn migrate_apply(
        &self,
        params: Parameters<MigrateApplyParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let preview: crate::migrate::MigrationPreview = serde_json::from_value(params.0.preview)
            .map_err(|e| mcp_err(&anyhow::anyhow!("Invalid preview JSON: {e}")))?;
        let result = crate::migrate::apply_preview(&preview, &store, &self.vault_path)
            .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "migrate_undo",
        description = "Undo the most recent PARA migration, restoring all moved files to their original locations."
    )]
    async fn migrate_undo(
        &self,
        _params: Parameters<MigrateUndoParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let result =
            crate::migrate::undo_last(&store, &self.vault_path).map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "delete",
        description = "Delete a note. Soft mode (default) moves it to the archive folder. Hard mode permanently removes it from disk and index."
    )]
    async fn delete(&self, params: Parameters<DeleteParams>) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let mode = match params.0.mode.as_deref().unwrap_or("soft") {
            "hard" => crate::writer::DeleteMode::Hard,
            _ => crate::writer::DeleteMode::Soft,
        };
        let archive_folder = self
            .profile
            .as_ref()
            .as_ref()
            .and_then(|p| p.structure.folders.archive.as_deref())
            .unwrap_or("04-Archive");
        crate::writer::delete_note(
            &store,
            &self.vault_path,
            &params.0.file,
            mode,
            archive_folder,
        )
        .map_err(|e| mcp_err(&e))?;
        let result = serde_json::json!({
            "deleted": params.0.file,
            "mode": params.0.mode.as_deref().unwrap_or("soft"),
        });
        to_json_result(&result)
    }
}

#[tool_handler]
impl rmcp::handler::server::ServerHandler for EngraphServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "engraph: vault intelligence for Obsidian. \
                 Read: vault_map to orient, search to find, read/read_section for content, who/project for context bundles, health for vault diagnostics. \
                 Write: create for new notes, append to add content, edit to modify a section, rewrite to replace body, \
                 edit_frontmatter for tags/properties, update_metadata for bulk tag/alias replacement. \
                 Lifecycle: move_note to relocate, archive to soft-delete, unarchive to restore, delete for permanent removal. \
                 Migration: migrate_preview to classify notes into PARA folders, migrate_apply to execute the migration, migrate_undo to revert.",
        )
    }
}

// ---------------------------------------------------------------------------
// HTTP server options (populated by CLI flags in Task 7)
// ---------------------------------------------------------------------------

pub struct HttpServeOpts {
    pub port: u16,
    pub host: String,
    pub no_auth: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run_serve(
    data_dir: &Path,
    http_opts: Option<HttpServeOpts>,
    read_only: bool,
) -> Result<()> {
    if let Some(ref opts) = http_opts
        && opts.no_auth
        && opts.host != "127.0.0.1"
    {
        anyhow::bail!(
            "--no-auth cannot be used with --host {} (only 127.0.0.1 is allowed)",
            opts.host
        );
    }

    let db_path = data_dir.join("engraph.db");
    let models_dir = data_dir.join("models");

    let store = Store::open(&db_path)?;
    let config = Config::load()?;
    let embedder = crate::llm::LlamaEmbed::new(&models_dir, &config)?;

    let vault_path_str = store.get_meta("vault_path")?.ok_or_else(|| {
        anyhow::anyhow!("No vault path in index. Run 'engraph index <path>' first.")
    })?;
    let vault_path = PathBuf::from(&vault_path_str);

    let cleaned = crate::writer::cleanup_temp_files(&vault_path)?;
    if cleaned > 0 {
        eprintln!(
            "Cleaned up {} incomplete write(s) from previous run",
            cleaned
        );
    }

    let orphans = crate::writer::verify_index_integrity(&store, &vault_path)?;
    if orphans > 0 {
        eprintln!("Cleaned up {} orphan DB entries for missing files", orphans);
    }

    let profile = Config::load_vault_profile().ok().flatten();

    // Load intelligence models if enabled
    let orchestrator: Option<Arc<Mutex<Box<dyn OrchestratorModel + Send>>>> =
        if config.intelligence_enabled() {
            match crate::llm::LlamaOrchestrator::new(&models_dir, &config) {
                Ok(orch) => Some(Arc::new(Mutex::new(
                    Box::new(orch) as Box<dyn OrchestratorModel + Send>
                ))),
                Err(e) => {
                    tracing::warn!("failed to load orchestrator: {e}, intelligence disabled");
                    None
                }
            }
        } else {
            None
        };

    let reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>> = if config.intelligence_enabled()
    {
        match crate::llm::LlamaRerank::new(&models_dir, &config) {
            Ok(rerank) => Some(Arc::new(Mutex::new(
                Box::new(rerank) as Box<dyn RerankModel + Send>
            ))),
            Err(e) => {
                tracing::warn!("failed to load reranker: {e}, reranking disabled");
                None
            }
        }
    } else {
        None
    };

    let store_arc = Arc::new(Mutex::new(store));
    let embedder_arc: Arc<Mutex<Box<dyn EmbedModel + Send>>> =
        Arc::new(Mutex::new(Box::new(embedder) as Box<dyn EmbedModel + Send>));
    let vault_path_arc = Arc::new(vault_path);
    let profile_arc = Arc::new(profile);
    let recent_writes: RecentWrites = Arc::new(Mutex::new(HashMap::new()));

    // Clone Arcs for HTTP server before MCP consumes them
    let http_store = store_arc.clone();
    let http_embedder = embedder_arc.clone();
    let http_vault_path = vault_path_arc.clone();
    let http_profile = profile_arc.clone();
    let http_orchestrator = orchestrator.as_ref().map(Arc::clone);
    let http_reranker = reranker.as_ref().map(Arc::clone);
    let http_recent_writes = recent_writes.clone();

    // Start file watcher for real-time index updates
    let mut exclude = config.exclude.clone();
    if let Some(ref prof) = *profile_arc
        && let Some(ref archive) = prof.structure.folders.archive
    {
        let pattern = format!("{}/", archive);
        if !exclude.contains(&pattern) {
            exclude.push(pattern);
        }
    }
    let (watcher_handle, watcher_shutdown) = crate::watcher::start_watcher(
        store_arc.clone(),
        embedder_arc.clone(),
        vault_path_arc.clone(),
        profile_arc.clone(),
        config,
        exclude,
        recent_writes.clone(),
    )?;

    if read_only {
        eprintln!("Read-only mode: write tools disabled");
    }

    let server = EngraphServer {
        store: store_arc,
        embedder: embedder_arc,
        vault_path: vault_path_arc,
        profile: profile_arc,
        tool_router: EngraphServer::tool_router(),
        orchestrator,
        reranker,
        recent_writes,
        read_only,
    };

    // Cancellation token for coordinated shutdown of HTTP + MCP
    let cancel_token = tokio_util::sync::CancellationToken::new();

    // Spawn HTTP server as a background task (before MCP blocks on stdio)
    if let Some(ref opts) = http_opts {
        let config = Config::load()?;
        let api_state = crate::http::ApiState {
            store: http_store,
            embedder: http_embedder,
            vault_path: http_vault_path,
            profile: http_profile,
            orchestrator: http_orchestrator,
            reranker: http_reranker,
            http_config: Arc::new(config.http.clone()),
            no_auth: opts.no_auth,
            recent_writes: http_recent_writes,
            rate_limiter: Arc::new(crate::http::RateLimiter::new(config.http.rate_limit)),
            read_only,
        };
        let router = crate::http::build_router(api_state);
        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let cancel = cancel_token.clone();
        eprintln!("HTTP server listening on http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(cancel.cancelled_owned())
                .await
                .ok();
        });
    }

    eprintln!("engraph MCP server starting...");

    let transport = rmcp::transport::io::stdio();
    match server.serve(transport).await {
        Ok(server_handle) => {
            server_handle.waiting().await?;
        }
        Err(e) => {
            if http_opts.is_some() {
                // MCP transport failed (e.g., no stdin) but HTTP is running — stay alive
                eprintln!("MCP transport unavailable ({e:#}), HTTP server still running...");
                cancel_token.cancelled().await;
            } else {
                return Err(anyhow::anyhow!("{e}"));
            }
        }
    }

    cancel_token.cancel(); // triggers HTTP graceful shutdown

    // Shut down watcher cleanly after MCP transport exits
    let _ = watcher_shutdown.send(());
    if let Err(e) = watcher_handle.join() {
        tracing::warn!("Watcher thread panicked: {:?}", e);
    }

    Ok(())
}
