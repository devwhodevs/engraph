use std::path::{Path, PathBuf};
use std::sync::Arc;

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
use crate::embedder::Embedder;
use crate::profile::VaultProfile;
use crate::search;
use crate::store::Store;

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

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct EngraphServer {
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Embedder>>,
    vault_path: Arc<PathBuf>,
    profile: Arc<Option<VaultProfile>>,
    tool_router: ToolRouter<Self>,
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
        let output = search::search_internal(&params.0.query, top_n, &store, &mut embedder)
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
        let items = context::context_list(&ctx, params.0.folder.as_deref(), &tags, limit)
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
            context::context_topic_with_search(&ctx, &params.0.topic, budget, &mut embedder)
                .map_err(|e| mcp_err(&e))?;
        to_json_result(&bundle)
    }

    #[tool(
        name = "create",
        description = "Create a new note with automatic tag resolution, link discovery, and folder placement. Returns the created file's path, docid, and what was auto-resolved."
    )]
    async fn create(&self, params: Parameters<CreateParams>) -> Result<CallToolResult, McpError> {
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
            &mut embedder,
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
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let input = crate::writer::AppendInput {
            file: params.0.file,
            content: params.0.content,
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::append_to_note(input, &store, &mut embedder, &self.vault_path)
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
}

#[tool_handler]
impl rmcp::handler::server::ServerHandler for EngraphServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "engraph: vault intelligence for Obsidian. \
                 Read: vault_map to orient, search to find, read for content, who/project for context. \
                 Write: create for new notes, append to add content, update_metadata for tags/aliases, move_note to relocate.",
        )
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run_serve(data_dir: &Path) -> Result<()> {
    let db_path = data_dir.join("engraph.db");
    let models_dir = data_dir.join("models");

    let store = Store::open(&db_path)?;
    let embedder = Embedder::new(&models_dir)?;

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

    let profile = Config::load_vault_profile().ok().flatten();

    let server = EngraphServer {
        store: Arc::new(Mutex::new(store)),
        embedder: Arc::new(Mutex::new(embedder)),
        vault_path: Arc::new(vault_path),
        profile: Arc::new(profile),
        tool_router: EngraphServer::tool_router(),
    };

    eprintln!("engraph MCP server starting...");

    let transport = rmcp::transport::io::stdio();
    let server_handle = server.serve(transport).await?;
    server_handle.waiting().await?;
    Ok(())
}
