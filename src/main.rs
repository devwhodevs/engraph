use engraph::config;
use engraph::indexer;
use engraph::profile;
use engraph::search;
use engraph::store;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Read as _, Write};
use std::path::PathBuf;

use config::Config;

#[derive(Parser, Debug)]
#[command(
    name = "engraph",
    version,
    about = "Local semantic search for Obsidian vaults"
)]
struct Cli {
    /// Output results as JSON.
    #[arg(long, global = true)]
    json: bool,

    /// Enable verbose logging.
    #[arg(long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Index a vault directory for semantic search.
    Index {
        /// Path to the vault (overrides config).
        path: Option<PathBuf>,

        /// Rebuild the index from scratch.
        #[arg(long)]
        rebuild: bool,
    },

    /// Search the indexed vault.
    Search {
        /// The search query.
        query: String,

        /// Number of results to return.
        #[arg(short = 'n', long)]
        top_n: Option<usize>,

        /// Show per-lane RRF score breakdown for each result.
        #[arg(long, conflicts_with = "json")]
        explain: bool,
    },

    /// Show index status and statistics.
    Status,

    /// Clear cached data.
    Clear {
        /// Remove everything including the database and embeddings.
        #[arg(long)]
        all: bool,
    },

    /// Initialize vault profile with auto-detection.
    Init {
        /// Path to the vault (defaults to current directory).
        path: Option<PathBuf>,
    },

    /// Interactively configure vault profile.
    Configure,

    /// Manage embedding models.
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Start MCP stdio server for AI agent access.
    Serve,

    /// Inspect vault graph connections.
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },

    /// Query vault context.
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },

    /// Write a note to the vault.
    Write {
        #[command(subcommand)]
        action: WriteAction,
    },
}

#[derive(Subcommand, Debug)]
enum GraphAction {
    /// Show connections for a note.
    Show {
        /// File path or #docid.
        file: String,
    },
    /// Show vault graph statistics.
    Stats,
}

#[derive(Subcommand, Debug)]
enum ContextAction {
    /// Read a note's full content with metadata.
    Read {
        /// File path, basename, or #docid.
        file: String,
    },
    /// List notes by metadata filters.
    List {
        /// Filter to folder path prefix.
        #[arg(long)]
        folder: Option<String>,
        /// Filter to notes with all listed tags (comma-separated).
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Filter to notes created by a specific agent.
        #[arg(long)]
        created_by: Option<String>,
        /// Maximum results.
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Vault structure overview.
    VaultMap,
    /// Person context bundle.
    Who {
        /// Person name (matches filename in People folder).
        name: String,
    },
    /// Project context bundle.
    Project {
        /// Project name (matches filename).
        name: String,
    },
    /// Rich topic context with budget.
    Topic {
        /// Search query for the topic.
        query: String,
        /// Character budget (default 32000, ~8000 tokens).
        #[arg(long, default_value = "32000")]
        budget: usize,
    },
}

#[derive(Subcommand, Debug)]
enum WriteAction {
    /// Create a new note.
    Create {
        /// Note content (reads from stdin if omitted).
        #[arg(long)]
        content: Option<String>,
        /// Filename (without .md).
        #[arg(long)]
        filename: Option<String>,
        /// Type hint for placement.
        #[arg(long)]
        type_hint: Option<String>,
        /// Tags (comma-separated).
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Explicit folder (skips placement).
        #[arg(long)]
        folder: Option<String>,
    },
    /// Append content to an existing note.
    Append {
        /// Target note (path, basename, or #docid).
        file: String,
        /// Content to append (reads from stdin if omitted).
        #[arg(long)]
        content: Option<String>,
    },
    /// Archive a note (soft delete — moves to archive, removes from index).
    Archive {
        /// Target note (path, basename, or #docid).
        file: String,
    },
    /// Restore an archived note to its original location.
    Unarchive {
        /// Archived note path (e.g., "04-Archive/01-Projects/note.md").
        file: String,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsAction {
    /// List available models.
    List,
    /// Show info about a model.
    Info { name: String },
}

/// Check whether an index has been built by looking for engraph.db in data_dir.
fn index_exists(data_dir: &std::path::Path) -> bool {
    data_dir.join("engraph.db").exists()
}

/// Remove a file, ignoring NotFound errors.
fn remove_if_exists(path: &std::path::Path) -> Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Remove a directory recursively, ignoring NotFound errors.
fn remove_dir_if_exists(path: &std::path::Path) -> Result<bool> {
    match std::fs::remove_dir_all(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up tracing. Default: suppress all logs (ort is very noisy).
    // --verbose enables debug for engraph, info for everything else.
    let filter = if cli.verbose {
        "engraph=debug,info"
    } else {
        "error"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .init();

    let mut cfg = Config::load()?;
    let data_dir = Config::data_dir()?;

    match cli.command {
        Command::Index { path, rebuild } => {
            // Merge CLI vault path over config.
            cfg.merge_vault_path(path);

            // Fall back to current directory if neither CLI nor config provides a vault path.
            let vault_path = match &cfg.vault_path {
                Some(p) => p.clone(),
                None => {
                    let cwd = std::env::current_dir()?;
                    cfg.vault_path = Some(cwd.clone());
                    cwd
                }
            };

            // Canonicalize to resolve symlinks and relative paths.
            let vault_path = vault_path.canonicalize().unwrap_or(vault_path);

            // Ensure data directory exists.
            std::fs::create_dir_all(&data_dir)?;

            // Check for vault mismatch: if store has a different vault path, warn.
            let db_path = data_dir.join("engraph.db");
            if db_path.exists() && !rebuild {
                let store = store::Store::open(&db_path)?;
                if let Some(stored_vault) = store.get_meta("vault_path")? {
                    let stored = PathBuf::from(&stored_vault);
                    if stored != vault_path {
                        eprint!(
                            "Warning: Index was built for '{}'. Re-indexing will replace it. Continue? [y/N] ",
                            stored.display()
                        );
                        io::stderr().flush()?;
                        let mut answer = String::new();
                        io::stdin().lock().read_line(&mut answer)?;
                        if !answer.trim().eq_ignore_ascii_case("y") {
                            println!("Aborted.");
                            return Ok(());
                        }
                    }
                }
            }

            let result = indexer::run_index(&vault_path, &cfg, rebuild)?;

            println!(
                "Indexed {} new, {} updated, {} deleted files ({} chunks) in {:.1}s",
                result.new_files,
                result.updated_files,
                result.deleted_files,
                result.total_chunks,
                result.duration.as_secs_f64(),
            );
        }

        Command::Search {
            query,
            top_n,
            explain,
        } => {
            cfg.merge_top_n(top_n);

            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }

            search::run_search(&query, cfg.top_n, cli.json, explain, &data_dir, &cfg)?;
        }

        Command::Status => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }

            search::run_status(cli.json, &data_dir)?;
        }

        Command::Clear { all } => {
            if all {
                // Delete entire ~/.engraph/ directory.
                if remove_dir_if_exists(&data_dir)? {
                    println!("Removed {}", data_dir.display());
                } else {
                    println!("Nothing to clear (data directory does not exist).");
                }
            } else {
                // Delete only index files: engraph.db.
                let db_path = data_dir.join("engraph.db");
                if remove_if_exists(&db_path)? {
                    println!("Removed {}", db_path.display());
                } else {
                    println!("Nothing to clear (no index files found).");
                }
            }
        }

        Command::Init { path } => {
            // Resolve vault path: CLI arg > config > cwd.
            cfg.merge_vault_path(path);
            let vault_path = match &cfg.vault_path {
                Some(p) => p.clone(),
                None => std::env::current_dir()?,
            };
            let vault_path = vault_path.canonicalize().unwrap_or(vault_path);

            println!("Detecting vault profile for: {}", vault_path.display());

            let vault_type = profile::detect_vault_type(&vault_path);
            let structure = profile::detect_structure(&vault_path)?;
            let stats = profile::scan_vault_stats(&vault_path)?;

            // Print detection results.
            println!();
            println!("  Vault type:   {:?}", vault_type);
            println!("  Structure:    {:?}", structure.method);
            if let Some(ref inbox) = structure.folders.inbox {
                println!("    inbox:      {}", inbox);
            }
            if let Some(ref projects) = structure.folders.projects {
                println!("    projects:   {}", projects);
            }
            if let Some(ref areas) = structure.folders.areas {
                println!("    areas:      {}", areas);
            }
            if let Some(ref resources) = structure.folders.resources {
                println!("    resources:  {}", resources);
            }
            if let Some(ref archive) = structure.folders.archive {
                println!("    archive:    {}", archive);
            }
            if let Some(ref templates) = structure.folders.templates {
                println!("    templates:  {}", templates);
            }
            if let Some(ref daily) = structure.folders.daily {
                println!("    daily:      {}", daily);
            }
            if let Some(ref people) = structure.folders.people {
                println!("    people:     {}", people);
            }
            println!();
            println!("  Total .md files:    {}", stats.total_files);
            println!("  With frontmatter:   {}", stats.files_with_frontmatter);
            println!("  Wikilinks:          {}", stats.wikilink_count);
            println!("  Unique tags:        {}", stats.unique_tags);
            println!("  Folders:            {}", stats.folder_count);
            println!("  Max folder depth:   {}", stats.folder_depth);

            let vault_profile = profile::VaultProfile {
                vault_path,
                vault_type,
                structure,
                stats,
            };

            // Ensure data dir exists and write vault.toml.
            std::fs::create_dir_all(&data_dir)?;
            profile::write_vault_toml(&vault_profile, &data_dir)?;

            println!();
            println!("Wrote {}", data_dir.join("vault.toml").display());
        }

        Command::Configure => {
            println!(
                "Interactive configuration not yet implemented. Run 'engraph init' for auto-detection."
            );
        }

        Command::Graph { action } => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }
            let db_path = data_dir.join("engraph.db");
            let store = store::Store::open(&db_path)?;

            match action {
                GraphAction::Show { file } => {
                    // Resolve: docid first, then exact path, then basename
                    let record = if file.starts_with('#') && file.len() == 7 {
                        store.get_file_by_docid(&file[1..])?
                    } else if let Some(f) = store.get_file(&file)? {
                        Some(f)
                    } else {
                        store.find_file_by_basename(&file)?
                    };

                    let record = match record {
                        Some(r) => r,
                        None => {
                            eprintln!("File not found: {file}");
                            std::process::exit(1);
                        }
                    };

                    let docid_str = record
                        .docid
                        .as_deref()
                        .map(|d| format!(" (#{d})"))
                        .unwrap_or_default();
                    println!("{}{}\n", record.path, docid_str);

                    let outgoing_wl = store.get_outgoing(record.id, Some("wikilink"))?;
                    println!("Outgoing wikilinks ({}):", outgoing_wl.len());
                    for (fid, _) in &outgoing_wl {
                        if let Some(f) = store.get_file_by_id(*fid)? {
                            let did = f
                                .docid
                                .as_deref()
                                .map(|d| format!(" (#{d})"))
                                .unwrap_or_default();
                            println!("  → {}{}", f.path, did);
                        }
                    }

                    println!();
                    let incoming_wl = store.get_incoming(record.id, Some("wikilink"))?;
                    println!("Incoming wikilinks ({}):", incoming_wl.len());
                    for (fid, _) in &incoming_wl {
                        if let Some(f) = store.get_file_by_id(*fid)? {
                            let did = f
                                .docid
                                .as_deref()
                                .map(|d| format!(" (#{d})"))
                                .unwrap_or_default();
                            println!("  ← {}{}", f.path, did);
                        }
                    }

                    println!();
                    let mentions_out = store.get_outgoing(record.id, Some("mention"))?;
                    let mentions_in = store.get_incoming(record.id, Some("mention"))?;
                    println!("Mentions out ({}):", mentions_out.len());
                    for (fid, _) in &mentions_out {
                        if let Some(f) = store.get_file_by_id(*fid)? {
                            let did = f
                                .docid
                                .as_deref()
                                .map(|d| format!(" (#{d})"))
                                .unwrap_or_default();
                            println!("  → {}{}", f.path, did);
                        }
                    }
                    if !mentions_in.is_empty() {
                        println!("Mentioned by ({}):", mentions_in.len());
                        for (fid, _) in &mentions_in {
                            if let Some(f) = store.get_file_by_id(*fid)? {
                                let did = f
                                    .docid
                                    .as_deref()
                                    .map(|d| format!(" (#{d})"))
                                    .unwrap_or_default();
                                println!("  ← {}{}", f.path, did);
                            }
                        }
                    }
                }

                GraphAction::Stats => {
                    let stats = store.get_edge_stats()?;
                    println!("Vault Graph:");
                    println!(
                        "  Wikilink edges: {} ({} bidirectional pairs)",
                        stats.wikilink_count,
                        stats.wikilink_count / 2
                    );
                    println!("  Mention edges:  {}", stats.mention_count);
                    println!("  Total edges:    {}", stats.total_edges);
                    let total_files = stats.connected_file_count + stats.isolated_file_count;
                    let pct = if total_files > 0 {
                        stats.connected_file_count as f64 / total_files as f64 * 100.0
                    } else {
                        0.0
                    };
                    println!(
                        "  Connected files: {} / {} ({:.1}%)",
                        stats.connected_file_count, total_files, pct
                    );
                    println!("  Isolated files:  {}", stats.isolated_file_count);
                }
            }
        }

        Command::Context { action } => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }
            let db_path = data_dir.join("engraph.db");
            let store = store::Store::open(&db_path)?;
            let vault_path_str = store.get_meta("vault_path")?.ok_or_else(|| {
                anyhow::anyhow!("No vault path in index. Run 'engraph index <path>' first.")
            })?;
            let vault_path = PathBuf::from(&vault_path_str);
            let profile = config::Config::load_vault_profile().ok().flatten();

            let params = engraph::context::ContextParams {
                store: &store,
                vault_path: &vault_path,
                profile: profile.as_ref(),
            };

            match action {
                ContextAction::Read { file } => {
                    let note = engraph::context::context_read(&params, &file)?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&note)?);
                    } else {
                        println!(
                            "{} {}",
                            note.path,
                            note.docid
                                .as_deref()
                                .map(|d| format!("(#{})", d))
                                .unwrap_or_default()
                        );
                        println!("Tags: {}", note.tags.join(", "));
                        println!("Outgoing links: {}", note.outgoing_links.len());
                        println!("Incoming links: {}", note.incoming_links.len());
                        println!("Bytes: {}\n", note.byte_count);
                        println!("{}", note.body);
                    }
                }
                ContextAction::List {
                    folder,
                    tags,
                    created_by,
                    limit,
                } => {
                    let items = engraph::context::context_list(
                        &params,
                        folder.as_deref(),
                        &tags,
                        created_by.as_deref(),
                        limit,
                    )?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&items)?);
                    } else {
                        for item in &items {
                            let did = item
                                .docid
                                .as_deref()
                                .map(|d| format!(" #{d}"))
                                .unwrap_or_default();
                            let tags_str = if item.tags.is_empty() {
                                String::new()
                            } else {
                                format!(" [{}]", item.tags.join(", "))
                            };
                            println!(
                                "{}{}{} ({} edges)",
                                item.path, did, tags_str, item.edge_count
                            );
                        }
                        println!("\n{} notes", items.len());
                    }
                }
                ContextAction::VaultMap => {
                    let map = engraph::context::vault_map(&params)?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&map)?);
                    } else {
                        println!("Vault: {}", map.vault_path);
                        println!("Type: {}, Structure: {}", map.vault_type, map.structure);
                        println!(
                            "Files: {}, Chunks: {}, Edges: {}\n",
                            map.total_files, map.total_chunks, map.total_edges
                        );
                        println!("Folders:");
                        for f in &map.folders {
                            println!("  {}: {} notes", f.path, f.note_count);
                        }
                        println!("\nTop tags:");
                        for (tag, count) in &map.top_tags {
                            println!("  {}: {}", tag, count);
                        }
                        println!("\nRecent files:");
                        for path in &map.recent_files {
                            println!("  {}", path);
                        }
                    }
                }
                ContextAction::Who { name } => {
                    let person = engraph::context::context_who(&params, &name)?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&person)?);
                    } else {
                        println!("# {}\n", person.name);
                        if let Some(note) = &person.note {
                            println!(
                                "Note: {} {}",
                                note.path,
                                note.docid
                                    .as_deref()
                                    .map(|d| format!("(#{})", d))
                                    .unwrap_or_default()
                            );
                            println!("Tags: {}\n", note.tags.join(", "));
                            println!("{}\n", note.body);
                        } else {
                            println!("(No person note found)\n");
                        }
                        if !person.mentioned_in.is_empty() {
                            println!("Mentioned in ({} notes):", person.mentioned_in.len());
                            for m in &person.mentioned_in {
                                println!("  {} — {}", m.path, m.snippet);
                            }
                            println!();
                        }
                        if !person.linked_from.is_empty() {
                            println!("Linked from ({}):", person.linked_from.len());
                            for p in &person.linked_from {
                                println!("  {}", p);
                            }
                            println!();
                        }
                        println!("Total: {} chars", person.total_chars);
                    }
                }
                ContextAction::Project { name } => {
                    let proj = engraph::context::context_project(&params, &name)?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&proj)?);
                    } else {
                        println!("# {}\n", proj.name);
                        if let Some(note) = &proj.note {
                            println!("Note: {}\n", note.path);
                            println!("{}\n", note.body);
                        }
                        if !proj.active_tasks.is_empty() {
                            println!("Active tasks ({}):", proj.active_tasks.len());
                            for t in &proj.active_tasks {
                                println!("  - [ ] {} ({})", t.text, t.source_file);
                            }
                            println!();
                        }
                        if !proj.child_notes.is_empty() {
                            println!("Child notes ({}):", proj.child_notes.len());
                            for c in &proj.child_notes {
                                println!("  {}", c.path);
                            }
                            println!();
                        }
                        if !proj.team.is_empty() {
                            println!("Team:");
                            for p in &proj.team {
                                println!("  {}", p);
                            }
                            println!();
                        }
                        if !proj.recent_mentions.is_empty() {
                            println!("Recent daily mentions:");
                            for m in &proj.recent_mentions {
                                println!("  {} — {}", m.path, m.snippet);
                            }
                            println!();
                        }
                    }
                }
                ContextAction::Topic { query, budget } => {
                    let models_dir = data_dir.join("models");
                    let mut embedder = engraph::llm::CandleEmbed::new(&models_dir, &cfg)?;

                    let bundle = engraph::context::context_topic_with_search(
                        &params,
                        &query,
                        budget,
                        &mut embedder,
                    )?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&bundle)?);
                    } else {
                        println!("# Context: {}\n", bundle.topic);
                        println!(
                            "Budget: {} / {} chars{}\n",
                            bundle.total_chars,
                            bundle.budget_chars,
                            if bundle.truncated { " (truncated)" } else { "" }
                        );
                        for s in &bundle.sections {
                            let did = s
                                .docid
                                .as_deref()
                                .map(|d| format!(" #{d}"))
                                .unwrap_or_default();
                            println!("## {} — {}{}", s.label, s.path, did);
                            println!("[{}]\n", s.relevance);
                            println!("{}\n", s.content);
                        }
                    }
                }
            }
        }

        Command::Serve => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }
            engraph::serve::run_serve(&data_dir).await?;
        }

        Command::Write { action } => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }
            let db_path = data_dir.join("engraph.db");
            let store = store::Store::open(&db_path)?;
            let vault_path_str = store
                .get_meta("vault_path")?
                .ok_or_else(|| anyhow::anyhow!("No vault path in index."))?;
            let vault_path = PathBuf::from(&vault_path_str);
            let models_dir = data_dir.join("models");
            let mut embedder = engraph::llm::CandleEmbed::new(&models_dir, &cfg)?;
            let profile = config::Config::load_vault_profile().ok().flatten();

            match action {
                WriteAction::Create {
                    content,
                    filename,
                    type_hint,
                    tags,
                    folder,
                } => {
                    let content = match content {
                        Some(c) => c,
                        None => {
                            let mut buf = String::new();
                            io::stdin().lock().read_to_string(&mut buf)?;
                            buf
                        }
                    };
                    let input = engraph::writer::CreateNoteInput {
                        content,
                        filename,
                        type_hint,
                        tags,
                        folder,
                        created_by: "cli".into(),
                    };
                    let result = engraph::writer::create_note(
                        input,
                        &store,
                        &mut embedder,
                        &vault_path,
                        profile.as_ref(),
                    )?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    } else {
                        println!(
                            "Created: {} (#{}) [{}]",
                            result.path, result.docid, result.strategy
                        );
                        if !result.links_added.is_empty() {
                            println!("Links: {}", result.links_added.join(", "));
                        }
                        if !result.links_suggested.is_empty() {
                            println!("Suggested: {}", result.links_suggested.join(", "));
                        }
                    }
                }
                WriteAction::Append { file, content } => {
                    let content = match content {
                        Some(c) => c,
                        None => {
                            let mut buf = String::new();
                            io::stdin().lock().read_to_string(&mut buf)?;
                            buf
                        }
                    };
                    let input = engraph::writer::AppendInput {
                        file,
                        content,
                        modified_by: "cli".into(),
                    };
                    let result =
                        engraph::writer::append_to_note(input, &store, &mut embedder, &vault_path)?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    } else {
                        println!("Appended to: {} (#{})", result.path, result.docid);
                    }
                }
                WriteAction::Archive { file } => {
                    let result = engraph::writer::archive_note(
                        &file,
                        &store,
                        &vault_path,
                        profile.as_ref(),
                    )?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    } else {
                        println!("Archived: {} → {}", file, result.path);
                    }
                }
                WriteAction::Unarchive { file } => {
                    let result =
                        engraph::writer::unarchive_note(&file, &store, &mut embedder, &vault_path)?;
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    } else {
                        println!("Restored: {} → {}", file, result.path);
                    }
                }
            }
        }

        Command::Models { action } => {
            let defaults = engraph::llm::ModelDefaults::default();
            match action {
                ModelsAction::List => {
                    println!("{:<30} {:>5}  DESCRIPTION", "NAME", "DIM");
                    println!("{}", "-".repeat(70));
                    let desc = "Default embedding model (GGUF)";
                    println!(
                        "{:<30} {:>5}  {}",
                        defaults.embed_uri, defaults.embed_dim, desc
                    );
                }
                ModelsAction::Info { name } => {
                    if name == defaults.embed_uri {
                        println!("Name:        {}", defaults.embed_uri);
                        println!("Format:      GGUF");
                        println!("Dimensions:  {}", defaults.embed_dim);
                        println!("Description: Default embedding model (GGUF)");
                    } else {
                        eprintln!("Unknown model: {name}");
                        eprintln!("Run 'engraph models list' to see available models.");
                        std::process::exit(1);
                    }
                }
            }
        }
    }

    Ok(())
}
