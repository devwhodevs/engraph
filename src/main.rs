use engraph::config;
use engraph::indexer;
use engraph::profile;
use engraph::search;
use engraph::store;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
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
        /// Remove everything including the HNSW index and embeddings.
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up tracing. Default: suppress all logs (ort and hnsw_rs are very noisy).
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

        Command::Search { query, top_n, explain } => {
            cfg.merge_top_n(top_n);

            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'engraph index <path>' first.");
                std::process::exit(1);
            }

            search::run_search(&query, cfg.top_n, cli.json, explain, &data_dir)?;
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
                // Delete only index files: engraph.db and hnsw directory.
                let mut deleted_any = false;

                let db_path = data_dir.join("engraph.db");
                if remove_if_exists(&db_path)? {
                    println!("Removed {}", db_path.display());
                    deleted_any = true;
                }

                let hnsw_dir = data_dir.join("hnsw");
                if remove_dir_if_exists(&hnsw_dir)? {
                    println!("Removed {}", hnsw_dir.display());
                    deleted_any = true;
                }

                if !deleted_any {
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
    }

    Ok(())
}
