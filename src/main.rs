mod config;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::info;

use config::Config;

#[derive(Parser, Debug)]
#[command(name = "engraph", version, about = "Local semantic search for Obsidian vaults")]
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
    },

    /// Show index status and statistics.
    Status,

    /// Clear cached data.
    Clear {
        /// Remove everything including the HNSW index and embeddings.
        #[arg(long)]
        all: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up tracing.
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .init();

    let mut cfg = Config::load()?;
    info!(data_dir = %Config::data_dir()?.display(), "loaded config");

    match cli.command {
        Command::Index { path, rebuild } => {
            cfg.merge_vault_path(path);
            let vault = cfg
                .vault_path
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<not set>".into());
            println!("Would index: {vault} (rebuild={rebuild})");
        }
        Command::Search { query, top_n } => {
            cfg.merge_top_n(top_n);
            println!("Would search: \"{query}\" (top_n={})", cfg.top_n);
        }
        Command::Status => {
            let json_flag = cli.json;
            println!("Would show status (json={json_flag})");
        }
        Command::Clear { all } => {
            println!("Would clear cache (all={all})");
        }
    }

    Ok(())
}
