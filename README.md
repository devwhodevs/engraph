# engraph

Local semantic search for Obsidian vaults. Runs entirely offline — no API keys, no cloud services.

engraph indexes your markdown notes into a local vector database and lets you search by meaning, not just keywords. It uses a small ONNX model (`all-MiniLM-L6-v2`, ~23MB) that runs on your machine.

## Install

**Homebrew (macOS):**

```bash
brew install devwhodevs/tap/engraph
```

**Pre-built binaries:**

Download from [Releases](https://github.com/devwhodevs/engraph/releases) (macOS arm64, Linux x86_64).

**From source:**

```bash
cargo install --git https://github.com/devwhodevs/engraph
```

## Usage

```bash
# Index your vault (downloads the embedding model on first run, ~23MB)
engraph index ~/path/to/vault

# Search
engraph search "how does error handling work in Rust"

# Check what's indexed
engraph status

# Re-index after changes
engraph index ~/path/to/vault

# Full rebuild (discard incremental state)
engraph index ~/path/to/vault --rebuild

# JSON output (for scripts/tools)
engraph search "query" --json
engraph status --json

# Clear index data (keeps downloaded model)
engraph clear

# Clear everything including model
engraph clear --all
```

## How it works

1. **Walk** the vault collecting `.md` files (respects `.gitignore` and exclude patterns)
2. **Chunk** each file by `##` heading boundaries. Oversized chunks are sub-split at sentence boundaries with token overlap
3. **Embed** chunks locally using `all-MiniLM-L6-v2` via ONNX Runtime (384-dim vectors)
4. **Store** vectors and metadata in SQLite (`~/.engraph/engraph.db`)
5. **Build** an HNSW index for fast approximate nearest-neighbor search

Re-indexing is incremental — only new or modified files are re-embedded. The HNSW index is rebuilt from stored vectors each run (necessary because `hnsw_rs` doesn't support append-after-load).

## Search output

```
 1. [0.87] 02-Areas/Development/Rust Tips.md > ## Error Handling
    Use thiserror for library errors and anyhow for application errors...

 2. [0.82] 03-Resources/Code-Snippets/WASM Setup.md
    Setting up wasm-pack with Rust requires...

 3. [0.74] 07-Daily/2026-03-15.md > ## Notes
    Looked into embedding models for local inference...
```

## Commands

| Command | Description | Options |
|---------|-------------|---------|
| `engraph index [PATH]` | Index a vault (default: current dir) | `--rebuild` force full rebuild |
| `engraph search <QUERY>` | Semantic search | `-n <N>` number of results (default: 5) |
| `engraph status` | Show index stats | |
| `engraph clear` | Delete index (keeps model) | `--all` delete everything |

Global flags: `--json` for machine-readable output, `--verbose` for debug logging.

## Configuration

Optional config file at `~/.engraph/config.toml`:

```toml
vault_path = "~/Documents/MyVault"
top_n = 5
exclude = [".obsidian/", "node_modules/"]
batch_size = 64
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vault_path` | string | current dir | Default vault path |
| `top_n` | integer | `5` | Number of search results |
| `exclude` | string[] | `[".obsidian/"]` | Patterns to exclude from indexing |
| `batch_size` | integer | `64` | Embedding batch size |

## Data directory

Everything is stored in `~/.engraph/`:

```
~/.engraph/
  engraph.db      # SQLite: file metadata, chunks, vectors
  hnsw/           # HNSW index files
  models/         # Downloaded ONNX model + tokenizer
  config.toml     # Optional configuration
```

## Development

```bash
# Run all unit tests
cargo test --lib

# Run integration tests (requires ~23MB model download)
cargo test --test integration -- --ignored

# Lint
cargo fmt --check
cargo clippy -- -D warnings
```

## Architecture

```
src/
  main.rs       # CLI entry point (clap)
  lib.rs        # Public module re-exports
  config.rs     # Config loading and merging
  chunker.rs    # Markdown parsing, heading-based chunking
  embedder.rs   # ONNX model download + inference
  store.rs      # SQLite persistence (files, chunks, vectors, metadata)
  hnsw.rs       # HNSW index wrapper
  indexer.rs    # Vault walking, incremental sync orchestration
  search.rs     # Query pipeline and output formatting
```

## License

MIT
