# engraph

Local semantic search for Obsidian vaults.

## Install

**Homebrew (macOS):**

```bash
brew install devwhodevs/tap/engraph
```

**Cargo:**

```bash
cargo install engraph
```

**Binary download:**

Pre-built binaries for macOS (arm64/x86_64) and Linux (x86_64) are available on the [Releases](https://github.com/devwhodevs/engraph/releases) page.

## Quick start

```bash
engraph index ~/vault
engraph search "query"
```

## Commands

| Command  | Description                        | Flags                          |
|----------|------------------------------------|--------------------------------|
| `index`  | Index a vault for semantic search  | `[path]`, `--rebuild`          |
| `search` | Search the indexed vault           | `<query>`, `-n/--top-n <N>`   |
| `status` | Show index status and statistics   |                                |
| `clear`  | Clear cached data                  | `--all`                        |

## Configuration

engraph reads `~/.config/engraph/config.toml`:

```toml
vault_path = "~/Documents/vault"
top_n = 5
exclude = [".obsidian/*", ".trash/*"]
batch_size = 64
```

| Key          | Description                                  | Default                          |
|--------------|----------------------------------------------|----------------------------------|
| `vault_path` | Path to Obsidian vault                       | None (must specify via CLI/config) |
| `top_n`      | Number of search results to return           | `5`                              |
| `exclude`    | Glob patterns to exclude from indexing       | `[".obsidian/*", ".trash/*"]`    |
| `batch_size` | Files per embedding batch                    | `64`                             |

## How it works

engraph splits your vault's markdown files into heading-based chunks, generates embeddings locally using an ONNX runtime model (all-MiniLM-L6-v2), and stores them in an HNSW index for fast approximate nearest-neighbor search. Everything runs on your machine -- no API keys, no network calls after the initial one-time model download.

The indexing pipeline:

1. Walk the vault, respecting `.gitignore` and exclude patterns
2. Split each markdown file into chunks by heading boundaries
3. Sub-split oversized chunks to stay within the model's token limit
4. Embed chunks in batches via ONNX Runtime
5. Insert embeddings into an HNSW graph stored alongside a SQLite metadata database

Subsequent runs are incremental -- only new or modified files are re-processed.

## Contributing

Contributions are welcome. Please open an issue to discuss larger changes before submitting a PR.

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test --lib
```

## License

MIT
