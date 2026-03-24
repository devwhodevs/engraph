# engraph

Local semantic search CLI for Obsidian vaults. Rust, MIT licensed.

## Architecture

Single binary with 11 modules behind a lib crate:

- `config.rs` — loads `~/.engraph/config.toml` and `vault.toml`, merges CLI args, provides `data_dir()`
- `chunker.rs` — smart chunking with break-point scoring algorithm. Finds optimal split points considering headings, code fences, blank lines, and thematic breaks. `split_oversized_chunks()` handles token-aware secondary splitting with overlap
- `docid.rs` — deterministic 6-char hex IDs for files (SHA-256 of path, truncated). Shown in search results for quick reference
- `embedder.rs` — downloads and runs `all-MiniLM-L6-v2` ONNX model (384-dim). SHA256-verified on download. Uses `ort` for inference, `tokenizers` for tokenization. Implements `ModelBackend` trait
- `model.rs` — pluggable `ModelBackend` trait, model registry, and `parse_model_spec()`. Enables future model swapping without changing consumer code
- `fts.rs` — FTS5 full-text search support. Re-exports `FtsResult` from store. BM25-ranked keyword search
- `fusion.rs` — Reciprocal Rank Fusion (RRF) engine. Merges semantic + FTS5 results. Supports lane weighting and `--explain` output
- `profile.rs` — vault profile detection. Auto-detects PARA/Folders/Flat structure, vault type (Obsidian/Logseq/Plain), wikilinks, frontmatter, tags. Writes/loads `vault.toml`
- `store.rs` — SQLite persistence. Tables: `meta`, `files` (with docid), `chunks` (with vector BLOBs), `chunks_fts` (FTS5 virtual table), `tombstones`. Handles incremental diffing via content hashes
- `hnsw.rs` — thin wrapper around `hnsw_rs`. **Important:** `hnsw_rs` does not support inserting after `load_hnsw()`. The index is rebuilt from vectors stored in SQLite on every index run
- `indexer.rs` — orchestrates vault walking (via `ignore` crate for `.gitignore` support), diffing, chunking, embedding (Rayon for parallel chunking, serial embedding since `Embedder` is not `Send`), and serial writes to store + HNSW + FTS5

`main.rs` is a thin clap CLI that wires the modules together. Subcommands: `index`, `search` (with `--explain`), `status`, `clear`, `init`, `configure`, `models`.

## Key patterns

- **Hybrid search:** Queries run through two lanes — semantic (HNSW embeddings) and keyword (FTS5 BM25). Results are fused via Reciprocal Rank Fusion (RRF) with configurable lane weights
- **Smart chunking:** Break-point scoring algorithm assigns scores to potential split points (headings 50-100, code fences 80, thematic breaks 60, blank lines 20). Chunks split at the highest-scored break point near the token target. Code fence protection prevents splitting inside code blocks
- **Incremental indexing:** `diff_vault()` compares file content hashes in SQLite against disk. Changed files have their old chunks deleted (cascade), then are re-embedded as new. FTS5 entries are cleaned up alongside vector entries
- **HNSW rebuild on every run:** Vectors are stored as BLOBs in the `chunks` table. After SQLite is updated, the full HNSW index is rebuilt from `store.get_all_vectors()`. This is necessary because `hnsw_rs` doesn't support append-after-load
- **Docids:** Each file gets a deterministic 6-char hex ID (SHA-256 of relative path). Displayed in search results for quick reference
- **Vault profiles:** `engraph init` auto-detects vault structure and writes `vault.toml`
- **Pluggable models:** `ModelBackend` trait enables future model swapping. Current implementation uses ONNX all-MiniLM-L6-v2

## Data directory

`~/.engraph/` — hardcoded via `Config::data_dir()` (uses `dirs::home_dir()`). Contains `engraph.db` (SQLite with FTS5), `hnsw/` (index files), `models/` (ONNX model + tokenizer), `vault.toml` (vault profile), `config.toml` (user config).

Single vault only. Re-indexing a different vault path triggers a confirmation prompt.

## Dependencies to be aware of

- `ort` (2.0.0-rc.12) — ONNX Runtime Rust bindings. Pre-release API. `Session::builder()?.commit_from_file()` pattern. Does not provide prebuilt binaries for all targets (no x86_64-apple-darwin)
- `hnsw_rs` (0.3) — pure Rust HNSW. `Box::leak` used in `load()` to satisfy `'static` lifetime on the loaded index. Read-only after load
- `tokenizers` (0.22) — HuggingFace tokenizer. Needs `fancy-regex` feature
- `ignore` (0.4) — vault walking with automatic `.gitignore` support
- `rusqlite` (0.32) — bundled SQLite with FTS5 support

## Testing

- Unit tests in each module (`cargo test --lib`) — 91 tests, no network required
- 1 ignored smoke test (`test_embed_smoke`) — downloads ONNX model, verifies embedding
- Integration tests (`cargo test --test integration -- --ignored`) — 8 tests, require model download. Use `tempfile` for isolated data dirs

## CI/CD

- CI: `cargo fmt --check` + `cargo clippy -- -D warnings` + `cargo test --lib` on macOS + Ubuntu
- Release: native builds on macOS arm64 (macos-14) + Linux x86_64 (ubuntu-latest). Triggered by `v*` tags. No x86_64 macOS build (ort-sys limitation)
- Homebrew: `devwhodevs/homebrew-tap` — formula builds from source tarball

## Common tasks

```bash
# Run tests
cargo test --lib

# Run integration tests (downloads model)
cargo test --test integration -- --ignored

# Build release
cargo build --release

# Release: tag and push
git tag v0.x.y && git push origin v0.x.y
# Then update homebrew-tap formula with new SHA256
```
