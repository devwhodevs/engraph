# engraph

Local semantic search CLI for Obsidian vaults. Rust, MIT licensed.

## Architecture

Single binary with 7 modules behind a lib crate:

- `config.rs` — loads `~/.engraph/config.toml`, merges CLI args, provides `data_dir()`
- `chunker.rs` — splits markdown by `##` headings, strips YAML frontmatter, extracts tags. `split_oversized_chunks()` handles token-aware sub-splitting with overlap
- `embedder.rs` — downloads and runs `all-MiniLM-L6-v2` ONNX model (384-dim). SHA256-verified on download. Uses `ort` for inference, `tokenizers` for tokenization
- `store.rs` — SQLite persistence. Tables: `meta`, `files`, `chunks` (with vector BLOBs), `tombstones`. Handles incremental diffing via content hashes
- `hnsw.rs` — thin wrapper around `hnsw_rs`. **Important:** `hnsw_rs` does not support inserting after `load_hnsw()`. The index is rebuilt from vectors stored in SQLite on every index run
- `indexer.rs` — orchestrates vault walking (via `ignore` crate for `.gitignore` support), diffing, chunking, embedding (Rayon for parallel chunking, serial embedding since `Embedder` is not `Send`), and serial writes to store + HNSW
- `search.rs` — embeds query, searches HNSW with tombstone filtering, formats results (human + JSON). Also handles `status` formatting

`main.rs` is a thin clap CLI that wires the modules together.

## Key patterns

- **Incremental indexing:** `diff_vault()` compares file content hashes in SQLite against disk. Changed files have their old chunks deleted (cascade), then are re-embedded as new
- **HNSW rebuild on every run:** Vectors are stored as BLOBs in the `chunks` table. After SQLite is updated, the full HNSW index is rebuilt from `store.get_all_vectors()`. This is necessary because `hnsw_rs` doesn't support append-after-load
- **Vector IDs:** Assigned sequentially, stored in both SQLite and HNSW. `next_vector_id` is derived from `MAX(vector_id)` in SQLite
- **Tombstones:** Exist in the schema but are largely unused now that we rebuild HNSW each run. Kept for future use if switching to a vector store that supports deletion

## Data directory

`~/.engraph/` — hardcoded via `Config::data_dir()` (uses `dirs::home_dir()`). Contains `engraph.db` (SQLite), `hnsw/` (index files), `models/` (ONNX model + tokenizer).

Single vault only. Re-indexing a different vault path triggers a confirmation prompt.

## Dependencies to be aware of

- `ort` (2.0.0-rc.12) — ONNX Runtime Rust bindings. Pre-release API. `Session::builder()?.commit_from_file()` pattern. Does not provide prebuilt binaries for all targets (no x86_64-apple-darwin)
- `hnsw_rs` (0.3) — pure Rust HNSW. `Box::leak` used in `load()` to satisfy `'static` lifetime on the loaded index. Read-only after load
- `tokenizers` (0.22) — HuggingFace tokenizer. Needs `fancy-regex` feature
- `ignore` (0.4) — vault walking with automatic `.gitignore` support

## Testing

- Unit tests in each module (`cargo test --lib`) — 44 tests, no network required
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
