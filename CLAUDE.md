# engraph

Local hybrid search CLI for Obsidian vaults. Rust, MIT licensed.

## Architecture

Single binary with 20 modules behind a lib crate:

- `config.rs` — loads `~/.engraph/config.toml` and `vault.toml`, merges CLI args, provides `data_dir()`
- `chunker.rs` — smart chunking with break-point scoring algorithm. Finds optimal split points considering headings, code fences, blank lines, and thematic breaks. `split_oversized_chunks()` handles token-aware secondary splitting with overlap
- `docid.rs` — deterministic 6-char hex IDs for files (SHA-256 of path, truncated). Shown in search results for quick reference
- `embedder.rs` — downloads and runs `bge-small-en-v1.5` ONNX model (384-dim, 512 token context). SHA256-verified on download. Uses `ort` for inference, `tokenizers` for tokenization. Implements `ModelBackend` trait. **Not `Send`** — all embedding is serial
- `model.rs` — pluggable `ModelBackend` trait, model registry, and `parse_model_spec()`. Enables future model swapping without changing consumer code
- `fts.rs` — FTS5 full-text search support. Re-exports `FtsResult` from store. BM25-ranked keyword search
- `fusion.rs` — Reciprocal Rank Fusion (RRF) engine. Merges semantic + FTS5 + graph results. Supports lane weighting, `--explain` output with per-lane detail
- `context.rs` — context engine. Six functions: `read` (full note content + metadata), `list` (filtered note listing with `created_by` filter), `vault_map` (structure overview), `who` (person context bundle), `project` (project context bundle), `context_topic` (rich topic context with budget trimming). Pure functions taking `ContextParams` — no model loading except `context_topic` which reuses `search_internal`
- `vecstore.rs` — sqlite-vec virtual table integration. Manages the `vec_chunks` vec0 table for vector storage and KNN search. Handles insert, delete, and search operations against the virtual table
- `tags.rs` — tag registry module. Maintains a `tag_registry` table tracking known tags with source attribution. Supports fuzzy matching for tag suggestions during note creation
- `links.rs` — link discovery module. Three match types: exact basename, fuzzy (sliding window Levenshtein, 0.92 threshold), and first-name (People folder, suggestion-only at 650bp). Overlap resolution via type priority (exact > alias > fuzzy > first-name)
- `placement.rs` — folder placement engine. Uses folder centroids (online mean of embeddings per folder) to suggest the best folder for new notes. Falls back to inbox when confidence is low. Includes placement correction detection (`detect_correction_from_frontmatter`) and frontmatter stripping for moved files
- `writer.rs` — write pipeline orchestrator. 5-step pipeline: resolve tags (fuzzy match + register new), discover links (exact + fuzzy), place in folder, atomic file write (temp + rename), and index update. Supports create, append, update_metadata, move_note, archive, and unarchive operations with mtime-based conflict detection and crash recovery via temp file cleanup
- `watcher.rs` — file watcher for `engraph serve`. OS thread producer (notify-debouncer-full, 2s debounce) sends `Vec<WatchEvent>` over tokio::mpsc to async consumer task. Two-pass batch processing: mutations (index_file/remove_file/rename_file) then edge rebuild. Move detection via content hash matching. Placement correction on file moves. Centroid adjustment on file add/remove. Startup reconciliation via `run_index_shared`
- `serve.rs` — MCP stdio server via rmcp SDK. Exposes 13 tools: 7 read (search, read, list, vault_map, who, project, context) + 6 write (create, append, update_metadata, move_note, archive, unarchive). EngraphServer struct with Arc+Mutex wrapping for async handlers. Spawns file watcher on startup
- `graph.rs` — vault graph agent. Extracts wikilink targets, expands search results by following graph connections 1-2 hops. Relevance filtering via FTS5 term check and shared tags
- `profile.rs` — vault profile detection. Auto-detects PARA/Folders/Flat structure, vault type (Obsidian/Logseq/Plain), wikilinks, frontmatter, tags. Writes/loads `vault.toml`
- `store.rs` — SQLite persistence. Tables: `meta`, `files` (with docid, created_by), `chunks` (with vector BLOBs), `chunks_fts` (FTS5), `edges` (vault graph), `tombstones`, `tag_registry`, `folder_centroids`, `placement_corrections`, `link_skiplist` (reserved). `vec_chunks` virtual table (sqlite-vec) for KNN search. Handles incremental diffing via content hashes
- `indexer.rs` — orchestrates vault walking (via `ignore` crate for `.gitignore` support), diffing, chunking, embedding, writes to store + sqlite-vec + FTS5, vault graph edge building (wikilinks + people detection), and folder centroid computation. Exposes `index_file`, `remove_file`, `rename_file` as public per-file functions. `run_index_shared` accepts external store/embedder for watcher FullRescan
- `search.rs` — hybrid search orchestrator. Runs semantic (sqlite-vec KNN), keyword (FTS5 BM25), and graph expansion lanes, then fuses via RRF

`main.rs` is a thin clap CLI (async via `#[tokio::main]`). Subcommands: `index`, `search` (with `--explain`), `status`, `clear`, `init`, `configure`, `models`, `graph` (show/stats), `context` (read/list/vault-map/who/project/topic), `write` (create/append/update-metadata/move), `serve` (MCP stdio server with file watcher).

## Key patterns

- **3-lane hybrid search:** Queries run through three lanes — semantic (sqlite-vec KNN embeddings), keyword (FTS5 BM25), and graph (wikilink expansion). Results are fused via Reciprocal Rank Fusion (RRF) with configurable lane weights (semantic 1.0, FTS 1.0, graph 0.8)
- **Vault graph:** `edges` table stores bidirectional wikilink edges and mention edges. Built during indexing after all files are written. People detection scans for person name/alias mentions using notes from the configured People folder
- **Graph agent:** Expands seed results by following wikilinks 1-2 hops. Decay: 0.8x for 1-hop, 0.5x for 2-hop. Relevance filter: must contain query term (FTS5) or share tags with seed. Multi-parent merge takes highest score
- **Smart chunking:** Break-point scoring algorithm assigns scores to potential split points (headings 50-100, code fences 80, thematic breaks 60, blank lines 20). Code fence protection prevents splitting inside code blocks
- **Incremental indexing:** `diff_vault()` compares file content hashes in SQLite against disk. Changed files have their old chunks, vectors, and edges deleted, then are re-processed. FTS5 and sqlite-vec entries cleaned up alongside store entries
- **sqlite-vec for vector search:** Vectors stored in a `vec_chunks` virtual table (vec0). KNN search via `vec_distance_cosine()`. Real deletes — no tombstone filtering needed during search
- **Write pipeline:** 5-step process for creating/modifying notes: (1) resolve tags via fuzzy matching against tag registry, (2) discover potential wikilinks via exact + fuzzy matching, (3) suggest folder placement via centroid similarity, (4) atomic file write (temp + rename for crash safety), (5) immediate index update (embed + insert into sqlite-vec + FTS5 + edges)
- **Warm sync (file watcher):** OS thread watches vault via `notify-debouncer-full` (2s debounce). Events sent over `tokio::mpsc` to async consumer. Two-pass processing: mutations then edge rebuild. Move detection via content hash matching. Placement correction learning on file moves (centroid adjustment + frontmatter stripping). Startup reconciliation catches changes since last shutdown
- **Fuzzy link matching:** Sliding window of N words over content, compared via `strsim::normalized_levenshtein` with 0.92 threshold. First-name matching for People notes (uniqueness check, 650bp confidence, suggestion-only). Overlap resolution: exact > alias > fuzzy > first-name
- **Centroid updates:** Online mean math (`adjust_folder_centroid`). Incremented on file add, decremented on file remove. Full recompute during bulk indexing
- **Docids:** Each file gets a deterministic 6-char hex ID. Displayed in search results
- **Vault profiles:** `engraph init` auto-detects vault structure and writes `vault.toml`
- **Pluggable models:** `ModelBackend` trait enables future model swapping

## Data directory

`~/.engraph/` — hardcoded via `Config::data_dir()`. Contains `engraph.db` (SQLite with FTS5 + sqlite-vec + edges), `models/` (ONNX model + tokenizer), `vault.toml` (vault profile), `config.toml` (user config).

Single vault only. Re-indexing a different vault path triggers a confirmation prompt.

## Dependencies to be aware of

- `ort` (2.0.0-rc.12) — ONNX Runtime Rust bindings. Pre-release API. Does not provide prebuilt binaries for all targets
- `sqlite-vec` (0.1.8-alpha.1) — SQLite extension for vector search. Provides vec0 virtual tables with KNN via `vec_distance_cosine()`
- `zerocopy` (0.7) — zero-copy serialization for vector data passed to sqlite-vec
- `strsim` (0.11) — string similarity for fuzzy tag matching and fuzzy link matching
- `time` (0.3) — date/time handling for frontmatter timestamps
- `tokenizers` (0.22) — HuggingFace tokenizer. Needs `fancy-regex` feature
- `ignore` (0.4) — vault walking with `.gitignore` support
- `rusqlite` (0.32) — bundled SQLite with FTS5 support
- `rmcp` (1.2) — MCP server SDK for stdio transport
- `notify` (7.0) — cross-platform filesystem notification (FSEvents on macOS, inotify on Linux)
- `notify-debouncer-full` (0.4) — debouncing + best-effort inode-based rename tracking

## Testing

- Unit tests in each module (`cargo test --lib`) — 225 tests, no network required
- 1 ignored smoke test (`test_embed_smoke`) — downloads ONNX model, verifies embedding
- Integration tests (`cargo test --test integration -- --ignored`) — require model download

## CI/CD

- CI: `cargo fmt --check` + `cargo clippy -- -D warnings` + `cargo test --lib` on macOS + Ubuntu
- Release: native builds on macOS arm64 (macos-14) + Linux x86_64 (ubuntu-latest). Triggered by `v*` tags
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
```
