# engraph

Local knowledge graph + intelligence layer for Obsidian vaults. Rust CLI + MCP server. llama.cpp inference with Metal GPU. MIT licensed.

## Architecture

Single binary with 22 modules behind a lib crate:

- `config.rs` — loads `~/.engraph/config.toml` and `vault.toml`, merges CLI args, provides `data_dir()`. Includes `intelligence: Option<bool>`, `[models]` section for model overrides, `[obsidian]` section (CLI path, enabled flag), and `[agents]` section (registered AI agent names). `Config::save()` writes back to disk.
- `chunker.rs` — smart chunking with break-point scoring algorithm. Finds optimal split points considering headings, code fences, blank lines, and thematic breaks. `split_oversized_chunks()` handles token-aware secondary splitting with overlap
- `docid.rs` — deterministic 6-char hex IDs for files (SHA-256 of path, truncated). Shown in search results for quick reference
- `llm.rs` — ML inference via llama.cpp (Rust bindings: `llama-cpp-2`). Three traits: `EmbedModel` (embeddings), `RerankModel` (cross-encoder scoring), `OrchestratorModel` (query intent + expansion). Three llama.cpp implementations: `LlamaEmbed` (embeddinggemma-300M GGUF on Metal GPU), `LlamaOrchestrator` (Qwen3-0.6B for query analysis + expansion), `LlamaRerank` (Qwen3-Reranker-0.6B for relevance scoring). Global `LlamaBackend` via `OnceLock`. Also: `MockLlm` for testing, `HfModelUri` for model download, `FlexTokenizer` (HuggingFace tokenizers + shimmytok GGUF fallback), `PromptFormat` for model-family prompt templates, `heuristic_orchestrate()` fast path, `LaneWeights` per query intent
- `fts.rs` — FTS5 full-text search support. Re-exports `FtsResult` from store. BM25-ranked keyword search
- `fusion.rs` — Reciprocal Rank Fusion (RRF) engine. Merges semantic + FTS5 + graph + reranker results. Supports per-lane weighting, `--explain` output with intent + per-lane detail
- `markdown.rs` — section parser. Heading detection (ATX `#` headings with level tracking), section extraction by heading text, frontmatter splitting (YAML block between `---` fences). Powers section-level reading and editing
- `obsidian.rs` — Obsidian CLI wrapper. Process detection (checks if Obsidian is running), circuit breaker state machine (Closed/Degraded/Open) for resilient CLI delegation, async subprocess execution with timeout. Falls back gracefully when Obsidian is unavailable
- `health.rs` — vault health diagnostics. Orphan detection (notes with no incoming or outgoing wikilinks), broken link detection (wikilinks pointing to nonexistent notes), stale note detection (notes not modified within configurable threshold), tag hygiene (unused/rare tags). Returns structured health report
- `context.rs` — context engine. Seven functions: `read` (full note content + metadata), `read_section` (targeted section extraction by heading), `list` (filtered note listing with `created_by` filter), `vault_map` (structure overview), `who` (person context bundle), `project` (project context bundle), `context_topic` (rich topic context with budget trimming). Pure functions taking `ContextParams` — no model loading except `context_topic` which reuses `search_internal`
- `vecstore.rs` — sqlite-vec virtual table integration. Manages the `vec_chunks` vec0 table for vector storage and KNN search. Handles insert, delete, and search operations against the virtual table
- `tags.rs` — tag registry module. Maintains a `tag_registry` table tracking known tags with source attribution. Supports fuzzy matching for tag suggestions during note creation
- `links.rs` — link discovery module. Three match types: exact basename, fuzzy (sliding window Levenshtein, 0.92 threshold), and first-name (People folder, suggestion-only at 650bp). Overlap resolution via type priority (exact > alias > fuzzy > first-name)
- `placement.rs` — folder placement engine. Uses folder centroids (online mean of embeddings per folder) to suggest the best folder for new notes. Falls back to inbox when confidence is low. Includes placement correction detection (`detect_correction_from_frontmatter`) and frontmatter stripping for moved files
- `writer.rs` — write pipeline orchestrator. 5-step pipeline: resolve tags (fuzzy match + register new), discover links (exact + fuzzy), place in folder, atomic file write (temp + rename), and index update. Supports create, append, update_metadata, move_note, archive, unarchive, edit (section-level replace/prepend/append), rewrite (full content with frontmatter preservation), edit_frontmatter (granular set/remove/add_tag/remove_tag/add_alias/remove_alias ops), and delete (soft archive or hard permanent) operations with mtime-based conflict detection and crash recovery via temp file cleanup
- `watcher.rs` — file watcher for `engraph serve`. OS thread producer (notify-debouncer-full, 2s debounce) sends `Vec<WatchEvent>` over tokio::mpsc to async consumer task. Two-pass batch processing: mutations (index_file/remove_file/rename_file) then edge rebuild. Move detection via content hash matching. Placement correction on file moves. Centroid adjustment on file add/remove. Startup reconciliation via `run_index_shared`. `recent_writes` map coordination with MCP server to prevent double re-indexing of files written through the write pipeline
- `serve.rs` — MCP stdio server via rmcp SDK. Exposes 19 tools: 8 read (search, read, read_section, list, vault_map, who, project, context) + 10 write (create, append, update_metadata, move_note, archive, unarchive, edit, rewrite, edit_frontmatter, delete) + 1 diagnostic (health). `edit_frontmatter` replaces `update_metadata` for granular frontmatter mutations. EngraphServer struct with Arc+Mutex wrapping for async handlers. Loads intelligence models (orchestrator + reranker) when enabled, wires into `search_with_intelligence`. Spawns file watcher on startup. CLI events table provides audit log for write operations. `recent_writes` map prevents double re-indexing of MCP-written files
- `graph.rs` — vault graph agent. Extracts wikilink targets, expands search results by following graph connections 1-2 hops. Relevance filtering via FTS5 term check and shared tags
- `profile.rs` — vault profile detection. Auto-detects PARA/Folders/Flat structure, vault type (Obsidian/Logseq/Plain), wikilinks, frontmatter, tags. Content-based role detection for people/daily/archive folders by content patterns (not just names). Writes/loads `vault.toml`
- `store.rs` — SQLite persistence. Tables: `meta`, `files` (with docid, created_by), `chunks` (with vector BLOBs), `chunks_fts` (FTS5), `edges` (vault graph), `tombstones`, `tag_registry`, `folder_centroids`, `placement_corrections`, `link_skiplist` (reserved), `llm_cache` (orchestrator result cache), `cli_events` (audit log for CLI operations). `vec_chunks` virtual table (sqlite-vec) for KNN search. Dynamic embedding dimension stored in meta. `has_dimension_mismatch()` and `reset_for_reindex()` for migration. Enhanced `resolve_file()` with fuzzy Levenshtein matching as final fallback
- `indexer.rs` — orchestrates vault walking (via `ignore` crate for `.gitignore` support), diffing, chunking, embedding, writes to store + sqlite-vec + FTS5, vault graph edge building (wikilinks + people detection), and folder centroid computation. Exposes `index_file`, `remove_file`, `rename_file` as public per-file functions. `run_index_shared` accepts external store/embedder for watcher FullRescan. Dimension migration on model change.
- `search.rs` — hybrid search orchestrator. `search_with_intelligence()` runs the full pipeline: orchestrate (intent + expansions) → 3-lane retrieval per expansion → RRF pass 1 → reranker 4th lane → RRF pass 2. `search_internal()` is a thin wrapper without intelligence models. Adaptive lane weights per query intent.

`main.rs` is a thin clap CLI (async via `#[tokio::main]`). Subcommands: `index` (with progress bar), `search` (with `--explain`, loads intelligence models when enabled), `status` (shows intelligence state), `clear`, `init` (intelligence onboarding prompt, detects Obsidian CLI + AI agents), `configure` (`--enable-intelligence`, `--disable-intelligence`, `--model`, `--obsidian-cli`, `--no-obsidian-cli`, `--agent`), `models`, `graph` (show/stats), `context` (read/list/vault-map/who/project/topic), `write` (create/append/update-metadata/move/edit/rewrite/edit-frontmatter/delete), `serve` (MCP stdio server with file watcher + intelligence).

## Key patterns

- **4-lane hybrid search:** Queries run through up to four lanes — semantic (sqlite-vec KNN embeddings), keyword (FTS5 BM25), graph (wikilink expansion), and cross-encoder reranking. A research orchestrator classifies query intent and sets adaptive lane weights. Two-pass RRF: 3-lane retrieval → reranker scores top 30 → 4-lane fusion. When intelligence is off, falls back to heuristic intent classification with 3-lane search (v0.7 behavior)
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
- **LLM orchestrator:** Single Qwen3-0.6B call classifies query intent (exact/conceptual/relationship/exploratory), generates 2-4 query expansions, and sets adaptive lane weights. Results cached in `llm_cache` SQLite table (keyed by query SHA256). Falls back to heuristic pattern matching when intelligence is off
- **Cross-encoder reranker:** Qwen3-Reranker-0.6B scores query-document relevance via Yes/No logit softmax. Runs as 4th RRF lane on top-30 candidates from pass 1
- **llama.cpp backend:** Global `LlamaBackend` singleton via `OnceLock`. `LlamaModel` is `Send+Sync`, `LlamaContext` is `!Send` (created per-call). Metal GPU auto-detected on macOS. Models download from HuggingFace on first use with progress bar
- **Prompt formatting:** `PromptFormat` auto-detects model family from GGUF filename. embeddinggemma uses asymmetric prefixes (`search_query:` / `search_document:`). Applied in `embed_one` (queries) and `embed_batch` (documents)

## Data directory

`~/.engraph/` — hardcoded via `Config::data_dir()`. Contains `engraph.db` (SQLite with FTS5 + sqlite-vec + edges + llm_cache), `models/` (GGUF models + tokenizers), `vault.toml` (vault profile), `config.toml` (user config with intelligence toggle + model overrides).

Single vault only. Re-indexing a different vault path triggers a confirmation prompt.

## Dependencies to be aware of

- `llama-cpp-2` (0.1) — Rust bindings to llama.cpp. GGUF model loading + inference. Metal GPU on macOS, CUDA on Linux. Compiles llama.cpp C++ via build script (requires CMake)
- `shimmytok` (0.7) — pure Rust tokenizer that reads from GGUF metadata. Fallback when tokenizer.json is unavailable (gated HuggingFace repos)
- `tokenizers` (0.22) — HuggingFace tokenizer. Kept for FlexTokenizer HuggingFace backend
- `sqlite-vec` (0.1.8-alpha.1) — SQLite extension for vector search. Provides vec0 virtual tables with KNN via `vec_distance_cosine()`
- `zerocopy` (0.7) — zero-copy serialization for vector data passed to sqlite-vec
- `strsim` (0.11) — string similarity for fuzzy tag matching and fuzzy link matching
- `time` (0.3) — date/time handling for frontmatter timestamps
- `ignore` (0.4) — vault walking with `.gitignore` support
- `rusqlite` (0.32) — bundled SQLite with FTS5 support
- `rmcp` (1.2) — MCP server SDK for stdio transport
- `notify` (7.0) — cross-platform filesystem notification (FSEvents on macOS, inotify on Linux)
- `notify-debouncer-full` (0.4) — debouncing + best-effort inode-based rename tracking

## Testing

- Unit tests in each module (`cargo test --lib`) — 318 tests, no network required
- Integration tests (`cargo test --test integration -- --ignored`) — require GGUF model download
- Build requires CMake (for llama.cpp C++ compilation)

## CI/CD

- CI: `cargo fmt --check` + `cargo clippy -- -D warnings` + `cargo test --lib` on macOS + Ubuntu. Ubuntu step installs CMake.
- Release: native builds on macOS arm64 (macos-14) + Linux x86_64 (ubuntu-latest). Triggered by `v*` tags
- Homebrew: `devwhodevs/homebrew-tap` — formula builds from source tarball. Depends on `cmake` + `rust`.

## Common tasks

```bash
# Run tests (requires CMake)
cargo test --lib

# Run integration tests (downloads GGUF model)
cargo test --test integration -- --ignored

# Build release
cargo build --release

# Release: tag and push
git tag v1.x.y && git push origin v1.x.y

# Enable intelligence (downloads ~1.3GB)
engraph configure --enable-intelligence

# Re-record demo GIF
vhs assets/demo.tape
```
