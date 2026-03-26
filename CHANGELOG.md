# Changelog

## v1.5.0 — ChatGPT Actions (2026-03-26)

### Added
- **OpenAPI 3.1.0 spec** (`openapi.rs`) — hand-written spec for all 23 endpoints, served at `GET /openapi.json`
- **ChatGPT plugin manifest** — served at `GET /.well-known/ai-plugin.json`
- **`--setup-chatgpt` CLI helper** — interactive setup: enables HTTP, creates API key, configures CORS, prompts for public URL
- **Plugin config** — `[http.plugin]` section for name, description, contact_email, public_url

### Changed
- Module count: 25 → 26
- Test count: 417 → 426
- `/openapi.json` and `/.well-known/ai-plugin.json` routes require no authentication

## v1.4.0 — PARA Migration (2026-03-26)

### Added
- **PARA migration engine** (`migrate.rs`) — AI-assisted vault restructuring into Projects/Areas/Resources/Archive
- **Heuristic classification** — priority-ordered rules detect Projects (tasks, active status), Areas (recurring topics), Resources (people, reference), Archive (done, inactive)
- **Preview-then-apply workflow** — generates markdown + JSON preview for review before moving files
- **Migration rollback** — `engraph migrate para --undo` reverses the last migration
- **3 new MCP tools** — `migrate_preview`, `migrate_apply`, `migrate_undo`
- **3 new HTTP endpoints** — `POST /api/migrate/preview`, `/apply`, `/undo`
- **Migration log** — SQLite table tracks all moves for rollback support

### Changed
- Module count: 24 → 25
- MCP tools: 19 → 22
- HTTP endpoints: 20 → 23
- Test count: 385 → 417

## v1.3.0 — HTTP/REST Transport (2026-03-26)

### Added
- **HTTP REST API** (`http.rs`) — axum-based HTTP server alongside MCP, enabled via `engraph serve --http`
- **20 REST endpoints** mirroring all 19 MCP tools + update-metadata
- **API key authentication** — `eg_` prefixed keys with read/write permission levels
- **Rate limiting** — configurable per-key token bucket (requests/minute)
- **CORS** — configurable allowed origins for web-based agents
- **Graceful shutdown** — CancellationToken coordinates MCP + HTTP + watcher exit
- **API key management CLI** — `engraph configure --add-api-key/--list-api-keys/--revoke-api-key`
- **`--no-auth` mode** — local development without API keys (127.0.0.1 only)

### Changed
- `engraph serve` gains `--http`, `--port`, `--host`, `--no-auth` flags
- Module count: 23 → 24
- Test count: 361 → 385
- New dependencies: axum, tower-http, tower, rand, tokio-util

## v1.2.0 — Temporal Search (2026-03-26)

### Added
- **Temporal search lane** (`temporal.rs`) — 5th RRF lane for time-aware queries
- **Date extraction** — from frontmatter `date:` field or `YYYY-MM-DD` filename pattern
- **Heuristic date parsing** — "today", "yesterday", "last week", "this month", "recent", month names, ISO dates, date ranges
- **LLM date extraction** — orchestrator detects temporal intent and extracts date ranges from natural language
- **Temporal scoring** — smooth decay function for files near but outside the target date range
- **Temporal candidate injection** — date-matched files enter candidate pool as graph seeds
- **Confidence % display** — search results show normalized confidence (0-100%) instead of raw RRF scores
- **Date coverage stats** — `engraph status` shows how many files have extractable dates

### Changed
- `QueryIntent` gains `Temporal` variant with custom lane weights (temporal: 1.5)
- `OrchestrationResult` gains `date_range` field (backward-compatible serde)
- `LaneWeights` gains `temporal` field (0.0 for non-temporal intents)
- `insert_file` signature extended with `note_date` parameter
- Module count: 22 → 23
- Test count: 318 → 361

## [1.1.0] - 2026-03-26 — Complete Vault Gateway

### Added
- **Section parser** (`markdown.rs`) — heading detection, section extraction, frontmatter splitting
- **Obsidian CLI wrapper** (`obsidian.rs`) — process detection, circuit breaker (Closed/Degraded/Open), async CLI delegation
- **Vault health** (`health.rs`) — orphan detection, broken link detection, stale notes, tag hygiene
- **Section-level editing** — `edit_note()` with replace/prepend/append modes targeting specific headings
- **Note rewriting** — `rewrite_note()` with frontmatter preservation
- **Frontmatter mutations** — `edit_frontmatter()` with granular set/remove/add_tag/remove_tag/add_alias/remove_alias ops
- **Hard delete** — `delete_note()` with soft (archive) and hard (permanent) modes
- **Section reading** — `read_section()` in context engine for targeted note section access
- **Enhanced file resolution** — fuzzy Levenshtein matching as final fallback in `resolve_file()`
- **6 new MCP tools** — `read_section`, `health`, `edit`, `rewrite`, `edit_frontmatter`, `delete`
- **CLI events table** — audit log for CLI operations
- **Watcher coordination** — `recent_writes` map prevents double re-indexing of MCP-written files
- **Content-based role detection** — detect people/daily/archive folders by content patterns, not just names
- **Enhanced onboarding** — `engraph init` detects Obsidian CLI + AI agents, `engraph configure` has new flags
- **Config sections** — `[obsidian]` and `[agents]` in config.toml

### Changed
- Module count: 19 → 22
- MCP tools: 13 → 19
- Test count: 270 → 318

## [1.0.2] - 2026-03-26

### Fixed
- **Person search uses FTS** — `context who` now finds person notes via full-text search instead of exact filename matching. Handles hyphens, underscores, any vault structure. Prefers People folder → `person` tag → fuzzy filename.
- **llama.cpp logs suppressed** — `backend.void_logs()` silences Metal/model loading output. Clean terminal output by default.
- **Basename resolution** — `find_file_by_basename` normalizes hyphens/underscores/spaces for cross-format matching.

### Changed
- Re-recorded demo GIF with v1.0.2 brew binary (clean output, no `2>/dev/null` workarounds)

## [1.0.1] - 2026-03-26

### Changed
- **Inference backend switched from candle to llama.cpp** — via `llama-cpp-2` Rust bindings. Gets full Metal GPU acceleration on macOS (88 files indexed in 70s vs 37+ minutes on CPU with candle). Same backend as [qmd](https://github.com/tobi/qmd).
- Default embedding model produces 256-dim vectors via embeddinggemma-300M (Matryoshka truncation)
- BERT GGUF architecture support added alongside Gemma (future model flexibility)
- Progress bar during indexing via indicatif (was silent for minutes)
- CI workflow installs CMake on Ubuntu (required for llama.cpp build)

### Fixed
- **Prompt format applied during embedding** — `embed_one` uses search_query prefix, `embed_batch` uses search_document prefix. Without this, embeddinggemma operated in wrong symmetric mode.
- **GGUF tokenizer fallback** — added `shimmytok` crate to extract tokenizer from GGUF metadata when tokenizer.json is unavailable (Google Gemma repos are gated)
- **LlamaBackend singleton** — global `OnceLock` prevents double-initialization crash when loading multiple models
- **Orchestrator/reranker use built-in tokenizer** — llama.cpp reads tokenizer from GGUF metadata, no external tokenizer.json needed
- **Dimension migration clears FTS** — `reset_for_reindex` now also clears `chunks_fts` to prevent duplicate entries
- **LLM cache wired into search** — `search_with_intelligence` checks/populates `llm_cache` table
- **MCP server wires intelligence** — search handler passes orchestrator + reranker via `SearchConfig`
- **CLI search wires intelligence** — `run_search` loads models when intelligence enabled
- **Qwen3 GGUF filename** — fixed case sensitivity (was 404)
- **Embedding batch params** — `n_ubatch >= n_tokens` assertion, use `encode()` not `decode()`, `AddBos::Never` (PromptFormat adds `<bos>`)

### Removed
- `candle-core`, `candle-nn`, `candle-transformers` dependencies (replaced by `llama-cpp-2`)

## [1.0.0] - 2026-03-25

Intelligence release. Replaced ONNX with GGUF model inference, added LLM-powered search intelligence. Immediately followed by v1.0.1 which switched the inference backend from candle to llama.cpp for Metal GPU support.

### Added
- **GGUF model inference** — replaced ONNX (`ort`) with GGUF quantized models for all ML inference
- **Research orchestrator** — LLM-based query classification (exact/conceptual/relationship/exploratory) with adaptive lane weights. Single LLM call returns intent + 2-4 query expansions.
- **Cross-encoder reranker** — 4th RRF lane using Qwen3-Reranker for relevance scoring. Two-pass fusion: 3-lane retrieval → reranker scores top 30 → 4-lane RRF.
- **Query expansion** — each search runs multiple expanded queries through all retrieval lanes, merged via deduplication.
- **Heuristic orchestrator** — fast-path intent classification via pattern matching (docids, ticket IDs, "who" queries) when intelligence is disabled. Zero latency.
- **Intelligence onboarding** — opt-in prompt during `engraph init` and first `engraph index`. Downloads ~1.3GB of optional models.
- **`engraph configure` command** — `--enable-intelligence`, `--disable-intelligence`, `--model embed|rerank|expand <uri>` for model overrides.
- **Dimension migration** — auto-detects embedding dimension changes and triggers re-index.
- **LLM result cache** — SQLite cache for orchestrator results (keyed by query SHA256).
- **Model override support** — configurable embedding, reranker, and expansion model URIs for multilingual support.

### Changed
- Embedding model: `all-MiniLM-L6-v2` (ONNX, 384-dim, 23MB) → `embeddinggemma-300M` (GGUF, 256-dim, ~300MB)
- Search pipeline: hardcoded 3-lane weights → adaptive per-query-intent weights
- `--explain` output now shows query intent and 4-lane breakdown (semantic, FTS, graph, rerank)
- `status` command shows intelligence enabled/disabled state

### Removed
- `ort` (ONNX Runtime) dependency
- `ndarray` dependency
- `src/embedder.rs` and `src/model.rs` (replaced by `src/llm.rs`)
- `ModelBackend` trait (replaced by `EmbedModel`)

## [0.7.0] - 2026-03-25

### Added
- **File watcher** — `engraph serve` now watches the vault for changes and re-indexes automatically (2s debounce)
- **Placement correction learning** — detects when users move notes from suggested folders, updates centroids
- **Fuzzy link matching** — sliding window Levenshtein matching (0.92 threshold) during note creation
- **First-name matching** — matches "Steve" to `[[Steve Barbera]]` for People folder notes (suggestion-only)
- `created_by` column and filter — track note origin, filter with `engraph context list --created-by`
- `placement_corrections` table for observability
- `link_skiplist` table schema (reserved for future use)

### Changed
- Centroid updates use true online mean (was EMA 0.9/0.1)
- Indexer refactored: `index_file`, `remove_file`, `rename_file` extracted as public functions
- Bulk indexing uses batched transactions for performance
- `run_index_shared` variant accepts external store/embedder references

### Fixed
- Content hash consistency between `diff_vault` and `index_file` (BOM handling)

## [0.6.0] - 2026-03-25

### Added
- **Write pipeline** — create, append, update_metadata, move, archive, unarchive notes
- **sqlite-vec** replaces HNSW for vector search (single SQLite database)
- **Tag registry** with fuzzy Levenshtein resolution
- **Link discovery** — exact basename and alias matching during note creation
- **Folder placement** — type rules, semantic centroids, inbox fallback
- **Archive/unarchive** — soft delete with metadata preservation
- 6 new MCP write tools (13 total)

### Changed
- All vectors stored in SQLite vec0 virtual table (was HNSW + separate files)
- Atomic writes via temp file + rename for crash safety
- Mtime-based conflict detection for concurrent edits

## [0.5.0] - 2026-03-24

### Added
- **MCP server** — `engraph serve` starts stdio MCP server via rmcp SDK
- 7 read-only MCP tools: search, read, list, vault_map, who, project, context

## [0.4.0] - 2026-03-24

### Added
- **Context engine** — 6 functions: read, list, vault_map, who, project, topic
- Token-budgeted context bundles for AI agents
- Person and project context assembly from graph + search

## [0.3.0] - 2026-03-24

### Added
- **Vault graph** — bidirectional wikilink + mention edges built during indexing
- **Graph search agent** — 3rd RRF lane with 1-2 hop expansion
- People detection from configured People folder

## [0.2.0] - 2026-03-24

### Added
- **Hybrid search** — semantic (embeddings) + keyword (FTS5 BM25) fused via RRF
- Smart chunking with break-point scoring algorithm
- Docid system (6-char hex file IDs)
- Vault profiles with auto-detection (`engraph init`)
- Pluggable model layer (`ModelBackend` trait)
- `--explain` flag for per-lane score breakdown

## [0.1.0] - 2026-03-19

### Added
- Initial release
- ONNX embedding model (all-MiniLM-L6-v2, 384-dim)
- SQLite metadata storage
- Incremental indexing
- `.gitignore`-aware vault walking
