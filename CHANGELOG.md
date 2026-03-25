# Changelog

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
