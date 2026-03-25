# Changelog

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
