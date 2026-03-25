# engraph

**Local knowledge graph for AI agents.** Hybrid search, vault graph, and MCP server for Obsidian vaults — entirely offline.

[![CI](https://github.com/devwhodevs/engraph/actions/workflows/ci.yml/badge.svg)](https://github.com/devwhodevs/engraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/devwhodevs/engraph)](https://github.com/devwhodevs/engraph/releases)

engraph turns your markdown vault into a searchable knowledge graph that AI agents can query through [MCP](https://modelcontextprotocol.io). It combines semantic embeddings, full-text search, and wikilink graph traversal into a single local binary. No API keys, no cloud — everything runs on your machine.

<p align="center">
  <img src="assets/demo.gif" alt="engraph demo: hybrid search with 3-lane RRF, person context bundles" width="800">
</p>

## Why engraph?

Plain vector search treats your notes as isolated documents. But knowledge isn't flat — your notes link to each other, share tags, reference the same people and projects. engraph understands these connections.

- **3-lane hybrid search** — semantic embeddings + BM25 full-text + graph expansion, fused via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Finds things that pure vector search misses.
- **MCP server for AI agents** — `engraph serve` exposes 13 tools (search, read, context bundles, note creation) that Claude, Cursor, or any MCP client can call directly.
- **Real-time sync** — file watcher keeps the index fresh as you edit in Obsidian. No manual re-indexing needed.
- **Smart write pipeline** — AI agents can create notes with automatic tag resolution, wikilink discovery, and folder placement based on semantic similarity.
- **Fully local** — ONNX embeddings (`bge-small-en-v1.5`, 127MB), SQLite storage, no network required after initial model download.

## What problem it solves

You have hundreds of markdown notes. You want your AI coding assistant to understand what you've written — not just search keywords, but follow the connections between notes, understand context, and write new notes that fit your vault's structure.

Existing options are either cloud-dependent (Notion AI, Mem), limited to keyword search (Obsidian's built-in), or require you to copy-paste context manually. engraph gives AI agents direct, structured access to your entire vault through a standard protocol.

## How it works

```
Your vault (markdown files)
        │
        ▼
┌─────────────────────────────────────────┐
│              engraph index               │
│                                         │
│  Walk → Chunk → Embed → Store → Graph   │
│                                         │
│  SQLite: files, chunks, FTS5, vectors,  │
│          edges, centroids, tags         │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│              engraph serve               │
│                                         │
│  MCP Server (stdio) + File Watcher      │
│                                         │
│  13 tools: search, read, list, context, │
│  who, project, create, append, move...  │
└─────────────────────────────────────────┘
        │
        ▼
  Claude / Cursor / any MCP client
```

1. **Index** — walks your vault, chunks markdown by headings, embeds with a local ONNX model, stores everything in SQLite with FTS5 + sqlite-vec + a wikilink graph
2. **Search** — runs three lanes in parallel (semantic KNN, BM25 keyword, graph expansion), fuses results via RRF
3. **Serve** — starts an MCP server that AI agents connect to, with a file watcher that re-indexes changes in real time

## Quick start

**Install:**

```bash
# Homebrew (macOS)
brew install devwhodevs/tap/engraph

# Pre-built binaries (macOS arm64, Linux x86_64)
# → https://github.com/devwhodevs/engraph/releases

# From source
cargo install --git https://github.com/devwhodevs/engraph
```

**Index your vault:**

```bash
engraph index ~/path/to/vault
# Downloads embedding model on first run (~127MB)
# Incremental — only re-embeds changed files on subsequent runs
```

**Search:**

```bash
engraph search "how does the auth system work"
```

```
 1. [0.03] 02-Areas/Development/Auth Architecture.md > ## OAuth Flow  #a1b2c3
    The OAuth 2.0 implementation uses PKCE for public clients...

 2. [0.02] 01-Projects/Backend/API Design.md > ## Authentication  #d4e5f6
    All endpoints require Bearer token auth. Tokens issued by...

 3. [0.02] 03-Resources/People/Sarah Chen.md  #789abc
    Lead on the auth rewrite. Key decisions documented in...
```

Note how result #3 was found via **graph expansion** — Sarah's note doesn't mention "auth system" directly, but she's linked from the auth architecture doc.

**Connect to Claude Code:**

```bash
# Start the MCP server
engraph serve

# Or add to Claude Code's settings (~/.claude/settings.json):
{
  "mcpServers": {
    "engraph": {
      "command": "engraph",
      "args": ["serve"]
    }
  }
}
```

Now Claude can search your vault, read notes, build context bundles, and create new notes — all through structured tool calls.

## Example usage

**Hybrid search with score breakdown:**

```bash
engraph search "project deadlines" --explain
```
```
 1. [0.03] 01-Projects/Q2 Planning.md > ## Milestones  #abc123
    Semantic: 0.018 | FTS: 0.015 | Graph: 0.008
    Q2 deliverables: auth rewrite by April 15, API v2 by May 1...
```

**Rich context for AI agents:**

```bash
engraph context topic "authentication" --budget 8000
```

Returns a token-budgeted context bundle: relevant notes, connected people, related projects — ready to paste into a prompt or serve via MCP.

**Person context:**

```bash
engraph context who "Sarah Chen"
```

Returns Sarah's note, all mentions across the vault, connected notes via wikilinks, and recent activity.

**Vault structure overview:**

```bash
engraph context vault-map
```

Returns folder counts, top tags, recent files — gives an AI agent orientation before it starts searching.

**Create a note via the write pipeline:**

```bash
engraph write create --content "# Meeting Notes\n\nDiscussed auth timeline with Sarah." --tags meeting,auth
```

engraph resolves tags against the registry (fuzzy matching), discovers potential wikilinks (`[[Sarah Chen]]`), suggests the best folder based on semantic similarity to existing notes, and writes atomically.

## Use cases

**AI-assisted knowledge work** — Give Claude or Cursor deep access to your personal knowledge base. Instead of copy-pasting context, the agent searches, reads, and cross-references your notes directly.

**Developer second brain** — Index architecture docs, decision records, meeting notes, and code snippets. Search by concept across all of them.

**Research and writing** — Find connections between notes that you didn't explicitly link. The graph lane surfaces related content through shared wikilinks and mentions.

**Team knowledge graphs** — Index a shared docs vault. AI agents can answer "who knows about X?" and "what decisions were made about Y?" by traversing the note graph.

## How it compares

| | engraph | Basic RAG (vector-only) | Obsidian search |
|---|---|---|---|
| Search method | Semantic + BM25 + graph (3-lane RRF) | Vector similarity only | Keyword only |
| Understands note links | Yes (wikilink graph traversal) | No | Limited (backlinks panel) |
| AI agent access | MCP server (13 tools) | Custom API needed | No |
| Write capability | Create/append/move with smart filing | No | Manual |
| Real-time sync | File watcher, 2s debounce | Manual re-index | N/A |
| Runs locally | Yes, fully offline | Depends | Yes |
| Setup | One binary, one command | Framework + code | Built-in |

engraph is not a replacement for Obsidian — it's the intelligence layer that sits between your vault and your AI tools.

## Current capabilities

- 3-lane hybrid search (semantic + FTS5 + graph expansion) with RRF fusion
- MCP server with 13 tools (7 read, 6 write) via stdio
- Real-time file watching with 2s debounce and startup reconciliation
- Write pipeline: tag resolution, fuzzy link discovery, semantic folder placement
- Context engine: topic bundles, person bundles, project bundles with token budgets
- Vault graph: bidirectional wikilink + mention edges with multi-hop expansion
- Placement correction learning from user file moves
- Fuzzy link matching (Levenshtein) + first-name matching for People notes
- Smart chunking with break-point scoring
- Vault profile auto-detection (PARA, folders, flat)
- 225 unit tests, CI on macOS + Ubuntu

## Roadmap

- [ ] Research orchestrator — query classification and adaptive lane weighting
- [ ] Temporal search — find notes by time period, detect trends
- [ ] LLM reranker — optional local model for result quality
- [ ] HTTP/REST API — complement MCP with a standard web API
- [ ] Multi-vault — search across multiple vaults
- [ ] Vault health monitor — surface orphan notes, broken links, stale content

## Configuration

Optional config at `~/.engraph/config.toml`:

```toml
vault_path = "~/Documents/MyVault"
top_n = 10
exclude = [".obsidian/", "node_modules/", ".git/"]
```

All data stored in `~/.engraph/` — single SQLite database (~10MB typical), ONNX model, and vault profile.

## Development

```bash
cargo test --lib          # 225 unit tests, no network
cargo clippy -- -D warnings
cargo fmt --check

# Integration tests (downloads ONNX model)
cargo test --test integration -- --ignored
```

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

The codebase is 20 Rust modules behind a lib crate. `CLAUDE.md` in the repo root has detailed architecture documentation for AI-assisted development.

## License

MIT
