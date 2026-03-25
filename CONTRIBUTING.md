# Contributing to engraph

Thanks for your interest in contributing to engraph.

## Getting started

```bash
git clone https://github.com/devwhodevs/engraph
cd engraph
cargo test --lib    # 225 tests, no network required
```

## Before you start

**Open an issue first.** For anything beyond a typo fix, please open an issue to discuss what you'd like to change. This avoids wasted effort if the change doesn't fit the project direction.

## Development workflow

1. Fork the repo and create a branch from `main`
2. Write tests for any new functionality
3. Run the full check suite:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test --lib
   ```
4. Open a pull request with a clear description of what and why

## Architecture

The codebase is 20 Rust modules behind a lib crate. See `CLAUDE.md` for detailed architecture documentation — it's designed for AI-assisted development but serves as a thorough codebase guide for human contributors too.

Key modules:
- `store.rs` — SQLite persistence (all tables, queries, migrations)
- `indexer.rs` — vault walking, chunking, embedding, index updates
- `serve.rs` — MCP server with 13 tools
- `watcher.rs` — file change detection and real-time re-indexing
- `search.rs` — 3-lane hybrid search orchestration

## What makes a good contribution

- Bug fixes with tests
- Performance improvements with benchmarks
- New MCP tools that expose existing functionality
- Documentation improvements
- Platform support (Windows, other architectures)

## Code style

- Run `cargo fmt` before committing
- No clippy warnings (`cargo clippy -- -D warnings`)
- Prefer small, focused PRs over large changes
- Tests are expected for new functionality
