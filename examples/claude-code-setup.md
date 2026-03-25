# Using engraph with Claude Code

Connect engraph to Claude Code so it can search, read, and write to your Obsidian vault.

## Setup

### 1. Install and index

```bash
brew install devwhodevs/tap/engraph
engraph index ~/path/to/vault
```

### 2. Add to Claude Code settings

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "engraph": {
      "command": "engraph",
      "args": ["serve"]
    }
  }
}
```

### 3. Start using

Claude Code now has access to 13 vault tools:

**Read tools:**
- `search` — hybrid search across the vault
- `read` — read a full note with metadata
- `list` — filtered note listing (by folder, tags, creator)
- `vault_map` — vault structure overview
- `who` — person context bundle (note + mentions + connections)
- `project` — project context bundle
- `context` — rich topic context with token budget

**Write tools:**
- `create` — create a note with smart filing
- `append` — append content to an existing note
- `update_metadata` — update tags and aliases
- `move_note` — move a note to a different folder
- `archive` — soft-delete to archive folder
- `unarchive` — restore from archive

## Example interactions

**"What do I know about authentication?"**
Claude will call `search("authentication")` and get results from semantic, keyword, and graph lanes.

**"Who is working on the API project?"**
Claude will call `project("API")` to get the project bundle — related notes, team members, active tasks.

**"Create a meeting note for today's standup"**
Claude will call `create` with content, tags, and type hint. engraph resolves tags against your registry, discovers wikilinks in the content, and places the note in the best folder.

## Real-time sync

The MCP server includes a file watcher. When you edit notes in Obsidian, engraph re-indexes them automatically (2-second debounce). No need to manually re-run `engraph index`.

## Tips

- Use `context("topic", budget=8000)` for token-budgeted context bundles — great for feeding context into prompts
- `vault_map` helps Claude understand your vault structure before searching
- `who("Person Name")` is powerful for understanding someone's involvement across projects
- The `--explain` flag on CLI search shows per-lane score breakdown — useful for debugging search quality
