use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::profile::VaultProfile;
use crate::store::Store;

/// Shared context for all context engine functions.
pub struct ContextParams<'a> {
    pub store: &'a Store,
    pub vault_path: &'a Path,
    pub profile: Option<&'a VaultProfile>,
}

#[derive(Debug, Serialize)]
pub struct NoteContent {
    pub path: String,
    pub docid: Option<String>,
    pub content: String,
    pub tags: Vec<String>,
    pub frontmatter: String,
    pub body: String,
    pub outgoing_links: Vec<String>,
    pub incoming_links: Vec<String>,
    pub mentions_people: Vec<String>,
    pub mentioned_by: Vec<String>,
    pub char_count: usize,
}

#[derive(Debug, Serialize)]
pub struct NoteListItem {
    pub path: String,
    pub docid: Option<String>,
    pub tags: Vec<String>,
    pub indexed_at: String,
    pub edge_count: usize,
}

#[derive(Debug, Serialize)]
pub struct VaultMap {
    pub vault_path: String,
    pub vault_type: String,
    pub structure: String,
    pub total_files: usize,
    pub total_chunks: usize,
    pub total_edges: usize,
    pub folders: Vec<FolderInfo>,
    pub top_tags: Vec<(String, usize)>,
    pub recent_files: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct FolderInfo {
    pub path: String,
    pub note_count: usize,
}

#[derive(Debug, Serialize)]
pub struct PersonContext {
    pub name: String,
    pub note: Option<NoteContent>,
    pub mentioned_in: Vec<MentionInfo>,
    pub linked_from: Vec<String>,
    pub linked_to: Vec<String>,
    pub total_chars: usize,
}

#[derive(Debug, Serialize)]
pub struct MentionInfo {
    pub path: String,
    pub docid: Option<String>,
    pub snippet: String,
}

#[derive(Debug, Serialize)]
pub struct ProjectContext {
    pub name: String,
    pub note: Option<NoteContent>,
    pub child_notes: Vec<NoteListItem>,
    pub active_tasks: Vec<TaskItem>,
    pub team: Vec<String>,
    pub recent_mentions: Vec<MentionInfo>,
    pub total_chars: usize,
}

#[derive(Debug, Serialize)]
pub struct TaskItem {
    pub text: String,
    pub source_file: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a file by docid (#abcdef), exact path, or basename match.
fn resolve_file(
    params: &ContextParams,
    file_or_docid: &str,
) -> Result<Option<crate::store::FileRecord>> {
    // Docid lookup: #abcdef
    if file_or_docid.starts_with('#') && file_or_docid.len() == 7 {
        return params.store.get_file_by_docid(&file_or_docid[1..]);
    }

    // Exact path lookup
    if let Some(f) = params.store.get_file(file_or_docid)? {
        return Ok(Some(f));
    }

    // Basename fallback: append .md if needed, then case-insensitive suffix match
    let target = if file_or_docid.ends_with(".md") {
        file_or_docid.to_string()
    } else {
        format!("{}.md", file_or_docid)
    };
    let target_lower = target.to_lowercase();
    let all = params.store.get_all_files()?;
    Ok(all.into_iter().find(|f| {
        let p = f.path.to_lowercase();
        p == target_lower || p.ends_with(&format!("/{}", target_lower))
    }))
}

/// Split content into (frontmatter YAML, body) parts.
fn split_frontmatter(content: &str) -> (String, String) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (String::new(), content.to_string());
    }
    let after = &trimmed[3..];
    let after = after.trim_start_matches('-');
    let after = after.strip_prefix('\n').unwrap_or(after);
    if let Some(end) = after.find("\n---") {
        let fm = after[..end].to_string();
        let body = after[end + 4..]
            .strip_prefix('\n')
            .unwrap_or(&after[end + 4..]);
        (fm, body.to_string())
    } else {
        (String::new(), content.to_string())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read a single note with full content, metadata, and graph edges.
pub fn context_read(params: &ContextParams, file_or_docid: &str) -> Result<NoteContent> {
    let record = resolve_file(params, file_or_docid)?
        .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_or_docid))?;

    let full_path = params.vault_path.join(&record.path);
    let (content, body, frontmatter) = match std::fs::read_to_string(&full_path) {
        Ok(c) => {
            let (fm, b) = split_frontmatter(&c);
            (c, b, fm)
        }
        Err(_) => {
            let msg = "[File not found on disk. Re-run 'engraph index' to update.]".to_string();
            (String::new(), msg, String::new())
        }
    };

    let outgoing_links: Vec<String> = params
        .store
        .get_outgoing(record.id, Some("wikilink"))?
        .iter()
        .filter_map(|(fid, _)| params.store.get_file_path_by_id(*fid).ok().flatten())
        .collect();
    let incoming_links: Vec<String> = params
        .store
        .get_incoming(record.id, Some("wikilink"))?
        .iter()
        .filter_map(|(fid, _)| params.store.get_file_path_by_id(*fid).ok().flatten())
        .collect();
    let mentions_people: Vec<String> = params
        .store
        .get_outgoing(record.id, Some("mention"))?
        .iter()
        .filter_map(|(fid, _)| params.store.get_file_path_by_id(*fid).ok().flatten())
        .collect();
    let mentioned_by: Vec<String> = params
        .store
        .get_incoming(record.id, Some("mention"))?
        .iter()
        .filter_map(|(fid, _)| params.store.get_file_path_by_id(*fid).ok().flatten())
        .collect();

    let char_count = content.len();
    Ok(NoteContent {
        path: record.path,
        docid: record.docid,
        content,
        tags: record.tags,
        frontmatter,
        body,
        outgoing_links,
        incoming_links,
        mentions_people,
        mentioned_by,
        char_count,
    })
}

/// List notes with optional folder/tag filters and edge counts.
pub fn context_list(
    params: &ContextParams,
    folder: Option<&str>,
    tags: &[String],
    limit: usize,
) -> Result<Vec<NoteListItem>> {
    let files = params.store.list_files(folder, tags, limit)?;
    let mut items = Vec::new();
    for f in files {
        let edge_count = params.store.edge_count_for_file(f.id).unwrap_or(0);
        items.push(NoteListItem {
            path: f.path,
            docid: f.docid,
            tags: f.tags,
            indexed_at: f.indexed_at,
            edge_count,
        });
    }
    Ok(items)
}

/// High-level vault overview: folders, tags, recent files, counts.
pub fn vault_map(params: &ContextParams) -> Result<VaultMap> {
    let stats = params.store.stats()?;
    let edge_stats = params.store.get_edge_stats().ok();

    let (vault_type, structure) = match params.profile {
        Some(p) => (
            format!("{:?}", p.vault_type),
            format!("{:?}", p.structure.method),
        ),
        None => ("Unknown".into(), "Unknown".into()),
    };

    let folder_counts = params.store.folder_note_counts()?;
    let folders: Vec<FolderInfo> = folder_counts
        .into_iter()
        .map(|(path, count)| FolderInfo {
            path,
            note_count: count,
        })
        .collect();

    let top_tags = params.store.top_tags(20)?;

    let recent = params.store.recent_files(10)?;
    let recent_files: Vec<String> = recent.into_iter().map(|f| f.path).collect();

    Ok(VaultMap {
        vault_path: params.vault_path.to_string_lossy().to_string(),
        vault_type,
        structure,
        total_files: stats.file_count,
        total_chunks: stats.chunk_count,
        total_edges: edge_stats.map(|e| e.total_edges).unwrap_or(0),
        folders,
        top_tags,
        recent_files,
    })
}

/// Build a person context bundle: note content, mentions, wikilink connections.
pub fn context_who(params: &ContextParams, name: &str) -> Result<PersonContext> {
    let name_md = format!("{}.md", name);
    let name_lower = name_md.to_lowercase();
    let all_files = params.store.get_all_files()?;
    let person_file = all_files.iter().find(|f| {
        let basename = f.path.rsplit('/').next().unwrap_or(&f.path).to_lowercase();
        basename == name_lower
    });

    let (note, person_id) = if let Some(pf) = person_file {
        let n = context_read(params, &pf.path)?;
        (Some(n), Some(pf.id))
    } else {
        (None, None)
    };

    let mut mentioned_in = Vec::new();
    let mut linked_from = Vec::new();
    let mut linked_to = Vec::new();

    if let Some(pid) = person_id {
        // Mention edges
        let mentions = params.store.get_incoming(pid, Some("mention"))?;
        for (fid, _) in &mentions {
            if let Some(path) = params.store.get_file_path_by_id(*fid).ok().flatten() {
                let docid = params
                    .store
                    .get_file_by_id(*fid)
                    .ok()
                    .flatten()
                    .and_then(|f| f.docid);
                let snippet = get_mention_snippet(params, *fid, name);
                mentioned_in.push(MentionInfo {
                    path,
                    docid,
                    snippet,
                });
            }
        }
        // Wikilink edges
        let incoming_wl = params.store.get_incoming(pid, Some("wikilink"))?;
        for (fid, _) in &incoming_wl {
            if let Some(path) = params.store.get_file_path_by_id(*fid).ok().flatten() {
                linked_from.push(path);
            }
        }
        let outgoing_wl = params.store.get_outgoing(pid, Some("wikilink"))?;
        for (fid, _) in &outgoing_wl {
            if let Some(path) = params.store.get_file_path_by_id(*fid).ok().flatten() {
                linked_to.push(path);
            }
        }
    }

    let total_chars = note.as_ref().map(|n| n.char_count).unwrap_or(0)
        + mentioned_in.iter().map(|m| m.snippet.len()).sum::<usize>();

    Ok(PersonContext {
        name: name.to_string(),
        note,
        mentioned_in,
        linked_from,
        linked_to,
        total_chars,
    })
}

/// Get a snippet from a file mentioning a name. Try FTS first, fall back to disk read.
fn get_mention_snippet(params: &ContextParams, file_id: i64, name: &str) -> String {
    if let Ok(results) = params.store.fts_search(name, 5)
        && let Some(r) = results.iter().find(|r| r.file_id == file_id)
    {
        return r.snippet.clone();
    }
    if let Some(path) = params.store.get_file_path_by_id(file_id).ok().flatten() {
        let full_path = params.vault_path.join(&path);
        if let Ok(content) = std::fs::read_to_string(&full_path) {
            let name_lower = name.to_lowercase();
            for line in content.lines() {
                if line.to_lowercase().contains(&name_lower) {
                    let truncated: String = line.chars().take(200).collect();
                    return if line.len() > 200 {
                        format!("{}...", truncated)
                    } else {
                        truncated
                    };
                }
            }
        }
    }
    String::new()
}

/// Build a project context bundle: note, child notes, tasks, team, recent mentions.
pub fn context_project(params: &ContextParams, name: &str) -> Result<ProjectContext> {
    let name_md = format!("{}.md", name);
    let name_lower = name_md.to_lowercase();
    let all_files = params.store.get_all_files()?;
    let project_file = all_files.iter().find(|f| {
        let basename = f.path.rsplit('/').next().unwrap_or(&f.path).to_lowercase();
        basename == name_lower
    });

    let (note, project_id, project_folder) = if let Some(pf) = project_file {
        let n = context_read(params, &pf.path)?;
        let folder = pf.path.rsplit_once('/').map(|(f, _)| f.to_string());
        (Some(n), Some(pf.id), folder)
    } else {
        (None, None, None)
    };

    let mut child_ids = HashSet::new();
    let mut child_notes = Vec::new();

    // Files in same folder
    if let Some(folder) = &project_folder {
        let folder_files = params.store.list_files(Some(folder), &[], 50)?;
        for f in folder_files {
            if Some(f.id) != project_id && child_ids.insert(f.id) {
                let ec = params.store.edge_count_for_file(f.id).unwrap_or(0);
                child_notes.push(NoteListItem {
                    path: f.path,
                    docid: f.docid,
                    tags: f.tags,
                    indexed_at: f.indexed_at,
                    edge_count: ec,
                });
            }
        }
    }

    // Files linking to project
    if let Some(pid) = project_id {
        let incoming = params.store.get_incoming(pid, Some("wikilink"))?;
        for (fid, _) in &incoming {
            if child_ids.insert(*fid)
                && let Some(f) = params.store.get_file_by_id(*fid).ok().flatten()
            {
                let ec = params.store.edge_count_for_file(*fid).unwrap_or(0);
                child_notes.push(NoteListItem {
                    path: f.path,
                    docid: f.docid,
                    tags: f.tags,
                    indexed_at: f.indexed_at,
                    edge_count: ec,
                });
            }
        }
    }

    // Active tasks
    let mut active_tasks = Vec::new();
    let scan_tasks = |path: &str, tasks: &mut Vec<TaskItem>| {
        let full = params.vault_path.join(path);
        if let Ok(content) = std::fs::read_to_string(&full) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("- [ ] ") {
                    tasks.push(TaskItem {
                        text: trimmed
                            .strip_prefix("- [ ] ")
                            .unwrap_or(trimmed)
                            .to_string(),
                        source_file: path.to_string(),
                    });
                }
            }
        }
    };
    if let Some(n) = &note {
        scan_tasks(&n.path, &mut active_tasks);
    }
    for child in &child_notes {
        scan_tasks(&child.path, &mut active_tasks);
    }

    // Team: people linked from project
    let mut team = Vec::new();
    if let Some(pid) = project_id {
        let outgoing = params.store.get_outgoing(pid, Some("wikilink"))?;
        for (fid, _) in &outgoing {
            if let Some(path) = params.store.get_file_path_by_id(*fid).ok().flatten()
                && path.to_lowercase().contains("people")
            {
                team.push(path);
            }
        }
    }

    // Recent mentions in daily notes
    let mut recent_mentions = Vec::new();
    if let Ok(fts_results) = params.store.fts_search(name, 10) {
        for r in fts_results {
            if let Some(path) = params.store.get_file_path_by_id(r.file_id).ok().flatten()
                && (path.contains("Daily") || path.contains("daily"))
            {
                let docid = params
                    .store
                    .get_file_by_id(r.file_id)
                    .ok()
                    .flatten()
                    .and_then(|f| f.docid);
                recent_mentions.push(MentionInfo {
                    path,
                    docid,
                    snippet: r.snippet,
                });
                if recent_mentions.len() >= 5 {
                    break;
                }
            }
        }
    }

    let total_chars = note.as_ref().map(|n| n.char_count).unwrap_or(0);

    Ok(ProjectContext {
        name: name.to_string(),
        note,
        child_notes,
        active_tasks,
        team,
        recent_mentions,
        total_chars,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::store::Store;
    use tempfile::TempDir;

    fn setup_vault() -> (TempDir, Store, std::path::PathBuf) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();

        std::fs::write(
            root.join("note.md"),
            "---\ntags:\n  - rust\n---\n# Note\n\nContent here.\n\nSee [[other]].",
        )
        .unwrap();
        std::fs::write(root.join("other.md"), "# Other\n\nMore content.").unwrap();

        let store = Store::open_memory().unwrap();
        let d1 = generate_docid("note.md");
        let d2 = generate_docid("other.md");
        store
            .insert_file("note.md", "h1", 100, &["rust".into()], &d1)
            .unwrap();
        store.insert_file("other.md", "h2", 100, &[], &d2).unwrap();

        let f1 = store.get_file("note.md").unwrap().unwrap().id;
        let f2 = store.get_file("other.md").unwrap().unwrap().id;
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f1, "wikilink").unwrap();

        (tmp, store, root)
    }

    #[test]
    fn test_read_by_path() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "note.md").unwrap();
        assert_eq!(note.path, "note.md");
        assert!(note.content.contains("Content here."));
        assert!(note.body.contains("Content here."));
        assert!(note.frontmatter.contains("tags:"));
        assert!(note.tags.contains(&"rust".to_string()));
        assert_eq!(note.outgoing_links.len(), 1);
        assert_eq!(note.incoming_links.len(), 1);
        assert!(note.char_count > 0);
    }

    #[test]
    fn test_read_by_docid() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let docid = generate_docid("note.md");
        let note = context_read(&params, &format!("#{}", docid)).unwrap();
        assert_eq!(note.path, "note.md");
    }

    #[test]
    fn test_read_file_not_on_disk() {
        let (_tmp, store, root) = setup_vault();
        store
            .insert_file("ghost.md", "h3", 100, &[], "ggg333")
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "ghost.md").unwrap();
        assert!(note.body.contains("File not found on disk"));
    }

    #[test]
    fn test_read_by_basename() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "note").unwrap();
        assert_eq!(note.path, "note.md");
    }

    #[test]
    fn test_context_list_no_filter() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(&params, None, &[], 20).unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_context_list_tag_filter() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(&params, None, &["rust".into()], 20).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "note.md");
    }

    #[test]
    fn test_vault_map() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let map = vault_map(&params).unwrap();
        assert_eq!(map.total_files, 2);
        assert!(!map.folders.is_empty());
        assert!(map.top_tags.iter().any(|(t, _)| t == "rust"));
    }

    #[test]
    fn test_split_frontmatter() {
        let (fm, body) = split_frontmatter("---\ntags:\n  - rust\n---\n# Hello\nWorld");
        assert!(fm.contains("tags:"));
        assert!(body.contains("# Hello"));
        assert!(!body.contains("---"));
    }

    #[test]
    fn test_split_frontmatter_no_fm() {
        let (fm, body) = split_frontmatter("# Just content\nHere.");
        assert!(fm.is_empty());
        assert!(body.contains("# Just content"));
    }

    #[test]
    fn test_who_finds_person() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("People")).unwrap();
        std::fs::write(
            root.join("People/John.md"),
            "---\naliases:\n  - JN\n---\n# John\nDeveloper.",
        )
        .unwrap();
        std::fs::write(root.join("daily.md"), "# Daily\nTalked to John about Rust.").unwrap();

        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("People/John.md", "h1", 100, &["person".into()], "aaa111")
            .unwrap();
        let f2 = store
            .insert_file("daily.md", "h2", 100, &[], "bbb222")
            .unwrap();
        store.insert_edge(f2, f1, "mention").unwrap();
        store
            .insert_chunk(f2, "# Daily", "Talked to John about Rust.", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f2, 0, "Talked to John about Rust.")
            .unwrap();

        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let person = context_who(&params, "John").unwrap();
        assert!(person.note.is_some());
        assert_eq!(person.name, "John");
        assert_eq!(person.mentioned_in.len(), 1);
        assert!(person.mentioned_in[0].path.contains("daily"));
    }

    #[test]
    fn test_who_not_found() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let person = context_who(&params, "NonExistent").unwrap();
        assert!(person.note.is_none());
        assert!(person.mentioned_in.is_empty());
    }

    #[test]
    fn test_project_context() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("01-Projects")).unwrap();
        std::fs::write(
            root.join("01-Projects/MyProject.md"),
            "# MyProject\n\n- [ ] Task one\n- [x] Done task\n- [ ] Task two",
        )
        .unwrap();
        std::fs::write(
            root.join("01-Projects/child.md"),
            "# Child\nRelated to [[MyProject]].\n- [ ] Sub task",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "01-Projects/MyProject.md",
                "h1",
                100,
                &["project".into()],
                "aaa111",
            )
            .unwrap();
        let f2 = store
            .insert_file("01-Projects/child.md", "h2", 100, &[], "bbb222")
            .unwrap();
        store.insert_edge(f2, f1, "wikilink").unwrap();
        store.insert_edge(f1, f2, "wikilink").unwrap();

        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let proj = context_project(&params, "MyProject").unwrap();
        assert!(proj.note.is_some());
        assert!(!proj.child_notes.is_empty());
        // Should find "Task one" and "Task two" (not "Done task")
        assert!(proj.active_tasks.len() >= 2);
        assert!(proj.active_tasks.iter().any(|t| t.text == "Task one"));
        assert!(proj.active_tasks.iter().any(|t| t.text == "Task two"));
        assert!(!proj.active_tasks.iter().any(|t| t.text.contains("Done")));
    }

    #[test]
    fn test_project_not_found() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let proj = context_project(&params, "NonExistent").unwrap();
        assert!(proj.note.is_none());
        assert!(proj.child_notes.is_empty());
    }
}
