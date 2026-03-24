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
}
