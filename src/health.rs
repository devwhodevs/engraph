use anyhow::Result;

use crate::store::Store;

/// Full vault health report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthReport {
    pub orphans: Vec<String>,
    pub broken_links: Vec<BrokenLink>,
    pub stale_notes: Vec<String>,
    pub inbox_pending: Vec<String>,
    pub tag_issues: Vec<TagIssue>,
    pub index_age_seconds: u64,
    pub total_files: usize,
}

/// A wikilink that could not be resolved to any indexed file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrokenLink {
    pub source: String,
    pub target: String,
}

/// A tag-related problem in a file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagIssue {
    pub file: String,
    pub issue: String,
}

/// Configuration controlling which folders are excluded from health checks.
pub struct HealthConfig {
    pub daily_folder: Option<String>,
    pub inbox_folder: Option<String>,
}

/// Find files with no edges (neither incoming nor outgoing).
///
/// Excludes files whose path starts with the configured daily or inbox folder
/// prefixes — those are expected to be unlinked.
pub fn find_orphans(store: &Store, config: &HealthConfig) -> Result<Vec<String>> {
    let mut exclude = Vec::new();
    if let Some(ref daily) = config.daily_folder {
        exclude.push(daily.as_str());
    }
    if let Some(ref inbox) = config.inbox_folder {
        exclude.push(inbox.as_str());
    }
    let isolated = store.find_isolated_files(&exclude)?;
    Ok(isolated.into_iter().map(|f| f.path).collect())
}

/// Find wikilink references that could not be resolved to any indexed file.
///
/// These are recorded in the `unresolved_links` table during indexing.
pub fn find_broken_links(store: &Store) -> Result<Vec<BrokenLink>> {
    let unresolved = store.get_unresolved_links()?;
    Ok(unresolved
        .into_iter()
        .map(|(source, target)| BrokenLink { source, target })
        .collect())
}

/// Find notes that haven't been updated in the given number of days.
///
/// Stub — returns an empty vec for now. A full implementation would check
/// `mtime` or a `reviewed_at` frontmatter field.
pub fn find_stale_notes(_store: &Store, _days: u32) -> Result<Vec<String>> {
    Ok(Vec::new())
}

/// Generate a combined health report for the vault.
pub fn generate_health_report(store: &Store, config: &HealthConfig) -> Result<HealthReport> {
    let orphans = find_orphans(store, config)?;
    let broken_links = find_broken_links(store)?;
    let stale_notes = find_stale_notes(store, 90)?;

    // Inbox pending: files in the inbox folder.
    let inbox_pending = if let Some(ref inbox) = config.inbox_folder {
        store
            .find_files_by_prefix(&format!("{}%", inbox))?
            .into_iter()
            .map(|f| f.path)
            .collect()
    } else {
        Vec::new()
    };

    let all_files = store.get_all_files()?;
    let total_files = all_files.len();

    // Tag issues: find work notes missing required tags.
    let tag_issues = all_files
        .iter()
        .filter(|f| f.path.contains("Work/") || f.path.contains("01-Projects/Work/"))
        .filter(|f| !f.tags.iter().any(|t| t == "work"))
        .map(|f| TagIssue {
            file: f.path.clone(),
            issue: "work note missing 'work' tag".to_string(),
        })
        .collect();

    // Index age: seconds since the most recent indexed_at timestamp.
    let index_age_seconds = {
        let last = all_files
            .iter()
            .filter_map(|f| f.indexed_at.parse::<u64>().ok())
            .max()
            .unwrap_or(0);
        if last == 0 {
            0
        } else {
            use std::time::SystemTime;
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now.saturating_sub(last)
        }
    };

    Ok(HealthReport {
        orphans,
        broken_links,
        stale_notes,
        inbox_pending,
        tag_issues,
        index_age_seconds,
        total_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn setup_health_store() -> Store {
        let store = Store::open_memory().unwrap();
        // Insert files with edges to test orphan detection.
        let linked_id = store
            .insert_file("linked.md", "aaa111", 100, &[], "aaa111", None, None)
            .unwrap();
        let orphan_id = store
            .insert_file("orphan.md", "bbb222", 100, &[], "bbb222", None, None)
            .unwrap();
        let _daily_id = store
            .insert_file(
                "daily/2026-03-26.md",
                "ccc333",
                100,
                &[],
                "ccc333",
                None,
                None,
            )
            .unwrap();
        // Add edge: linked.md → orphan.md (both files are "connected")
        store.insert_edge(linked_id, orphan_id, "wikilink").unwrap();
        store
    }

    #[test]
    fn test_find_orphans_excludes_daily() {
        let store = setup_health_store();
        let config = HealthConfig {
            daily_folder: Some("daily/".to_string()),
            inbox_folder: None,
        };
        let orphans = find_orphans(&store, &config).unwrap();
        // linked.md has outgoing edge, orphan.md has incoming edge — both connected.
        // daily note is excluded by prefix. Result should be empty.
        assert!(orphans.is_empty());
    }

    #[test]
    fn test_find_orphans_detects_isolated() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("connected.md", "h1", 100, &[], "d1", None, None)
            .unwrap();
        let iso_id = store
            .insert_file("island.md", "h2", 100, &[], "d2", None, None)
            .unwrap();
        let other_id = store
            .insert_file("other.md", "h3", 100, &[], "d3", None, None)
            .unwrap();
        store.insert_edge(iso_id, other_id, "wikilink").unwrap();

        let config = HealthConfig {
            daily_folder: None,
            inbox_folder: None,
        };
        let orphans = find_orphans(&store, &config).unwrap();
        // connected.md has no edges at all — it's the orphan.
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0], "connected.md");
    }

    #[test]
    fn test_find_broken_links() {
        let store = setup_health_store();
        // Record an unresolved link (wikilink target that doesn't exist).
        store
            .insert_unresolved_link("linked.md", "nonexistent.md")
            .unwrap();
        let broken = find_broken_links(&store).unwrap();
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0].source, "linked.md");
        assert_eq!(broken[0].target, "nonexistent.md");
    }

    #[test]
    fn test_find_broken_links_empty_when_none() {
        let store = setup_health_store();
        let broken = find_broken_links(&store).unwrap();
        assert!(broken.is_empty());
    }

    #[test]
    fn test_generate_health_report() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("note.md", "h1", 100, &[], "d1", None, None)
            .unwrap();
        store
            .insert_file("00-Inbox/unsorted.md", "h2", 100, &[], "d2", None, None)
            .unwrap();
        store
            .insert_unresolved_link("note.md", "missing.md")
            .unwrap();

        let config = HealthConfig {
            daily_folder: Some("daily/".to_string()),
            inbox_folder: Some("00-Inbox/".to_string()),
        };
        let report = generate_health_report(&store, &config).unwrap();
        assert_eq!(report.total_files, 2);
        // note.md has no edges and is not in daily/ or inbox/ — it's an orphan.
        assert_eq!(report.orphans.len(), 1);
        assert_eq!(report.orphans[0], "note.md");
        // One broken link recorded.
        assert_eq!(report.broken_links.len(), 1);
        // One file in inbox.
        assert_eq!(report.inbox_pending.len(), 1);
        assert_eq!(report.inbox_pending[0], "00-Inbox/unsorted.md");
    }
}
