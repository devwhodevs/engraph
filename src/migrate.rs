//! Heuristic PARA classification engine for vault migration.
//!
//! Classifies notes into PARA categories (Project, Area, Resource, Archive)
//! using priority-ordered heuristic rules, generates migration previews,
//! and formats them as markdown for user review.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::markdown::split_frontmatter;
use crate::profile::VaultProfile;
use crate::store::Store;

// ── Core types ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Category {
    Project,
    Area,
    Resource,
    Archive,
    Skip,
    Uncertain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub category: Category,
    pub confidence: f64,
    pub signal: String,
    pub suggested_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileClassification {
    pub path: String,
    pub classification: Classification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPreview {
    pub migration_id: String,
    pub files: Vec<FileClassification>,
    pub uncertain: Vec<FileClassification>,
    pub skipped: usize,
}

#[derive(Debug, Serialize)]
pub struct MigrationResult {
    pub migration_id: String,
    pub moved: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct UndoResult {
    pub migration_id: String,
    pub restored: usize,
    pub errors: Vec<String>,
}

// ── Heuristic classifier ───────────────────────────────────────

/// Classify a note using heuristic rules only (no LLM).
/// Rules run in priority order — first match wins.
///
/// Parameters:
/// - content: full note content
/// - filename: relative path (e.g., "07-Daily/2026-03-26.md")
/// - frontmatter_str: raw frontmatter YAML (without --- delimiters), or None
/// - edge_count: incoming + outgoing edges from the store
/// - has_recent_mentions: whether the note was mentioned in notes from the last 30 days
pub fn classify_heuristic(
    content: &str,
    filename: &str,
    frontmatter_str: Option<&str>,
    edge_count: usize,
    has_recent_mentions: bool,
) -> Classification {
    // Extract basename (without extension) for pattern matching
    let basename = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Rule 1: Daily note — basename matches YYYY-MM-DD pattern
    if is_daily_note(basename) {
        return Classification {
            category: Category::Skip,
            confidence: 1.0,
            signal: "daily note filename pattern".into(),
            suggested_path: None,
        };
    }

    // Rule 2: Template — path contains "template" (case-insensitive)
    if filename.to_lowercase().contains("template") {
        return Classification {
            category: Category::Skip,
            confidence: 1.0,
            signal: "template path".into(),
            suggested_path: None,
        };
    }

    // Rule 3: Canvas — filename ends with .canvas
    if filename.ends_with(".canvas") {
        return Classification {
            category: Category::Skip,
            confidence: 1.0,
            signal: "canvas file".into(),
            suggested_path: None,
        };
    }

    let fm = frontmatter_str.unwrap_or("");

    // Rule 4: Status active/in-progress → Project (90%)
    if fm.contains("status: active") || fm.contains("status: in-progress") {
        return Classification {
            category: Category::Project,
            confidence: 0.9,
            signal: "frontmatter status active/in-progress".into(),
            suggested_path: Some("01-Projects/".into()),
        };
    }

    // Rule 5: Unchecked tasks → Project (80%)
    if content.contains("- [ ]") {
        return Classification {
            category: Category::Project,
            confidence: 0.8,
            signal: "unchecked tasks found".into(),
            suggested_path: Some("01-Projects/".into()),
        };
    }

    // Rule 6: Status done/completed → Archive (85%)
    if fm.contains("status: done") || fm.contains("status: completed") {
        return Classification {
            category: Category::Archive,
            confidence: 0.85,
            signal: "frontmatter status done/completed".into(),
            suggested_path: Some("04-Archive/".into()),
        };
    }

    // Rule 7: Person tag → Resource (90%)
    if fm.contains("- person") || fm.contains("- people") {
        return Classification {
            category: Category::Resource,
            confidence: 0.9,
            signal: "person/people tag in frontmatter".into(),
            suggested_path: Some("03-Resources/People/".into()),
        };
    }

    // Rule 8: No edges + no recent mentions → Archive (75%)
    if edge_count == 0 && !has_recent_mentions {
        return Classification {
            category: Category::Archive,
            confidence: 0.75,
            signal: "no edges and no recent mentions".into(),
            suggested_path: Some("04-Archive/".into()),
        };
    }

    // Rule 9: High edges + no tasks → Resource (70%)
    if edge_count >= 3 && !content.contains("- [ ]") {
        return Classification {
            category: Category::Resource,
            confidence: 0.7,
            signal: "high edge count with no open tasks".into(),
            suggested_path: Some("03-Resources/".into()),
        };
    }

    // Rule 10: Area keywords in filename or first 200 chars of content
    let area_keywords = [
        "health",
        "finance",
        "career",
        "learning",
        "fitness",
        "nutrition",
        "budget",
    ];
    let filename_lower = filename.to_lowercase();
    let content_prefix: String = content.chars().take(200).collect::<String>().to_lowercase();
    for keyword in &area_keywords {
        if filename_lower.contains(keyword) || content_prefix.contains(keyword) {
            return Classification {
                category: Category::Area,
                confidence: 0.6,
                signal: format!("area keyword '{keyword}' found"),
                suggested_path: Some("02-Areas/".into()),
            };
        }
    }

    // Rule 11: Nothing matched → Uncertain
    Classification {
        category: Category::Uncertain,
        confidence: 0.0,
        signal: "no heuristic rules matched".into(),
        suggested_path: None,
    }
}

/// Check if a basename matches the YYYY-MM-DD date pattern.
fn is_daily_note(basename: &str) -> bool {
    let bytes = basename.as_bytes();
    if bytes.len() != 10 {
        return false;
    }
    // Check format: DDDD-DD-DD where D is digit
    bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[0..4].iter().all(|b| b.is_ascii_digit())
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
        && bytes[8..10].iter().all(|b| b.is_ascii_digit())
}

// ── Path suggestion ───────────────────────────────────────────

/// Suggest a PARA-compliant destination path for a classified note.
///
/// Uses the `VaultProfile` folder mappings if available, otherwise falls
/// back to standard PARA folder names. Returns the current path unchanged
/// if the category is Skip/Uncertain, or if the file is already under the
/// correct PARA folder.
fn suggest_path(current_path: &str, category: &Category, profile: Option<&VaultProfile>) -> String {
    let basename = std::path::Path::new(current_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(current_path);

    let folder = match category {
        Category::Project => profile
            .and_then(|p| p.structure.folders.projects.as_deref())
            .unwrap_or("01-Projects"),
        Category::Area => profile
            .and_then(|p| p.structure.folders.areas.as_deref())
            .unwrap_or("02-Areas"),
        Category::Resource => profile
            .and_then(|p| p.structure.folders.resources.as_deref())
            .unwrap_or("03-Resources"),
        Category::Archive => profile
            .and_then(|p| p.structure.folders.archive.as_deref())
            .unwrap_or("04-Archive"),
        _ => return current_path.to_string(), // Skip/Uncertain don't move
    };

    let trimmed = folder.trim_end_matches('/');

    // If the file is already under the target folder, keep it where it is.
    if current_path.starts_with(&format!("{}/", trimmed))
        || current_path.starts_with(&format!("{}/", folder))
    {
        return current_path.to_string();
    }

    format!("{}/{}", trimmed, basename)
}

// ── Preview generation ────────────────────────────────────────

/// Generate a migration preview by classifying all indexed files.
///
/// Reads file content from disk, runs heuristic classification, computes
/// suggested paths, and partitions results into confident moves vs
/// uncertain notes that need manual review.
pub fn generate_preview(
    store: &Store,
    vault_path: &Path,
    profile: Option<&VaultProfile>,
) -> Result<MigrationPreview> {
    let migration_id = uuid::Uuid::new_v4().to_string();
    let all_files = store.get_all_files()?;
    let mut files = Vec::new();
    let mut uncertain = Vec::new();
    let mut skipped = 0;

    let thirty_days_ago = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
        - 30 * 86400;

    for file in &all_files {
        let full_path = vault_path.join(&file.path);
        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                skipped += 1;
                continue; // skip unreadable files
            }
        };

        let (fm, _body) = split_frontmatter(&content);
        let edge_count = store.edge_count_for_file(file.id).unwrap_or(0);

        // A note is "recently active" if it has a note_date within 30 days or has edges.
        let has_recent = file
            .note_date
            .map(|d| d >= thirty_days_ago)
            .unwrap_or(false)
            || edge_count > 0;

        let mut classification =
            classify_heuristic(&content, &file.path, fm.as_deref(), edge_count, has_recent);

        if classification.category == Category::Skip {
            skipped += 1;
            continue;
        }

        // Compute suggested path.
        let suggested = suggest_path(&file.path, &classification.category, profile);

        // If the file is already in the right place, skip it.
        if suggested == file.path {
            skipped += 1;
            continue;
        }

        classification.suggested_path = Some(suggested);

        let fc = FileClassification {
            path: file.path.clone(),
            classification,
        };

        if fc.classification.category == Category::Uncertain {
            uncertain.push(fc);
        } else {
            files.push(fc);
        }
    }

    // Sort by confidence descending.
    files.sort_by(|a, b| {
        b.classification
            .confidence
            .partial_cmp(&a.classification.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(MigrationPreview {
        migration_id,
        files,
        uncertain,
        skipped,
    })
}

// ── Markdown formatting ───────────────────────────────────────

/// Extract the filename (last path component) from a path string.
fn basename(path: &str) -> &str {
    std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
}

/// Extract the parent folder from a path string, or "-" if at root.
fn folder(path: &str) -> &str {
    std::path::Path::new(path)
        .parent()
        .and_then(|p| p.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("-")
}

/// Format a `MigrationPreview` as a markdown document for user review.
///
/// Groups files by category in tables showing current path, proposed path,
/// confidence, and the heuristic signal that triggered the classification.
pub fn format_preview_markdown(preview: &MigrationPreview) -> String {
    let now = OffsetDateTime::now_utc();
    let date_str = format!(
        "{:04}-{:02}-{:02}",
        now.year(),
        now.month() as u8,
        now.day()
    );

    let mut out = String::new();
    out.push_str(&format!(
        "# PARA Migration Preview\n\n\
         Generated: {} | Files to move: {} | Uncertain: {} | Skipped: {}\n\n",
        date_str,
        preview.files.len(),
        preview.uncertain.len(),
        preview.skipped,
    ));

    // Group files by category.
    for category in &[
        Category::Project,
        Category::Area,
        Category::Resource,
        Category::Archive,
    ] {
        let cat_files: Vec<_> = preview
            .files
            .iter()
            .filter(|f| f.classification.category == *category)
            .collect();
        if cat_files.is_empty() {
            continue;
        }

        out.push_str(&format!(
            "## {:?} ({} files)\n\n",
            category,
            cat_files.len()
        ));
        out.push_str("| File | Current | Proposed | Confidence | Signal |\n");
        out.push_str("|------|---------|----------|------------|--------|\n");
        for f in &cat_files {
            out.push_str(&format!(
                "| {} | {} | {} | {:.0}% | {} |\n",
                basename(&f.path),
                folder(&f.path),
                f.classification.suggested_path.as_deref().unwrap_or("?"),
                f.classification.confidence * 100.0,
                f.classification.signal,
            ));
        }
        out.push('\n');
    }

    if !preview.uncertain.is_empty() {
        out.push_str(&format!(
            "## Uncertain ({} files)\n\n",
            preview.uncertain.len()
        ));
        out.push_str("| File | Current | Best Guess | Signal |\n");
        out.push_str("|------|---------|------------|--------|\n");
        for f in &preview.uncertain {
            out.push_str(&format!(
                "| {} | {} | ? | {} |\n",
                basename(&f.path),
                folder(&f.path),
                f.classification.signal,
            ));
        }
        out.push('\n');
    }

    out
}

// ── Apply / Undo / Persistence ────────────────────────────────

/// Execute a migration preview: move each file to its suggested path.
///
/// Skips files with no suggested path or that are already in the correct
/// location. Logs each successful move to the store's migration log so it
/// can be undone later.
pub fn apply_preview(
    preview: &MigrationPreview,
    store: &Store,
    vault_path: &Path,
) -> Result<MigrationResult> {
    let mut moved = 0;
    let mut errors = Vec::new();

    for fc in &preview.files {
        let target = match &fc.classification.suggested_path {
            Some(p) => p,
            None => continue,
        };
        // Skip if already in correct location
        if fc.path == *target {
            continue;
        }

        // Extract target folder from the suggested path
        let folder = std::path::Path::new(target)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("");

        match crate::writer::move_note(&fc.path, folder, store, vault_path) {
            Ok(_) => {
                store.log_migration(
                    &preview.migration_id,
                    &fc.path,
                    target,
                    &format!("{:?}", fc.classification.category),
                    fc.classification.confidence,
                )?;
                moved += 1;
            }
            Err(e) => errors.push(format!("{}: {e:#}", fc.path)),
        }
    }

    Ok(MigrationResult {
        migration_id: preview.migration_id.clone(),
        moved,
        skipped: errors.len(),
        errors,
    })
}

/// Rollback the most recent migration by moving files back to their
/// original locations and deleting the migration log entries.
pub fn undo_last(store: &Store, vault_path: &Path) -> Result<UndoResult> {
    let migration_id = store
        .get_last_migration_id()?
        .ok_or_else(|| anyhow::anyhow!("No migration to undo"))?;
    let entries = store.get_migration(&migration_id)?;

    let mut restored = 0;
    let mut errors = Vec::new();

    // Reverse order to undo correctly
    for entry in entries.iter().rev() {
        let old_folder = std::path::Path::new(&entry.old_path)
            .parent()
            .and_then(|p| p.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(".");
        match crate::writer::move_note(&entry.new_path, old_folder, store, vault_path) {
            Ok(_) => restored += 1,
            Err(e) => errors.push(format!("{}: {e:#}", entry.new_path)),
        }
    }

    store.delete_migration(&migration_id)?;

    Ok(UndoResult {
        migration_id,
        restored,
        errors,
    })
}

/// Write a migration preview to disk as both JSON and markdown files.
pub fn save_preview(preview: &MigrationPreview, data_dir: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(preview)?;
    std::fs::write(data_dir.join("migration-preview.json"), json)?;
    let md = format_preview_markdown(preview);
    std::fs::write(data_dir.join("migration-preview.md"), md)?;
    Ok(())
}

/// Load a previously saved migration preview from disk.
pub fn load_preview(data_dir: &Path) -> Result<MigrationPreview> {
    let json = std::fs::read_to_string(data_dir.join("migration-preview.json"))?;
    let preview: MigrationPreview = serde_json::from_str(&json)?;
    Ok(preview)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_project_by_status() {
        let c = classify_heuristic(
            "---\nstatus: active\n---\n# Sprint 6\n",
            "sprint-6.md",
            Some("status: active"),
            5,
            true,
        );
        assert_eq!(c.category, Category::Project);
        assert!(c.confidence >= 0.9);
    }

    #[test]
    fn test_classify_project_by_tasks() {
        let c = classify_heuristic(
            "# Todo\n- [ ] Fix bug\n- [x] Done\n",
            "todo.md",
            None,
            2,
            true,
        );
        assert_eq!(c.category, Category::Project);
        assert!(c.confidence >= 0.8);
    }

    #[test]
    fn test_classify_archive_by_status() {
        let c = classify_heuristic(
            "---\nstatus: done\n---\n# Old\n",
            "old.md",
            Some("status: done"),
            0,
            false,
        );
        assert_eq!(c.category, Category::Archive);
    }

    #[test]
    fn test_classify_resource_person() {
        let c = classify_heuristic(
            "---\ntags:\n  - person\n---\n# John\n",
            "john.md",
            Some("tags:\n  - person"),
            3,
            true,
        );
        assert_eq!(c.category, Category::Resource);
    }

    #[test]
    fn test_classify_area_keywords() {
        let c = classify_heuristic(
            "# Health\n\nTreadmill training\n",
            "health.md",
            None,
            2,
            true,
        );
        assert_eq!(c.category, Category::Area);
    }

    #[test]
    fn test_skip_daily_note() {
        let c = classify_heuristic("# Daily\n", "2026-03-26.md", None, 0, true);
        assert_eq!(c.category, Category::Skip);
    }

    #[test]
    fn test_skip_daily_note_in_folder() {
        let c = classify_heuristic("# Daily\n", "07-Daily/2026-03-26.md", None, 0, true);
        assert_eq!(c.category, Category::Skip);
    }

    #[test]
    fn test_classify_archive_no_edges() {
        let c = classify_heuristic("# Random\nSome content\n", "random.md", None, 0, false);
        assert_eq!(c.category, Category::Archive);
    }

    #[test]
    fn test_uncertain_when_ambiguous() {
        // Has edges and recent mentions, but no tasks, no status, no person tag, no area keywords.
        // edge_count=2 avoids Rule 9 (high edges >= 3 → Resource).
        let c = classify_heuristic(
            "# Meeting notes\nDiscussed roadmap\n",
            "meeting.md",
            None,
            2,
            true,
        );
        assert_eq!(c.category, Category::Uncertain);
    }

    #[test]
    fn test_skip_template() {
        let c = classify_heuristic("# Template\n", "05-Templates/Daily Note.md", None, 0, false);
        assert_eq!(c.category, Category::Skip);
    }

    // ── suggest_path tests ────────────────────────────────────

    #[test]
    fn test_suggest_path_project() {
        let path = suggest_path("random/sprint.md", &Category::Project, None);
        assert_eq!(path, "01-Projects/sprint.md");
    }

    #[test]
    fn test_suggest_path_area() {
        let path = suggest_path("misc/health.md", &Category::Area, None);
        assert_eq!(path, "02-Areas/health.md");
    }

    #[test]
    fn test_suggest_path_resource() {
        let path = suggest_path("notes/article.md", &Category::Resource, None);
        assert_eq!(path, "03-Resources/article.md");
    }

    #[test]
    fn test_suggest_path_archive() {
        let path = suggest_path("old/done.md", &Category::Archive, None);
        assert_eq!(path, "04-Archive/done.md");
    }

    #[test]
    fn test_suggest_path_already_correct() {
        let path = suggest_path("01-Projects/sprint.md", &Category::Project, None);
        assert_eq!(path, "01-Projects/sprint.md");
    }

    #[test]
    fn test_suggest_path_skip_unchanged() {
        let path = suggest_path("some/note.md", &Category::Skip, None);
        assert_eq!(path, "some/note.md");
    }

    #[test]
    fn test_suggest_path_uncertain_unchanged() {
        let path = suggest_path("some/note.md", &Category::Uncertain, None);
        assert_eq!(path, "some/note.md");
    }

    #[test]
    fn test_suggest_path_with_profile() {
        use crate::profile::*;
        let profile = VaultProfile {
            vault_path: std::path::PathBuf::from("/test"),
            vault_type: VaultType::Obsidian,
            structure: StructureDetection {
                method: StructureMethod::Para,
                folders: FolderMap {
                    inbox: None,
                    projects: Some("Projects".into()),
                    areas: Some("Areas".into()),
                    resources: Some("Resources".into()),
                    archive: Some("Archive".into()),
                    templates: None,
                    daily: None,
                    people: None,
                },
            },
            stats: VaultStats::default(),
        };
        let path = suggest_path("random/sprint.md", &Category::Project, Some(&profile));
        assert_eq!(path, "Projects/sprint.md");
    }

    #[test]
    fn test_suggest_path_root_file() {
        let path = suggest_path("todo.md", &Category::Project, None);
        assert_eq!(path, "01-Projects/todo.md");
    }

    // ── basename / folder tests ───────────────────────────────

    #[test]
    fn test_basename_simple() {
        assert_eq!(basename("01-Projects/sprint.md"), "sprint.md");
    }

    #[test]
    fn test_basename_root() {
        assert_eq!(basename("note.md"), "note.md");
    }

    #[test]
    fn test_folder_nested() {
        assert_eq!(folder("01-Projects/Work/sprint.md"), "01-Projects/Work");
    }

    #[test]
    fn test_folder_root() {
        assert_eq!(folder("note.md"), "-");
    }

    // ── format_preview_markdown tests ─────────────────────────

    #[test]
    fn test_format_preview_markdown_structure() {
        let preview = MigrationPreview {
            migration_id: "test".into(),
            files: vec![FileClassification {
                path: "todo.md".into(),
                classification: Classification {
                    category: Category::Project,
                    confidence: 0.8,
                    signal: "has tasks".into(),
                    suggested_path: Some("01-Projects/todo.md".into()),
                },
            }],
            uncertain: vec![],
            skipped: 2,
        };
        let md = format_preview_markdown(&preview);
        assert!(md.contains("# PARA Migration Preview"));
        assert!(md.contains("Project (1 files)"));
        assert!(md.contains("todo.md"));
        assert!(md.contains("80%"));
        assert!(md.contains("has tasks"));
        assert!(md.contains("Skipped: 2"));
    }

    #[test]
    fn test_format_preview_markdown_multiple_categories() {
        let preview = MigrationPreview {
            migration_id: "test2".into(),
            files: vec![
                FileClassification {
                    path: "sprint.md".into(),
                    classification: Classification {
                        category: Category::Project,
                        confidence: 0.9,
                        signal: "status active".into(),
                        suggested_path: Some("01-Projects/sprint.md".into()),
                    },
                },
                FileClassification {
                    path: "old/done.md".into(),
                    classification: Classification {
                        category: Category::Archive,
                        confidence: 0.85,
                        signal: "status done".into(),
                        suggested_path: Some("04-Archive/done.md".into()),
                    },
                },
            ],
            uncertain: vec![FileClassification {
                path: "mystery.md".into(),
                classification: Classification {
                    category: Category::Uncertain,
                    confidence: 0.0,
                    signal: "no heuristic rules matched".into(),
                    suggested_path: None,
                },
            }],
            skipped: 5,
        };
        let md = format_preview_markdown(&preview);
        assert!(md.contains("Project (1 files)"));
        assert!(md.contains("Archive (1 files)"));
        assert!(md.contains("Uncertain (1 files)"));
        assert!(md.contains("Files to move: 2"));
        assert!(md.contains("Uncertain: 1"));
        assert!(md.contains("Skipped: 5"));
    }

    #[test]
    fn test_format_preview_markdown_empty() {
        let preview = MigrationPreview {
            migration_id: "empty".into(),
            files: vec![],
            uncertain: vec![],
            skipped: 10,
        };
        let md = format_preview_markdown(&preview);
        assert!(md.contains("# PARA Migration Preview"));
        assert!(md.contains("Files to move: 0"));
        assert!(md.contains("Skipped: 10"));
        // Should NOT contain any category section headers.
        assert!(!md.contains("## Project"));
        assert!(!md.contains("## Uncertain"));
    }

    // ── apply / undo / save+load tests ────────────────────────

    #[test]
    fn test_apply_and_undo_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let store = crate::store::Store::open_memory().unwrap();

        // Create directory structure
        std::fs::create_dir_all(root.join("01-Projects")).unwrap();

        // Create a file at root level
        std::fs::write(root.join("todo.md"), "# Todo\n- [ ] task\n").unwrap();
        store
            .insert_file("todo.md", "hash1", 100, &[], "tod123", None, None)
            .unwrap();

        // Build a preview manually
        let preview = MigrationPreview {
            migration_id: "test-mig-001".into(),
            files: vec![FileClassification {
                path: "todo.md".into(),
                classification: Classification {
                    category: Category::Project,
                    confidence: 0.8,
                    signal: "has tasks".into(),
                    suggested_path: Some("01-Projects/todo.md".into()),
                },
            }],
            uncertain: vec![],
            skipped: 0,
        };

        // Apply
        let result = apply_preview(&preview, &store, &root).unwrap();
        assert_eq!(result.moved, 1);
        assert!(result.errors.is_empty());
        assert!(!root.join("todo.md").exists());
        assert!(root.join("01-Projects/todo.md").exists());

        // Undo
        let undo = undo_last(&store, &root).unwrap();
        assert_eq!(undo.restored, 1);
        assert!(undo.errors.is_empty());
        assert!(root.join("todo.md").exists());
        assert!(!root.join("01-Projects/todo.md").exists());
    }

    #[test]
    fn test_undo_no_migration() {
        let store = crate::store::Store::open_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let result = undo_last(&store, tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_save_and_load_preview() {
        let tmp = tempfile::tempdir().unwrap();
        let preview = MigrationPreview {
            migration_id: "test-001".into(),
            files: vec![],
            uncertain: vec![],
            skipped: 5,
        };
        save_preview(&preview, tmp.path()).unwrap();
        let loaded = load_preview(tmp.path()).unwrap();
        assert_eq!(loaded.migration_id, "test-001");
        assert_eq!(loaded.skipped, 5);
    }
}
