//! Heuristic PARA classification engine for vault migration.
//!
//! Classifies notes into PARA categories (Project, Area, Resource, Archive)
//! using priority-ordered heuristic rules. No LLM required.

use serde::{Deserialize, Serialize};

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
        "health", "finance", "career", "learning", "fitness", "nutrition", "budget",
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
        let c = classify_heuristic(
            "# Template\n",
            "05-Templates/Daily Note.md",
            None,
            0,
            false,
        );
        assert_eq!(c.category, Category::Skip);
    }
}
