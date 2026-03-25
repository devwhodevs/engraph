use anyhow::Result;

use crate::llm::EmbedModel;
use crate::profile::VaultProfile;
use crate::store::Store;
use crate::writer::split_frontmatter;

#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionInfo {
    pub suggested_folder: String,
    pub actual_folder: String,
}

const ENGRAPH_AGENTS: &[&str] = &["claude-code", "cli", "mcp-server"];

#[derive(Debug, Clone)]
pub struct PlacementResult {
    pub folder: String,
    pub confidence: f64,
    pub strategy: PlacementStrategy,
    pub reason: String,
    /// When strategy is InboxFallback and semantic matching was attempted,
    /// this holds the best-matching folder even though it was below threshold.
    /// Used to inject `suggested_folder` frontmatter for user triage.
    pub suggestion: Option<(String, f64)>, // (folder, confidence)
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlacementStrategy {
    TypeRule,
    SemanticCentroid,
    InboxFallback,
}

pub struct PlacementHints {
    pub type_hint: Option<String>,
    pub tags: Vec<String>,
}

/// Main entry point. Tries 3 strategies in order:
/// 1. Type-based rules
/// 2. Semantic centroid matching
/// 3. Inbox fallback
pub fn place_note(
    content: &str,
    hints: &PlacementHints,
    profile: Option<&VaultProfile>,
    store: &Store,
    embedder: Option<&mut impl EmbedModel>,
) -> Result<PlacementResult> {
    // Strategy A: Type-based rules
    if let Some(result) = try_type_rules(content, hints, profile) {
        return Ok(result);
    }

    // Strategy B: Semantic centroid matching
    let mut semantic_suggestion: Option<(String, f64)> = None;
    if let Some(embedder) = embedder
        && let Some(result) = try_semantic_placement(content, store, embedder)?
    {
        if result.strategy == PlacementStrategy::SemanticCentroid {
            return Ok(result);
        }
        // Below threshold — carry suggestion into inbox fallback
        semantic_suggestion = result.suggestion;
    }

    // Strategy C: Inbox fallback
    let inbox = profile
        .and_then(|p| p.structure.folders.inbox.clone())
        .unwrap_or_else(|| "00-Inbox".to_string());

    Ok(PlacementResult {
        folder: inbox,
        confidence: 0.0,
        strategy: PlacementStrategy::InboxFallback,
        reason: "No confident placement".to_string(),
        suggestion: semantic_suggestion,
    })
}

/// Strategy A: Type-based rules.
/// Maps explicit type hints or content patterns to known folder roles.
fn try_type_rules(
    content: &str,
    hints: &PlacementHints,
    profile: Option<&VaultProfile>,
) -> Option<PlacementResult> {
    let profile = profile?;
    let folders = &profile.structure.folders;

    // Check explicit type hints first
    if let Some(ref type_hint) = hints.type_hint {
        match type_hint.as_str() {
            "person" => {
                let folder = folders.people.clone()?;
                return Some(PlacementResult {
                    folder,
                    confidence: 0.95,
                    strategy: PlacementStrategy::TypeRule,
                    reason: "type_hint: person".to_string(),
                    suggestion: None,
                });
            }
            "daily" => {
                let folder = folders.daily.clone()?;
                return Some(PlacementResult {
                    folder,
                    confidence: 0.95,
                    strategy: PlacementStrategy::TypeRule,
                    reason: "type_hint: daily".to_string(),
                    suggestion: None,
                });
            }
            "workout" => {
                // areas/Health — need areas folder configured
                let areas = folders.areas.as_ref()?;
                let folder = format!("{areas}/Health");
                return Some(PlacementResult {
                    folder,
                    confidence: 0.90,
                    strategy: PlacementStrategy::TypeRule,
                    reason: "type_hint: workout".to_string(),
                    suggestion: None,
                });
            }
            "decision" => {
                // Decision records go to projects folder (or inbox if no projects folder)
                let folder = folders.projects.clone()?;
                return Some(PlacementResult {
                    folder,
                    confidence: 0.90,
                    strategy: PlacementStrategy::TypeRule,
                    reason: "type_hint: decision".to_string(),
                    suggestion: None,
                });
            }
            _ => {}
        }
    }

    // Content-based: person note detection
    // First line is "# First Last" (2-4 words) AND content contains "Role:" or "Company:"
    if let Some(first_line) = content.lines().next()
        && let Some(heading) = first_line.strip_prefix("# ")
    {
        let words: Vec<&str> = heading.split_whitespace().collect();
        if (2..=4).contains(&words.len())
            && (content.contains("Role:") || content.contains("Company:"))
        {
            let folder = folders.people.clone()?;
            return Some(PlacementResult {
                folder,
                confidence: 0.85,
                strategy: PlacementStrategy::TypeRule,
                reason: "content pattern: person note (heading + Role:/Company:)".to_string(),
                suggestion: None,
            });
        }
    }

    // Content-based: ticket/work note detection
    // BRE-XXXX or DRIFT-XXX patterns → projects folder
    if contains_ticket_id(content) {
        let folder = folders.projects.clone()?;
        return Some(PlacementResult {
            folder,
            confidence: 0.80,
            strategy: PlacementStrategy::TypeRule,
            reason: "content pattern: ticket ID detected".to_string(),
            suggestion: None,
        });
    }

    // Content-based: daily/meeting note detection
    // Has a date-like heading and action items (- [ ] checkboxes)
    if looks_like_meeting_note(content) {
        let folder = folders.daily.clone().or_else(|| folders.inbox.clone())?;
        return Some(PlacementResult {
            folder,
            confidence: 0.75,
            strategy: PlacementStrategy::TypeRule,
            reason: "content pattern: date heading + action items".to_string(),
            suggestion: None,
        });
    }

    None
}

/// Check if content contains ticket IDs like BRE-1234 or DRIFT-567.
fn contains_ticket_id(content: &str) -> bool {
    let bytes = content.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Look for uppercase letter sequences followed by -digits
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_uppercase() {
                i += 1;
            }
            let prefix_len = i - start;
            if prefix_len >= 2 && i < bytes.len() && bytes[i] == b'-' {
                i += 1; // skip '-'
                let digit_start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i - digit_start >= 2 {
                    return true; // Found pattern like XX-123
                }
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Check if content looks like a meeting/daily note (date-like heading + action items).
fn looks_like_meeting_note(content: &str) -> bool {
    let has_date_heading = content.lines().any(|l| {
        let t = l.trim();
        // "# 2026-03-25" or "# Meeting 2026-03-25" or "## Action Items"
        (t.starts_with("# ") || t.starts_with("## "))
            && (t.contains("202") || t.contains("action item") || t.contains("Action Item"))
    });
    let has_checkboxes = content.contains("- [ ]") || content.contains("- [x]");
    has_date_heading && has_checkboxes
}

/// Strategy B: Semantic centroid matching.
/// Embeds content and compares against precomputed folder centroids.
fn try_semantic_placement(
    content: &str,
    store: &Store,
    embedder: &mut impl EmbedModel,
) -> Result<Option<PlacementResult>> {
    let centroids = store.get_folder_centroids()?;
    if centroids.is_empty() {
        return Ok(None);
    }

    let embedding = embedder.embed_one(content)?;

    let mut best_folder = String::new();
    let mut best_sim = f64::NEG_INFINITY;

    for (folder, centroid) in &centroids {
        let sim = cosine_similarity(&embedding, centroid);
        if sim > best_sim {
            best_sim = sim;
            best_folder = folder.clone();
        }
    }

    if best_sim > 0.65 {
        Ok(Some(PlacementResult {
            folder: best_folder,
            confidence: best_sim,
            strategy: PlacementStrategy::SemanticCentroid,
            reason: format!("semantic similarity: {best_sim:.3}"),
            suggestion: None,
        }))
    } else if best_sim > 0.0 && !best_folder.is_empty() {
        // Below threshold but we have a candidate — store as suggestion
        // so the inbox fallback can surface it in frontmatter
        Ok(Some(PlacementResult {
            folder: String::new(), // will be overridden by inbox fallback
            confidence: 0.0,
            strategy: PlacementStrategy::InboxFallback,
            reason: "No confident placement".to_string(),
            suggestion: Some((best_folder, best_sim)),
        }))
    } else {
        Ok(None)
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Detect whether a note was placed by engraph and then moved by the user to a different folder.
///
/// Returns `Some(CorrectionInfo)` if the frontmatter contains `suggested_folder` and `created_by`
/// matching a known engraph agent, and the actual folder differs from the suggested one.
/// Returns `None` if there's no correction (confirmed placement, missing fields, or external tool).
pub fn detect_correction_from_frontmatter(
    content: &str,
    actual_folder: &str,
) -> Option<CorrectionInfo> {
    let (fm, _body) = split_frontmatter(content);
    if fm.is_empty() {
        return None;
    }

    let mut suggested_folder: Option<String> = None;
    let mut created_by: Option<String> = None;

    for line in fm.lines() {
        if let Some(val) = line.strip_prefix("suggested_folder:") {
            suggested_folder = Some(val.trim().trim_matches('"').trim_matches('\'').to_string());
        } else if let Some(val) = line.strip_prefix("created_by:") {
            created_by = Some(val.trim().trim_matches('"').trim_matches('\'').to_string());
        }
    }

    // Guard: created_by must exist and match a known engraph agent
    let agent = created_by?;
    if !ENGRAPH_AGENTS.contains(&agent.as_str()) {
        return None;
    }

    let suggested = suggested_folder?;

    // Compare trimming trailing slashes
    let suggested_trimmed = suggested.trim_end_matches('/');
    let actual_trimmed = actual_folder.trim_end_matches('/');

    if suggested_trimmed == actual_trimmed {
        return None; // Confirmation, not correction
    }

    Some(CorrectionInfo {
        suggested_folder: suggested,
        actual_folder: actual_folder.to_string(),
    })
}

/// Strip `suggested_folder:` and `confidence:` lines from frontmatter.
///
/// Guards against block scalars (lines ending with `|` or `>`), returning content unchanged
/// with a warning if detected.
pub fn strip_placement_frontmatter(content: &str) -> String {
    let (fm, body) = split_frontmatter(content);
    if fm.is_empty() {
        return content.to_string();
    }

    // The frontmatter includes --- delimiters. We need to filter lines between them.
    let mut filtered_lines: Vec<&str> = Vec::new();
    let mut any_stripped = false;

    for line in fm.lines() {
        if line.starts_with("suggested_folder:") || line.starts_with("confidence:") {
            // Guard: block scalar indicator
            let trimmed = line.trim_end();
            if trimmed.ends_with('|') || trimmed.ends_with('>') {
                tracing::warn!(
                    "Frontmatter line appears to use block scalar, skipping strip: {}",
                    line
                );
                return content.to_string();
            }
            any_stripped = true;
            continue;
        }
        filtered_lines.push(line);
    }

    if !any_stripped {
        return content.to_string();
    }

    // Reconstruct: filtered frontmatter + body
    let new_fm = filtered_lines.join("\n");
    format!("{}\n{}", new_fm, body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::MockLlm;
    use crate::profile::{
        FolderMap, StructureDetection, StructureMethod, VaultProfile, VaultStats,
    };
    use crate::store::Store;
    use std::path::PathBuf;

    fn make_profile(folders: FolderMap) -> VaultProfile {
        VaultProfile {
            vault_path: PathBuf::from("/test/vault"),
            vault_type: crate::profile::VaultType::Obsidian,
            structure: StructureDetection {
                method: StructureMethod::Para,
                folders,
            },
            stats: VaultStats::default(),
        }
    }

    #[test]
    fn test_inbox_fallback() {
        let store = Store::open_memory().unwrap();
        let hints = PlacementHints {
            type_hint: None,
            tags: vec![],
        };
        let result = place_note(
            "Some random note.",
            &hints,
            None,
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::InboxFallback);
        assert_eq!(result.folder, "00-Inbox");
    }

    #[test]
    fn test_type_rule_person_no_profile() {
        // Without a profile, type rules return None -> falls through to inbox
        let store = Store::open_memory().unwrap();
        let hints = PlacementHints {
            type_hint: Some("person".into()),
            tags: vec![],
        };
        let result = place_note("# John Doe", &hints, None, &store, None::<&mut MockLlm>).unwrap();
        assert_eq!(result.strategy, PlacementStrategy::InboxFallback);
    }

    #[test]
    fn test_type_rule_person_with_profile() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            people: Some("03-Resources/People".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: Some("person".into()),
            tags: vec![],
        };
        let result = place_note(
            "# John Doe",
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::TypeRule);
        assert_eq!(result.folder, "03-Resources/People");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_type_rule_daily() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            daily: Some("07-Daily".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: Some("daily".into()),
            tags: vec![],
        };
        let result = place_note(
            "Today's notes",
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::TypeRule);
        assert_eq!(result.folder, "07-Daily");
    }

    #[test]
    fn test_type_rule_workout() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            areas: Some("02-Areas".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: Some("workout".into()),
            tags: vec![],
        };
        let result = place_note(
            "Leg day workout",
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::TypeRule);
        assert_eq!(result.folder, "02-Areas/Health");
    }

    #[test]
    fn test_content_based_person_detection() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            people: Some("03-Resources/People".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: None,
            tags: vec![],
        };
        let content = "# Jane Smith\nRole: Engineering Manager\nCompany: Acme Corp";
        let result = place_note(
            content,
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::TypeRule);
        assert_eq!(result.folder, "03-Resources/People");
    }

    #[test]
    fn test_content_based_not_person_no_role() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            people: Some("03-Resources/People".into()),
            inbox: Some("00-Inbox".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: None,
            tags: vec![],
        };
        // Heading with 2 words but no Role: or Company:
        let content = "# Jane Smith\nJust some notes about a topic.";
        let result = place_note(
            content,
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::InboxFallback);
    }

    #[test]
    fn test_inbox_fallback_uses_profile_inbox() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            inbox: Some("Inbox".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: None,
            tags: vec![],
        };
        let result = place_note(
            "Random note",
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::InboxFallback);
        assert_eq!(result.folder, "Inbox");
    }

    #[test]
    fn test_ticket_id_detection() {
        assert!(contains_ticket_id("Working on BRE-1234 today"));
        assert!(contains_ticket_id("DRIFT-567 is in progress"));
        assert!(contains_ticket_id("See JIRA-99 for details"));
        assert!(!contains_ticket_id("No ticket here"));
        assert!(!contains_ticket_id("AB-1")); // digits too short
        assert!(!contains_ticket_id("a-1234")); // lowercase prefix
    }

    #[test]
    fn test_meeting_note_detection() {
        let meeting = "# Meeting 2026-03-25\n## Attendees\n- Alice\n## Action Items\n- [ ] Follow up\n- [x] Done";
        assert!(looks_like_meeting_note(meeting));

        let not_meeting = "# Just a heading\nSome notes without checkboxes";
        assert!(!looks_like_meeting_note(not_meeting));

        let only_checkboxes = "- [ ] a task\n- [x] done task";
        assert!(!looks_like_meeting_note(only_checkboxes));
    }

    #[test]
    fn test_decision_type_hint() {
        let store = Store::open_memory().unwrap();
        let folders = FolderMap {
            projects: Some("01-Projects".into()),
            ..FolderMap::default()
        };
        let profile = make_profile(folders);
        let hints = PlacementHints {
            type_hint: Some("decision".into()),
            tags: vec![],
        };
        let result = place_note(
            "Architecture decision",
            &hints,
            Some(&profile),
            &store,
            None::<&mut MockLlm>,
        )
        .unwrap();
        assert_eq!(result.strategy, PlacementStrategy::TypeRule);
        assert_eq!(result.folder, "01-Projects");
        assert!(result.confidence >= 0.90);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── Correction detection tests ──────────────────────────────

    #[test]
    fn test_detect_correction() {
        let content = "---\ntags:\n  - test\ncreated_by: claude-code\nsuggested_folder: 00-Inbox/\nconfidence: 0.45\n---\n\n# Note\n";
        let result = detect_correction_from_frontmatter(content, "01-Projects/");
        assert!(result.is_some());
        let info = result.unwrap();
        assert_eq!(info.suggested_folder, "00-Inbox/");
        assert_eq!(info.actual_folder, "01-Projects/");
    }

    #[test]
    fn test_no_correction_when_confirmed() {
        let content = "---\nsuggested_folder: 01-Projects/\ncreated_by: cli\n---\n\n# Note\n";
        assert!(detect_correction_from_frontmatter(content, "01-Projects/").is_none());
    }

    #[test]
    fn test_no_correction_without_created_by() {
        let content = "---\nsuggested_folder: 00-Inbox/\n---\n\n# Note\n";
        assert!(detect_correction_from_frontmatter(content, "01-Projects/").is_none());
    }

    #[test]
    fn test_no_correction_external_tool() {
        let content =
            "---\nsuggested_folder: 00-Inbox/\ncreated_by: some-other-tool\n---\n\n# Note\n";
        assert!(detect_correction_from_frontmatter(content, "01-Projects/").is_none());
    }

    #[test]
    fn test_strip_placement_frontmatter() {
        let content = "---\ntags:\n  - test\nsuggested_folder: 00-Inbox/\nconfidence: 0.45\ncreated_by: cli\n---\n\n# Note\nContent.";
        let result = strip_placement_frontmatter(content);
        assert!(!result.contains("suggested_folder"));
        assert!(!result.contains("confidence"));
        assert!(result.contains("tags:"));
        assert!(result.contains("created_by: cli"));
        assert!(result.contains("# Note"));
    }
}
