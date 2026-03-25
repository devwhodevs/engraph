use anyhow::Result;

use crate::embedder::Embedder;
use crate::profile::VaultProfile;
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct PlacementResult {
    pub folder: String,
    pub confidence: f64,
    pub strategy: PlacementStrategy,
    pub reason: String,
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
    embedder: Option<&mut Embedder>,
) -> Result<PlacementResult> {
    // Strategy A: Type-based rules
    if let Some(result) = try_type_rules(content, hints, profile) {
        return Ok(result);
    }

    // Strategy B: Semantic centroid matching
    if let Some(embedder) = embedder {
        if let Some(result) = try_semantic_placement(content, store, embedder)? {
            return Ok(result);
        }
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
                });
            }
            "daily" => {
                let folder = folders.daily.clone()?;
                return Some(PlacementResult {
                    folder,
                    confidence: 0.95,
                    strategy: PlacementStrategy::TypeRule,
                    reason: "type_hint: daily".to_string(),
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
                });
            }
            _ => {}
        }
    }

    // Content-based: person note detection
    // First line is "# First Last" (2-4 words) AND content contains "Role:" or "Company:"
    if let Some(first_line) = content.lines().next() {
        if let Some(heading) = first_line.strip_prefix("# ") {
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
                });
            }
        }
    }

    None
}

/// Strategy B: Semantic centroid matching.
/// Embeds content and compares against precomputed folder centroids.
fn try_semantic_placement(
    content: &str,
    store: &Store,
    embedder: &mut Embedder,
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
        }))
    } else {
        Ok(None)
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let result = place_note("Some random note.", &hints, None, &store, None).unwrap();
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
        let result = place_note("# John Doe", &hints, None, &store, None).unwrap();
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
        let result = place_note("# John Doe", &hints, Some(&profile), &store, None).unwrap();
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
        let result = place_note("Today's notes", &hints, Some(&profile), &store, None).unwrap();
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
        let result =
            place_note("Leg day workout", &hints, Some(&profile), &store, None).unwrap();
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
        let result = place_note(content, &hints, Some(&profile), &store, None).unwrap();
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
        let result = place_note(content, &hints, Some(&profile), &store, None).unwrap();
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
        let result =
            place_note("Random note", &hints, Some(&profile), &store, None).unwrap();
        assert_eq!(result.strategy, PlacementStrategy::InboxFallback);
        assert_eq!(result.folder, "Inbox");
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
}
