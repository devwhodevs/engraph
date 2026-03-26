use anyhow::Result;
use std::path::Path;

use crate::indexer::extract_aliases_from_frontmatter;
use crate::store::Store;
use strsim::normalized_levenshtein;

/// A potential wikilink discovered in note content.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredLink {
    pub matched_text: String,
    pub target_path: String,
    pub display: Option<String>,
    pub match_type: LinkMatchType,
}

/// How a link target was matched.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkMatchType {
    /// Matched note filename (basename without .md)
    ExactName,
    /// Matched an alias from frontmatter
    Alias,
    /// Fuzzy match on note name (0-1000 basis points confidence)
    FuzzyName { confidence_bp: u16 },
    /// First-name match for people notes (suggestion-only, 0-1000 basis points)
    FirstName { confidence_bp: u16 },
}

impl LinkMatchType {
    /// Priority for overlap resolution (lower = higher priority).
    pub fn priority(&self) -> u8 {
        match self {
            Self::ExactName => 0,
            Self::Alias => 1,
            Self::FuzzyName { .. } => 2,
            Self::FirstName { .. } => 3,
        }
    }
}

/// An entry in the name-to-path lookup table.
#[derive(Debug, Clone)]
pub(crate) struct NameEntry {
    pub(crate) name: String,
    pub(crate) name_lower: String,
    pub(crate) path: String,
    pub(crate) match_type: LinkMatchType,
}

/// Build a lookup table of (name, path, match_type) from all indexed files.
///
/// For each file: extract basename (without .md) as ExactName (if len >= 3),
/// then read the file from disk to extract aliases (each len >= 2) as Alias entries.
/// Results are sorted by name length descending so longer names match first.
pub(crate) fn build_name_index(store: &Store, vault_path: &Path) -> Result<Vec<NameEntry>> {
    let all_files = store.get_all_files()?;
    let mut entries = Vec::new();

    for file in &all_files {
        // Extract basename without .md
        let basename = file
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&file.path)
            .trim_end_matches(".md");

        if basename.len() >= 3 {
            entries.push(NameEntry {
                name: basename.to_string(),
                name_lower: basename.to_lowercase(),
                path: file.path.clone(),
                match_type: LinkMatchType::ExactName,
            });
        }

        // Read file from disk to extract aliases
        let full_path = vault_path.join(&file.path);
        if let Ok(content) = std::fs::read_to_string(&full_path)
            && let Some(aliases) = extract_aliases_from_frontmatter(&content)
        {
            for alias in aliases {
                if alias.len() >= 2 {
                    let alias_lower = alias.to_lowercase();
                    entries.push(NameEntry {
                        name: alias,
                        name_lower: alias_lower,
                        path: file.path.clone(),
                        match_type: LinkMatchType::Alias,
                    });
                }
            }
        }
    }

    // Sort by name length descending — match longer names first
    entries.sort_by(|a, b| b.name.len().cmp(&a.name.len()));
    Ok(entries)
}

/// Find byte ranges of existing `[[...]]` wikilinks in content.
///
/// Returns `(start, end)` pairs where start is the index of the first `[`
/// and end is one past the last `]`.
pub fn find_wikilink_regions(content: &str) -> Vec<(usize, usize)> {
    let bytes = content.as_bytes();
    let mut regions = Vec::new();
    let mut i = 0;

    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            // Find the closing ]]
            let start = i;
            let mut j = i + 2;
            while j + 1 < bytes.len() {
                if bytes[j] == b']' && bytes[j + 1] == b']' {
                    regions.push((start, j + 2));
                    i = j + 2;
                    break;
                }
                j += 1;
            }
            if j + 1 >= bytes.len() {
                // No closing ]] found
                i += 2;
            }
        } else {
            i += 1;
        }
    }

    regions
}

/// Check if a byte position falls inside any of the given regions.
fn inside_region(pos: usize, end: usize, regions: &[(usize, usize)]) -> bool {
    regions.iter().any(|&(rs, re)| pos >= rs && end <= re)
}

/// Check if a match position overlaps with any already-claimed range.
fn overlaps_claimed(pos: usize, end: usize, claimed: &[(usize, usize)]) -> bool {
    claimed.iter().any(|&(cs, ce)| pos < ce && end > cs)
}

/// Check word boundary at a byte position in content.
fn is_word_boundary(content: &[u8], pos: usize) -> bool {
    if pos == 0 {
        return true;
    }
    let ch = content[pos - 1];
    // Word boundary: previous char is not alphanumeric or underscore
    !ch.is_ascii_alphanumeric() && ch != b'_'
}

/// Check word boundary after a match ends.
fn is_word_boundary_after(content: &[u8], end: usize) -> bool {
    if end >= content.len() {
        return true;
    }
    let ch = content[end];
    !ch.is_ascii_alphanumeric() && ch != b'_'
}

/// Returns (start_byte, end_byte, cleaned_word) for each word in the text.
///
/// Splits on non-alphanumeric boundaries (except apostrophes within words).
/// Strips trailing possessives (`'s`) from the cleaned word for matching.
pub(crate) fn word_spans(text: &str) -> Vec<(usize, usize, String)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        // Skip non-word characters
        if !bytes[i].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }

        let start = i;
        // Consume word characters (alphanumeric + apostrophes within words)
        while i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric()
                || (bytes[i] == b'\''
                    && i + 1 < bytes.len()
                    && bytes[i + 1].is_ascii_alphanumeric()))
        {
            i += 1;
        }
        let end = i;

        let mut word = text[start..end].to_string();
        // Strip trailing possessive 's
        if word.ends_with("'s") || word.ends_with("'S") {
            word.truncate(word.len() - 2);
        }

        if !word.is_empty() {
            spans.push((start, end, word));
        }
    }

    spans
}

/// Find fuzzy matches for eligible names in content using a sliding window.
///
/// Uses normalized Levenshtein distance with a 0.92 (920bp) threshold.
/// Skips single-word names unless they come from the People folder.
/// Skips windows overlapping with `existing_regions` (from exact matches).
/// Only matches once per name per content.
pub(crate) fn find_fuzzy_matches(
    content: &str,
    eligible_names: &[NameEntry],
    existing_regions: &[(usize, usize)],
    people_folder: Option<&str>,
) -> Vec<DiscoveredLink> {
    let spans = word_spans(content);
    let mut results = Vec::new();

    for entry in eligible_names {
        let word_count = entry.name.split_whitespace().count();

        // Skip single-word names not from People folder
        if word_count <= 1 {
            let is_people = people_folder
                .map(|pf| entry.path.starts_with(pf))
                .unwrap_or(false);
            if !is_people {
                continue;
            }
            // People single-word names must be >= 3 chars (already enforced by build_name_index)
        }

        if spans.len() < word_count {
            continue;
        }

        let mut matched = false;
        for win_start in 0..=(spans.len() - word_count) {
            if matched {
                break;
            }

            let win_end_idx = win_start + word_count - 1;
            let byte_start = spans[win_start].0;
            let byte_end = spans[win_end_idx].1;

            // Skip if overlapping with existing regions
            if overlaps_claimed(byte_start, byte_end, existing_regions) {
                continue;
            }

            // Join cleaned words for comparison
            let window_text: String = spans[win_start..=win_end_idx]
                .iter()
                .map(|(_, _, w)| w.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");

            let sim = normalized_levenshtein(&window_text, &entry.name_lower);
            let confidence_bp = (sim * 1000.0) as u16;

            if confidence_bp >= 920 {
                // Use actual content bytes for matched_text
                let matched_text = content[byte_start..byte_end].to_string();

                results.push(DiscoveredLink {
                    matched_text,
                    target_path: entry.path.clone(),
                    display: Some(content[byte_start..byte_end].to_string()),
                    match_type: LinkMatchType::FuzzyName { confidence_bp },
                });
                matched = true;
            }
        }
    }

    results
}

const FIRST_NAME_CONFIDENCE_BP: u16 = 650;

/// Find first-name matches for People notes in content.
///
/// For each word in content, checks if it uniquely matches the first name of exactly
/// one People note. Ambiguous matches (0 or 2+ people sharing the same first name)
/// are skipped. Matches overlapping `existing_regions` are also skipped.
pub(crate) fn find_first_name_matches(
    content: &str,
    people_names: &[NameEntry],
    existing_regions: &[(usize, usize)],
) -> Vec<DiscoveredLink> {
    let spans = word_spans(content);
    let mut results = Vec::new();

    for &(start, end, ref word) in &spans {
        // Skip if overlapping with existing regions
        if overlaps_claimed(start, end, existing_regions) {
            continue;
        }

        let word_lower = word.to_lowercase();

        // Find all people whose name_lower starts with this word followed by a space
        let matching: Vec<&NameEntry> = people_names
            .iter()
            .filter(|e| {
                e.name_lower.starts_with(&word_lower)
                    && e.name_lower.len() > word_lower.len()
                    && e.name_lower.as_bytes()[word_lower.len()] == b' '
            })
            .collect();

        // Exactly one match → emit
        if matching.len() == 1 {
            let entry = matching[0];
            results.push(DiscoveredLink {
                matched_text: content[start..end].to_string(),
                target_path: entry.path.clone(),
                display: Some(content[start..end].to_string()),
                match_type: LinkMatchType::FirstName {
                    confidence_bp: FIRST_NAME_CONFIDENCE_BP,
                },
            });
        }
    }

    results
}

/// Discover potential wikilink targets in content by matching note names and aliases.
///
/// Builds a name index from the store, then scans content for case-insensitive
/// matches that aren't inside existing wikilinks and don't overlap with longer
/// already-matched names. After exact matching, runs fuzzy matching for eligible
/// names that weren't already matched.
pub fn discover_links(
    store: &Store,
    content: &str,
    vault_path: &Path,
    people_folder: Option<&str>,
) -> Result<Vec<DiscoveredLink>> {
    let name_index = build_name_index(store, vault_path)?;
    let wikilink_regions = find_wikilink_regions(content);
    let content_lower = content.to_lowercase();
    let content_bytes = content.as_bytes();

    let mut links = Vec::new();
    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut exact_matched_paths: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for entry in &name_index {
        let needle = &entry.name_lower;
        let mut search_from = 0;

        while let Some(rel_pos) = content_lower[search_from..].find(needle.as_str()) {
            let pos = search_from + rel_pos;
            let end = pos + needle.len();
            search_from = end;

            // Skip if inside existing wikilink
            if inside_region(pos, end, &wikilink_regions) {
                continue;
            }

            // Skip if overlapping with an already-claimed (longer) match
            if overlaps_claimed(pos, end, &claimed) {
                continue;
            }

            // Check word boundaries
            if !is_word_boundary(content_bytes, pos) || !is_word_boundary_after(content_bytes, end)
            {
                continue;
            }

            let matched_text = content[pos..end].to_string();

            let display = match entry.match_type {
                LinkMatchType::Alias
                | LinkMatchType::FuzzyName { .. }
                | LinkMatchType::FirstName { .. } => Some(matched_text.clone()),
                LinkMatchType::ExactName => None,
            };

            links.push(DiscoveredLink {
                matched_text,
                target_path: entry.path.clone(),
                display,
                match_type: entry.match_type.clone(),
            });

            claimed.push((pos, end));
            exact_matched_paths.insert(entry.path.clone());
        }
    }

    // --- Fuzzy matching phase ---
    // Build eligible names: multi-word names, or People folder notes >= 3 chars
    let eligible: Vec<NameEntry> = name_index
        .iter()
        .filter(|e| !exact_matched_paths.contains(&e.path))
        .filter(|e| {
            let word_count = e.name.split_whitespace().count();
            if word_count >= 2 {
                return true;
            }
            // Single-word: only if from People folder and >= 3 chars
            if let Some(pf) = people_folder {
                e.path.starts_with(pf) && e.name.len() >= 3
            } else {
                false
            }
        })
        .cloned()
        .collect();

    // Combine exact match regions and wikilink regions for fuzzy exclusion
    let mut fuzzy_excluded = claimed.clone();
    fuzzy_excluded.extend_from_slice(&wikilink_regions);
    let fuzzy_matches = find_fuzzy_matches(content, &eligible, &fuzzy_excluded, people_folder);

    // Track fuzzy match regions for first-name exclusion
    let mut first_name_excluded = fuzzy_excluded.clone();
    for fm in &fuzzy_matches {
        // Find position of the fuzzy match in content to exclude from first-name matching
        let needle = fm.matched_text.to_lowercase();
        let content_lower_bytes = content.to_lowercase();
        if let Some(pos) = content_lower_bytes.find(&needle) {
            first_name_excluded.push((pos, pos + needle.len()));
        }
    }
    links.extend(fuzzy_matches);

    // --- First-name matching phase ---
    // Filter people names from the name index
    if let Some(pf) = people_folder {
        let people_names: Vec<NameEntry> = name_index
            .iter()
            .filter(|e| e.path.starts_with(pf) && matches!(e.match_type, LinkMatchType::ExactName))
            .filter(|e| !exact_matched_paths.contains(&e.path))
            .cloned()
            .collect();

        let first_name_matches =
            find_first_name_matches(content, &people_names, &first_name_excluded);
        links.extend(first_name_matches);
    }

    // Sort: exact matches first (by priority), then by confidence descending
    links.sort_by(|a, b| {
        let pa = a.match_type.priority();
        let pb = b.match_type.priority();
        if pa != pb {
            return pa.cmp(&pb);
        }
        // For same priority, sort by confidence descending
        let ca = match &a.match_type {
            LinkMatchType::FuzzyName { confidence_bp } => *confidence_bp,
            LinkMatchType::FirstName { confidence_bp } => *confidence_bp,
            _ => 1000,
        };
        let cb = match &b.match_type {
            LinkMatchType::FuzzyName { confidence_bp } => *confidence_bp,
            LinkMatchType::FirstName { confidence_bp } => *confidence_bp,
            _ => 1000,
        };
        cb.cmp(&ca)
    });

    Ok(links)
}

/// Apply discovered links to content, replacing matched text with `[[wikilinks]]`.
///
/// For exact name matches: `[[TargetName]]`
/// For alias matches: `[[TargetName|DisplayText]]`
///
/// Replacements are applied from end to start to preserve byte positions.
pub fn apply_links(content: &str, links: &[DiscoveredLink]) -> String {
    if links.is_empty() {
        return content.to_string();
    }

    let content_lower = content.to_lowercase();
    let content_bytes = content.as_bytes();
    let wikilink_regions = find_wikilink_regions(content);

    // Find the position of each link in the content
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    let mut claimed: Vec<(usize, usize)> = Vec::new();

    for link in links {
        let needle = link.matched_text.to_lowercase();
        let mut search_from = 0;

        while let Some(rel_pos) = content_lower[search_from..].find(needle.as_str()) {
            let pos = search_from + rel_pos;
            let end = pos + needle.len();
            search_from = end;

            if inside_region(pos, end, &wikilink_regions) {
                continue;
            }
            if overlaps_claimed(pos, end, &claimed) {
                continue;
            }
            if !is_word_boundary(content_bytes, pos) || !is_word_boundary_after(content_bytes, end)
            {
                continue;
            }

            let target_name = link
                .target_path
                .rsplit('/')
                .next()
                .unwrap_or(&link.target_path)
                .trim_end_matches(".md");

            let replacement = match &link.display {
                Some(display) => format!("[[{}|{}]]", target_name, display),
                None => format!("[[{}]]", target_name),
            };

            replacements.push((pos, end, replacement));
            claimed.push((pos, end));
            break; // Only replace first occurrence per link
        }
    }

    // Sort by position descending so we can replace from end to start
    replacements.sort_by(|a, b| b.0.cmp(&a.0));

    let mut result = content.to_string();
    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn setup_store_and_vault() -> (Store, tempfile::TempDir) {
        let vault_dir = tempfile::TempDir::new().unwrap();
        let store = Store::open_memory().unwrap();

        // Insert files into store
        store
            .insert_file(
                "03-Resources/People/Steve Barbera.md",
                "h1",
                0,
                &[],
                "aaa111",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "03-Resources/Code-Snippets/Reciprocal Rank Fusion.md",
                "h2",
                0,
                &[],
                "bbb222",
                None,
                None,
            )
            .unwrap();

        // Create files on disk for alias reading
        let people = vault_dir.path().join("03-Resources/People");
        std::fs::create_dir_all(&people).unwrap();
        std::fs::write(people.join("Steve Barbera.md"), "# Steve Barbera\n").unwrap();

        let snippets = vault_dir.path().join("03-Resources/Code-Snippets");
        std::fs::create_dir_all(&snippets).unwrap();
        std::fs::write(
            snippets.join("Reciprocal Rank Fusion.md"),
            "---\naliases: [RRF]\n---\n# Reciprocal Rank Fusion\n",
        )
        .unwrap();

        (store, vault_dir)
    }

    #[test]
    fn test_exact_name_match() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "Talked to Steve Barbera";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].matched_text, "Steve Barbera");
        assert_eq!(links[0].match_type, LinkMatchType::ExactName);
    }

    #[test]
    fn test_skip_existing_wikilinks() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "Talked to [[Steve Barbera]]";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        assert_eq!(links.len(), 0);
    }

    #[test]
    fn test_multiple_matches() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "Steve Barbera explained Reciprocal Rank Fusion";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        assert_eq!(links.len(), 2);

        let names: Vec<&str> = links.iter().map(|l| l.matched_text.as_str()).collect();
        assert!(names.contains(&"Steve Barbera"));
        assert!(names.contains(&"Reciprocal Rank Fusion"));
    }

    #[test]
    fn test_alias_match() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "We use RRF for search";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].match_type, LinkMatchType::Alias);
        assert_eq!(
            links[0].target_path,
            "03-Resources/Code-Snippets/Reciprocal Rank Fusion.md"
        );
        assert_eq!(links[0].display, Some("RRF".to_string()));
    }

    #[test]
    fn test_apply_links() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "Steve Barbera explained RRF to me";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        let result = apply_links(content, &links);

        assert!(result.contains("[[Steve Barbera]]"));
        assert!(result.contains("[[Reciprocal Rank Fusion|RRF]]"));
    }

    #[test]
    fn test_find_wikilink_regions() {
        let content = "Hello [[World]] and [[Foo|Bar]]";
        let regions = find_wikilink_regions(content);
        assert_eq!(regions.len(), 2);
        assert_eq!(&content[regions[0].0..regions[0].1], "[[World]]");
        assert_eq!(&content[regions[1].0..regions[1].1], "[[Foo|Bar]]");
    }

    #[test]
    fn test_case_insensitive_match() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "talked to steve barbera today";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].matched_text, "steve barbera");
    }

    #[test]
    fn test_word_boundary_check() {
        let (store, vault_dir) = setup_store_and_vault();
        // "RRF" embedded inside a word should not match
        let content = "The xRRFy algorithm";
        let links = discover_links(&store, content, vault_dir.path(), None).unwrap();
        assert_eq!(links.len(), 0);
    }

    // --- word_spans tests ---

    #[test]
    fn test_word_spans_basic() {
        let spans = word_spans("hello world");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], (0, 5, "hello".to_string()));
        assert_eq!(spans[1], (6, 11, "world".to_string()));
    }

    #[test]
    fn test_word_spans_possessive() {
        let spans = word_spans("Steve's book");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].2, "Steve");
        assert_eq!(spans[1].2, "book");
    }

    #[test]
    fn test_word_spans_preserves_byte_positions() {
        let text = "I met Steeve Barbera yesterday";
        let spans = word_spans(text);
        assert_eq!(spans.len(), 5);
        // spans: I(0,1), met(2,5), Steeve(6,12), Barbera(13,20), yesterday(21,30)
        assert_eq!(spans[2].0, 6);
        assert_eq!(spans[2].1, 12);
        assert_eq!(&text[spans[2].0..spans[2].1], "Steeve");
    }

    // --- find_fuzzy_matches tests ---

    #[test]
    fn test_fuzzy_match_typo() {
        let _spans = word_spans("I met Steeve Barbera yesterday");
        let entries = vec![NameEntry {
            name: "Steve Barbera".into(),
            name_lower: "steve barbera".into(),
            path: "People/Steve Barbera.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        let matches = find_fuzzy_matches("I met Steeve Barbera yesterday", &entries, &[], None);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].target_path, "People/Steve Barbera.md");
        assert!(
            matches!(matches[0].match_type, LinkMatchType::FuzzyName { confidence_bp } if confidence_bp >= 920)
        );
    }

    #[test]
    fn test_fuzzy_no_match_below_threshold() {
        let entries = vec![NameEntry {
            name: "Steve Barbera".into(),
            name_lower: "steve barbera".into(),
            path: "People/Steve Barbera.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        let matches = find_fuzzy_matches("Steven Rogers was there", &entries, &[], None);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_fuzzy_skips_single_word_non_people() {
        let entries = vec![NameEntry {
            name: "Rust".into(),
            name_lower: "rust".into(),
            path: "Resources/Rust.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        let matches = find_fuzzy_matches("I love Rust programming", &entries, &[], None);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_fuzzy_skips_claimed_regions() {
        let entries = vec![NameEntry {
            name: "Steve Barbera".into(),
            name_lower: "steve barbera".into(),
            path: "People/Steve Barbera.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        // Claim the region where "Steve Barbera" would match
        let matches =
            find_fuzzy_matches("I met Steve Barbera yesterday", &entries, &[(6, 20)], None);
        assert_eq!(matches.len(), 0);
    }

    // --- find_first_name_matches tests ---

    #[test]
    fn test_first_name_unique_match() {
        let people = vec![NameEntry {
            name: "Steve Barbera".into(),
            name_lower: "steve barbera".into(),
            path: "People/Steve Barbera.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        let matches = find_first_name_matches("I talked to Steve about it.", &people, &[]);
        assert_eq!(matches.len(), 1);
        assert!(matches!(
            matches[0].match_type,
            LinkMatchType::FirstName { confidence_bp: 650 }
        ));
        assert_eq!(matches[0].display, Some("Steve".to_string()));
    }

    #[test]
    fn test_first_name_ambiguous() {
        let people = vec![
            NameEntry {
                name: "Steve Barbera".into(),
                name_lower: "steve barbera".into(),
                path: "People/Steve Barbera.md".into(),
                match_type: LinkMatchType::ExactName,
            },
            NameEntry {
                name: "Steve Rogers".into(),
                name_lower: "steve rogers".into(),
                path: "People/Steve Rogers.md".into(),
                match_type: LinkMatchType::ExactName,
            },
        ];
        let matches = find_first_name_matches("I talked to Steve about it.", &people, &[]);
        assert_eq!(matches.len(), 0); // ambiguous — no match
    }

    #[test]
    fn test_first_name_skips_existing_regions() {
        let people = vec![NameEntry {
            name: "Steve Barbera".into(),
            name_lower: "steve barbera".into(),
            path: "People/Steve Barbera.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        // "Steve" is at position 12..17
        let matches = find_first_name_matches("I talked to Steve about it.", &people, &[(12, 17)]);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_first_name_no_match_exact_name_only() {
        // A person with only a first name (no space) should not match
        let people = vec![NameEntry {
            name: "Steve".into(),
            name_lower: "steve".into(),
            path: "People/Steve.md".into(),
            match_type: LinkMatchType::ExactName,
        }];
        let matches = find_first_name_matches("I talked to Steve about it.", &people, &[]);
        assert_eq!(matches.len(), 0); // "steve" doesn't start with "steve " (no space after)
    }
}
