use anyhow::Result;
use std::path::Path;

use crate::indexer::extract_aliases_from_frontmatter;
use crate::store::Store;

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
    name: String,
    name_lower: String,
    path: String,
    match_type: LinkMatchType,
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

/// Discover potential wikilink targets in content by matching note names and aliases.
///
/// Builds a name index from the store, then scans content for case-insensitive
/// matches that aren't inside existing wikilinks and don't overlap with longer
/// already-matched names.
pub fn discover_links(
    store: &Store,
    content: &str,
    vault_path: &Path,
) -> Result<Vec<DiscoveredLink>> {
    let name_index = build_name_index(store, vault_path)?;
    let wikilink_regions = find_wikilink_regions(content);
    let content_lower = content.to_lowercase();
    let content_bytes = content.as_bytes();

    let mut links = Vec::new();
    let mut claimed: Vec<(usize, usize)> = Vec::new();

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
        }
    }

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
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].matched_text, "Steve Barbera");
        assert_eq!(links[0].match_type, LinkMatchType::ExactName);
    }

    #[test]
    fn test_skip_existing_wikilinks() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "Talked to [[Steve Barbera]]";
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
        assert_eq!(links.len(), 0);
    }

    #[test]
    fn test_multiple_matches() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "Steve Barbera explained Reciprocal Rank Fusion";
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
        assert_eq!(links.len(), 2);

        let names: Vec<&str> = links.iter().map(|l| l.matched_text.as_str()).collect();
        assert!(names.contains(&"Steve Barbera"));
        assert!(names.contains(&"Reciprocal Rank Fusion"));
    }

    #[test]
    fn test_alias_match() {
        let (store, vault_dir) = setup_store_and_vault();
        let content = "We use RRF for search";
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
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
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
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
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].matched_text, "steve barbera");
    }

    #[test]
    fn test_word_boundary_check() {
        let (store, vault_dir) = setup_store_and_vault();
        // "RRF" embedded inside a word should not match
        let content = "The xRRFy algorithm";
        let links = discover_links(&store, content, vault_dir.path()).unwrap();
        assert_eq!(links.len(), 0);
    }
}
