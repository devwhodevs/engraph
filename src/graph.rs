use std::collections::{HashMap, HashSet, hash_map::Entry};

use anyhow::Result;

use crate::fusion::RankedResult;
use crate::store::Store;

/// Extract unique wikilink targets from text.
/// Handles [[Target]], [[Target|Display]], [[Target#Heading]].
/// Skips embeds (![[...]]).
pub fn extract_wikilink_targets(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    let mut i = 0;

    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            // Check for embed prefix (! before [[)
            let is_embed = i > 0 && bytes[i - 1] == b'!';
            if let Some(rest) = text.get(i + 2..)
                && let Some(close) = rest.find("]]")
            {
                let inner = &rest[..close];
                if !is_embed && !inner.is_empty() && !inner.contains('\n') {
                    // Strip heading: [[Note#Section]] → "Note"
                    let target = inner.split('#').next().unwrap_or(inner);
                    // Strip display: [[Note|Display]] → "Note"
                    let target = target.split('|').next().unwrap_or(target);
                    let target = target.trim().to_string();
                    if !target.is_empty() && seen.insert(target.clone()) {
                        targets.push(target);
                    }
                }
                i += 2 + close + 2;
                continue;
            }
        }
        i += 1;
    }
    targets
}

/// Extract query terms for relevance filtering.
/// Splits on whitespace, lowercases, drops terms shorter than 3 chars.
pub fn extract_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 3)
        .collect()
}

/// Expand search results by following graph connections.
/// Seeds are the top results from semantic + FTS lanes.
/// Returns expanded results suitable for RRF fusion.
pub fn graph_expand(
    store: &Store,
    seeds: &[RankedResult],
    query: &str,
    max_hops: usize,
    max_expansions: usize,
) -> Result<Vec<RankedResult>> {
    let query_terms = extract_query_terms(query);
    let seed_ids: HashSet<i64> = seeds.iter().map(|s| s.file_id).collect();

    // Track best score per expanded file (multi-parent merge: take highest)
    // (file_id) → (best_score, hop_depth, seed_file_path)
    let mut expansions: HashMap<i64, (f64, usize, String)> = HashMap::new();

    for seed in seeds {
        let neighbors = store.get_neighbors(seed.file_id, max_hops)?;

        for (neighbor_id, hop) in neighbors {
            if seed_ids.contains(&neighbor_id) {
                continue;
            }

            let decay = match hop {
                1 => 0.8,
                2 => 0.5,
                _ => 0.3,
            };
            let mut expansion_score = seed.score * decay;

            // Relevance filter: must match a query term via FTS or share tags
            let term_match = query_terms
                .iter()
                .any(|t| store.file_contains_term(neighbor_id, t).unwrap_or(false));

            if !term_match {
                let shared = store
                    .get_shared_tags_files(neighbor_id, 100)
                    .unwrap_or_default();
                if shared.contains(&seed.file_id) {
                    expansion_score *= 0.7;
                } else {
                    continue; // tangential — skip
                }
            }

            // Multi-parent merge: keep highest score
            match expansions.entry(neighbor_id) {
                Entry::Occupied(mut e) => {
                    if expansion_score > e.get().0 {
                        e.insert((expansion_score, hop, seed.file_path.clone()));
                    }
                }
                Entry::Vacant(e) => {
                    e.insert((expansion_score, hop, seed.file_path.clone()));
                }
            }
        }
    }

    // Sort by score descending, cap at max_expansions
    let mut results: Vec<(i64, f64, usize, String)> = expansions
        .into_iter()
        .map(|(fid, (score, hop, seed))| (fid, score, hop, seed))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(max_expansions);

    // Convert to RankedResult
    let mut ranked = Vec::new();
    for (file_id, score, _hop, _seed) in results {
        let file = store.get_file_by_id(file_id)?;
        let (file_path, docid) = match file {
            Some(f) => (f.path, f.docid),
            None => continue,
        };
        let (heading, snippet) = store
            .get_best_chunk_for_file(file_id)?
            .unwrap_or_else(|| (String::new(), String::new()));
        let heading = if heading.is_empty() {
            None
        } else {
            Some(heading)
        };

        ranked.push(RankedResult {
            file_path,
            file_id,
            score,
            heading,
            snippet,
            docid,
        });
    }

    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::fusion::RankedResult;
    use crate::store::Store;

    #[test]
    fn test_extract_wikilink_targets() {
        let text =
            "See [[Note One]] and [[Note Two|display]] for details. Also [[Note One]] again.";
        let targets = extract_wikilink_targets(text);
        assert!(targets.contains(&"Note One".to_string()));
        assert!(targets.contains(&"Note Two".to_string()));
        assert_eq!(targets.len(), 2); // deduplicated
    }

    #[test]
    fn test_extract_wikilinks_with_headings() {
        let text = "Link to [[Note#Section]] here.";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Note"]);
    }

    #[test]
    fn test_extract_wikilinks_empty() {
        assert!(extract_wikilink_targets("no links here").is_empty());
        assert!(extract_wikilink_targets("").is_empty());
    }

    #[test]
    fn test_extract_wikilinks_skip_embeds() {
        let text = "![[embedded image.png]] and [[real link]]";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["real link"]);
    }

    #[test]
    fn test_extract_wikilinks_heading_and_display() {
        let text = "[[Note#Section|Custom Display]]";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Note"]); // strip both heading and display
    }

    #[test]
    fn test_extract_query_terms() {
        let terms = extract_query_terms("BRE-2579 delivery date");
        assert_eq!(terms, vec!["bre-2579", "delivery", "date"]);
    }

    #[test]
    fn test_extract_query_terms_short_words_dropped() {
        let terms = extract_query_terms("a is the big query");
        assert_eq!(terms, vec!["the", "big", "query"]);
    }

    #[test]
    fn test_graph_expand_basic() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "seed.md",
                "h1",
                100,
                &["rust".into()],
                &generate_docid("seed.md"),
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "linked.md",
                "h2",
                100,
                &["rust".into()],
                &generate_docid("linked.md"),
                None,
            )
            .unwrap();
        let _f3 = store
            .insert_file(
                "unlinked.md",
                "h3",
                100,
                &[],
                &generate_docid("unlinked.md"),
                None,
            )
            .unwrap();

        store.insert_edge(f1, f2, "wikilink").unwrap();
        store
            .insert_chunk(f2, "## Linked", "Linked content about delivery", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f2, 0, "Linked content about delivery")
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: f1,
            score: 0.85,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        let expanded = graph_expand(&store, &seeds, "delivery", 2, 20).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].file_path, "linked.md");
        assert!(expanded[0].score > 0.0 && expanded[0].score < 0.85);
    }

    #[test]
    fn test_graph_expand_skips_seeds() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], &generate_docid("a.md"), None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, &[], &generate_docid("b.md"), None)
            .unwrap();

        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_chunk(f2, "## B", "Content B", 10, 20).unwrap();
        store.insert_fts_chunk(f2, 0, "Content B").unwrap();

        let seeds = vec![
            RankedResult {
                file_path: "a.md".into(),
                file_id: f1,
                score: 0.9,
                heading: None,
                snippet: "A".into(),
                docid: None,
            },
            RankedResult {
                file_path: "b.md".into(),
                file_id: f2,
                score: 0.8,
                heading: None,
                snippet: "B".into(),
                docid: None,
            },
        ];

        let expanded = graph_expand(&store, &seeds, "content", 2, 20).unwrap();
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_graph_expand_multi_parent_takes_highest() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], &generate_docid("a.md"), None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, &[], &generate_docid("b.md"), None)
            .unwrap();
        let f3 = store
            .insert_file(
                "shared.md",
                "h3",
                100,
                &[],
                &generate_docid("shared.md"),
                None,
            )
            .unwrap();

        store.insert_edge(f1, f3, "wikilink").unwrap();
        store.insert_edge(f2, f3, "wikilink").unwrap();
        store
            .insert_chunk(f3, "## Shared", "Shared topic content", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f3, 0, "Shared topic content")
            .unwrap();

        let seeds = vec![
            RankedResult {
                file_path: "a.md".into(),
                file_id: f1,
                score: 0.9,
                heading: None,
                snippet: "A".into(),
                docid: None,
            },
            RankedResult {
                file_path: "b.md".into(),
                file_id: f2,
                score: 0.5,
                heading: None,
                snippet: "B".into(),
                docid: None,
            },
        ];

        let expanded = graph_expand(&store, &seeds, "topic", 1, 20).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].file_path, "shared.md");
        // Should use highest parent: 0.9 * 0.8 = 0.72
        assert!((expanded[0].score - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_graph_expand_empty_graph() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], "aaa111", None)
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "a.md".into(),
            file_id: f1,
            score: 0.9,
            heading: None,
            snippet: "A".into(),
            docid: None,
        }];

        let expanded = graph_expand(&store, &seeds, "query", 2, 20).unwrap();
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_graph_expand_tag_fallback() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "seed.md",
                "h1",
                100,
                &["rust".into(), "cli".into()],
                &generate_docid("seed.md"),
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "linked.md",
                "h2",
                100,
                &["rust".into()],
                &generate_docid("linked.md"),
                None,
            )
            .unwrap();

        store.insert_edge(f1, f2, "wikilink").unwrap();
        store
            .insert_chunk(f2, "## Linked", "Unrelated content", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f2, 0, "Unrelated content here")
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: f1,
            score: 0.85,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        // Query doesn't match FTS, but shared tag "rust" should keep it (with 0.7x penalty)
        let expanded = graph_expand(&store, &seeds, "nonexistent query term", 2, 20).unwrap();
        assert_eq!(expanded.len(), 1);
        // Score: 0.85 * 0.8 * 0.7 = 0.476
        assert!((expanded[0].score - 0.476).abs() < 0.01);
    }
}
