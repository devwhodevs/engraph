use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::embedder::Embedder;
use crate::fusion::{self, RankedResult};
use crate::graph;
use crate::store::{Store, StoreStats};

/// A single search result with metadata.
pub struct SearchResult {
    pub score: f32,
    pub file_path: String,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
}

/// Structured search result for internal use (no I/O).
#[derive(Debug, Clone, serde::Serialize)]
pub struct InternalSearchResult {
    pub file_path: String,
    pub file_id: i64,
    pub score: f64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
}

/// Output from `search_internal`: structured results plus raw fused data for --explain.
pub struct SearchOutput {
    pub results: Vec<InternalSearchResult>,
    pub fused: Vec<fusion::FusedResult>,
}

/// Run hybrid search and return structured results (no I/O).
/// Used by both `run_search` (CLI) and context engine.
pub fn search_internal(
    query: &str,
    top_n: usize,
    store: &Store,
    embedder: &mut Embedder,
) -> Result<SearchOutput> {
    // --- Semantic lane ---
    let query_vec = embedder.embed_one(query).context("embedding query")?;
    let tombstones = store.get_tombstones().context("loading tombstones")?;

    // Request extra results to account for tombstone filtering and file-level dedup.
    let raw_results = store.search_vec(&query_vec, top_n * 3, &tombstones)?;

    // Group semantic results by file_path, keeping best per file.
    let mut sem_by_file: HashMap<String, RankedResult> = HashMap::new();
    for (vector_id, distance) in raw_results {
        if let Some(chunk) = store.get_chunk_by_vector_id(vector_id)? {
            let (file_path, docid) = match store.get_file_by_id(chunk.file_id)? {
                Some(f) => (f.path, f.docid),
                None => ("<unknown>".to_string(), None),
            };
            let score = (1.0 - distance) as f64;
            let heading = if chunk.heading.is_empty() {
                None
            } else {
                Some(chunk.heading)
            };

            // Keep the best-scoring chunk per file.
            let better = match sem_by_file.get(&file_path) {
                Some(existing) => score > existing.score,
                None => true,
            };
            if better {
                sem_by_file.insert(
                    file_path.clone(),
                    RankedResult {
                        file_path,
                        file_id: chunk.file_id,
                        score,
                        heading,
                        snippet: chunk.snippet,
                        docid,
                    },
                );
            }
        }
    }

    // Sort semantic results by score descending for rank assignment.
    let mut semantic_results: Vec<RankedResult> = sem_by_file.into_values().collect();
    semantic_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- FTS lane ---
    let fts_raw = store.fts_search(query, top_n * 3).unwrap_or_default();

    // Group FTS results by file_path, keeping best per file.
    let mut fts_by_file: HashMap<String, RankedResult> = HashMap::new();
    for fr in fts_raw {
        let (file_path, docid) = match store.get_file_by_id(fr.file_id)? {
            Some(f) => (f.path, f.docid),
            None => continue,
        };

        let better = match fts_by_file.get(&file_path) {
            Some(existing) => fr.score > existing.score,
            None => true,
        };
        if better {
            fts_by_file.insert(
                file_path.clone(),
                RankedResult {
                    file_path,
                    file_id: fr.file_id,
                    score: fr.score,
                    heading: None, // FTS doesn't return headings
                    snippet: fr.snippet,
                    docid,
                },
            );
        }
    }

    let mut fts_results: Vec<RankedResult> = fts_by_file.into_values().collect();
    fts_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- Graph lane ---
    // Combine seeds from semantic + FTS (deduplicated by file_path, take higher score)
    let combined_seeds: Vec<RankedResult> = {
        let mut by_file: HashMap<String, RankedResult> = HashMap::new();
        for r in semantic_results.iter().chain(fts_results.iter()) {
            match by_file.get(&r.file_path) {
                Some(existing) if r.score <= existing.score => {}
                _ => {
                    by_file.insert(r.file_path.clone(), r.clone());
                }
            }
        }
        by_file.into_values().collect()
    };

    let graph_results =
        graph::graph_expand(store, &combined_seeds, query, 2, 20).unwrap_or_default();

    // --- RRF Fusion ---
    const RRF_K: usize = 60;
    let fused = fusion::rrf_fuse(
        &[
            ("semantic", &semantic_results, 1.0),
            ("fts", &fts_results, 1.0),
            ("graph", &graph_results, 0.8),
        ],
        RRF_K,
    );

    // Convert fused results to InternalSearchResult, taking top_n.
    let results: Vec<InternalSearchResult> = fused
        .iter()
        .take(top_n)
        .map(|f| InternalSearchResult {
            file_path: f.file_path.clone(),
            file_id: f.file_id,
            score: f.rrf_score,
            heading: f.heading.clone(),
            snippet: f.snippet.clone(),
            docid: f.docid.clone(),
        })
        .collect();

    Ok(SearchOutput { results, fused })
}

/// Run a search query and print results.
///
/// Performs both semantic (sqlite-vec) and keyword (FTS5) search, then fuses
/// results using Reciprocal Rank Fusion. When `explain` is true, each
/// result includes per-lane score breakdown.
pub fn run_search(
    query: &str,
    top_n: usize,
    json: bool,
    explain: bool,
    data_dir: &Path,
) -> Result<()> {
    let models_dir = data_dir.join("models");
    let mut embedder = Embedder::new(&models_dir).context("loading embedder")?;

    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path).context("opening store")?;

    let output = search_internal(query, top_n, &store, &mut embedder)?;

    let results: Vec<SearchResult> = output
        .results
        .iter()
        .map(|r| SearchResult {
            score: r.score as f32,
            file_path: r.file_path.clone(),
            heading: r.heading.clone(),
            snippet: r.snippet.clone(),
            docid: r.docid.clone(),
        })
        .collect();

    let mut out = format_results(&results, json);

    if explain && !json {
        let mut explain_out = String::from("\n--- Explain ---\n");
        for f in output.fused.iter().take(top_n) {
            explain_out.push_str(&format!("{}\n", f.file_path));
            explain_out.push_str(&fusion::format_explain(f));
        }
        out.push_str(&explain_out);
    }

    print!("{out}");
    Ok(())
}

/// Run the status command and print index information.
pub fn run_status(json: bool, data_dir: &Path) -> Result<()> {
    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path).context("opening store")?;
    let stats = store.stats()?;

    // Compute index size on disk (sqlite db file).
    let index_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    let model_name = "all-MiniLM-L6-v2";

    let output = format_status(&stats, index_size, model_name, json);
    print!("{output}");
    Ok(())
}

/// Format search results for display (pure function, no I/O).
pub fn format_results(results: &[SearchResult], json: bool) -> String {
    if results.is_empty() {
        return if json {
            "[]\n".to_string()
        } else {
            "No results found.\n".to_string()
        };
    }

    if json {
        let items: Vec<serde_json::Value> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                // Round score to 2 decimal places via f64 to avoid f32 precision artifacts.
                let score_rounded = ((r.score as f64) * 100.0).round() / 100.0;
                json!({
                    "rank": i + 1,
                    "score": score_rounded,
                    "file": r.file_path,
                    "heading": r.heading,
                    "snippet": r.snippet,
                    "docid": r.docid,
                })
            })
            .collect();
        format!("{}\n", serde_json::to_string_pretty(&items).unwrap())
    } else {
        let mut out = String::new();
        for (i, r) in results.iter().enumerate() {
            let heading_part = match &r.heading {
                Some(h) => format!(" > {h}"),
                None => String::new(),
            };
            let docid_part = match &r.docid {
                Some(d) => format!(" #{d}"),
                None => String::new(),
            };
            let snippet = truncate_snippet(&r.snippet, 200);
            out.push_str(&format!(
                "{:>2}. [{:.2}] {}{}{}\n    {}\n",
                i + 1,
                r.score,
                r.file_path,
                heading_part,
                docid_part,
                snippet,
            ));
        }
        out
    }
}

/// Format status information for display (pure function, no I/O).
pub fn format_status(stats: &StoreStats, index_size: u64, model_name: &str, json: bool) -> String {
    let vault = stats.vault_path.as_deref().unwrap_or("<not set>");
    let last_indexed = stats.last_indexed_at.as_deref().unwrap_or("never");

    if json {
        let mut obj = json!({
            "vault": vault,
            "files": stats.file_count,
            "chunks": stats.chunk_count,
            "tombstones": stats.tombstone_count,
            "last_indexed": last_indexed,
            "index_size": index_size,
            "model": model_name,
        });
        if let (Some(edges), Some(wl), Some(mn)) =
            (stats.edge_count, stats.wikilink_count, stats.mention_count)
        {
            obj["edges"] = json!(edges);
            obj["wikilink_edges"] = json!(wl);
            obj["mention_edges"] = json!(mn);
        }
        format!("{}\n", serde_json::to_string_pretty(&obj).unwrap())
    } else {
        let mut out = format!(
            "Vault:      {}\n\
             Files:      {}\n\
             Chunks:     {}\n",
            vault, stats.file_count, stats.chunk_count,
        );
        if let (Some(edges), Some(wl), Some(mn)) =
            (stats.edge_count, stats.wikilink_count, stats.mention_count)
        {
            out.push_str(&format!(
                "Edges:      {} ({} wikilinks, {} mentions)\n",
                edges, wl, mn
            ));
        }
        out.push_str(&format!(
            "Tombstones: {} (pending cleanup)\n\
             Last index: {}\n\
             Index size: {}\n\
             Model:      {}\n",
            stats.tombstone_count,
            last_indexed,
            format_bytes(index_size),
            model_name,
        ));
        out
    }
}

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
fn truncate_snippet(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find a char boundary near max_len.
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_human_result() {
        let results = vec![SearchResult {
            score: 0.87,
            file_path: "foo.md".to_string(),
            heading: Some("## Bar".to_string()),
            snippet: "Some text...".to_string(),
            docid: Some("ab12cd".to_string()),
        }];
        let output = format_results(&results, false);
        assert_eq!(
            output,
            " 1. [0.87] foo.md > ## Bar #ab12cd\n    Some text...\n"
        );
    }

    #[test]
    fn test_format_human_result_no_docid() {
        let results = vec![SearchResult {
            score: 0.87,
            file_path: "foo.md".to_string(),
            heading: Some("## Bar".to_string()),
            snippet: "Some text...".to_string(),
            docid: None,
        }];
        let output = format_results(&results, false);
        assert_eq!(output, " 1. [0.87] foo.md > ## Bar\n    Some text...\n");
    }

    #[test]
    fn test_format_json_result() {
        let results = vec![SearchResult {
            score: 0.87,
            file_path: "foo.md".to_string(),
            heading: Some("## Bar".to_string()),
            snippet: "Some text...".to_string(),
            docid: Some("ab12cd".to_string()),
        }];
        let output = format_results(&results, true);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["rank"], 1);
        assert_eq!(parsed[0]["score"], 0.87);
        assert_eq!(parsed[0]["file"], "foo.md");
        assert_eq!(parsed[0]["heading"], "## Bar");
        assert_eq!(parsed[0]["snippet"], "Some text...");
        assert_eq!(parsed[0]["docid"], "ab12cd");
    }

    #[test]
    fn test_no_results_message() {
        let output = format_results(&[], false);
        assert_eq!(output, "No results found.\n");

        let json_output = format_results(&[], true);
        assert_eq!(json_output, "[]\n");
    }

    #[test]
    fn test_format_status_human() {
        let stats = StoreStats {
            file_count: 42,
            chunk_count: 187,
            tombstone_count: 3,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
            edge_count: None,
            wikilink_count: None,
            mention_count: None,
        };
        let output = format_status(&stats, 2_516_582, "all-MiniLM-L6-v2", false);

        assert!(output.contains("/path/to/vault"), "missing vault path");
        assert!(output.contains("42"), "missing file count");
        assert!(output.contains("187"), "missing chunk count");
        assert!(output.contains("3"), "missing tombstone count");
        assert!(output.contains("2026-03-19 14:30:00"), "missing last index");
        assert!(output.contains("2.4 MB"), "missing index size");
        assert!(output.contains("all-MiniLM-L6-v2"), "missing model");
    }

    #[test]
    fn test_format_status_json() {
        let stats = StoreStats {
            file_count: 42,
            chunk_count: 187,
            tombstone_count: 3,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
            edge_count: None,
            wikilink_count: None,
            mention_count: None,
        };
        let output = format_status(&stats, 2_516_582, "all-MiniLM-L6-v2", true);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(parsed["vault"], "/path/to/vault");
        assert_eq!(parsed["files"], 42);
        assert_eq!(parsed["chunks"], 187);
        assert_eq!(parsed["tombstones"], 3);
        assert_eq!(parsed["last_indexed"], "2026-03-19 14:30:00");
        assert_eq!(parsed["index_size"], 2_516_582);
        assert_eq!(parsed["model"], "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_truncate_snippet() {
        let short = "hello";
        assert_eq!(truncate_snippet(short, 200), "hello");

        let long = "a".repeat(300);
        let truncated = truncate_snippet(&long, 200);
        assert!(truncated.ends_with("..."));
        assert_eq!(truncated.len(), 203); // 200 + "..."
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(2_516_582), "2.4 MB");
    }
}
