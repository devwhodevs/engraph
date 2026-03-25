use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::fusion::{self, RankedResult};
use crate::graph;
use crate::llm::{self, EmbedModel, OrchestratorModel, RerankModel};
use crate::store::{Store, StoreStats};

/// Compute cache key for orchestration results (SHA256 of query).
fn orchestration_cache_key(query: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(query.as_bytes());
    format!("{:x}", hash)
}

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
    pub intent: Option<crate::llm::QueryIntent>,
}

/// Configuration for the intelligence search pipeline.
pub struct SearchConfig<'a> {
    pub orchestrator: Option<&'a mut dyn OrchestratorModel>,
    pub reranker: Option<&'a mut dyn RerankModel>,
    pub store: &'a Store,
    pub rerank_candidates: usize,
}

/// Run hybrid search and return structured results (no I/O).
/// Used by both `run_search` (CLI) and context engine.
///
/// Thin wrapper around `search_with_intelligence` with no intelligence models,
/// preserving the existing heuristic-only behavior.
pub fn search_internal(
    query: &str,
    top_n: usize,
    store: &Store,
    embedder: &mut impl EmbedModel,
) -> Result<SearchOutput> {
    let mut config = SearchConfig {
        orchestrator: None,
        reranker: None,
        store,
        rerank_candidates: 30,
    };
    search_with_intelligence(query, top_n, embedder, &mut config)
}

/// Full intelligence search pipeline.
///
/// 1. Orchestrate (intent + expansions + weights) — LLM if available, else heuristic.
/// 2. 3-lane retrieval per expanded query (semantic, FTS, graph).
/// 3. RRF Pass 1 with top candidates.
/// 4. Reranker scores each candidate (4th lane) if available.
/// 5. RRF Pass 2 with all 4 lanes for final ranking.
pub fn search_with_intelligence(
    query: &str,
    top_n: usize,
    embedder: &mut impl EmbedModel,
    config: &mut SearchConfig<'_>,
) -> Result<SearchOutput> {
    // --- Step 1: Orchestrate (with LLM cache when orchestrator is present) ---
    let orchestration = match &mut config.orchestrator {
        Some(orch) => {
            let cache_key = orchestration_cache_key(query);
            if let Some(cached_json) = config.store.get_llm_cache(&cache_key)? {
                serde_json::from_str(&cached_json).unwrap_or_else(|_| {
                    orch.orchestrate(query)
                        .unwrap_or_else(|_| llm::heuristic_orchestrate(query))
                })
            } else {
                let result = orch.orchestrate(query)?;
                if let Ok(json) = serde_json::to_string(&result) {
                    let _ = config
                        .store
                        .set_llm_cache(&cache_key, &json, "orchestrator");
                }
                result
            }
        }
        None => llm::heuristic_orchestrate(query),
    };
    let weights = llm::LaneWeights::from_intent(&orchestration.intent);

    // --- Step 2: Run 3-lane retrieval for EACH expanded query ---
    let mut all_semantic: Vec<RankedResult> = Vec::new();
    let mut all_fts: Vec<RankedResult> = Vec::new();

    for expanded_query in &orchestration.expansions {
        // Semantic lane
        let query_vec = embedder
            .embed_one(expanded_query)
            .context("embedding query")?;
        let tombstones = std::collections::HashSet::new();
        let raw_results = config
            .store
            .search_vec(&query_vec, top_n * 3, &tombstones)?;

        // Group semantic results by file_path, keeping best per file.
        let mut sem_by_file: HashMap<String, RankedResult> = HashMap::new();
        for (vector_id, distance) in raw_results {
            if let Some(chunk) = config.store.get_chunk_by_vector_id(vector_id)? {
                let (file_path, docid) = match config.store.get_file_by_id(chunk.file_id)? {
                    Some(f) => (f.path, f.docid),
                    None => ("<unknown>".to_string(), None),
                };
                let score = (1.0 - distance) as f64;
                let heading = if chunk.heading.is_empty() {
                    None
                } else {
                    Some(chunk.heading)
                };

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
        all_semantic.extend(sem_by_file.into_values());

        // FTS lane
        let fts_raw = config
            .store
            .fts_search(expanded_query, top_n * 3)
            .unwrap_or_default();

        let mut fts_by_file: HashMap<String, RankedResult> = HashMap::new();
        for fr in fts_raw {
            let (file_path, docid) = match config.store.get_file_by_id(fr.file_id)? {
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
        all_fts.extend(fts_by_file.into_values());
    }

    // Deduplicate across expanded queries (keep best score per file)
    let semantic_results = dedup_by_file(all_semantic);
    let fts_results = dedup_by_file(all_fts);

    // --- Graph lane from combined seeds ---
    let combined_seeds = merge_seeds(&semantic_results, &fts_results);
    let graph_results =
        graph::graph_expand(config.store, &combined_seeds, query, 2, 20).unwrap_or_default();

    // --- Step 3: RRF Pass 1 (3-lane) ---
    const RRF_K: usize = 60;
    let fused_pass1 = fusion::rrf_fuse(
        &[
            ("semantic", &semantic_results, weights.semantic),
            ("fts", &fts_results, weights.fts),
            ("graph", &graph_results, weights.graph),
        ],
        RRF_K,
    );

    // --- Step 4: Reranker (4th lane) if available ---
    let final_fused = if let Some(reranker) = &mut config.reranker {
        let mut rerank_results: Vec<RankedResult> = Vec::new();
        for candidate in fused_pass1.iter().take(config.rerank_candidates) {
            let score = reranker
                .rerank_score(query, &candidate.snippet)
                .unwrap_or(0.0) as f64;
            rerank_results.push(RankedResult {
                file_path: candidate.file_path.clone(),
                file_id: candidate.file_id,
                score,
                heading: candidate.heading.clone(),
                snippet: candidate.snippet.clone(),
                docid: candidate.docid.clone(),
            });
        }
        rerank_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // RRF Pass 2 (4-lane)
        fusion::rrf_fuse(
            &[
                ("semantic", &semantic_results, weights.semantic),
                ("fts", &fts_results, weights.fts),
                ("graph", &graph_results, weights.graph),
                ("rerank", &rerank_results, weights.rerank),
            ],
            RRF_K,
        )
    } else {
        fused_pass1
    };

    // Convert fused results to InternalSearchResult, taking top_n.
    let results: Vec<InternalSearchResult> = final_fused
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

    Ok(SearchOutput {
        results,
        fused: final_fused,
        intent: Some(orchestration.intent),
    })
}

/// Deduplicate ranked results by file path, keeping the highest score per file.
fn dedup_by_file(results: Vec<RankedResult>) -> Vec<RankedResult> {
    let mut by_file: HashMap<String, RankedResult> = HashMap::new();
    for r in results {
        let dominated = by_file
            .get(&r.file_path)
            .is_some_and(|existing| existing.score >= r.score);
        if !dominated {
            by_file.insert(r.file_path.clone(), r);
        }
    }
    let mut deduped: Vec<RankedResult> = by_file.into_values().collect();
    deduped.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    deduped
}

/// Merge semantic and FTS seed results, keeping the highest score per file.
fn merge_seeds(semantic: &[RankedResult], fts: &[RankedResult]) -> Vec<RankedResult> {
    let mut by_file: HashMap<String, RankedResult> = HashMap::new();
    for r in semantic.iter().chain(fts.iter()) {
        let dominated = by_file
            .get(&r.file_path)
            .is_some_and(|existing| existing.score >= r.score);
        if !dominated {
            by_file.insert(r.file_path.clone(), r.clone());
        }
    }
    by_file.into_values().collect()
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
    config: &crate::config::Config,
) -> Result<()> {
    let models_dir = data_dir.join("models");
    let mut embedder =
        crate::llm::CandleEmbed::new(&models_dir, config).context("loading embedder")?;

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
        let mut explain_out = String::new();
        if let Some(ref intent) = output.intent {
            explain_out.push_str(&format!("Intent: {:?}\n\n", intent));
        }
        explain_out.push_str("--- Explain ---\n");
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

    let config = crate::config::Config::load().unwrap_or_default();
    let intelligence = if config.intelligence_enabled() {
        "enabled"
    } else {
        "disabled"
    };

    let output = format_status(&stats, index_size, model_name, intelligence, json);
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
pub fn format_status(
    stats: &StoreStats,
    index_size: u64,
    model_name: &str,
    intelligence: &str,
    json: bool,
) -> String {
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
            "intelligence": intelligence,
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
             Model:      {}\n\
             Intelligence: {}\n",
            stats.tombstone_count,
            last_indexed,
            format_bytes(index_size),
            model_name,
            intelligence,
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
        let output = format_status(&stats, 2_516_582, "all-MiniLM-L6-v2", "disabled", false);

        assert!(output.contains("/path/to/vault"), "missing vault path");
        assert!(output.contains("42"), "missing file count");
        assert!(output.contains("187"), "missing chunk count");
        assert!(output.contains("3"), "missing tombstone count");
        assert!(output.contains("2026-03-19 14:30:00"), "missing last index");
        assert!(output.contains("2.4 MB"), "missing index size");
        assert!(output.contains("all-MiniLM-L6-v2"), "missing model");
        assert!(output.contains("disabled"), "missing intelligence");
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
        let output = format_status(&stats, 2_516_582, "all-MiniLM-L6-v2", "enabled", true);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(parsed["vault"], "/path/to/vault");
        assert_eq!(parsed["files"], 42);
        assert_eq!(parsed["chunks"], 187);
        assert_eq!(parsed["tombstones"], 3);
        assert_eq!(parsed["last_indexed"], "2026-03-19 14:30:00");
        assert_eq!(parsed["index_size"], 2_516_582);
        assert_eq!(parsed["model"], "all-MiniLM-L6-v2");
        assert_eq!(parsed["intelligence"], "enabled");
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

    #[test]
    fn test_cache_key_deterministic() {
        let key1 = super::orchestration_cache_key("how does auth work");
        let key2 = super::orchestration_cache_key("how does auth work");
        assert_eq!(key1, key2);

        let key3 = super::orchestration_cache_key("different query");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_search_output_has_intent() {
        let output = SearchOutput {
            results: vec![],
            fused: vec![],
            intent: Some(crate::llm::QueryIntent::Conceptual),
        };
        assert_eq!(output.intent, Some(crate::llm::QueryIntent::Conceptual));
    }

    #[test]
    fn test_search_output_intent_none() {
        let output = SearchOutput {
            results: vec![],
            fused: vec![],
            intent: None,
        };
        assert!(output.intent.is_none());
    }

    #[test]
    fn test_dedup_by_file_keeps_best() {
        let results = vec![
            RankedResult {
                file_path: "a.md".to_string(),
                file_id: 1,
                score: 0.5,
                heading: None,
                snippet: "low".to_string(),
                docid: None,
            },
            RankedResult {
                file_path: "a.md".to_string(),
                file_id: 1,
                score: 0.9,
                heading: None,
                snippet: "high".to_string(),
                docid: None,
            },
            RankedResult {
                file_path: "b.md".to_string(),
                file_id: 2,
                score: 0.7,
                heading: None,
                snippet: "only".to_string(),
                docid: None,
            },
        ];
        let deduped = dedup_by_file(results);
        assert_eq!(deduped.len(), 2);
        // Sorted by score descending
        assert_eq!(deduped[0].file_path, "a.md");
        assert!((deduped[0].score - 0.9).abs() < 1e-10);
        assert_eq!(deduped[0].snippet, "high");
        assert_eq!(deduped[1].file_path, "b.md");
    }

    #[test]
    fn test_dedup_by_file_empty() {
        let deduped = dedup_by_file(vec![]);
        assert!(deduped.is_empty());
    }

    #[test]
    fn test_merge_seeds_deduplicates() {
        let semantic = vec![RankedResult {
            file_path: "shared.md".to_string(),
            file_id: 1,
            score: 0.8,
            heading: None,
            snippet: "sem".to_string(),
            docid: None,
        }];
        let fts = vec![
            RankedResult {
                file_path: "shared.md".to_string(),
                file_id: 1,
                score: 0.9,
                heading: None,
                snippet: "fts".to_string(),
                docid: None,
            },
            RankedResult {
                file_path: "fts_only.md".to_string(),
                file_id: 2,
                score: 0.6,
                heading: None,
                snippet: "fts only".to_string(),
                docid: None,
            },
        ];
        let merged = merge_seeds(&semantic, &fts);
        assert_eq!(merged.len(), 2);
        // "shared.md" should have the FTS score (0.9 > 0.8)
        let shared = merged.iter().find(|r| r.file_path == "shared.md").unwrap();
        assert!((shared.score - 0.9).abs() < 1e-10);
    }
}
