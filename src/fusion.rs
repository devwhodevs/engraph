/// Reciprocal Rank Fusion (RRF) engine.
///
/// Merges ranked results from multiple search lanes (e.g. semantic HNSW
/// and FTS5 keyword search) into a single ranked list using the RRF formula:
///
///   rrf_score = sum( weight_i / (k + rank_i) )
///
/// A ranked result from a single search lane.
pub struct RankedResult {
    pub file_path: String,
    pub file_id: i64,
    pub score: f64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
}

/// A fused result after RRF merging across lanes.
pub struct FusedResult {
    pub file_path: String,
    pub file_id: i64,
    pub rrf_score: f64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
    pub lane_contributions: Vec<LaneContribution>,
}

/// Per-lane contribution details for --explain output.
pub struct LaneContribution {
    pub lane_name: String,
    pub rank: usize,
    pub raw_score: f64,
    pub weighted_contribution: f64,
}

use std::collections::HashMap;

/// Fuse ranked results from multiple search lanes using Reciprocal Rank Fusion.
///
/// Each lane is a tuple of `(lane_name, results, weight)`.
/// Results are grouped by `file_path` (file-level deduplication).
/// The best snippet/heading per file is kept from the highest-ranked lane.
///
/// `k` is the RRF constant (typically 60).
pub fn rrf_fuse(lanes: &[(&str, &[RankedResult], f64)], k: usize) -> Vec<FusedResult> {
    // Track per-file: rrf_score, best snippet info, lane contributions
    struct Accumulator {
        file_path: String,
        file_id: i64,
        rrf_score: f64,
        heading: Option<String>,
        snippet: String,
        docid: Option<String>,
        best_rank: usize, // lowest rank seen (for picking best snippet)
        lane_contributions: Vec<LaneContribution>,
    }

    let mut acc_map: HashMap<String, Accumulator> = HashMap::new();

    for &(lane_name, results, weight) in lanes {
        for (idx, r) in results.iter().enumerate() {
            let rank = idx + 1; // 1-based
            let contribution = weight / (k as f64 + rank as f64);

            let acc = acc_map
                .entry(r.file_path.clone())
                .or_insert_with(|| Accumulator {
                    file_path: r.file_path.clone(),
                    file_id: r.file_id,
                    rrf_score: 0.0,
                    heading: r.heading.clone(),
                    snippet: r.snippet.clone(),
                    docid: r.docid.clone(),
                    best_rank: rank,
                    lane_contributions: Vec::new(),
                });

            acc.rrf_score += contribution;

            // Keep snippet from the best-ranked appearance
            if rank < acc.best_rank {
                acc.best_rank = rank;
                acc.heading = r.heading.clone();
                acc.snippet = r.snippet.clone();
                if r.docid.is_some() {
                    acc.docid = r.docid.clone();
                }
            }

            acc.lane_contributions.push(LaneContribution {
                lane_name: lane_name.to_string(),
                rank,
                raw_score: r.score,
                weighted_contribution: contribution,
            });
        }
    }

    let mut results: Vec<FusedResult> = acc_map
        .into_values()
        .map(|a| FusedResult {
            file_path: a.file_path,
            file_id: a.file_id,
            rrf_score: a.rrf_score,
            heading: a.heading,
            snippet: a.snippet,
            docid: a.docid,
            lane_contributions: a.lane_contributions,
        })
        .collect();

    // Sort by rrf_score descending
    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Format explain output for a single fused result.
pub fn format_explain(result: &FusedResult) -> String {
    let mut out = format!("  RRF: {:.4}\n", result.rrf_score);
    for lc in &result.lane_contributions {
        out.push_str(&format!(
            "    {}: rank #{}, raw {:.2}, +{:.4}\n",
            lc.lane_name, lc.rank, lc.raw_score, lc.weighted_contribution,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(file_path: &str, score: f64) -> RankedResult {
        RankedResult {
            file_path: file_path.to_string(),
            file_id: 0,
            score,
            heading: Some(format!("heading for {}", file_path)),
            snippet: format!("snippet for {}", file_path),
            docid: None,
        }
    }

    #[test]
    fn test_rrf_basic() {
        // Item appearing in both lanes should rank highest
        let semantic = vec![
            make_result("both.md", 0.87),
            make_result("sem_only.md", 0.75),
        ];
        let fts = vec![make_result("fts_only.md", 5.0), make_result("both.md", 3.2)];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0), ("fts", &fts, 1.0)], 60);

        assert_eq!(fused.len(), 3);
        // "both.md" should be first because it appears in both lanes
        assert_eq!(fused[0].file_path, "both.md");

        // Verify the RRF score for "both.md":
        // semantic rank 1: 1.0 / (60 + 1) = 0.01639...
        // fts rank 2: 1.0 / (60 + 2) = 0.01613...
        // total = 0.03252...
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].rrf_score - expected).abs() < 1e-10);

        // Both single-lane items should have lower scores
        assert!(fused[0].rrf_score > fused[1].rrf_score);
        assert!(fused[0].rrf_score > fused[2].rrf_score);

        // "both.md" should have 2 lane contributions
        assert_eq!(fused[0].lane_contributions.len(), 2);
    }

    #[test]
    fn test_rrf_weighted() {
        // FTS weighted 3x should make FTS-only item win over semantic-only item
        let semantic = vec![make_result("sem.md", 0.95)];
        let fts = vec![make_result("fts.md", 8.0)];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0), ("fts", &fts, 3.0)], 60);

        assert_eq!(fused.len(), 2);
        // FTS item at rank 1 with weight 3.0: 3.0 / 61 = 0.04918...
        // Semantic item at rank 1 with weight 1.0: 1.0 / 61 = 0.01639...
        assert_eq!(fused[0].file_path, "fts.md");
        assert_eq!(fused[1].file_path, "sem.md");

        let fts_expected = 3.0 / 61.0;
        let sem_expected = 1.0 / 61.0;
        assert!((fused[0].rrf_score - fts_expected).abs() < 1e-10);
        assert!((fused[1].rrf_score - sem_expected).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_single_lane() {
        let semantic = vec![
            make_result("a.md", 0.9),
            make_result("b.md", 0.8),
            make_result("c.md", 0.7),
        ];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0)], 60);

        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].file_path, "a.md");
        assert_eq!(fused[1].file_path, "b.md");
        assert_eq!(fused[2].file_path, "c.md");

        // Each should have exactly 1 lane contribution
        for f in &fused {
            assert_eq!(f.lane_contributions.len(), 1);
            assert_eq!(f.lane_contributions[0].lane_name, "semantic");
        }
    }

    #[test]
    fn test_format_explain() {
        let result = FusedResult {
            file_path: "test.md".to_string(),
            file_id: 1,
            rrf_score: 0.0328,
            heading: None,
            snippet: "test".to_string(),
            docid: None,
            lane_contributions: vec![
                LaneContribution {
                    lane_name: "semantic".to_string(),
                    rank: 1,
                    raw_score: 0.87,
                    weighted_contribution: 0.0164,
                },
                LaneContribution {
                    lane_name: "fts".to_string(),
                    rank: 3,
                    raw_score: 5.23,
                    weighted_contribution: 0.0159,
                },
            ],
        };

        let output = format_explain(&result);
        assert!(output.contains("RRF: 0.0328"));
        assert!(output.contains("semantic: rank #1, raw 0.87, +0.0164"));
        assert!(output.contains("fts: rank #3, raw 5.23, +0.0159"));
    }

    #[test]
    fn test_rrf_empty_lanes() {
        let fused = rrf_fuse(&[], 60);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_empty_results() {
        let empty: Vec<RankedResult> = vec![];
        let fused = rrf_fuse(&[("semantic", &empty, 1.0), ("fts", &empty, 1.0)], 60);
        assert!(fused.is_empty());
    }
}
