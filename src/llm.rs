use anyhow::Result;
use sha2::{Digest, Sha256};

// ── Types ────────────────────────────────────────────────────────────────────

/// Classified intent of an incoming search query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryIntent {
    /// User wants a precise fact or term match.
    Exact,
    /// User wants related ideas and concepts.
    Conceptual,
    /// User wants to explore connections between entities.
    Relationship,
    /// User is browsing without a clear target.
    Exploratory,
}

/// Output produced by an orchestrator model for a query.
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    /// Classified query intent.
    pub intent: QueryIntent,
    /// Query string(s) to actually run (original + any expansions).
    pub expansions: Vec<String>,
}

/// Per-lane weights for the RRF fusion step.
#[derive(Debug, Clone)]
pub struct LaneWeights {
    pub semantic: f64,
    pub fts: f64,
    pub graph: f64,
    pub rerank: f64,
}

impl LaneWeights {
    /// Map a classified intent to recommended lane weights.
    pub fn from_intent(intent: &QueryIntent) -> Self {
        match intent {
            QueryIntent::Exact => Self {
                fts: 1.5,
                semantic: 0.6,
                graph: 0.6,
                rerank: 0.8,
            },
            QueryIntent::Conceptual => Self {
                semantic: 1.2,
                fts: 0.8,
                graph: 1.0,
                rerank: 1.2,
            },
            QueryIntent::Relationship => Self {
                graph: 1.5,
                semantic: 0.8,
                fts: 0.8,
                rerank: 1.0,
            },
            QueryIntent::Exploratory => Self {
                semantic: 1.0,
                fts: 1.0,
                graph: 0.8,
                rerank: 1.0,
            },
        }
    }

    /// Weights used when no intelligence layer is available (legacy mode).
    pub fn default_no_intelligence() -> Self {
        Self {
            semantic: 1.0,
            fts: 1.0,
            graph: 0.8,
            rerank: 0.0,
        }
    }
}

// ── Traits ───────────────────────────────────────────────────────────────────

/// Embedding backend — converts text into dense float vectors.
pub trait EmbedModel: Send {
    /// Embed a batch of texts in one call.
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Convenience wrapper for a single text.
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text])?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embed_batch returned empty results"))
    }

    /// Approximate token count for `text` (used for chunk-size budgeting).
    fn token_count(&self, text: &str) -> usize;

    /// Dimensionality of vectors produced by this model.
    fn dim(&self) -> usize;
}

/// Cross-encoder reranker — scores a (query, document) pair.
pub trait RerankModel: Send {
    /// Return a relevance score in [0.0, 1.0].
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32>;
}

/// Orchestrator — interprets a query and produces an enriched search plan.
pub trait OrchestratorModel: Send {
    fn orchestrate(&mut self, query: &str) -> Result<OrchestrationResult>;
}

// ── MockLlm ──────────────────────────────────────────────────────────────────

/// Deterministic in-process implementation of all three traits.
/// Suitable for unit tests and CI runs — no model files required.
pub struct MockLlm {
    dim: usize,
}

impl MockLlm {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Produce a deterministic L2-normalised vector from `text` via SHA-256.
    pub fn hash_to_vector(&self, text: &str) -> Vec<f32> {
        let mut raw: Vec<f32> = Vec::with_capacity(self.dim);
        // Seed the first hash from the text itself, then chain hashes to fill
        // vectors wider than 32 bytes (8 f32s per 256-bit hash).
        let mut seed = text.to_owned();
        while raw.len() < self.dim {
            let mut hasher = Sha256::new();
            hasher.update(seed.as_bytes());
            let hash = hasher.finalize();
            // Each hash gives 32 bytes → 8 f32 values.
            for chunk in hash.chunks(4) {
                if raw.len() >= self.dim {
                    break;
                }
                let bytes: [u8; 4] = chunk.try_into().expect("chunk is always 4 bytes");
                // Map u32 → [-1.0, 1.0] for a reasonable spread before normalisation.
                let u = u32::from_le_bytes(bytes);
                let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
                raw.push(f);
            }
            // Next round: hash the previous hash digest (as hex) so values differ.
            seed = format!("{:x}", {
                let mut h2 = Sha256::new();
                h2.update(hash);
                h2.finalize()
            });
        }

        // L2-normalise so the mock behaves like a real embedding model.
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            raw.iter_mut().for_each(|x| *x /= norm);
        }
        raw
    }
}

impl EmbedModel for MockLlm {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vector(t)).collect())
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_vector(text))
    }

    fn token_count(&self, text: &str) -> usize {
        text.len() / 4 + 1
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

impl RerankModel for MockLlm {
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32> {
        // Deterministic score: Jaccard overlap of character 4-grams, clamped to [0,1].
        let ngrams = |s: &str| -> std::collections::HashSet<String> {
            s.chars()
                .collect::<Vec<_>>()
                .windows(4)
                .map(|w| w.iter().collect())
                .collect()
        };

        let q_set = ngrams(&query.to_lowercase());
        let d_set = ngrams(&document.to_lowercase());

        if q_set.is_empty() && d_set.is_empty() {
            return Ok(0.5);
        }

        let intersection = q_set.intersection(&d_set).count();
        let union = q_set.union(&d_set).count();

        let score = intersection as f32 / union as f32;
        Ok(score.clamp(0.0, 1.0))
    }
}

impl OrchestratorModel for MockLlm {
    fn orchestrate(&mut self, query: &str) -> Result<OrchestrationResult> {
        Ok(OrchestrationResult {
            intent: QueryIntent::Exploratory,
            expansions: vec![query.to_owned()],
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embed_deterministic() {
        let mut mock = MockLlm::new(256);
        let v1 = mock.embed_one("hello").unwrap();
        let v2 = mock.embed_one("hello").unwrap();
        assert_eq!(v1.len(), 256);
        assert_eq!(v1, v2, "same input must produce same output");
    }

    #[test]
    fn test_mock_embed_different_inputs() {
        let mut mock = MockLlm::new(256);
        let v1 = mock.embed_one("hello").unwrap();
        let v2 = mock.embed_one("world").unwrap();
        assert_ne!(v1, v2, "different inputs should produce different vectors");
    }

    #[test]
    fn test_mock_embed_batch() {
        let mut mock = MockLlm::new(256);
        let vecs = mock.embed_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 256));
    }

    #[test]
    fn test_mock_embed_normalized() {
        let mut mock = MockLlm::new(256);
        let v = mock.embed_one("test").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "mock vectors should be L2-normalized"
        );
    }

    #[test]
    fn test_mock_rerank() {
        let mut mock = MockLlm::new(256);
        let score = mock.rerank_score("query", "document text").unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_mock_orchestrate() {
        let mut mock = MockLlm::new(256);
        let result = mock.orchestrate("how does auth work").unwrap();
        assert_eq!(result.intent, QueryIntent::Exploratory);
        assert!(!result.expansions.is_empty());
        assert_eq!(result.expansions[0], "how does auth work");
    }

    #[test]
    fn test_mock_rerank_empty_query() {
        let mut mock = MockLlm::new(256);
        let score = mock.rerank_score("", "document text").unwrap();
        assert_eq!(score, 0.0, "empty query should score 0.0");
    }

    #[test]
    fn test_lane_weights_from_intent() {
        let exact = LaneWeights::from_intent(&QueryIntent::Exact);
        assert!(exact.fts > exact.semantic, "exact intent should favor FTS");

        let conceptual = LaneWeights::from_intent(&QueryIntent::Conceptual);
        assert!(
            conceptual.semantic > conceptual.fts,
            "conceptual should favor semantic"
        );

        let relationship = LaneWeights::from_intent(&QueryIntent::Relationship);
        assert!(
            relationship.graph > relationship.semantic,
            "relationship should favor graph"
        );
    }
}
