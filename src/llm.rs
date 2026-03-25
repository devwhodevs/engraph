use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
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

// Blanket impl: `Box<dyn EmbedModel + Send>` itself implements `EmbedModel`.
// This lets `Arc<Mutex<Box<dyn EmbedModel + Send>>>` callers pass
// `&mut *guard` (which is `&mut Box<dyn EmbedModel + Send>`) to any
// function taking `&mut impl EmbedModel`.
impl EmbedModel for Box<dyn EmbedModel + Send> {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_one(text)
    }

    fn token_count(&self, text: &str) -> usize {
        (**self).token_count(text)
    }

    fn dim(&self) -> usize {
        (**self).dim()
    }
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

// ── HuggingFace model download infrastructure ─────────────────────────────────

/// Parsed HuggingFace model URI: "hf:org/repo/filename.gguf"
#[derive(Debug, Clone)]
pub struct HfModelUri {
    pub repo: String,
    pub filename: String,
}

impl HfModelUri {
    pub fn parse(uri: &str) -> Result<Self> {
        let rest = uri
            .strip_prefix("hf:")
            .ok_or_else(|| anyhow::anyhow!("model URI must start with 'hf:', got: {uri}"))?;
        let last_slash = rest
            .rfind('/')
            .ok_or_else(|| anyhow::anyhow!("model URI must be 'hf:org/repo/file.gguf', got: {uri}"))?;
        let repo = &rest[..last_slash];
        let filename = &rest[last_slash + 1..];
        if repo.is_empty() || filename.is_empty() || !repo.contains('/') {
            bail!("invalid model URI format: {uri}");
        }
        Ok(Self {
            repo: repo.to_string(),
            filename: filename.to_string(),
        })
    }

    pub fn download_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.repo, self.filename
        )
    }

    /// Local cache path: models_dir/repo--filename (slashes replaced with --)
    pub fn cache_path(&self, models_dir: &Path) -> PathBuf {
        let safe_name = format!("{}--{}", self.repo.replace('/', "--"), self.filename);
        models_dir.join(safe_name)
    }
}

/// Download a file with progress bar and optional SHA256 verification. Retries once on failure.
pub fn download_model(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
    fn try_download(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
        tracing::info!("downloading {} -> {}", url, dest.display());

        let resp = ureq::get(url)
            .call()
            .map_err(|e| anyhow::anyhow!("HTTP GET {url}: {e}"))?;

        let total_size: u64 = resp
            .header("Content-Length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
        );
        pb.set_message(format!(
            "downloading {}",
            dest.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model")
        ));

        // Write to a temp file alongside dest, then rename for crash safety.
        let tmp_path = dest.with_extension("tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)
                .map_err(|e| anyhow::anyhow!("creating {}: {e}", tmp_path.display()))?;
            let mut reader = resp.into_reader();
            let mut buffer = [0u8; 8192];
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                std::io::Write::write_all(&mut file, &buffer[..n])?;
                pb.inc(n as u64);
            }
        }
        pb.finish_with_message("done");

        // Verify hash if provided.
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(&tmp_path)?;
            if actual != expected {
                let _ = std::fs::remove_file(&tmp_path);
                bail!(
                    "SHA-256 mismatch for {}: expected {expected}, got {actual}",
                    dest.display()
                );
            }
        }

        std::fs::rename(&tmp_path, dest)
            .map_err(|e| anyhow::anyhow!("renaming temp file: {e}"))?;

        Ok(())
    }

    // Try once, retry on failure.
    match try_download(url, dest, expected_sha256) {
        Ok(()) => Ok(()),
        Err(first_err) => {
            tracing::warn!("download failed, retrying: {first_err:#}");
            let _ = std::fs::remove_file(dest);
            try_download(url, dest, expected_sha256)
        }
    }
}

/// Compute SHA-256 hex digest of a file.
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Ensure a model is present locally, downloading if not cached.
pub fn ensure_model(uri: &HfModelUri, models_dir: &Path) -> Result<PathBuf> {
    let path = uri.cache_path(models_dir);
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        download_model(&uri.download_url(), &path, None)?;
    }
    Ok(path)
}

/// Default model URIs for the intelligence layer.
pub struct ModelDefaults {
    pub embed_uri: String,
    pub embed_dim: usize,
    pub rerank_uri: String,
    pub expand_uri: String,
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            embed_uri: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".into(),
            embed_dim: 256,
            rerank_uri: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf".into(),
            expand_uri: "hf:Qwen/Qwen3-0.6B-GGUF/qwen3-0.6b-q8_0.gguf".into(),
        }
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

    #[test]
    fn test_parse_hf_uri() {
        let uri = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
        let parsed = HfModelUri::parse(uri).unwrap();
        assert_eq!(parsed.repo, "ggml-org/embeddinggemma-300M-GGUF");
        assert_eq!(parsed.filename, "embeddinggemma-300M-Q8_0.gguf");
        assert_eq!(
            parsed.download_url(),
            "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf"
        );
    }

    #[test]
    fn test_parse_hf_uri_invalid() {
        assert!(HfModelUri::parse("not-a-hf-uri").is_err());
        assert!(HfModelUri::parse("hf:only-repo").is_err());
    }

    #[test]
    fn test_model_cache_path() {
        let uri = HfModelUri::parse("hf:ggml-org/embeddinggemma-300M-GGUF/model.gguf").unwrap();
        let cache_dir = std::path::Path::new("/tmp/models");
        let path = uri.cache_path(cache_dir);
        assert_eq!(
            path,
            cache_dir.join("ggml-org--embeddinggemma-300M-GGUF--model.gguf")
        );
    }

    #[test]
    fn test_model_defaults() {
        let defaults = ModelDefaults::default();
        assert!(defaults.embed_uri.starts_with("hf:"));
        assert_eq!(defaults.embed_dim, 256);
    }
}
