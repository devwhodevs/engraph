use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};

// ── Device selection ─────────────────────────────────────────────────────────

/// Select best available device: Metal on macOS (with `metal` feature), CPU elsewhere.
fn select_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }
    Ok(Device::Cpu)
}

// ── Prompt format ────────────────────────────────────────────────────────────

/// Model-family-specific prompt templates for embedding models.
#[derive(Debug, Clone)]
pub enum PromptFormat {
    /// Google embeddinggemma family: uses `<bos>search_query:` / `<bos>search_document:` prefixes.
    EmbeddingGemma,
    /// Qwen embedding family: uses `Instruct:` / `Query:` format.
    QwenEmbedding,
    /// No special formatting — pass text as-is.
    Raw,
}

impl PromptFormat {
    /// Auto-detect prompt format from a GGUF filename.
    pub fn detect(filename: &str) -> Self {
        let lower = filename.to_lowercase();
        if lower.contains("embeddinggemma") {
            Self::EmbeddingGemma
        } else if lower.contains("qwen") && lower.contains("embed") {
            Self::QwenEmbedding
        } else {
            Self::Raw
        }
    }

    /// Format text for a search query.
    pub fn format_query(&self, query: &str) -> String {
        match self {
            Self::EmbeddingGemma => format!("<bos>search_query: {query}"),
            Self::QwenEmbedding => {
                format!("Instruct: Retrieve relevant passages\nQuery: {query}")
            }
            Self::Raw => query.to_string(),
        }
    }

    /// Format text for a document to be indexed.
    pub fn format_document(&self, title: &str, text: &str) -> String {
        match self {
            Self::EmbeddingGemma => format!("<bos>search_document: {title} {text}"),
            Self::QwenEmbedding | Self::Raw => format!("{title}\n{text}"),
        }
    }
}

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
        let last_slash = rest.rfind('/').ok_or_else(|| {
            anyhow::anyhow!("model URI must be 'hf:org/repo/file.gguf', got: {uri}")
        })?;
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
            dest.file_name().and_then(|n| n.to_str()).unwrap_or("model")
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

        std::fs::rename(&tmp_path, dest).map_err(|e| anyhow::anyhow!("renaming temp file: {e}"))?;

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
            rerank_uri: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
                .into(),
            expand_uri: "hf:Qwen/Qwen3-0.6B-GGUF/qwen3-0.6b-q8_0.gguf".into(),
        }
    }
}

// ── CandleEmbed — GGUF embedding model via candle ──────────────────────────

/// Quantized matrix multiplication wrapper (mirrors candle-transformers pattern).
#[derive(Debug, Clone)]
struct CandleQMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl CandleQMatMul {
    fn from_qtensor(qtensor: candle_core::quantized::QTensor) -> candle_core::Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(xs)
    }
}

/// Single transformer layer for the embedding model.
#[derive(Debug, Clone)]
struct EmbedLayer {
    attention_wq: CandleQMatMul,
    attention_wk: CandleQMatMul,
    attention_wv: CandleQMatMul,
    attention_wo: CandleQMatMul,
    attention_q_norm: candle_transformers::quantized_nn::RmsNorm,
    attention_k_norm: candle_transformers::quantized_nn::RmsNorm,
    attention_norm: candle_transformers::quantized_nn::RmsNorm,
    post_attention_norm: candle_transformers::quantized_nn::RmsNorm,
    ffn_norm: candle_transformers::quantized_nn::RmsNorm,
    post_ffn_norm: candle_transformers::quantized_nn::RmsNorm,
    ffn_gate: CandleQMatMul,
    ffn_up: CandleQMatMul,
    ffn_down: CandleQMatMul,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,
    rotary_sin: Tensor,
    rotary_cos: Tensor,
}

impl EmbedLayer {
    /// Bidirectional forward pass — no causal mask, no KV cache.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // --- Attention block ---
        let residual = x;
        let x = self.attention_norm.forward(x)?;

        let q = self.attention_wq.forward(&x)?;
        let k = self.attention_wk.forward(&x)?;
        let v = self.attention_wv.forward(&x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        // Apply rotary embeddings (truncated to seq_len).
        let q = Self::apply_rotary(&q, &self.rotary_cos, &self.rotary_sin, seq_len)?;
        let k = Self::apply_rotary(&k, &self.rotary_cos, &self.rotary_sin, seq_len)?;

        // Repeat KV heads for GQA.
        let n_rep = self.n_head / self.n_kv_head;
        let k = candle_transformers::utils::repeat_kv(k, n_rep)?;
        let v = candle_transformers::utils::repeat_kv(v, n_rep)?;

        // Scaled dot-product attention — BIDIRECTIONAL (no mask).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;
        let attn_output = self.attention_wo.forward(&attn_output)?;
        let x = self.post_attention_norm.forward(&attn_output)?;
        let x = (x + residual)?;

        // --- FFN block ---
        let residual = &x;
        let h = self.ffn_norm.forward(&x)?;
        let gate = self.ffn_gate.forward(&h)?;
        let up = self.ffn_up.forward(&h)?;
        let h = (candle_nn::ops::silu(&gate)? * up)?;
        let h = self.ffn_down.forward(&h)?;
        let h = self.post_ffn_norm.forward(&h)?;
        h + residual
    }

    /// Apply rotary embeddings to a [batch, heads, seq, dim] tensor.
    fn apply_rotary(
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seq_len: usize,
    ) -> candle_core::Result<Tensor> {
        let cos = cos.i(..seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.i(..seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let dim = x.dim(D::Minus1)?;
        let half = dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let out = (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
        Ok(out)
    }
}

/// GGUF embedding model loaded via candle.
///
/// Loads a quantized Gemma-family embedding model (e.g., embeddinggemma-300M)
/// from a GGUF file and produces dense float vectors via bidirectional attention
/// + mean pooling + L2 normalization.
pub struct CandleEmbed {
    layers: Vec<EmbedLayer>,
    tok_embeddings: Embedding,
    norm: candle_transformers::quantized_nn::RmsNorm,
    embedding_length: usize,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dim: usize,
    prompt_format: PromptFormat,
}

impl std::fmt::Debug for CandleEmbed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbed")
            .field("dim", &self.dim)
            .field("embedding_length", &self.embedding_length)
            .field("num_layers", &self.layers.len())
            .field("prompt_format", &self.prompt_format)
            .finish()
    }
}

impl CandleEmbed {
    /// Load a GGUF embedding model from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load tokenizer (try same repo's tokenizer.json, then repo without -GGUF suffix)
    /// 4. Load GGUF and build layer structs for bidirectional embedding
    /// 5. Detect prompt format from filename
    pub fn new(models_dir: &Path, config: &crate::config::Config) -> Result<Self> {
        let defaults = ModelDefaults::default();
        let uri_str = config
            .models
            .embed
            .as_deref()
            .unwrap_or(&defaults.embed_uri);
        let uri = HfModelUri::parse(uri_str)?;
        let model_path = ensure_model(&uri, models_dir)?;

        // Load tokenizer: try from the same HF repo, then from the non-GGUF variant.
        let tokenizer = Self::load_tokenizer(&uri, models_dir)?;

        // Detect prompt format from filename.
        let prompt_format = PromptFormat::detect(&uri.filename);

        // Target output dimensionality.
        let dim = defaults.embed_dim;

        // Load GGUF and build model.
        let device = select_device()?;
        let (layers, tok_embeddings, norm, embedding_length) =
            Self::load_gguf(&model_path, &device)?;

        tracing::info!(
            "loaded CandleEmbed: {} layers, embedding_length={}, target_dim={}, device={:?}",
            layers.len(),
            embedding_length,
            dim,
            device
        );

        Ok(Self {
            layers,
            tok_embeddings,
            norm,
            embedding_length,
            tokenizer,
            device,
            dim,
            prompt_format,
        })
    }

    /// Try to load tokenizer.json from the same HF repo, or from repo without "-GGUF" suffix.
    fn load_tokenizer(uri: &HfModelUri, models_dir: &Path) -> Result<tokenizers::Tokenizer> {
        // Try 1: tokenizer.json from the same repo.
        let tok_uri = HfModelUri {
            repo: uri.repo.clone(),
            filename: "tokenizer.json".to_string(),
        };
        let tok_path = tok_uri.cache_path(models_dir);
        if tok_path.exists() {
            return tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| {
                anyhow::anyhow!("loading tokenizer from {}: {e}", tok_path.display())
            });
        }

        // Try 2: download from the same repo.
        if let Ok(p) = ensure_model(&tok_uri, models_dir) {
            return tokenizers::Tokenizer::from_file(&p)
                .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", p.display()));
        }

        // Try 3: non-GGUF variant of the repo (e.g., "org/model-GGUF" -> "org/model").
        let base_repo = uri.repo.trim_end_matches("-GGUF").to_string();
        if base_repo != uri.repo {
            let base_tok_uri = HfModelUri {
                repo: base_repo,
                filename: "tokenizer.json".to_string(),
            };
            if let Ok(p) = ensure_model(&base_tok_uri, models_dir) {
                return tokenizers::Tokenizer::from_file(&p)
                    .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", p.display()));
            }
        }

        bail!(
            "could not find or download tokenizer for model repo '{}'",
            uri.repo
        );
    }

    /// Load GGUF file and construct layer structs for bidirectional embedding.
    fn load_gguf(
        path: &Path,
        device: &Device,
    ) -> Result<(
        Vec<EmbedLayer>,
        Embedding,
        candle_transformers::quantized_nn::RmsNorm,
        usize,
    )> {
        use candle_core::quantized::gguf_file;

        let mut file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("opening GGUF {}: {e}", path.display()))?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("reading GGUF {}: {e}", path.display()))?;

        // Detect architecture prefix (same probe as candle-transformers quantized_gemma3).
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| -> Result<&gguf_file::Value> {
            let key = format!("{prefix}.{s}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("cannot find {key} in GGUF metadata"))
        };

        let head_count = md_get("attention.head_count")?
            .to_u32()
            .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?
            .to_u32()
            .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
        let block_count = md_get("block_count")?
            .to_u32()
            .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
        let embedding_length = md_get("embedding_length")?
            .to_u32()
            .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
        let key_length = md_get("attention.key_length")?
            .to_u32()
            .map_err(|e| anyhow::anyhow!("{e}"))? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?
            .to_f32()
            .map_err(|e| anyhow::anyhow!("{e}"))? as f64;
        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|v| v.to_f32().map_err(|e| anyhow::anyhow!("{e}")))
            .unwrap_or(10_000.0);

        let q_dim = head_count * key_length;

        // Build rotary embedding tables (shared by all layers for the base freq).
        let max_seq_len: usize = 8192; // Sufficient for embedding inputs.
        let (rotary_sin, rotary_cos) =
            Self::build_rotary_tables(key_length, rope_freq_base, max_seq_len, device)?;

        // Load token embeddings.
        let tok_embd = ct
            .tensor(&mut file, "token_embd.weight", device)
            .map_err(|e| anyhow::anyhow!("loading token_embd.weight: {e}"))?;
        let tok_embd_deq = tok_embd
            .dequantize(device)
            .map_err(|e| anyhow::anyhow!("dequantizing token_embd: {e}"))?;
        let tok_embeddings = Embedding::new(tok_embd_deq, embedding_length);

        // Final norm.
        let norm_qt = ct
            .tensor(&mut file, "output_norm.weight", device)
            .map_err(|e| anyhow::anyhow!("loading output_norm.weight: {e}"))?;
        let norm = candle_transformers::quantized_nn::RmsNorm::from_qtensor(norm_qt, rms_norm_eps)
            .map_err(|e| anyhow::anyhow!("creating RmsNorm: {e}"))?;

        // Load transformer layers.
        let mut layers = Vec::with_capacity(block_count);
        for idx in 0..block_count {
            let p = format!("blk.{idx}");

            // Helper: load a quantized weight tensor as QMatMul.
            macro_rules! load_q {
                ($name:expr) => {{
                    let full = format!("{}.{}", p, $name);
                    let qt = ct
                        .tensor(&mut file, &full, device)
                        .map_err(|e| anyhow::anyhow!("loading {full}: {e}"))?;
                    CandleQMatMul::from_qtensor(qt)
                        .map_err(|e| anyhow::anyhow!("QMatMul for {full}: {e}"))?
                }};
            }

            // Helper: load a norm weight tensor as RmsNorm.
            macro_rules! load_norm {
                ($name:expr) => {{
                    let full = format!("{}.{}", p, $name);
                    let qt = ct
                        .tensor(&mut file, &full, device)
                        .map_err(|e| anyhow::anyhow!("loading {full}: {e}"))?;
                    candle_transformers::quantized_nn::RmsNorm::from_qtensor(qt, rms_norm_eps)
                        .map_err(|e| anyhow::anyhow!("RmsNorm for {full}: {e}"))?
                }};
            }

            layers.push(EmbedLayer {
                attention_wq: load_q!("attn_q.weight"),
                attention_wk: load_q!("attn_k.weight"),
                attention_wv: load_q!("attn_v.weight"),
                attention_wo: load_q!("attn_output.weight"),
                attention_q_norm: load_norm!("attn_q_norm.weight"),
                attention_k_norm: load_norm!("attn_k_norm.weight"),
                attention_norm: load_norm!("attn_norm.weight"),
                post_attention_norm: load_norm!("post_attention_norm.weight"),
                ffn_norm: load_norm!("ffn_norm.weight"),
                post_ffn_norm: load_norm!("post_ffw_norm.weight"),
                ffn_gate: load_q!("ffn_gate.weight"),
                ffn_up: load_q!("ffn_up.weight"),
                ffn_down: load_q!("ffn_down.weight"),
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                rotary_sin: rotary_sin.clone(),
                rotary_cos: rotary_cos.clone(),
            });
        }

        Ok((layers, tok_embeddings, norm, embedding_length))
    }

    /// Build sin/cos rotary embedding tables of shape [max_seq_len, head_dim].
    fn build_rotary_tables(
        head_dim: usize,
        freq_base: f32,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let half = head_dim / 2;
        let theta: Vec<f32> = (0..half)
            .map(|i| 1.0 / freq_base.powf(i as f32 / half as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)
            .map_err(|e| anyhow::anyhow!("rotary theta: {e}"))?;
        let positions = Tensor::arange(0, max_seq_len as u32, device)
            .map_err(|e| anyhow::anyhow!("rotary positions: {e}"))?
            .to_dtype(DType::F32)
            .map_err(|e| anyhow::anyhow!("rotary positions dtype: {e}"))?;
        // [max_seq_len, half]
        let freqs = positions
            .unsqueeze(1)
            .map_err(|e| anyhow::anyhow!("rotary unsqueeze: {e}"))?
            .broadcast_mul(&theta.unsqueeze(0).map_err(|e| anyhow::anyhow!("{e}"))?)
            .map_err(|e| anyhow::anyhow!("rotary freqs: {e}"))?;
        // Duplicate to [max_seq_len, head_dim] to match x1,x2 concatenation.
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)
            .map_err(|e| anyhow::anyhow!("rotary cat: {e}"))?;
        let sin = freqs
            .sin()
            .map_err(|e| anyhow::anyhow!("rotary sin: {e}"))?;
        let cos = freqs
            .cos()
            .map_err(|e| anyhow::anyhow!("rotary cos: {e}"))?;
        Ok((sin, cos))
    }

    /// Run a bidirectional forward pass and return the mean-pooled, truncated,
    /// L2-normalized embedding.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let token_ids = encoding.get_ids();
        if token_ids.is_empty() {
            bail!("tokenizer returned empty token sequence");
        }

        let input = Tensor::new(token_ids, &self.device)
            .map_err(|e| anyhow::anyhow!("creating input tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("unsqueeze: {e}"))?;

        // Token embeddings, scaled by sqrt(embedding_length) (Gemma convention).
        let mut hidden = self
            .tok_embeddings
            .forward(&input)
            .map_err(|e| anyhow::anyhow!("token embedding forward: {e}"))?;
        hidden = (hidden * (self.embedding_length as f64).sqrt())
            .map_err(|e| anyhow::anyhow!("scaling embeddings: {e}"))?;

        // Forward through all transformer layers (bidirectional — no causal mask).
        for layer in &self.layers {
            hidden = layer
                .forward(&hidden)
                .map_err(|e| anyhow::anyhow!("layer forward: {e}"))?;
        }

        // Final layer norm.
        hidden = self
            .norm
            .forward(&hidden)
            .map_err(|e| anyhow::anyhow!("final norm: {e}"))?;

        // Mean pool across sequence dimension: [1, seq_len, hidden] -> [1, hidden].
        let seq_len = hidden
            .dim(1)
            .map_err(|e| anyhow::anyhow!("getting seq dim: {e}"))?;
        let pooled = (hidden.sum(1).map_err(|e| anyhow::anyhow!("sum: {e}"))? / (seq_len as f64))
            .map_err(|e| anyhow::anyhow!("mean div: {e}"))?;

        // Squeeze batch dimension: [1, hidden] -> [hidden].
        let pooled = pooled
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("squeeze: {e}"))?;

        // Truncate to target dimensionality.
        let full_dim = pooled
            .dim(0)
            .map_err(|e| anyhow::anyhow!("dim check: {e}"))?;
        let truncated = if full_dim > self.dim {
            pooled
                .narrow(0, 0, self.dim)
                .map_err(|e| anyhow::anyhow!("truncate: {e}"))?
        } else {
            pooled
        };

        // L2 normalize.
        let norm_val = truncated
            .sqr()
            .map_err(|e| anyhow::anyhow!("sqr: {e}"))?
            .sum_all()
            .map_err(|e| anyhow::anyhow!("sum_all: {e}"))?
            .sqrt()
            .map_err(|e| anyhow::anyhow!("sqrt: {e}"))?;
        let norm_scalar: f32 = norm_val
            .to_scalar()
            .map_err(|e| anyhow::anyhow!("norm scalar: {e}"))?;

        let normalized = if norm_scalar > 0.0 {
            (truncated / norm_scalar as f64).map_err(|e| anyhow::anyhow!("normalize: {e}"))?
        } else {
            truncated
        };

        let vec: Vec<f32> = normalized
            .to_vec1()
            .map_err(|e| anyhow::anyhow!("to_vec1: {e}"))?;
        Ok(vec)
    }
}

impl EmbedModel for CandleEmbed {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Process texts sequentially — candle quantized ops are single-threaded.
        texts.iter().map(|t| self.embed_text(t)).collect()
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_text(text)
    }

    fn token_count(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().len())
            .unwrap_or(text.len() / 4 + 1)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ── Heuristic orchestrator ───────────────────────────────────────────────────

/// Heuristic orchestrator — no LLM, fast path when intelligence is off.
pub fn heuristic_orchestrate(query: &str) -> OrchestrationResult {
    let trimmed = query.trim();

    // Exact: docids (#abc123) or ticket IDs (ABC-1234)
    if trimmed.starts_with('#') && trimmed.len() <= 8 {
        return OrchestrationResult {
            intent: QueryIntent::Exact,
            expansions: vec![trimmed.to_string()],
        };
    }
    // Ticket ID pattern: PREFIX-1234
    if trimmed.contains('-')
        && let Some(prefix) = trimmed.split('-').next()
        && prefix.chars().all(|c| c.is_ascii_uppercase())
    {
        let after = trimmed.split('-').nth(1).unwrap_or("");
        if after.chars().all(|c| c.is_ascii_digit()) && !after.is_empty() {
            return OrchestrationResult {
                intent: QueryIntent::Exact,
                expansions: vec![trimmed.to_string()],
            };
        }
    }

    // Relationship: "who" queries
    let lower = trimmed.to_lowercase();
    if lower.starts_with("who ") || lower.contains(" who ") {
        return OrchestrationResult {
            intent: QueryIntent::Relationship,
            expansions: vec![trimmed.to_string()],
        };
    }

    // Default: exploratory with word splitting for multi-word queries
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let mut expansions = vec![trimmed.to_string()];
    if words.len() > 2 {
        let stopwords = [
            "how", "does", "the", "a", "an", "is", "are", "was", "to", "in", "on", "for",
            "with", "what", "when", "where",
        ];
        for word in &words {
            if word.len() > 2 && !stopwords.contains(&word.to_lowercase().as_str()) {
                expansions.push(word.to_string());
            }
        }
    }

    OrchestrationResult {
        intent: QueryIntent::Exploratory,
        expansions,
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

    // ── CandleEmbed / PromptFormat tests ────────────────────────────────────

    #[test]
    fn test_candle_embed_struct_exists() {
        fn assert_embed_model<E: EmbedModel>(_e: &E) {}
        let mock = MockLlm::new(256);
        assert_embed_model(&mock);
        // CandleEmbed also implements EmbedModel — verified at compile time.
        // We can't instantiate CandleEmbed without a real GGUF model,
        // but the trait bound compiles.
    }

    #[test]
    fn test_prompt_format_embeddinggemma_query() {
        let fmt = PromptFormat::detect("embeddinggemma-300M-Q8_0.gguf");
        let formatted = fmt.format_query("how does auth work");
        assert!(formatted.contains("search_query"));
        assert!(formatted.contains("how does auth work"));
    }

    #[test]
    fn test_prompt_format_embeddinggemma_document() {
        let fmt = PromptFormat::detect("embeddinggemma-300M-Q8_0.gguf");
        let formatted = fmt.format_document("Note Title", "some content");
        assert!(formatted.contains("Note Title"));
        assert!(formatted.contains("some content"));
        assert!(formatted.contains("search_document"));
    }

    #[test]
    fn test_prompt_format_unknown_model() {
        let fmt = PromptFormat::detect("unknown-model.gguf");
        let formatted = fmt.format_query("test query");
        assert_eq!(formatted, "test query");
    }

    #[test]
    fn test_prompt_format_qwen_embedding() {
        let fmt = PromptFormat::detect("qwen-embed-v2.gguf");
        let formatted = fmt.format_query("find me something");
        assert!(formatted.contains("Instruct:"));
        assert!(formatted.contains("Query:"));
        assert!(formatted.contains("find me something"));
    }

    #[test]
    fn test_prompt_format_qwen_document() {
        let fmt = PromptFormat::detect("qwen-embed-v2.gguf");
        let formatted = fmt.format_document("Title", "Body text");
        assert_eq!(formatted, "Title\nBody text");
    }

    #[test]
    fn test_prompt_format_raw_document() {
        let fmt = PromptFormat::detect("random-model.gguf");
        let formatted = fmt.format_document("Title", "Body");
        assert_eq!(formatted, "Title\nBody");
    }

    #[test]
    fn test_select_device_returns_cpu_by_default() {
        // Without the `metal` feature, select_device should return CPU.
        let device = select_device().unwrap();
        // On CI/test without metal feature, this should be CPU.
        // With metal feature on macOS, it could be Metal — both are valid.
        let _ = device; // Just verify it doesn't error.
    }

    // ── heuristic_orchestrate tests ──────────────────────────────────────────

    #[test]
    fn test_heuristic_orchestrate_single_word() {
        let result = heuristic_orchestrate("auth");
        assert_eq!(result.intent, QueryIntent::Exploratory);
        assert_eq!(result.expansions, vec!["auth"]);
    }

    #[test]
    fn test_heuristic_orchestrate_multi_word() {
        let result = heuristic_orchestrate("how does auth work");
        assert_eq!(result.intent, QueryIntent::Exploratory);
        assert!(result.expansions.contains(&"how does auth work".to_string()));
        assert!(result.expansions.len() > 1);
    }

    #[test]
    fn test_heuristic_orchestrate_docid() {
        let result = heuristic_orchestrate("#ab12cd");
        assert_eq!(result.intent, QueryIntent::Exact);
    }

    #[test]
    fn test_heuristic_orchestrate_ticket_id() {
        let result = heuristic_orchestrate("BRE-1234");
        assert_eq!(result.intent, QueryIntent::Exact);
    }

    #[test]
    fn test_heuristic_orchestrate_who_query() {
        let result = heuristic_orchestrate("who works on checkout");
        assert_eq!(result.intent, QueryIntent::Relationship);
    }
}
