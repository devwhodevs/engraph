use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Tokenizer that can be backed by either HuggingFace tokenizers crate or shimmytok (GGUF-embedded).
pub enum FlexTokenizer {
    HuggingFace(Box<tokenizers::Tokenizer>),
    Gguf(Box<shimmytok::Tokenizer>),
}

impl FlexTokenizer {
    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(t) => {
                let enc = t
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
                Ok(enc.get_ids().to_vec())
            }
            Self::Gguf(t) => {
                let ids = t
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
                Ok(ids)
            }
        }
    }

    /// Count tokens in text.
    pub fn token_count(&self, text: &str) -> usize {
        self.encode(text, false).map(|ids| ids.len()).unwrap_or(0)
    }

    /// Look up a token's ID by string (only available with HuggingFace backend).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::HuggingFace(t) => t.token_to_id(token),
            Self::Gguf(_) => None,
        }
    }

    /// Decode token IDs back to text (only available with HuggingFace backend).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t
                .decode(ids, skip_special)
                .map_err(|e| anyhow::anyhow!("decode: {e}")),
            Self::Gguf(_) => bail!("decode not supported with GGUF tokenizer"),
        }
    }
}

/// Load tokenizer for a model. Tries external tokenizer.json first, falls back to GGUF-embedded.
fn load_tokenizer_for_model(uri: &HfModelUri, models_dir: &Path) -> Result<FlexTokenizer> {
    // First try: external tokenizer.json from candidate repos.
    if let Some(tok) = try_external_tokenizer(uri, models_dir) {
        return Ok(FlexTokenizer::HuggingFace(Box::new(tok)));
    }

    // Fallback: load tokenizer from GGUF file metadata.
    let model_path = uri.cache_path(models_dir);
    if model_path.exists() {
        tracing::info!(
            "no external tokenizer found, loading from GGUF: {}",
            model_path.display()
        );
        let tok = shimmytok::Tokenizer::from_gguf_file(&model_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer from GGUF metadata: {e}"))?;
        return Ok(FlexTokenizer::Gguf(Box::new(tok)));
    }

    bail!(
        "could not find tokenizer for model '{}': no external tokenizer.json \
         and GGUF file not yet downloaded",
        uri.repo
    )
}

/// Load tokenizer as HuggingFace `tokenizers::Tokenizer` specifically.
/// Used by LlamaOrchestrator and LlamaRerank which need decode/token_to_id.
fn load_hf_tokenizer(uri: &HfModelUri, models_dir: &Path) -> Result<tokenizers::Tokenizer> {
    try_external_tokenizer(uri, models_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "could not find tokenizer.json for model '{}'. \
             Orchestrator/reranker models require tokenizer.json (not GGUF-embedded).",
            uri.repo
        )
    })
}

/// Try downloading tokenizer.json from candidate HuggingFace repos.
fn try_external_tokenizer(uri: &HfModelUri, models_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let mut candidates: Vec<String> = vec![uri.repo.clone()];

    // Non-GGUF variant: "org/model-GGUF" → "org/model"
    let base_repo = uri.repo.trim_end_matches("-GGUF").to_string();
    if base_repo != uri.repo {
        candidates.push(base_repo);
    }

    // Known upstream repos for default models (GGUF repos rarely ship tokenizers).
    let model_lower = uri.repo.to_lowercase();
    if model_lower.contains("all-minilm") {
        candidates.push("sentence-transformers/all-MiniLM-L6-v2".to_string());
    } else if model_lower.contains("embeddinggemma") {
        candidates.push("google/embeddinggemma-300m".to_string());
        candidates.push("google/gemma-2b".to_string());
    } else if model_lower.contains("qwen3") {
        let base_name = uri
            .repo
            .rsplit('/')
            .next()
            .unwrap_or("")
            .trim_end_matches("-GGUF")
            .trim_end_matches("-Q8_0-GGUF");
        if !base_name.is_empty() {
            candidates.push(format!("Qwen/{base_name}"));
        }
    }

    for repo in &candidates {
        let tok_uri = HfModelUri {
            repo: repo.clone(),
            filename: "tokenizer.json".to_string(),
        };
        let tok_path = tok_uri.cache_path(models_dir);

        if tok_path.exists()
            && let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path)
        {
            return Some(tok);
        }

        if let Ok(p) = ensure_model(&tok_uri, models_dir)
            && let Ok(tok) = tokenizers::Tokenizer::from_file(&p)
        {
            return Some(tok);
        }
    }

    None
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
            expand_uri: "hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf".into(),
        }
    }
}

// ── LlamaEmbed — GGUF embedding model via llama.cpp ──────────────────────────

/// GGUF embedding model loaded via llama.cpp.
///
/// Loads a quantized embedding model from a GGUF file and produces dense float
/// vectors via llama.cpp's built-in embedding support with mean pooling + L2
/// normalization. Supports Metal acceleration on macOS automatically.
///
/// `LlamaModel` is `Send + Sync`, so this struct is `Send`. `LlamaContext` is
/// `!Send`, so we create it per-call from the stored model and backend.
pub struct LlamaEmbed {
    model: LlamaModel,
    backend: LlamaBackend,
    tokenizer: FlexTokenizer,
    dim: usize,
    prompt_format: PromptFormat,
}

// Safety: LlamaModel is Send+Sync per llama-cpp-2 docs. LlamaBackend is Send+Sync.
// FlexTokenizer contains only Send types (tokenizers::Tokenizer is Send, shimmytok::Tokenizer is Send).
// We never store a LlamaContext (which is !Send) — it is created per-call.
unsafe impl Send for LlamaEmbed {}

impl std::fmt::Debug for LlamaEmbed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaEmbed")
            .field("dim", &self.dim)
            .field("prompt_format", &self.prompt_format)
            .finish()
    }
}

impl LlamaEmbed {
    /// Load a GGUF embedding model from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load tokenizer (try same repo's tokenizer.json, then repo without -GGUF suffix)
    /// 4. Load GGUF model via llama.cpp
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
        let tokenizer = load_tokenizer_for_model(&uri, models_dir)?;

        // Detect prompt format from filename.
        let prompt_format = PromptFormat::detect(&uri.filename);

        // Target output dimensionality.
        let dim = defaults.embed_dim;

        // Initialize llama.cpp backend and load model.
        let backend =
            LlamaBackend::init().map_err(|e| anyhow::anyhow!("initializing llama backend: {e}"))?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("loading GGUF model {}: {e}", model_path.display()))?;

        tracing::info!(
            "loaded LlamaEmbed from {}, target_dim={}",
            uri_str,
            dim
        );

        Ok(Self {
            model,
            backend,
            tokenizer,
            dim,
            prompt_format,
        })
    }

    /// Run embedding inference and return the truncated, L2-normalized embedding.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize using llama.cpp's built-in tokenizer.
        // Use AddBos::Never because PromptFormat already adds <bos> for embeddinggemma.
        let tokens = self
            .model
            .str_to_token(text, AddBos::Never)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        if tokens.is_empty() {
            bail!("tokenizer returned empty token sequence");
        }

        // Create a context with embeddings enabled (per-call, since LlamaContext is !Send).
        // n_ubatch must be >= n_tokens for the encoder, and n_ctx must fit all tokens.
        let n_tokens = tokens.len() as u32;
        let n_ctx = std::num::NonZeroU32::new(n_tokens.max(64) + 16);
        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_n_ctx(n_ctx)
            .with_n_ubatch(n_tokens.max(512))
            .with_n_batch(n_tokens.max(512));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("creating embedding context: {e}"))?;

        // Create batch and add tokens — mark all as outputs for embedding.
        let mut batch = LlamaBatch::new(tokens.len() + 16, 1);
        batch
            .add_sequence(&tokens, 0, true)
            .map_err(|e| anyhow::anyhow!("adding sequence to batch: {e}"))?;

        // Encode (compute embeddings). Use encode() for embedding models.
        ctx.encode(&mut batch)
            .map_err(|e| anyhow::anyhow!("embedding encode failed: {e}"))?;

        // Get embeddings for sequence 0 (mean pooled by llama.cpp).
        let embeddings = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("getting embeddings: {e}"))?;

        // Truncate to target dimensionality.
        let full_dim = embeddings.len();
        let truncated: Vec<f32> = if full_dim > self.dim {
            embeddings[..self.dim].to_vec()
        } else {
            embeddings.to_vec()
        };

        // L2 normalize.
        let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized = if norm > 0.0 {
            truncated.iter().map(|x| x / norm).collect()
        } else {
            truncated
        };

        Ok(normalized)
    }
}

impl EmbedModel for LlamaEmbed {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Process texts sequentially — llama.cpp context is per-call.
        // Apply document prompt format for indexing (asymmetric models need this).
        texts
            .iter()
            .map(|t| {
                let formatted = self.prompt_format.format_document("", t);
                self.embed_text(&formatted)
            })
            .collect()
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        // Apply query prompt format (asymmetric models like embeddinggemma need this).
        let formatted = self.prompt_format.format_query(text);
        self.embed_text(&formatted)
    }

    fn token_count(&self, text: &str) -> usize {
        self.tokenizer.token_count(text)
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
            "how", "does", "the", "a", "an", "is", "are", "was", "to", "in", "on", "for", "with",
            "what", "when", "where",
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

// ── Orchestration JSON parsing ────────────────────────────────────────────────

/// Parse orchestration JSON from LLM output.
/// Handles: raw JSON, JSON embedded in text, and partial/malformed responses.
pub fn parse_orchestration_json(text: &str) -> Result<OrchestrationResult> {
    let json_str = extract_json_object(text)
        .ok_or_else(|| anyhow::anyhow!("no JSON object found in LLM response"))?;

    let parsed: serde_json::Value =
        serde_json::from_str(json_str).with_context(|| "parsing orchestration JSON")?;

    let intent_str = parsed["intent"].as_str().unwrap_or("exploratory");
    let intent = match intent_str {
        "exact" => QueryIntent::Exact,
        "conceptual" => QueryIntent::Conceptual,
        "relationship" => QueryIntent::Relationship,
        _ => QueryIntent::Exploratory,
    };

    let expansions: Vec<String> = parsed["expansions"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if expansions.is_empty() {
        anyhow::bail!("no expansions in orchestration response");
    }

    Ok(OrchestrationResult { intent, expansions })
}

/// Extract the first JSON object ({...}) from text, handling nested braces.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0;
    for (i, b) in text[start..].bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

// ── LlamaOrchestrator — GGUF text generation via llama.cpp ─────────────────────

const ORCHESTRATOR_SYSTEM_PROMPT: &str = r#"You are a search query analyzer. Given a user's search query, classify it and expand it.

Return JSON with:
- "intent": one of "exact", "conceptual", "relationship", "exploratory"
- "expansions": 2-4 alternative phrasings (always include the original query first)

Be concise. Only return the JSON object."#;

/// Quantized Qwen3 model for query orchestration and expansion via llama.cpp.
///
/// Loads a Qwen3 GGUF model and performs autoregressive generation to classify
/// queries and produce expansions. Falls back to `heuristic_orchestrate` if
/// generation or JSON parsing fails. Uses Metal acceleration on macOS automatically.
pub struct LlamaOrchestrator {
    model: LlamaModel,
    backend: LlamaBackend,
    tokenizer: tokenizers::Tokenizer,
}

// Safety: LlamaModel and LlamaBackend are Send+Sync. tokenizers::Tokenizer is Send.
// LlamaContext is created per-call and never stored.
unsafe impl Send for LlamaOrchestrator {}

impl std::fmt::Debug for LlamaOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaOrchestrator").finish()
    }
}

impl LlamaOrchestrator {
    /// Load a Qwen3 GGUF model for orchestration from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load tokenizer from the model repo (or the non-GGUF base repo)
    /// 4. Load GGUF model via llama.cpp
    pub fn new(models_dir: &Path, config: &crate::config::Config) -> Result<Self> {
        let defaults = ModelDefaults::default();
        let uri_str = config
            .models
            .expand
            .as_deref()
            .unwrap_or(&defaults.expand_uri);
        let uri = HfModelUri::parse(uri_str)?;
        let model_path = ensure_model(&uri, models_dir)?;

        // Orchestrator needs HF tokenizer (for decode + token_to_id).
        let tokenizer = load_hf_tokenizer(&uri, models_dir)?;

        let backend =
            LlamaBackend::init().map_err(|e| anyhow::anyhow!("initializing llama backend: {e}"))?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("loading orchestrator model {}: {e}", model_path.display()))?;

        tracing::info!("loaded LlamaOrchestrator from {}", uri_str);

        Ok(Self {
            model,
            backend,
            tokenizer,
        })
    }

    /// Format a chat prompt in Qwen3 ChatML format.
    fn format_prompt(query: &str) -> String {
        format!(
            "<|im_start|>system\n{ORCHESTRATOR_SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n{query}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    /// Run autoregressive generation (greedy decode) up to `max_tokens`.
    /// Returns the generated text (excluding the prompt).
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Tokenize using llama.cpp's tokenizer.
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        if tokens.is_empty() {
            bail!("tokenizer returned empty token sequence");
        }

        // Create context per-call (LlamaContext is !Send).
        let n_ctx = (tokens.len() + max_tokens + 16) as u32;
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("creating orchestrator context: {e}"))?;

        // Process prompt tokens in a batch.
        let mut batch = LlamaBatch::new(tokens.len() + max_tokens + 16, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| anyhow::anyhow!("adding prompt token to batch: {e}"))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("prompt decode failed: {e}"))?;

        // Autoregressive generation loop.
        let mut sampler = LlamaSampler::greedy();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            // Check for end-of-generation.
            if self.model.is_eog_token(new_token) {
                break;
            }

            generated_tokens.push(new_token.0 as u32);

            // Add token to batch for next iteration.
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| anyhow::anyhow!("adding generated token to batch: {e}"))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("generation decode failed: {e}"))?;
        }

        // Decode generated tokens back to text using HF tokenizer.
        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("decoding generated tokens: {e}"))?;
        Ok(text)
    }
}

impl OrchestratorModel for LlamaOrchestrator {
    fn orchestrate(&mut self, query: &str) -> Result<OrchestrationResult> {
        let prompt = Self::format_prompt(query);

        match self.generate(&prompt, 256) {
            Ok(text) => match parse_orchestration_json(&text) {
                Ok(result) => Ok(result),
                Err(e) => {
                    tracing::warn!(
                        "orchestrator JSON parse failed, falling back to heuristic: {e:#}"
                    );
                    Ok(heuristic_orchestrate(query))
                }
            },
            Err(e) => {
                tracing::warn!("orchestrator generation failed, falling back to heuristic: {e:#}");
                Ok(heuristic_orchestrate(query))
            }
        }
    }
}

// ── LlamaRerank — GGUF cross-encoder reranker via llama.cpp ─────────────────────

/// Format query+document for cross-encoder reranking.
pub fn format_reranker_input(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the document is relevant to the search query. \
         Respond only with \"Yes\" or \"No\".<|im_end|>\n\
         <|im_start|>user\nSearch query: {query}\nDocument: {document}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Quantized Qwen3 cross-encoder for reranking search results via llama.cpp.
///
/// Loads a Qwen3-Reranker GGUF model and scores (query, document) pairs by
/// running a single forward pass and extracting Yes/No logit probabilities.
/// Unlike `LlamaOrchestrator`, this does NOT do autoregressive generation —
/// just one pass through the full input to get logits at the last position.
pub struct LlamaRerank {
    model: LlamaModel,
    backend: LlamaBackend,
    yes_token_id: i32,
    no_token_id: i32,
}

// Safety: LlamaModel and LlamaBackend are Send+Sync.
// LlamaContext is created per-call and never stored.
unsafe impl Send for LlamaRerank {}

impl std::fmt::Debug for LlamaRerank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaRerank")
            .field("yes_token_id", &self.yes_token_id)
            .field("no_token_id", &self.no_token_id)
            .finish()
    }
}

impl LlamaRerank {
    /// Load a Qwen3-Reranker GGUF model from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults::default().rerank_uri`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load tokenizer from the model repo to look up Yes/No token IDs
    /// 4. Load GGUF model via llama.cpp
    pub fn new(models_dir: &Path, config: &crate::config::Config) -> Result<Self> {
        let defaults = ModelDefaults::default();
        let uri_str = config
            .models
            .rerank
            .as_deref()
            .unwrap_or(&defaults.rerank_uri);
        let uri = HfModelUri::parse(uri_str)?;
        let model_path = ensure_model(&uri, models_dir)?;

        // Reranker needs HF tokenizer to look up Yes/No token IDs.
        let hf_tokenizer = load_hf_tokenizer(&uri, models_dir)?;

        let yes_token_id = hf_tokenizer
            .token_to_id("Yes")
            .ok_or_else(|| anyhow::anyhow!("tokenizer has no 'Yes' token"))?
            as i32;
        let no_token_id = hf_tokenizer
            .token_to_id("No")
            .ok_or_else(|| anyhow::anyhow!("tokenizer has no 'No' token"))?
            as i32;

        let backend =
            LlamaBackend::init().map_err(|e| anyhow::anyhow!("initializing llama backend: {e}"))?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("loading reranker model {}: {e}", model_path.display()))?;

        tracing::info!(
            "loaded LlamaRerank from {}, yes_id={}, no_id={}",
            uri_str,
            yes_token_id,
            no_token_id
        );

        Ok(Self {
            model,
            backend,
            yes_token_id,
            no_token_id,
        })
    }
}

impl RerankModel for LlamaRerank {
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32> {
        let input_text = format_reranker_input(query, document);

        // Tokenize using llama.cpp's built-in tokenizer.
        let tokens = self
            .model
            .str_to_token(&input_text, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        if tokens.is_empty() {
            bail!("tokenizer returned empty token sequence");
        }

        // Create context per-call (LlamaContext is !Send).
        let n_ctx = (tokens.len() + 16) as u32;
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("creating reranker context: {e}"))?;

        // Create batch with all tokens; mark last as logit-producing.
        let mut batch = LlamaBatch::new(tokens.len() + 16, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| anyhow::anyhow!("adding token to reranker batch: {e}"))?;
        }

        // Single forward pass through the full input.
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("reranker decode failed: {e}"))?;

        // Get logits for the last token position.
        let logits = ctx.get_logits_ith(batch.n_tokens() - 1);

        // Extract Yes/No logits and compute softmax probability.
        let yes_logit = logits[self.yes_token_id as usize];
        let no_logit = logits[self.no_token_id as usize];

        let max_logit = yes_logit.max(no_logit);
        let yes_exp = (yes_logit - max_logit).exp();
        let no_exp = (no_logit - max_logit).exp();
        let score = yes_exp / (yes_exp + no_exp);

        Ok(score)
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
        assert!(
            defaults.embed_uri.contains("embeddinggemma"),
            "default embed model should be embeddinggemma"
        );
    }

    // ── LlamaEmbed / PromptFormat tests ────────────────────────────────────

    #[test]
    fn test_llama_embed_struct_exists() {
        fn assert_embed_model<E: EmbedModel>(_e: &E) {}
        let mock = MockLlm::new(256);
        assert_embed_model(&mock);
        // LlamaEmbed also implements EmbedModel — verified at compile time.
        // We can't instantiate LlamaEmbed without a real GGUF model,
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
        assert!(
            result
                .expansions
                .contains(&"how does auth work".to_string())
        );
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

    // ── parse_orchestration_json tests ───────────────────────────────────────

    #[test]
    fn test_parse_orchestration_json_valid() {
        let json =
            r#"{"intent": "conceptual", "expansions": ["auth work", "authentication design"]}"#;
        let result = parse_orchestration_json(json).unwrap();
        assert_eq!(result.intent, QueryIntent::Conceptual);
        assert_eq!(result.expansions.len(), 2);
    }

    #[test]
    fn test_parse_orchestration_json_with_surrounding_text() {
        let text =
            "Here is the analysis:\n{\"intent\": \"exact\", \"expansions\": [\"BRE-1234\"]}\nDone.";
        let result = parse_orchestration_json(text).unwrap();
        assert_eq!(result.intent, QueryIntent::Exact);
    }

    #[test]
    fn test_parse_orchestration_json_invalid() {
        let bad = "not json at all";
        assert!(parse_orchestration_json(bad).is_err());
    }

    #[test]
    fn test_parse_orchestration_json_unknown_intent() {
        let json = r#"{"intent": "unknown_type", "expansions": ["query"]}"#;
        let result = parse_orchestration_json(json).unwrap();
        assert_eq!(result.intent, QueryIntent::Exploratory);
    }

    #[test]
    fn test_extract_json_object_nested() {
        let text = r#"prefix {"a": {"b": 1}} suffix"#;
        let extracted = extract_json_object(text).unwrap();
        assert_eq!(extracted, r#"{"a": {"b": 1}}"#);
    }

    #[test]
    fn test_extract_json_object_none() {
        assert!(extract_json_object("no braces here").is_none());
    }

    #[test]
    fn test_extract_json_object_unclosed() {
        assert!(extract_json_object("{ open but never closed").is_none());
    }

    #[test]
    fn test_parse_orchestration_json_empty_expansions() {
        let json = r#"{"intent": "exact", "expansions": []}"#;
        assert!(parse_orchestration_json(json).is_err());
    }

    #[test]
    fn test_parse_orchestration_json_missing_expansions() {
        let json = r#"{"intent": "exact"}"#;
        assert!(parse_orchestration_json(json).is_err());
    }

    // ── LlamaOrchestrator tests ─────────────────────────────────────────────

    #[test]
    fn test_llama_orchestrator_format_prompt() {
        let prompt = LlamaOrchestrator::format_prompt("how does auth work");
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("how does auth work"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_llama_orchestrator_implements_trait() {
        // Compile-time check: LlamaOrchestrator implements OrchestratorModel.
        fn assert_orchestrator<O: OrchestratorModel>() {}
        assert_orchestrator::<LlamaOrchestrator>();
    }

    // ── LlamaRerank tests ──────────────────────────────────────────────────

    #[test]
    fn test_format_reranker_input() {
        let formatted = format_reranker_input("auth system", "The auth module handles OAuth");
        assert!(formatted.contains("auth system"));
        assert!(formatted.contains("The auth module handles OAuth"));
        assert!(formatted.contains("Respond only with"));
    }

    #[test]
    fn test_llama_rerank_trait_compliance() {
        // Verify MockLlm still satisfies RerankModel.
        fn assert_rerank<R: RerankModel>(_r: &R) {}
        let mock = MockLlm::new(256);
        assert_rerank(&mock);
    }
}
