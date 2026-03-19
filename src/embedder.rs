use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tracing::info;

const MODEL_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";
/// SHA-256 of the ONNX model file. Set to empty string to skip verification
/// until we can compute the real hash from a download.
const MODEL_SHA256: &str = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452";
pub const EMBEDDING_DIM: usize = 384;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Create a new Embedder, downloading the model and tokenizer into
    /// `models_dir` if they are not already present.
    pub fn new(models_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(models_dir)
            .with_context(|| format!("creating models dir {}", models_dir.display()))?;

        let model_path = models_dir.join("model.onnx");
        let tokenizer_path = models_dir.join("tokenizer.json");

        // Download model if missing.
        if !model_path.exists() {
            download_file(MODEL_URL, &model_path, Some(MODEL_SHA256))?;
        }

        // Download tokenizer if missing.
        if !tokenizer_path.exists() {
            download_file(TOKENIZER_URL, &tokenizer_path, None)?;
        }

        // Verify model hash.
        verify_sha256(&model_path, MODEL_SHA256)?;

        let session = Session::builder()
            .with_context(|| "creating ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("loading ONNX model from {}", model_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        info!("embedder loaded from {}", models_dir.display());

        Ok(Self { session, tokenizer })
    }

    /// Embed a batch of texts, returning one L2-normalised vector per text.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build padded input arrays.
        let mut input_ids_vec = vec![0i64; batch_size * max_len];
        let mut attention_mask_vec = vec![0i64; batch_size * max_len];
        let mut token_type_ids_vec = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let type_ids = enc.get_type_ids();
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids_vec[i * max_len + j] = id as i64;
                attention_mask_vec[i * max_len + j] = m as i64;
                if j < type_ids.len() {
                    token_type_ids_vec[i * max_len + j] = type_ids[j] as i64;
                }
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, max_len), input_ids_vec)?;
        let attention_mask = Array2::from_shape_vec((batch_size, max_len), attention_mask_vec)?;
        let token_type_ids = Array2::from_shape_vec((batch_size, max_len), token_type_ids_vec)?;

        let input_ids_tensor = Tensor::from_array(input_ids)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask.clone())?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;

        // The model outputs token_embeddings of shape (batch, seq_len, 384).
        // We need mean pooling with the attention mask.
        let token_embeddings = outputs[0].try_extract_array::<f32>()?;
        let token_embeddings = token_embeddings.to_owned(); // into owned array

        // Mean pooling: sum embeddings where attention_mask == 1, divide by count.
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut sum = vec![0f32; EMBEDDING_DIM];
            let mut count = 0f32;
            for j in 0..max_len {
                if attention_mask[[i, j]] == 1 {
                    count += 1.0;
                    for k in 0..EMBEDDING_DIM {
                        sum[k] += token_embeddings[[i, j, k]];
                    }
                }
            }
            if count > 0.0 {
                for v in &mut sum {
                    *v /= count;
                }
            }
            let normalized = normalize_vector(&sum);
            results.push(normalized);
        }

        Ok(results)
    }

    /// Embed a single text.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text])?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty result from embed_batch"))
    }

    /// Return the number of tokens in a text string.
    pub fn token_count(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
    }
}

/// L2-normalize a vector. Returns a zero vector if input norm is zero.
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
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

/// Verify that a file matches an expected SHA-256 hash.
fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual != expected {
        bail!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

/// Compute SHA-256 hex digest of a byte slice.
#[cfg(test)]
fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Download a file from `url` to `dest`, optionally verifying SHA-256.
/// Retries once on failure.
fn download_file(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
    fn try_download(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
        info!("downloading {} -> {}", url, dest.display());

        let resp = ureq::get(url)
            .call()
            .with_context(|| format!("HTTP GET {url}"))?;

        let total_size: u64 = resp
            .header("Content-Length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_message(
            dest.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        );

        let mut reader = resp.into_reader();
        let mut file = std::fs::File::create(dest)?;
        let mut buffer = [0u8; 8192];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut file, &buffer[..n])?;
            pb.inc(n as u64);
        }
        pb.finish_with_message("done");

        // Verify hash if provided.
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(dest)?;
            if actual != expected {
                let _ = std::fs::remove_file(dest);
                bail!(
                    "SHA-256 mismatch for {}: expected {expected}, got {actual}",
                    dest.display()
                );
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_verification() {
        let data = b"hello world";
        let hash = sha256_bytes(data);
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let n = normalize_vector(&v);
        assert_eq!(n.len(), 2);
        // Should be [0.6, 0.8].
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        // L2 norm should be ~1.0.
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize_vector(&v);
        assert!(n.iter().all(|x| *x == 0.0));
    }

    #[test]
    #[ignore]
    fn test_embed_smoke() {
        let dir = tempfile::tempdir().unwrap();
        let mut embedder = Embedder::new(dir.path()).unwrap();
        let vec = embedder.embed_one("hello world").unwrap();
        assert_eq!(vec.len(), EMBEDDING_DIM);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
