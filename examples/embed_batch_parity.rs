//! Compare shared-context document embeddings with fresh-context embeddings.
//!
//! Usage: `embed_batch_parity <models_dir>`

use std::path::PathBuf;
use std::time::Instant;

use engraph::llm::{EmbedModel, LlamaEmbed};

fn main() -> anyhow::Result<()> {
    let models_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("models_dir argument is required"),
    );
    let config = engraph::config::Config::default();
    let mut embedder = LlamaEmbed::new(&models_dir, &config)?;

    let texts: Vec<String> = (0..32)
        .map(|index| {
            format!(
                "Note {index}: this chunk discusses retrieval pipelines, wikilink graphs, \
                 and reciprocal rank fusion in a knowledge vault. Sample {index} varies \
                 the content slightly to avoid identical sequences."
            )
        })
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let started = Instant::now();
    let shared_context = embedder.embed_batch(&text_refs)?;
    let shared_context_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let mut fresh_context = Vec::with_capacity(text_refs.len());
    for text in &text_refs {
        fresh_context.extend(embedder.embed_batch(&[text])?);
    }
    let fresh_context_ms = started.elapsed().as_millis();

    let mut max_abs_diff = 0.0_f32;
    let mut vectors_beyond_tolerance = 0_usize;
    for (shared, fresh) in shared_context.iter().zip(&fresh_context) {
        assert_eq!(shared.len(), fresh.len());
        let difference = shared
            .iter()
            .zip(fresh)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        max_abs_diff = max_abs_diff.max(difference);
        if difference > 1e-6 {
            vectors_beyond_tolerance += 1;
        }
    }

    println!(
        "{{\"n\":{},\"shared_context_ms\":{},\"fresh_context_ms\":{},\"max_abs_diff\":{:e},\"vectors_beyond_1e-6\":{}}}",
        text_refs.len(),
        shared_context_ms,
        fresh_context_ms,
        max_abs_diff,
        vectors_beyond_tolerance
    );

    if vectors_beyond_tolerance > 0 {
        anyhow::bail!("shared-context embeddings exceeded the 1e-6 parity tolerance");
    }
    Ok(())
}
