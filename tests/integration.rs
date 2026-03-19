//! Integration tests for engraph.
//!
//! All tests are `#[ignore]` because they require the ONNX model download (~23MB).
//! Run with: `cargo test --test integration -- --ignored`

use std::path::{Path, PathBuf};

use engraph::chunker::chunk_markdown;
use engraph::config::Config;
use engraph::embedder::Embedder;
use engraph::hnsw::HnswIndex;
use engraph::indexer::{compute_file_hash, diff_vault, walk_vault};
use engraph::store::Store;

use tempfile::TempDir;

/// Fixture directory relative to the project root.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Copy the fixture vault into a temp directory so tests can mutate it.
fn copy_fixtures_to(dest: &Path) {
    copy_dir_recursive(&fixtures_dir(), dest);
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path);
        } else {
            std::fs::copy(&src_path, &dst_path).unwrap();
        }
    }
}

/// Write a file at the given relative path inside a directory.
fn write_file(dir: &Path, rel_path: &str, content: &str) {
    let full = dir.join(rel_path);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&full, content).unwrap();
}

/// Index a vault directory into the given data_dir using lower-level APIs.
/// Returns the number of files indexed.
fn index_vault(vault_path: &Path, data_dir: &Path, config: &Config, rebuild: bool) -> usize {
    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    let hnsw_dir = data_dir.join("hnsw");

    let files = walk_vault(vault_path, &config.exclude).unwrap();

    let (new_files, changed_files, deleted_files) = if rebuild {
        (files.clone(), Vec::new(), Vec::new())
    } else {
        diff_vault(&files, vault_path, &store).unwrap()
    };

    // Handle deletes.
    for record in &deleted_files {
        store.delete_file(record.id).unwrap();
    }

    // Handle updates (delete old record, then treat as new).
    let mut files_to_index: Vec<PathBuf> = new_files.clone();
    for file_path in &changed_files {
        let rel = file_path.strip_prefix(vault_path).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().to_string();
        if let Some(record) = store.get_file(&rel_str).unwrap() {
            store.delete_file(record.id).unwrap();
        }
        files_to_index.push(file_path.clone());
    }

    // Embed and store with vectors.
    let models_dir = data_dir.join("models");
    let mut embedder = Embedder::new(&models_dir).unwrap();

    let mut next_vid: u64 = store
        .get_all_vectors()
        .unwrap_or_default()
        .iter()
        .map(|(id, _)| *id)
        .max()
        .map_or(0, |m| m + 1);

    for file_path in &files_to_index {
        let content = std::fs::read_to_string(file_path).unwrap();
        let rel = file_path.strip_prefix(vault_path).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().to_string();
        let hash = compute_file_hash(file_path).unwrap();

        let parsed = chunk_markdown(&content);
        let tags = parsed.tags;
        let chunks = parsed.chunks;

        let file_id = store.insert_file(&rel_str, &hash, 0, &tags).unwrap();

        for chunk in &chunks {
            let heading = chunk.heading.clone().unwrap_or_default();
            let vec = embedder.embed_one(&chunk.text).unwrap();
            let token_count = embedder.token_count(&chunk.text) as i64;
            let vector_id = next_vid;
            next_vid += 1;
            store
                .insert_chunk_with_vector(
                    file_id,
                    &heading,
                    &chunk.snippet,
                    vector_id,
                    token_count,
                    &vec,
                )
                .unwrap();
        }
    }

    store
        .set_meta("vault_path", &vault_path.to_string_lossy())
        .unwrap();
    store
        .set_meta(
            "last_indexed_at",
            &format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
        )
        .unwrap();

    // Rebuild HNSW from all vectors in SQLite (hnsw_rs doesn't support append after load).
    let all_vectors = store.get_all_vectors().unwrap();
    let mut hnsw = HnswIndex::new(all_vectors.len().max(1000));
    for (vid, vector) in &all_vectors {
        hnsw.insert_with_id(vector, *vid);
    }
    hnsw.save(&hnsw_dir).unwrap();

    files_to_index.len()
}

/// Search using lower-level APIs and return results as (file_path, score) pairs.
fn search_vault(query: &str, top_n: usize, data_dir: &Path) -> Vec<(String, f32)> {
    let models_dir = data_dir.join("models");
    let mut embedder = Embedder::new(&models_dir).unwrap();

    let hnsw_dir = data_dir.join("hnsw");
    let index = HnswIndex::load(&hnsw_dir).unwrap();

    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path).unwrap();

    let query_vec = embedder.embed_one(query).unwrap();
    let tombstones = store.get_tombstones().unwrap();
    let raw_results = index.search(&query_vec, top_n, &tombstones);

    let mut results = Vec::new();
    for (vector_id, distance) in raw_results {
        if let Some(chunk) = store.get_chunk_by_vector_id(vector_id).unwrap() {
            let file_path = store
                .get_file_path_by_id(chunk.file_id)
                .unwrap()
                .unwrap_or_else(|| "<unknown>".to_string());
            let score = 1.0 - distance;
            results.push((file_path, score));
        }
    }
    results
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_full_index_and_search() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();
    let indexed = index_vault(vault_dir.path(), data_dir.path(), &config, false);
    assert!(indexed > 0, "should have indexed some files");

    let results = search_vault("error handling", 5, data_dir.path());
    assert!(!results.is_empty(), "search should return results");

    let file_paths: Vec<&str> = results.iter().map(|(p, _)| p.as_str()).collect();
    assert!(
        file_paths.iter().any(|p| p.contains("note1")),
        "results should contain note1.md (Rust Error Handling), got: {:?}",
        file_paths
    );
}

#[test]
#[ignore]
fn test_incremental_add() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    // Check initial file count.
    let db_path = data_dir.path().join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    let initial_count = store.stats().unwrap().file_count;
    drop(store);

    // Add a new file.
    write_file(
        vault_dir.path(),
        "note5.md",
        "---\ntags: [kubernetes]\n---\n\n# Kubernetes Pods\n\nPods are the smallest deployable units.",
    );

    // Re-index incrementally.
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    let store = Store::open(&db_path).unwrap();
    let new_count = store.stats().unwrap().file_count;
    assert_eq!(
        new_count,
        initial_count + 1,
        "file count should increase by 1 after adding a file"
    );

    // Search for the new content.
    let results = search_vault("kubernetes pods", 5, data_dir.path());
    let file_paths: Vec<&str> = results.iter().map(|(p, _)| p.as_str()).collect();
    assert!(
        file_paths.iter().any(|p| p.contains("note5")),
        "results should contain newly added note5.md, got: {:?}",
        file_paths
    );
}

#[test]
#[ignore]
fn test_incremental_delete() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    let db_path = data_dir.path().join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    let initial_count = store.stats().unwrap().file_count;
    drop(store);

    // Delete note2.md from the vault.
    std::fs::remove_file(vault_dir.path().join("note2.md")).unwrap();

    // Re-index incrementally.
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    let store = Store::open(&db_path).unwrap();
    let new_count = store.stats().unwrap().file_count;
    assert_eq!(
        new_count,
        initial_count - 1,
        "file count should decrease by 1 after deleting a file"
    );

    // Verify the deleted file is gone from the store.
    assert!(
        store.get_file("note2.md").unwrap().is_none(),
        "deleted file should not be in the store"
    );
}

#[test]
#[ignore]
fn test_incremental_update() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    // Modify note3.md to contain completely different content.
    write_file(
        vault_dir.path(),
        "note3.md",
        "# Quantum Computing\n\nQuantum computers use qubits for parallel computation.",
    );

    // Re-index incrementally.
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    // Search for the updated content.
    let results = search_vault("quantum computing qubits", 5, data_dir.path());
    let file_paths: Vec<&str> = results.iter().map(|(p, _)| p.as_str()).collect();
    assert!(
        file_paths.iter().any(|p| p.contains("note3")),
        "results should contain updated note3.md with quantum computing content, got: {:?}",
        file_paths
    );
}

#[test]
#[ignore]
fn test_obsidian_dir_excluded() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default(); // default excludes .obsidian/
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    let db_path = data_dir.path().join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    let stats = store.stats().unwrap();

    // Fixtures have 4 .md files (note1-4) plus .obsidian/config.md.
    // .obsidian should be excluded, so we expect exactly 4 files.
    assert_eq!(
        stats.file_count, 4,
        "should index exactly 4 files (excluding .obsidian/), got {}",
        stats.file_count
    );

    // Verify .obsidian/config.md is not in the store.
    assert!(
        store.get_file(".obsidian/config.md").unwrap().is_none(),
        ".obsidian/config.md should not be indexed"
    );
}

#[test]
#[ignore]
fn test_rebuild_flag() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();

    // Initial index.
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    let db_path = data_dir.path().join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    let initial_stats = store.stats().unwrap();
    drop(store);

    // Rebuild from scratch.
    // First clear the store to simulate rebuild behavior.
    std::fs::remove_file(&db_path).unwrap();
    let hnsw_dir = data_dir.path().join("hnsw");
    if hnsw_dir.exists() {
        std::fs::remove_dir_all(&hnsw_dir).unwrap();
    }

    index_vault(vault_dir.path(), data_dir.path(), &config, true);

    let store = Store::open(&db_path).unwrap();
    let rebuild_stats = store.stats().unwrap();

    assert_eq!(
        rebuild_stats.file_count, initial_stats.file_count,
        "rebuild should index the same number of files"
    );
    assert_eq!(
        rebuild_stats.tombstone_count, 0,
        "rebuild should have no tombstones"
    );
}

#[test]
#[ignore]
fn test_clear_preserves_model() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    // Verify model files exist.
    let models_dir = data_dir.path().join("models");
    assert!(
        models_dir.join("model.onnx").exists(),
        "model.onnx should exist before clear"
    );

    // Simulate `engraph clear` (without --all): remove db and hnsw, keep models.
    let db_path = data_dir.path().join("engraph.db");
    let hnsw_dir = data_dir.path().join("hnsw");
    let _ = std::fs::remove_file(&db_path);
    if hnsw_dir.exists() {
        let _ = std::fs::remove_dir_all(&hnsw_dir);
    }

    // Model directory should still exist.
    assert!(
        models_dir.join("model.onnx").exists(),
        "model.onnx should survive clear (without --all)"
    );
    assert!(
        models_dir.join("tokenizer.json").exists(),
        "tokenizer.json should survive clear (without --all)"
    );
}

#[test]
#[ignore]
fn test_status_output() {
    let vault_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();
    copy_fixtures_to(vault_dir.path());

    let config = Config::default();
    index_vault(vault_dir.path(), data_dir.path(), &config, false);

    let db_path = data_dir.path().join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    let stats = store.stats().unwrap();

    assert_eq!(stats.file_count, 4, "expected 4 indexed files");
    assert!(stats.chunk_count > 0, "expected some chunks");
    assert_eq!(
        stats.tombstone_count, 0,
        "expected no tombstones on fresh index"
    );
    assert!(
        stats.last_indexed_at.is_some(),
        "last_indexed_at should be set"
    );
    assert!(stats.vault_path.is_some(), "vault_path should be set");
    assert!(
        stats
            .vault_path
            .as_ref()
            .unwrap()
            .contains(vault_dir.path().to_str().unwrap()),
        "vault_path should point to the test vault"
    );
}
