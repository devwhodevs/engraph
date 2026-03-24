use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tracing::info;

use crate::chunker::{chunk_markdown, split_oversized_chunks};
use crate::config::Config;
use crate::docid::generate_docid;
use crate::embedder::Embedder;
use crate::hnsw::HnswIndex;
use crate::store::{FileRecord, Store};

/// Summary of an indexing run.
pub struct IndexResult {
    pub new_files: usize,
    pub updated_files: usize,
    pub deleted_files: usize,
    pub total_chunks: usize,
    pub duration: Duration,
}

/// Walk a vault directory and collect all `.md` file paths.
///
/// Uses the `ignore` crate so `.gitignore` rules are respected automatically.
/// Additional exclude patterns (e.g. `.obsidian/`) are applied on top.
pub fn walk_vault(path: &Path, exclude: &[String]) -> Result<Vec<PathBuf>> {
    let walker = WalkBuilder::new(path)
        .standard_filters(true) // respect .gitignore, .ignore, etc.
        .build();

    let mut files = Vec::new();
    for entry in walker {
        let entry = entry.context("error reading directory entry")?;
        let entry_path = entry.path();

        // Only regular files.
        if !entry_path.is_file() {
            continue;
        }

        // Only .md files.
        match entry_path.extension() {
            Some(ext) if ext == "md" => {}
            _ => continue,
        }

        // Check exclude patterns.
        let rel = entry_path.strip_prefix(path).unwrap_or(entry_path);
        let rel_str = rel.to_string_lossy();

        let excluded = exclude.iter().any(|pattern| {
            // Support simple prefix/contains matching for directory patterns like ".obsidian/"
            if pattern.ends_with('/') {
                let dir_name = pattern.trim_end_matches('/');
                rel_str.split('/').any(|component| component == dir_name)
            } else {
                rel_str.contains(pattern.as_str())
            }
        });

        if excluded {
            continue;
        }

        files.push(entry_path.to_path_buf());
    }

    files.sort();
    Ok(files)
}

/// Compute the SHA-256 hash of a file's contents, returned as a hex string.
pub fn compute_file_hash(path: &Path) -> Result<String> {
    let content = std::fs::read(path)
        .with_context(|| format!("reading file for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compare vault files against the store to find new, changed, and deleted files.
///
/// Returns `(new_files, changed_files, deleted_file_records)`.
pub fn diff_vault(
    files: &[PathBuf],
    vault_root: &Path,
    store: &Store,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>, Vec<FileRecord>)> {
    let stored_files = store.get_all_files()?;
    let stored_map: HashMap<String, &FileRecord> =
        stored_files.iter().map(|f| (f.path.clone(), f)).collect();

    let mut new_files = Vec::new();
    let mut changed_files = Vec::new();

    // Track which stored paths we've seen on disk.
    let mut seen_paths = std::collections::HashSet::new();

    for file_path in files {
        let rel = file_path.strip_prefix(vault_root).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().to_string();

        seen_paths.insert(rel_str.clone());

        match stored_map.get(&rel_str) {
            None => {
                new_files.push(file_path.clone());
            }
            Some(record) => {
                let current_hash = compute_file_hash(file_path)?;
                if current_hash != record.content_hash {
                    changed_files.push(file_path.clone());
                }
            }
        }
    }

    // Files in store but not on disk are deleted.
    let deleted: Vec<FileRecord> = stored_files
        .into_iter()
        .filter(|f| !seen_paths.contains(&f.path))
        .collect();

    Ok((new_files, changed_files, deleted))
}

/// Main indexing orchestrator.
///
/// Walks the vault, diffs against the store, processes new/changed/deleted files,
/// embeds chunks in parallel, and writes everything to the store and HNSW index.
pub fn run_index(vault_path: &Path, config: &Config, rebuild: bool) -> Result<IndexResult> {
    let start = Instant::now();
    let data_dir = Config::data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path)?;

    let hnsw_dir = data_dir.join("hnsw");

    // If rebuild, treat everything as new.
    let files = walk_vault(vault_path, &config.exclude)?;

    let (new_files, changed_files, deleted_files) = if rebuild {
        // On rebuild we skip diffing — all files are "new".
        (files.clone(), Vec::new(), Vec::new())
    } else {
        let (n, c, d) = diff_vault(&files, vault_path, &store)?;
        (n, c, d)
    };

    info!(
        new = new_files.len(),
        changed = changed_files.len(),
        deleted = deleted_files.len(),
        "diff complete"
    );

    // Step 4: Handle deleted files — tombstone their vectors and remove from store.
    for record in &deleted_files {
        let vector_ids = store.get_vector_ids_for_file(record.id)?;
        if !vector_ids.is_empty() {
            store.add_tombstones(&vector_ids)?;
        }
        store.delete_fts_chunks_for_file(record.id)?;
        store.delete_file(record.id)?;
    }

    // Step 5: Handle changed files — tombstone old vectors, delete, then treat as new.
    let mut files_to_index: Vec<PathBuf> = new_files.clone();
    for file_path in &changed_files {
        let rel = file_path.strip_prefix(vault_path).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().to_string();
        if let Some(record) = store.get_file(&rel_str)? {
            let vector_ids = store.get_vector_ids_for_file(record.id)?;
            if !vector_ids.is_empty() {
                store.add_tombstones(&vector_ids)?;
            }
            store.delete_fts_chunks_for_file(record.id)?;
            store.delete_file(record.id)?;
        }
        files_to_index.push(file_path.clone());
    }

    // Step 6: Read content, chunk, and embed in parallel.
    let models_dir = data_dir.join("models");
    let mut embedder = Embedder::new(&models_dir)?;

    // Determine max tokens from embedder (use 512 as default for all-MiniLM-L6-v2).
    let max_tokens = 512;
    let overlap_tokens = 50;

    // Read all file contents sequentially, then process in parallel.
    let file_contents: Vec<(PathBuf, String)> = files_to_index
        .iter()
        .filter_map(|p| {
            std::fs::read_to_string(p)
                .ok()
                .map(|content| (p.clone(), content))
        })
        .collect();

    // Parallel chunking (embedding is serial since Embedder is not Send+Sync).
    let chunked_files: Vec<_> = file_contents
        .par_iter()
        .map(|(path, content)| {
            let parsed = chunk_markdown(content);
            // We can't call embedder.token_count across threads, so we defer
            // oversized splitting to serial phase.
            let rel = path.strip_prefix(vault_path).unwrap_or(path);
            let rel_str = rel.to_string_lossy().to_string();
            let hash = {
                let mut hasher = Sha256::new();
                hasher.update(content.as_bytes());
                format!("{:x}", hasher.finalize())
            };
            (path.clone(), rel_str, hash, parsed.tags, parsed.chunks)
        })
        .collect();

    // Serial: split oversized chunks, embed, and collect results.
    struct FileResult {
        rel_path: String,
        hash: String,
        tags: Vec<String>,
        mtime: i64,
        chunks: Vec<(String, String, Vec<f32>, usize)>, // (heading, snippet, vector, token_count)
    }

    let mut results: Vec<FileResult> = Vec::new();
    let mut total_chunks = 0usize;

    for (path, rel_str, hash, tags, chunks) in chunked_files {
        // Use a closure that borrows embedder for token counting in split phase.
        let chunks = {
            let tc = |s: &str| embedder.token_count(s);
            split_oversized_chunks(chunks, &tc, max_tokens, overlap_tokens)
        };

        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        // Count tokens before embedding (while embedder is borrowed immutably).
        let token_counts: Vec<usize> = chunks
            .iter()
            .map(|c| embedder.token_count(&c.text))
            .collect();

        // Embed in batches (borrows embedder mutably).
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let mut all_vectors = Vec::with_capacity(texts.len());

        for batch in texts.chunks(config.batch_size) {
            let vectors = embedder.embed_batch(batch)?;
            all_vectors.extend(vectors);
        }

        let mut chunk_results = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let heading = chunk.heading.clone().unwrap_or_default();
            let snippet = chunk.snippet.clone();
            chunk_results.push((heading, snippet, all_vectors[i].clone(), token_counts[i]));
        }

        total_chunks += chunk_results.len();
        results.push(FileResult {
            rel_path: rel_str,
            hash,
            tags,
            mtime,
            chunks: chunk_results,
        });
    }

    // Step 8: Serial write — insert files + chunks into store with vectors.
    let mut next_vector_id: u64 = {
        // Get the max existing vector_id to avoid collisions.
        let all_existing = store.get_all_vectors().unwrap_or_default();
        all_existing
            .iter()
            .map(|(id, _)| *id)
            .max()
            .map_or(0, |m| m + 1)
    };

    for result in &results {
        let docid = generate_docid(&result.rel_path);
        let file_id = store.insert_file(
            &result.rel_path,
            &result.hash,
            result.mtime,
            &result.tags,
            &docid,
        )?;

        for (chunk_seq, (heading, snippet, vector, token_count)) in result.chunks.iter().enumerate()
        {
            let vector_id = next_vector_id;
            next_vector_id += 1;
            store.insert_chunk_with_vector(
                file_id,
                heading,
                snippet,
                vector_id,
                *token_count as i64,
                vector,
            )?;
            store.insert_fts_chunk(file_id, chunk_seq as i64, snippet)?;
        }
    }

    // Step 9: Store vault path in meta.
    store.set_meta("vault_path", &vault_path.to_string_lossy())?;
    store.set_meta(
        "last_indexed_at",
        &format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
    )?;

    // Step 10: Rebuild HNSW index from all vectors in SQLite.
    // hnsw_rs doesn't support appending after load, so we always rebuild.
    let all_vectors = store.get_all_vectors()?;
    let mut hnsw = HnswIndex::new(all_vectors.len().max(1000));
    for (vid, vector) in &all_vectors {
        hnsw.insert_with_id(vector, *vid);
    }

    info!(
        vectors = all_vectors.len(),
        "rebuilt HNSW index from stored vectors"
    );

    // Step 11: Save HNSW index to disk.
    hnsw.save(&hnsw_dir)?;

    let duration = start.elapsed();
    info!(
        new = new_files.len(),
        updated = changed_files.len(),
        deleted = deleted_files.len(),
        chunks = total_chunks,
        duration_secs = duration.as_secs_f64(),
        "indexing complete"
    );

    Ok(IndexResult {
        new_files: new_files.len(),
        updated_files: changed_files.len(),
        deleted_files: deleted_files.len(),
        total_chunks,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a file with given content inside a temp directory.
    fn write_file(dir: &Path, rel_path: &str, content: &str) {
        let full = dir.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full, content).unwrap();
    }

    #[test]
    fn test_walk_collects_md_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note1.md", "# Note 1");
        write_file(root, "note2.md", "# Note 2");
        write_file(root, "sub/note3.md", "# Note 3");
        write_file(root, "image.png", "not markdown");
        write_file(root, "readme.txt", "text file");

        let files = walk_vault(root, &[]).unwrap();
        assert_eq!(files.len(), 3, "expected 3 .md files, got {:?}", files);
        for f in &files {
            assert_eq!(f.extension().unwrap(), "md");
        }
    }

    #[test]
    fn test_walk_excludes_patterns() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note.md", "# Note");
        write_file(root, ".obsidian/workspace.md", "obsidian internal");
        write_file(root, ".obsidian/plugins/plugin.md", "plugin data");

        let files = walk_vault(root, &[".obsidian/".to_string()]).unwrap();
        assert_eq!(files.len(), 1, "expected 1 file, got {:?}", files);
        assert!(files[0].ends_with("note.md"));
    }

    #[test]
    fn test_walk_respects_gitignore() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Initialize a git repo so the ignore crate respects .gitignore.
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(root)
            .output()
            .expect("git init failed");

        write_file(root, ".gitignore", "drafts/\n");
        write_file(root, "note.md", "# Note");
        write_file(root, "drafts/note.md", "# Draft");

        let files = walk_vault(root, &[]).unwrap();
        assert_eq!(
            files.len(),
            1,
            "expected 1 file (drafts/ gitignored), got {:?}",
            files
        );
        assert!(files[0].ends_with("note.md"));
    }

    #[test]
    fn test_detect_new_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A");
        write_file(root, "b.md", "# B");

        let store = Store::open_memory().unwrap();
        let files = walk_vault(root, &[]).unwrap();
        let (new, changed, deleted) = diff_vault(&files, root, &store).unwrap();

        assert_eq!(new.len(), 2, "all files should be new");
        assert_eq!(changed.len(), 0);
        assert_eq!(deleted.len(), 0);
    }

    #[test]
    fn test_detect_changed_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note.md", "# Original content");

        let store = Store::open_memory().unwrap();
        // Insert file with an old/different hash.
        store
            .insert_file(
                "note.md",
                "old_hash_that_wont_match",
                100,
                &[],
                &generate_docid("note.md"),
            )
            .unwrap();

        let files = walk_vault(root, &[]).unwrap();
        let (new, changed, deleted) = diff_vault(&files, root, &store).unwrap();

        assert_eq!(new.len(), 0);
        assert_eq!(
            changed.len(),
            1,
            "file with different hash should be changed"
        );
        assert_eq!(deleted.len(), 0);
    }

    #[test]
    fn test_detect_deleted_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "surviving.md", "# I exist");

        let store = Store::open_memory().unwrap();
        // Insert a file that no longer exists on disk.
        store
            .insert_file(
                "surviving.md",
                &compute_file_hash(&root.join("surviving.md")).unwrap(),
                100,
                &[],
                &generate_docid("surviving.md"),
            )
            .unwrap();
        store
            .insert_file(
                "deleted.md",
                "some_hash",
                100,
                &[],
                &generate_docid("deleted.md"),
            )
            .unwrap();

        let files = walk_vault(root, &[]).unwrap();
        let (new, changed, deleted) = diff_vault(&files, root, &store).unwrap();

        assert_eq!(new.len(), 0);
        assert_eq!(changed.len(), 0);
        assert_eq!(
            deleted.len(),
            1,
            "missing file should be detected as deleted"
        );
        assert_eq!(deleted[0].path, "deleted.md");
    }

    #[test]
    fn test_compute_file_hash_deterministic() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.md");
        std::fs::write(&path, "hello world").unwrap();

        let h1 = compute_file_hash(&path).unwrap();
        let h2 = compute_file_hash(&path).unwrap();
        assert_eq!(h1, h2, "same content should produce same hash");

        // Verify it's the known SHA-256 of "hello world".
        assert_eq!(
            h1,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
}
