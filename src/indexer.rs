use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use tracing::info;

use crate::chunker::{chunk_markdown, split_oversized_chunks};
use crate::config::Config;
use crate::docid::generate_docid;
use crate::embedder::Embedder;
use crate::graph::extract_wikilink_targets;
use crate::store::{FileRecord, Store};

/// Summary of an indexing run.
pub struct IndexResult {
    pub new_files: usize,
    pub updated_files: usize,
    pub deleted_files: usize,
    pub total_chunks: usize,
    pub duration: Duration,
}

/// Result of indexing a single file.
pub struct IndexFileResult {
    pub file_id: i64,
    pub total_chunks: usize,
    pub docid: String,
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

/// Resolve a wikilink target name to a file ID in the store.
fn resolve_link_target(store: &Store, target: &str) -> Result<Option<i64>> {
    let with_ext = if target.ends_with(".md") {
        target.to_string()
    } else {
        format!("{}.md", target)
    };

    // Try exact path match
    if let Some(f) = store.get_file(&with_ext)? {
        return Ok(Some(f.id));
    }

    // Try basename match (case-insensitive)
    let all_files = store.get_all_files()?;
    let target_lower = with_ext.to_lowercase();
    let mut matches: Vec<&FileRecord> = all_files
        .iter()
        .filter(|f| {
            let path_lower = f.path.to_lowercase();
            path_lower == target_lower || path_lower.ends_with(&format!("/{}", target_lower))
        })
        .collect();

    matches.sort_by_key(|f| f.path.len());
    Ok(matches.first().map(|f| f.id))
}

/// Build wikilink edges for a single file.
pub fn build_edges_for_file(store: &Store, file_id: i64, content: &str) -> Result<()> {
    let targets = extract_wikilink_targets(content);
    for target in targets {
        if let Some(target_id) = resolve_link_target(store, &target)?
            && target_id != file_id
        {
            store.insert_edge(file_id, target_id, "wikilink")?;
            store.insert_edge(target_id, file_id, "wikilink")?;
        }
    }
    Ok(())
}

/// Load people entities from the People folder.
/// Returns (file_id, [name, aliases...]) for each person note.
pub fn load_people_entities(
    store: &Store,
    people_folder: &str,
    content_by_path: &HashMap<String, String>,
) -> Result<Vec<(i64, Vec<String>)>> {
    let all_files = store.get_all_files()?;
    let mut people = Vec::new();
    for file in &all_files {
        if file.path.contains(people_folder) {
            let basename = file.path.rsplit('/').next().unwrap_or(&file.path);
            let name = basename.trim_end_matches(".md").to_string();
            let mut names = vec![name];

            // Extract aliases from frontmatter
            if let Some(content) = content_by_path.get(&file.path)
                && let Some(aliases) = extract_aliases_from_frontmatter(content)
            {
                names.extend(aliases);
            }

            people.push((file.id, names));
        }
    }
    Ok(people)
}

/// Extract aliases from YAML frontmatter.
pub fn extract_aliases_from_frontmatter(content: &str) -> Option<Vec<String>> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }
    let after = trimmed[3..].trim_start_matches('-').strip_prefix('\n')?;
    let end = after.find("\n---")?;
    let yaml = &after[..end];

    let lines: Vec<&str> = yaml.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with("aliases:") {
            let after_colon = t.strip_prefix("aliases:")?.trim();
            let mut aliases = Vec::new();
            if after_colon.starts_with('[') {
                let inner = after_colon.trim_start_matches('[').trim_end_matches(']');
                for a in inner.split(',') {
                    let a = a.trim().trim_matches('"').trim_matches('\'').to_string();
                    if !a.is_empty() {
                        aliases.push(a);
                    }
                }
            } else if after_colon.is_empty() {
                for sub in &lines[i + 1..] {
                    let st = sub.trim();
                    if st.starts_with("- ") {
                        aliases.push(st.strip_prefix("- ").unwrap().trim().to_string());
                    } else if !st.is_empty() {
                        break;
                    }
                }
            }
            return Some(aliases);
        }
    }
    None
}

/// Detect people mentions and create edges.
pub fn build_people_edges(
    store: &Store,
    file_id: i64,
    content: &str,
    people: &[(i64, Vec<String>)],
) -> Result<()> {
    let content_lower = content.to_lowercase();
    for (person_id, names) in people {
        if *person_id == file_id {
            continue;
        }
        let mentioned = names
            .iter()
            .any(|name| content_lower.contains(&name.to_lowercase()));
        if mentioned {
            store.insert_edge(file_id, *person_id, "mention")?;
        }
    }
    Ok(())
}

/// Process a single file: chunk, embed, and store in a single transaction.
///
/// This is the self-contained per-file indexing unit. If the file already exists
/// in the store, old entries (vec, FTS, file record) are cleaned up first.
pub fn index_file(
    rel_path: &str,
    content: &str,
    content_hash: &str,
    store: &Store,
    embedder: &mut Embedder,
    vault_path: &Path,
    config: &Config,
) -> Result<IndexFileResult> {
    let max_tokens = 512;
    let overlap_tokens = 50;

    // 1. Parse frontmatter for tags and created_by
    let parsed = chunk_markdown(content);
    let tags = parsed.tags;
    let chunks = {
        let tc = |s: &str| embedder.token_count(s);
        split_oversized_chunks(parsed.chunks, &tc, max_tokens, overlap_tokens)
    };

    // Extract created_by from frontmatter
    let (frontmatter, _body) = crate::writer::split_frontmatter(content);
    let created_by: Option<String> = frontmatter.lines().find_map(|line| {
        let trimmed = line.trim();
        if let Some(val) = trimmed.strip_prefix("created_by:") {
            let val = val.trim().trim_matches('"').trim_matches('\'');
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
        None
    });

    // 2. Embed all chunks
    let token_counts: Vec<usize> = chunks.iter().map(|c| embedder.token_count(&c.text)).collect();
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let mut all_vectors = Vec::with_capacity(texts.len());
    for batch in texts.chunks(config.batch_size) {
        let vectors = embedder.embed_batch(batch)?;
        all_vectors.extend(vectors);
    }

    // 3. Compute mtime
    let mtime = std::fs::metadata(vault_path.join(rel_path))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let docid = generate_docid(rel_path);

    // 4. Begin transaction
    store.conn().execute_batch("BEGIN DEFERRED")?;

    // 5. If file already exists, clean up old entries
    if let Some(record) = store.get_file(rel_path)? {
        let vector_ids = store.get_vector_ids_for_file(record.id)?;
        for &vid in &vector_ids {
            store.delete_vec(vid)?;
        }
        store.delete_fts_chunks_for_file(record.id)?;
        store.delete_file(record.id)?;
    }

    // 6. Insert new file and chunks
    let file_id = store.insert_file(
        rel_path,
        content_hash,
        mtime,
        &tags,
        &docid,
        created_by.as_deref(),
    )?;

    let mut next_vector_id: u64 = store.next_vector_id()?;
    let total_chunks = chunks.len();

    for (chunk_seq, chunk) in chunks.iter().enumerate() {
        let heading = chunk.heading.clone().unwrap_or_default();
        let snippet = &chunk.snippet;
        let vector = &all_vectors[chunk_seq];
        let vector_id = next_vector_id;
        next_vector_id += 1;

        store.insert_chunk_with_vector(
            file_id,
            &heading,
            snippet,
            vector_id,
            token_counts[chunk_seq] as i64,
            vector,
        )?;
        store.insert_vec(vector_id, vector)?;
        store.insert_fts_chunk(file_id, chunk_seq as i64, snippet)?;
    }

    // 7. Register tags
    for tag in &tags {
        store.register_tag(tag, "indexer")?;
    }

    // 8. Commit
    store.commit()?;

    Ok(IndexFileResult {
        file_id,
        total_chunks,
        docid,
    })
}

/// Remove a file from the store, cleaning up vec, FTS, and cascading chunks/edges.
///
/// sqlite-vec virtual tables don't participate in CASCADE deletes, so we must
/// manually delete vector entries before removing the file record.
pub fn remove_file(rel_path: &str, store: &Store) -> Result<()> {
    let file = store
        .get_file(rel_path)?
        .ok_or_else(|| anyhow!("File not found: '{}'", rel_path))?;
    let vector_ids = store.get_vector_ids_for_file(file.id)?;
    for &vid in &vector_ids {
        store.delete_vec(vid)?;
    }
    store.delete_fts_chunks_for_file(file.id)?;
    store.delete_file(file.id)?;
    Ok(())
}

/// Rename a file in the store, preserving its file_id and all edge integrity.
///
/// Recomputes the docid from the new path and delegates to `Store::update_file_path`
/// which performs a collision check and updates the path in place.
pub fn rename_file(old_rel: &str, new_rel: &str, store: &Store) -> Result<()> {
    let new_docid = generate_docid(new_rel);
    store.update_file_path(old_rel, new_rel, &new_docid)?;
    Ok(())
}

/// Main indexing orchestrator.
///
/// Walks the vault, diffs against the store, processes new/changed/deleted files,
/// embeds chunks in parallel, and writes everything to the store.
pub fn run_index(vault_path: &Path, config: &Config, rebuild: bool) -> Result<IndexResult> {
    let data_dir = Config::data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path)?;

    let models_dir = data_dir.join("models");
    let mut embedder = Embedder::new(&models_dir)?;

    run_index_inner(vault_path, config, &store, &mut embedder, rebuild)
}

/// Like [`run_index`], but accepts shared `Store` and `Embedder` references.
///
/// Useful when the caller already owns these resources (e.g. a file watcher
/// performing a full rescan without re-opening the database or reloading the model).
pub fn run_index_shared(
    vault_path: &Path,
    config: &Config,
    store: &Store,
    embedder: &mut Embedder,
    rebuild: bool,
) -> Result<IndexResult> {
    run_index_inner(vault_path, config, store, embedder, rebuild)
}

/// Shared implementation for [`run_index`] and [`run_index_shared`].
fn run_index_inner(
    vault_path: &Path,
    config: &Config,
    store: &Store,
    embedder: &mut Embedder,
    rebuild: bool,
) -> Result<IndexResult> {
    let start = Instant::now();

    let cleaned = crate::writer::cleanup_temp_files(vault_path)?;
    if cleaned > 0 {
        info!(cleaned, "cleaned up incomplete writes from previous run");
    }

    let orphans = crate::writer::verify_index_integrity(store, vault_path)?;
    if orphans > 0 {
        info!(orphans, "cleaned up orphan DB entries for missing files");
    }

    // Build exclude list: config excludes + archive folder (if detected)
    let mut exclude = config.exclude.clone();
    if let Ok(Some(profile)) = crate::config::Config::load_vault_profile()
        && let Some(archive) = &profile.structure.folders.archive
    {
        let archive_pattern = format!("{}/", archive);
        if !exclude.contains(&archive_pattern) {
            exclude.push(archive_pattern);
        }
    }

    // If rebuild, treat everything as new.
    let files = walk_vault(vault_path, &exclude)?;

    let (new_files, changed_files, deleted_files) = if rebuild {
        // On rebuild we skip diffing — all files are "new".
        store.clear_vec()?;
        (files.clone(), Vec::new(), Vec::new())
    } else {
        let (n, c, d) = diff_vault(&files, vault_path, store)?;
        (n, c, d)
    };

    info!(
        new = new_files.len(),
        changed = changed_files.len(),
        deleted = deleted_files.len(),
        "diff complete"
    );

    // Step 4: Handle deleted files — remove vectors from vec0, FTS, and store.
    for record in &deleted_files {
        remove_file(&record.path, store)?;
    }

    // Step 5: Handle changed files — delete old entries, then treat as new.
    let mut files_to_index: Vec<PathBuf> = new_files.clone();
    for file_path in &changed_files {
        let rel = file_path.strip_prefix(vault_path).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().to_string();
        if store.get_file(&rel_str)?.is_some() {
            remove_file(&rel_str, store)?;
        }
        files_to_index.push(file_path.clone());
    }

    // Step 6: Read content, index each file via index_file.
    // Read all file contents and compute hashes.
    let file_contents: Vec<(String, String, String)> = files_to_index
        .iter()
        .filter_map(|p| {
            let content = std::fs::read_to_string(p).ok()?;
            let rel = p.strip_prefix(vault_path).unwrap_or(p);
            let rel_str = rel.to_string_lossy().to_string();
            let hash = {
                let mut hasher = Sha256::new();
                hasher.update(content.as_bytes());
                format!("{:x}", hasher.finalize())
            };
            Some((rel_str, content, hash))
        })
        .collect();

    // Preserve raw content for edge building (wikilink extraction needs full text).
    let content_by_path: HashMap<String, String> = file_contents
        .iter()
        .map(|(rel_str, content, _hash)| (rel_str.clone(), content.clone()))
        .collect();

    // Serial: chunk, embed, and write each file via index_file.
    let mut total_chunks = 0usize;
    let mut indexed_rel_paths: Vec<String> = Vec::new();

    for (rel_str, content, hash) in &file_contents {
        let result = index_file(rel_str, content, hash, store, embedder, vault_path, config)?;
        total_chunks += result.total_chunks;
        indexed_rel_paths.push(rel_str.clone());
    }

    // Step 9: Build vault graph edges.
    info!("building vault graph edges");
    if rebuild {
        store.clear_edges()?;
    }

    for rel_path in &indexed_rel_paths {
        if let Some(file_record) = store.get_file(rel_path)?
            && let Some(content) = content_by_path.get(rel_path)
        {
            build_edges_for_file(store, file_record.id, content)?;
        }
    }

    // People detection (if configured via vault profile)
    if let Ok(Some(profile)) = crate::config::Config::load_vault_profile()
        && let Some(people_folder) = &profile.structure.folders.people
    {
        let people = load_people_entities(store, people_folder, &content_by_path)?;
        if !people.is_empty() {
            info!(people_count = people.len(), "detecting people mentions");
            for rel_path in &indexed_rel_paths {
                if let Some(file_record) = store.get_file(rel_path)?
                    && let Some(content) = content_by_path.get(rel_path)
                {
                    // Skip files in the People folder itself
                    if !rel_path.contains(people_folder.as_str()) {
                        build_people_edges(store, file_record.id, content, &people)?;
                    }
                }
            }
        }
    }

    // Step 10: Store vault path in meta.
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

    // Step 11: Compute folder centroids for placement engine.
    // Recompute from all chunks in the store for indexed files.
    info!("computing folder centroids");
    let mut folder_vecs: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for rel_path in &indexed_rel_paths {
        let folder = rel_path
            .split('/')
            .next()
            .unwrap_or("(root)")
            .to_string();
        if let Some(file_record) = store.get_file(rel_path)? {
            let chunk_vectors = store.get_chunk_vectors_for_file(file_record.id)?;
            for vector in chunk_vectors {
                folder_vecs.entry(folder.clone()).or_default().push(vector);
            }
        }
    }

    for (folder, vectors) in &folder_vecs {
        if vectors.is_empty() {
            continue;
        }
        let dim = 384;
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            for (i, val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let n = vectors.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }
        store.upsert_folder_centroid(folder, &centroid, vectors.len())?;
    }

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
                None,
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
                None,
            )
            .unwrap();
        store
            .insert_file(
                "deleted.md",
                "some_hash",
                100,
                &[],
                &generate_docid("deleted.md"),
                None,
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

    #[test]
    fn test_edge_building_during_index() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[b]] for details.");
        write_file(root, "b.md", "# B\nLinks to [[a]].");
        write_file(root, "c.md", "# C\nNo links here.");

        let store = Store::open_memory().unwrap();
        let f_a = store.insert_file("a.md", "h1", 100, &[], "aaa111", None).unwrap();
        let f_b = store.insert_file("b.md", "h2", 100, &[], "bbb222", None).unwrap();
        let _f_c = store.insert_file("c.md", "h3", 100, &[], "ccc333", None).unwrap();

        let content_a = std::fs::read_to_string(root.join("a.md")).unwrap();
        let content_b = std::fs::read_to_string(root.join("b.md")).unwrap();

        build_edges_for_file(&store, f_a, &content_a).unwrap();
        build_edges_for_file(&store, f_b, &content_b).unwrap();

        let a_out = store.get_outgoing(f_a, Some("wikilink")).unwrap();
        assert_eq!(a_out.len(), 1);
        assert_eq!(a_out[0].0, f_b);

        let b_out = store.get_outgoing(f_b, Some("wikilink")).unwrap();
        assert_eq!(b_out.len(), 1);
        assert_eq!(b_out[0].0, f_a);
    }

    #[test]
    fn test_extract_aliases_from_frontmatter() {
        let content = "---\ntags:\n  - person\naliases:\n  - Johnny\n  - JN\n---\n# John Nelson";
        let aliases = extract_aliases_from_frontmatter(content).unwrap();
        assert_eq!(aliases, vec!["Johnny", "JN"]);
    }

    #[test]
    fn test_extract_aliases_inline() {
        let content = "---\naliases: [Max, MD]\n---\n# Max Darski";
        let aliases = extract_aliases_from_frontmatter(content).unwrap();
        assert_eq!(aliases, vec!["Max", "MD"]);
    }

    #[test]
    fn test_extract_aliases_no_frontmatter() {
        assert!(extract_aliases_from_frontmatter("# Just a heading").is_none());
    }

    #[test]
    fn test_people_mention_detection() {
        let store = Store::open_memory().unwrap();
        let person = store
            .insert_file("People/John Nelson.md", "h1", 100, &[], "aaa111", None)
            .unwrap();
        let note = store
            .insert_file("daily.md", "h2", 100, &[], "bbb222", None)
            .unwrap();

        let people = vec![(person, vec!["John Nelson".to_string()])];
        let content = "Discussed with John Nelson about the architecture.";

        build_people_edges(&store, note, content, &people).unwrap();

        let mentions = store.get_outgoing(note, Some("mention")).unwrap();
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].0, person);
    }
}
