use std::path::Path;

use anyhow::{Result, bail};
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::chunker::{chunk_markdown, split_oversized_chunks};
use crate::docid::generate_docid;
use crate::embedder::Embedder;
use crate::indexer::build_edges_for_file;
use crate::links;
use crate::placement::{self, PlacementHints};
use crate::profile::VaultProfile;
use crate::store::Store;

// ── Input / Output types ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CreateNoteInput {
    pub content: String,
    pub filename: Option<String>,
    pub type_hint: Option<String>,
    pub tags: Vec<String>,
    pub folder: Option<String>,
    pub created_by: String,
}

#[derive(Debug, Clone)]
pub struct AppendInput {
    pub file: String,
    pub content: String,
    pub modified_by: String,
}

#[derive(Debug, Clone)]
pub struct UpdateMetadataInput {
    pub file: String,
    pub tags: Option<Vec<String>>,
    pub aliases: Option<Vec<String>>,
    pub modified_by: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WriteResult {
    pub path: String,
    pub docid: String,
    pub tags: Vec<String>,
    pub links_added: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub links_suggested: Vec<String>,
    pub folder: String,
    pub confidence: f64,
    pub strategy: String,
}

// ── Helper functions ────────────────────────────────────────────

/// Strip characters that are invalid in filenames, keeping alphanumeric, spaces, dashes, underscores, and dots.
pub fn generate_filename(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_' || *c == '.')
        .collect()
}

/// Extract a title from content: first `# heading` or first non-empty line, truncated to 50 chars.
pub fn extract_title(content: &str) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("# ") {
            let heading = heading.trim();
            if heading.len() > 50 {
                return heading[..50].to_string();
            }
            return heading.to_string();
        }
        // First non-empty line
        if trimmed.len() > 50 {
            return trimmed[..50].to_string();
        }
        return trimmed.to_string();
    }
    "Untitled".to_string()
}

/// Optional placement suggestion metadata for inbox notes.
pub struct PlacementSuggestion {
    pub suggested_folder: String,
    pub confidence: f64,
    pub reason: String,
}

/// Build YAML frontmatter string.
pub fn build_frontmatter(
    tags: &[String],
    created_by: Option<&str>,
    aliases: Option<&[String]>,
    suggestion: Option<&PlacementSuggestion>,
) -> String {
    let mut fm = String::from("---\n");

    if !tags.is_empty() {
        fm.push_str("tags:\n");
        for tag in tags {
            fm.push_str(&format!("  - {}\n", tag));
        }
    }

    if let Some(aliases) = aliases
        && !aliases.is_empty()
    {
        fm.push_str("aliases:\n");
        for alias in aliases {
            fm.push_str(&format!("  - {}\n", alias));
        }
    }

    fm.push_str(&format!("created: {}\n", today_date()));

    if let Some(by) = created_by {
        fm.push_str(&format!("created_by: {}\n", by));
    }

    // Placement suggestion for inbox notes — user sees why it landed here
    if let Some(s) = suggestion {
        fm.push_str(&format!("suggested_folder: {}\n", s.suggested_folder));
        fm.push_str(&format!("confidence: {:.2}\n", s.confidence));
        fm.push_str(&format!("reason: \"{}\"\n", s.reason));
    }

    fm.push_str("---\n\n");
    fm
}

/// Split content into (frontmatter_string, body_string).
/// If no frontmatter, returns ("", content).
pub fn split_frontmatter(content: &str) -> (String, String) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (String::new(), content.to_string());
    }

    // Find the closing ---
    let after_open = &trimmed[3..];
    // Skip past any remaining dashes and the newline
    let after_open = after_open.trim_start_matches('-');
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);

    if let Some(end_pos) = after_open.find("\n---") {
        let fm_content = &after_open[..end_pos];
        let rest_start = end_pos + 4; // "\n---"
        let rest = &after_open[rest_start..];
        // Skip trailing dashes and newline after closing ---
        let rest = rest.trim_start_matches('-');
        let rest = rest.strip_prefix('\n').unwrap_or(rest);

        let fm = format!("---\n{}\n---\n", fm_content);
        (fm, rest.to_string())
    } else {
        (String::new(), content.to_string())
    }
}

/// Returns today's date as "YYYY-MM-DD".
pub fn today_date() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}",
        now.year(),
        now.month() as u8,
        now.day()
    )
}

/// Compute SHA-256 hash of content bytes, returned as hex string.
fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Get file mtime as seconds since epoch.
fn file_mtime(path: &Path) -> Result<i64> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Ok(mtime.as_secs() as i64)
}

/// Pre-computed chunk data ready for store insertion.
type ChunkData = (String, String, Vec<f32>, i64); // (heading, snippet, vector, token_count)

/// Chunk content, embed, and return pre-computed data ready for store insertion.
fn precompute_chunks(content: &str, embedder: &mut Embedder) -> Result<Vec<ChunkData>> {
    let parsed = chunk_markdown(content);
    let chunks = split_oversized_chunks(parsed.chunks, &|s| s.split_whitespace().count(), 512, 50);

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts)?;

    let mut results = Vec::with_capacity(chunks.len());
    for (chunk, embedding) in chunks.into_iter().zip(embeddings) {
        let heading = chunk.heading.unwrap_or_default();
        let token_count = chunk.text.split_whitespace().count() as i64;
        results.push((heading, chunk.snippet, embedding, token_count));
    }
    Ok(results)
}

/// Write content to a temp file and atomically rename to final path.
/// Returns error if final_path already exists and `allow_overwrite` is false.
fn atomic_write(final_path: &Path, content: &str, allow_overwrite: bool) -> Result<()> {
    if !allow_overwrite && final_path.exists() {
        bail!(
            "file already exists at {}, refusing to overwrite",
            final_path.display()
        );
    }

    // Ensure parent directory exists
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let temp_path = final_path.with_extension("md.tmp");
    std::fs::write(&temp_path, content)?;
    std::fs::rename(&temp_path, final_path)?;
    Ok(())
}

/// Clean up incomplete writes from a previous crash.
/// Scans vault for .md.tmp files and removes them.
pub fn cleanup_temp_files(vault_path: &Path) -> Result<usize> {
    let mut cleaned = 0;
    for entry in WalkBuilder::new(vault_path).standard_filters(true).build() {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path.extension().is_some_and(|e| e == "tmp")
            && path.to_string_lossy().ends_with(".md.tmp")
        {
            std::fs::remove_file(path)?;
            cleaned += 1;
        }
    }
    Ok(cleaned)
}

// ── Pipeline functions ──────────────────────────────────────────

/// Create a new note via the 5-step write pipeline.
pub fn create_note(
    input: CreateNoteInput,
    store: &Store,
    embedder: &mut Embedder,
    vault_path: &Path,
    profile: Option<&VaultProfile>,
) -> Result<WriteResult> {
    // Step 1: Determine filename
    let title = if let Some(ref name) = input.filename {
        name.clone()
    } else {
        extract_title(&input.content)
    };
    let filename = generate_filename(&title);
    let filename = if filename.ends_with(".md") {
        filename
    } else {
        format!("{}.md", filename)
    };

    // Step 2: Resolve tags
    let resolved_tags = store.resolve_tags(&input.tags)?;

    // Step 3: Discover links and apply them
    let people_folder = profile
        .and_then(|p| p.structure.folders.people.as_deref());
    let discovered = links::discover_links(store, &input.content, vault_path, people_folder)?;

    // Split discovered links into auto-apply and suggestion-only
    let (auto_apply, suggestions): (Vec<_>, Vec<_>) = discovered.into_iter().partition(|l| {
        match &l.match_type {
            links::LinkMatchType::ExactName | links::LinkMatchType::Alias => true,
            links::LinkMatchType::FuzzyName { confidence_bp } => *confidence_bp >= 920,
            links::LinkMatchType::FirstName { .. } => false,
        }
    });

    let links_added: Vec<String> = auto_apply.iter().map(|l| l.target_path.clone()).collect();
    let links_suggested: Vec<String> = suggestions.iter().map(|l| {
        let target_name = l.target_path.rsplit('/').next().unwrap_or(&l.target_path).trim_end_matches(".md");
        if let Some(ref display) = l.display {
            format!("[[{}|{}]]", target_name, display)
        } else {
            format!("[[{}]]", target_name)
        }
    }).collect();

    // Apply auto-apply links to content — wrap matched text in [[wikilinks]]
    let mut content_with_links = input.content.clone();
    // Apply in reverse order of position to preserve offsets
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    let content_lower = content_with_links.to_lowercase();
    for link in &auto_apply {
        let search_lower = link.matched_text.to_lowercase();
        if let Some(pos) = content_lower.find(&search_lower) {
            let end = pos + link.matched_text.len();
            let original_text = &content_with_links[pos..end];
            let wikilink = if let Some(ref display) = link.display {
                format!(
                    "[[{}|{}]]",
                    link.target_path.trim_end_matches(".md"),
                    display
                )
            } else {
                format!("[[{}]]", original_text)
            };
            replacements.push((pos, end, wikilink));
        }
    }
    // Sort by position descending so replacements don't shift offsets
    replacements.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, end, replacement) in replacements {
        content_with_links.replace_range(start..end, &replacement);
    }

    // Step 4: Determine folder placement
    let placement_result = if let Some(ref folder) = input.folder {
        placement::PlacementResult {
            folder: folder.clone(),
            confidence: 1.0,
            strategy: placement::PlacementStrategy::TypeRule,
            reason: "Explicit folder".to_string(),
            suggestion: None,
        }
    } else {
        let hints = PlacementHints {
            type_hint: input.type_hint.clone(),
            tags: resolved_tags.clone(),
        };
        placement::place_note(&content_with_links, &hints, profile, store, Some(embedder))?
    };

    // Step 5: Build frontmatter and assemble content
    // If placement fell back to inbox with a suggestion, inject suggested_folder metadata
    let suggestion = if placement_result.strategy == placement::PlacementStrategy::InboxFallback {
        placement_result
            .suggestion
            .as_ref()
            .map(|(folder, conf)| PlacementSuggestion {
                suggested_folder: folder.clone(),
                confidence: *conf,
                reason: format!("semantic similarity: {conf:.3}"),
            })
    } else {
        None
    };
    let frontmatter = build_frontmatter(
        &resolved_tags,
        Some(&input.created_by),
        None,
        suggestion.as_ref(),
    );
    let full_content = format!("{}{}", frontmatter, content_with_links);

    let rel_path = format!("{}/{}", placement_result.folder, filename);
    let final_path = vault_path.join(&rel_path);

    // Check for existing file before doing expensive work
    if final_path.exists() {
        bail!(
            "file already exists at {}, refusing to overwrite",
            final_path.display()
        );
    }

    // Step 6: Pre-compute chunks + embeddings BEFORE transaction
    let chunk_data = precompute_chunks(&full_content, embedder)?;

    let content_hash = compute_content_hash(&full_content);
    let docid = generate_docid(&rel_path);

    // Write to temp file first
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = final_path.with_extension("md.tmp");
    std::fs::write(&temp_path, &full_content)?;

    // Step 7: BEGIN IMMEDIATE transaction
    store.begin_transaction()?;
    let result = (|| -> Result<i64> {
        let mtime = file_mtime(&temp_path).unwrap_or(0);
        let file_id = store.insert_file(&rel_path, &content_hash, mtime, &resolved_tags, &docid, Some(&input.created_by))?;

        let mut next_vid = store.next_vector_id()?;
        for (chunk_seq, (heading, snippet, vector, token_count)) in chunk_data.iter().enumerate() {
            let vid = next_vid;
            next_vid += 1;
            store.insert_chunk_with_vector(file_id, heading, snippet, vid, *token_count, vector)?;
            store.insert_vec(vid, vector)?;
            store.insert_fts_chunk(file_id, chunk_seq as i64, snippet)?;
        }

        build_edges_for_file(store, file_id, &full_content)?;

        // Register new tags
        for tag in &resolved_tags {
            store.register_tag(tag, &input.created_by)?;
        }

        Ok(file_id)
    })();

    match result {
        Ok(_) => {
            // Step 8: COMMIT
            store.commit()?;
            // Step 9: Atomic rename temp → final
            std::fs::rename(&temp_path, &final_path)?;
            // Update stored mtime to match the actual file after rename
            // (OS may adjust mtime during rename)
            let actual_mtime = file_mtime(&final_path).unwrap_or(0);
            store.insert_file(
                &rel_path,
                &content_hash,
                actual_mtime,
                &resolved_tags,
                &docid,
                Some(&input.created_by),
            )?;

            // Incrementally update folder centroid with new note's mean vector
            {
                let folder = &placement_result.folder;
                let new_vecs: Vec<&[f32]> =
                    chunk_data.iter().map(|(_, _, v, _)| v.as_slice()).collect();
                if !new_vecs.is_empty() {
                    let dim = new_vecs[0].len();
                    let mut mean_vec = vec![0.0f32; dim];
                    for v in &new_vecs {
                        for (i, val) in v.iter().enumerate() {
                            mean_vec[i] += val;
                        }
                    }
                    let n = new_vecs.len() as f32;
                    for val in &mut mean_vec {
                        *val /= n;
                    }
                    let _ = store.adjust_folder_centroid(folder, &mean_vec, true);
                }
            }
        }
        Err(e) => {
            let _ = store.rollback();
            let _ = std::fs::remove_file(&temp_path);
            return Err(e);
        }
    }

    let strategy_name = format!("{:?}", placement_result.strategy);
    Ok(WriteResult {
        path: rel_path,
        docid,
        tags: resolved_tags,
        links_added,
        links_suggested,
        folder: placement_result.folder,
        confidence: placement_result.confidence,
        strategy: strategy_name,
    })
}

/// Append content to an existing note.
pub fn append_to_note(
    input: AppendInput,
    store: &Store,
    embedder: &mut Embedder,
    vault_path: &Path,
) -> Result<WriteResult> {
    // Step 1: Resolve file
    let file_record = store
        .resolve_file(&input.file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", input.file))?;

    let full_path = vault_path.join(&file_record.path);

    // Step 2: Mtime conflict check
    let disk_mtime = file_mtime(&full_path)?;
    if disk_mtime != file_record.mtime {
        bail!(
            "mtime conflict: file {} was modified outside engraph (disk={}, indexed={})",
            file_record.path,
            disk_mtime,
            file_record.mtime
        );
    }

    // Step 3: Append content
    let existing_content = std::fs::read_to_string(&full_path)?;
    let new_content = format!("{}\n{}", existing_content.trim_end(), input.content);

    // Step 4: Pre-compute new chunks + embeddings
    let chunk_data = precompute_chunks(&new_content, embedder)?;

    let content_hash = compute_content_hash(&new_content);
    let docid = file_record
        .docid
        .clone()
        .unwrap_or_else(|| generate_docid(&file_record.path));

    // Write to temp file
    let temp_path = full_path.with_extension("md.tmp");
    std::fs::write(&temp_path, &new_content)?;

    // Step 5: Transaction — delete old data, re-insert
    store.begin_transaction()?;
    let result = (|| -> Result<i64> {
        // Tombstone old vectors
        let old_vids = store.get_vector_ids_for_file(file_record.id)?;

        for vid in &old_vids {
            store.delete_vec(*vid)?;
        }

        // Delete old chunks, FTS, edges
        store.delete_fts_chunks_for_file(file_record.id)?;
        store.delete_edges_for_file(file_record.id)?;
        store.delete_file(file_record.id)?;

        // Re-insert file
        let mtime = file_mtime(&temp_path).unwrap_or(0);
        let file_id = store.insert_file(
            &file_record.path,
            &content_hash,
            mtime,
            &file_record.tags,
            &docid,
            file_record.created_by.as_deref(),
        )?;

        let mut next_vid = store.next_vector_id()?;
        for (chunk_seq, (heading, snippet, vector, token_count)) in chunk_data.iter().enumerate() {
            let vid = next_vid;
            next_vid += 1;
            store.insert_chunk_with_vector(file_id, heading, snippet, vid, *token_count, vector)?;
            store.insert_vec(vid, vector)?;
            store.insert_fts_chunk(file_id, chunk_seq as i64, snippet)?;
        }

        build_edges_for_file(store, file_id, &new_content)?;
        Ok(file_id)
    })();

    match result {
        Ok(_) => {
            store.commit()?;
            // Step 6: Rename temp → final
            std::fs::rename(&temp_path, &full_path)?;
            // Update stored mtime to match actual file after rename
            let actual_mtime = file_mtime(&full_path).unwrap_or(0);
            store.insert_file(
                &file_record.path,
                &content_hash,
                actual_mtime,
                &file_record.tags,
                &docid,
                file_record.created_by.as_deref(),
            )?;
        }
        Err(e) => {
            let _ = store.rollback();
            let _ = std::fs::remove_file(&temp_path);
            return Err(e);
        }
    }

    let folder = file_record
        .path
        .rsplit_once('/')
        .map(|(f, _)| f.to_string())
        .unwrap_or_default();

    Ok(WriteResult {
        path: file_record.path,
        docid,
        tags: file_record.tags,
        links_added: vec![],
        links_suggested: vec![],
        folder,
        confidence: 1.0,
        strategy: "Append".to_string(),
    })
}

/// Update frontmatter metadata only (tags, aliases).
pub fn update_metadata(
    input: UpdateMetadataInput,
    store: &Store,
    vault_path: &Path,
) -> Result<WriteResult> {
    // Step 1: Resolve file
    let file_record = store
        .resolve_file(&input.file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", input.file))?;

    let full_path = vault_path.join(&file_record.path);

    // Step 2: Mtime conflict check
    let disk_mtime = file_mtime(&full_path)?;
    if disk_mtime != file_record.mtime {
        bail!(
            "mtime conflict: file {} was modified outside engraph (disk={}, indexed={})",
            file_record.path,
            disk_mtime,
            file_record.mtime
        );
    }

    // Step 3: Parse existing frontmatter and build new
    let existing_content = std::fs::read_to_string(&full_path)?;
    let (_old_fm, body) = split_frontmatter(&existing_content);

    let tags = input.tags.unwrap_or_else(|| file_record.tags.clone());
    let aliases_vec = input.aliases.unwrap_or_default();
    let aliases_ref: Option<&[String]> = if aliases_vec.is_empty() {
        None
    } else {
        Some(&aliases_vec)
    };

    let new_fm = build_frontmatter(&tags, Some(&input.modified_by), aliases_ref, None);
    let new_content = format!("{}{}", new_fm, body);

    // Step 4: Write via temp + rename
    let content_hash = compute_content_hash(&new_content);
    let docid = file_record
        .docid
        .clone()
        .unwrap_or_else(|| generate_docid(&file_record.path));

    atomic_write(&full_path, &new_content, true)?;

    // Step 5: Update store record (metadata-only, no re-chunking)
    let mtime = file_mtime(&full_path)?;
    store.insert_file(&file_record.path, &content_hash, mtime, &tags, &docid, file_record.created_by.as_deref())?;

    // Register tags
    for tag in &tags {
        store.register_tag(tag, &input.modified_by)?;
    }

    let folder = file_record
        .path
        .rsplit_once('/')
        .map(|(f, _)| f.to_string())
        .unwrap_or_default();

    Ok(WriteResult {
        path: file_record.path,
        docid,
        tags,
        links_added: vec![],
        links_suggested: vec![],
        folder,
        confidence: 1.0,
        strategy: "UpdateMetadata".to_string(),
    })
}

/// Move a note to a new folder.
pub fn move_note(
    file: &str,
    new_folder: &str,
    store: &Store,
    vault_path: &Path,
) -> Result<WriteResult> {
    // Step 1: Resolve file
    let file_record = store
        .resolve_file(file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", file))?;

    let old_path = vault_path.join(&file_record.path);
    let basename = file_record
        .path
        .rsplit('/')
        .next()
        .unwrap_or(&file_record.path);
    let new_rel_path = format!("{}/{}", new_folder, basename);
    let new_full_path = vault_path.join(&new_rel_path);

    if new_full_path.exists() {
        bail!("target path already exists: {}", new_full_path.display());
    }

    // Read content for re-indexing
    let content = std::fs::read_to_string(&old_path)?;
    let content_hash = compute_content_hash(&content);
    let new_docid = generate_docid(&new_rel_path);

    // Ensure target directory exists
    if let Some(parent) = new_full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Step 2: Transaction — delete old record, insert new
    store.begin_transaction()?;
    let result = (|| -> Result<()> {
        // Tombstone old vectors
        let old_vids = store.get_vector_ids_for_file(file_record.id)?;

        for vid in &old_vids {
            store.delete_vec(*vid)?;
        }
        store.delete_fts_chunks_for_file(file_record.id)?;
        store.delete_edges_for_file(file_record.id)?;
        store.delete_file(file_record.id)?;

        // Insert with new path (reuse existing chunks data via insert_file only for the record)
        let mtime = file_mtime(&old_path)?;
        store.insert_file(
            &new_rel_path,
            &content_hash,
            mtime,
            &file_record.tags,
            &new_docid,
            file_record.created_by.as_deref(),
        )?;

        Ok(())
    })();

    match result {
        Ok(()) => {
            store.commit()?;
            // Step 3: Rename file on disk
            std::fs::rename(&old_path, &new_full_path)?;
        }
        Err(e) => {
            let _ = store.rollback();
            return Err(e);
        }
    }

    Ok(WriteResult {
        path: new_rel_path,
        docid: new_docid,
        tags: file_record.tags,
        links_added: vec![],
        links_suggested: vec![],
        folder: new_folder.to_string(),
        confidence: 1.0,
        strategy: "Move".to_string(),
    })
}

// ── Archive / Unarchive ─────────────────────────────────────────

/// Archive a note: move to archive folder, add archived frontmatter, remove from index.
/// The note becomes invisible to search/context but is physically preserved.
pub fn archive_note(
    file: &str,
    store: &Store,
    vault_path: &Path,
    profile: Option<&crate::profile::VaultProfile>,
) -> Result<WriteResult> {
    let file_record = store
        .resolve_file(file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", file))?;

    let archive_folder = profile
        .and_then(|p| p.structure.folders.archive.as_deref())
        .unwrap_or("04-Archive");

    // Don't archive something already in the archive
    if file_record.path.starts_with(archive_folder) {
        bail!("note is already archived: {}", file_record.path);
    }

    let old_path = vault_path.join(&file_record.path);
    let new_rel_path = format!("{}/{}", archive_folder, file_record.path);
    let new_full_path = vault_path.join(&new_rel_path);

    // Read content and inject archive frontmatter
    let content = std::fs::read_to_string(&old_path)?;
    let (_old_fm, body) = split_frontmatter(&content);

    // Preserve existing tags, add archived metadata
    let mut tags = file_record.tags.clone();
    if !tags.contains(&"archived".to_string()) {
        tags.push("archived".to_string());
    }

    let archive_fm = format!(
        "---\n\
         archived: true\n\
         archived_at: {}\n\
         archived_from: {}\n\
         tags:\n{}\
         ---\n\n",
        today_date(),
        file_record.path,
        tags.iter()
            .map(|t| format!("  - {}\n", t))
            .collect::<String>(),
    );
    let new_content = format!("{}{}", archive_fm, body);

    // Ensure target directory
    if let Some(parent) = new_full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write archived file to new location
    atomic_write(&new_full_path, &new_content, false)?;

    // Remove from index (note disappears from search)
    let old_vids = store.get_vector_ids_for_file(file_record.id)?;
    for vid in &old_vids {
        store.delete_vec(*vid)?;
    }
    store.delete_fts_chunks_for_file(file_record.id)?;
    store.delete_edges_for_file(file_record.id)?;
    store.delete_file(file_record.id)?;

    // Remove original file
    std::fs::remove_file(&old_path)?;

    let docid = file_record.docid.unwrap_or_default();

    Ok(WriteResult {
        path: new_rel_path,
        docid,
        tags,
        links_added: vec![],
        links_suggested: vec![],
        folder: archive_folder.to_string(),
        confidence: 1.0,
        strategy: "Archive".to_string(),
    })
}

/// Unarchive a note: move back to original location, strip archive frontmatter, re-index.
pub fn unarchive_note(
    file: &str,
    store: &Store,
    embedder: &mut Embedder,
    vault_path: &Path,
) -> Result<WriteResult> {
    // Resolve — the file may not be in the index (archived notes are excluded).
    // Try resolving by direct path on disk.
    let archive_path = vault_path.join(file);
    if !archive_path.exists() {
        bail!("archived note not found: {}", file);
    }

    let content = std::fs::read_to_string(&archive_path)?;
    let (fm_str, body) = split_frontmatter(&content);

    // Extract archived_from from frontmatter
    let original_path = fm_str
        .lines()
        .find(|l| l.starts_with("archived_from:"))
        .and_then(|l| l.strip_prefix("archived_from:"))
        .map(|s| s.trim().to_string())
        .ok_or_else(|| {
            anyhow::anyhow!("no archived_from in frontmatter — cannot determine original location")
        })?;

    let restore_full_path = vault_path.join(&original_path);

    if restore_full_path.exists() {
        bail!(
            "cannot unarchive: a file already exists at {}",
            original_path
        );
    }

    // Rebuild frontmatter without archive fields
    let mut tags: Vec<String> = fm_str
        .lines()
        .skip_while(|l| !l.starts_with("tags:"))
        .skip(1)
        .take_while(|l| l.starts_with("  - "))
        .filter_map(|l| l.strip_prefix("  - "))
        .map(|s| s.trim().to_string())
        .filter(|t| t != "archived")
        .collect();
    if tags.is_empty() {
        // Try inline tags format
        if let Some(line) = fm_str.lines().find(|l| l.starts_with("tags:"))
            && let Some(rest) = line.strip_prefix("tags:")
        {
            let rest = rest.trim().trim_start_matches('[').trim_end_matches(']');
            tags = rest
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "archived")
                .collect();
        }
    }

    let new_fm = build_frontmatter(&tags, None, None, None);
    let restored_content = format!("{}{}", new_fm, body);

    // Ensure target directory
    if let Some(parent) = restore_full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write restored file
    atomic_write(&restore_full_path, &restored_content, false)?;

    // Index the restored note
    let chunk_data = precompute_chunks(&restored_content, embedder)?;
    let content_hash = compute_content_hash(&restored_content);
    let docid = generate_docid(&original_path);
    let mtime = file_mtime(&restore_full_path).unwrap_or(0);

    store.begin_transaction()?;
    let result = (|| -> Result<()> {
        let file_id = store.insert_file(&original_path, &content_hash, mtime, &tags, &docid, Some("unarchive"))?;

        let mut next_vid = store.next_vector_id()?;
        for (seq, (heading, snippet, vector, token_count)) in chunk_data.iter().enumerate() {
            let vid = next_vid;
            next_vid += 1;
            store.insert_chunk_with_vector(file_id, heading, snippet, vid, *token_count, vector)?;
            store.insert_vec(vid, vector)?;
            store.insert_fts_chunk(file_id, seq as i64, snippet)?;
        }

        build_edges_for_file(store, file_id, &restored_content)?;

        for tag in &tags {
            store.register_tag(tag, "unarchive")?;
        }

        Ok(())
    })();

    match result {
        Ok(()) => store.commit()?,
        Err(e) => {
            let _ = store.rollback();
            let _ = std::fs::remove_file(&restore_full_path);
            return Err(e);
        }
    }

    // Remove archived file
    std::fs::remove_file(&archive_path)?;

    let folder = original_path
        .rsplit_once('/')
        .map(|(f, _)| f.to_string())
        .unwrap_or_default();

    Ok(WriteResult {
        path: original_path,
        docid,
        tags,
        links_added: vec![],
        links_suggested: vec![],
        folder,
        confidence: 1.0,
        strategy: "Unarchive".to_string(),
    })
}

// ── Index integrity ─────────────────────────────────────────────

/// Verify that all indexed files still exist on disk.
/// Removes orphan DB entries for files that no longer exist.
/// Returns the number of orphan entries cleaned up.
pub fn verify_index_integrity(store: &Store, vault_path: &Path) -> Result<usize> {
    let all_files = store.get_all_files()?;
    let mut orphans = 0;
    for file in &all_files {
        let full_path = vault_path.join(&file.path);
        if !full_path.exists() {
            // Clean up orphan: vectors, FTS, edges, file record
            let vids = store.get_vector_ids_for_file(file.id)?;
            for vid in &vids {
                store.delete_vec(*vid)?;
            }
            store.delete_fts_chunks_for_file(file.id)?;
            store.delete_edges_for_file(file.id)?;
            store.delete_file(file.id)?;
            orphans += 1;
        }
    }
    Ok(orphans)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_filename() {
        assert_eq!(generate_filename("My Great Note"), "My Great Note");
        assert_eq!(generate_filename("Note/With:Bad*Chars"), "NoteWithBadChars");
    }

    #[test]
    fn test_extract_title() {
        assert_eq!(extract_title("# Hello World\nBody"), "Hello World");
        assert_eq!(extract_title("Just some text"), "Just some text");
    }

    #[test]
    fn test_extract_title_empty() {
        assert_eq!(extract_title(""), "Untitled");
    }

    #[test]
    fn test_extract_title_truncation() {
        let long_title = "a".repeat(100);
        let content = format!("# {}\nBody", long_title);
        assert_eq!(extract_title(&content).len(), 50);
    }

    #[test]
    fn test_build_frontmatter() {
        let fm = build_frontmatter(
            &["work".to_string(), "engraph".to_string()],
            Some("claude-code"),
            None,
            None,
        );
        assert!(fm.starts_with("---\n"));
        assert!(fm.ends_with("---\n\n"));
        assert!(fm.contains("work"));
        assert!(fm.contains("created_by: claude-code"));
    }

    #[test]
    fn test_build_frontmatter_with_aliases() {
        let fm = build_frontmatter(
            &["test".to_string()],
            Some("writer"),
            Some(&["alias1".to_string(), "alias2".to_string()]),
            None,
        );
        assert!(fm.contains("aliases:"));
        assert!(fm.contains("  - alias1"));
        assert!(fm.contains("  - alias2"));
    }

    #[test]
    fn test_split_frontmatter() {
        let content = "---\ntags: [a]\n---\n\nBody text";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.contains("tags"));
        assert_eq!(body.trim(), "Body text");
    }

    #[test]
    fn test_split_frontmatter_no_fm() {
        let content = "Just body text";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_empty());
        assert_eq!(body, "Just body text");
    }

    #[test]
    fn test_cleanup_temp_files() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("note.md.tmp"), "incomplete").unwrap();
        std::fs::write(dir.path().join("good.md"), "complete").unwrap();
        std::fs::write(dir.path().join("other.tmp"), "not md tmp").unwrap();

        let cleaned = cleanup_temp_files(dir.path()).unwrap();
        assert_eq!(cleaned, 1);
        assert!(!dir.path().join("note.md.tmp").exists());
        assert!(dir.path().join("good.md").exists());
        assert!(dir.path().join("other.tmp").exists()); // .tmp but not .md.tmp
    }

    #[test]
    fn test_today_date_format() {
        let date = today_date();
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        assert_eq!(&date[7..8], "-");
    }

    #[test]
    fn test_build_frontmatter_with_suggestion() {
        let suggestion = PlacementSuggestion {
            suggested_folder: "02-Areas/Development".to_string(),
            confidence: 0.58,
            reason: "semantic similarity: 0.580".to_string(),
        };
        let fm = build_frontmatter(
            &["work".to_string()],
            Some("claude-code"),
            None,
            Some(&suggestion),
        );
        assert!(fm.contains("suggested_folder: 02-Areas/Development"));
        assert!(fm.contains("confidence: 0.58"));
        assert!(fm.contains("reason: \"semantic similarity: 0.580\""));
    }

    #[test]
    fn test_verify_index_integrity() {
        let dir = tempfile::TempDir::new().unwrap();
        let vault = dir.path();
        std::fs::create_dir_all(vault.join("notes")).unwrap();
        std::fs::write(vault.join("notes/existing.md"), "# Exists").unwrap();

        let store = crate::store::Store::open_memory().unwrap();
        // Insert two files: one exists on disk, one does not
        store
            .insert_file(
                "notes/existing.md",
                "hash1",
                100,
                &[],
                &crate::docid::generate_docid("notes/existing.md"),
                None,
            )
            .unwrap();
        store
            .insert_file(
                "notes/gone.md",
                "hash2",
                100,
                &[],
                &crate::docid::generate_docid("notes/gone.md"),
                None,
            )
            .unwrap();

        let orphans = verify_index_integrity(&store, vault).unwrap();
        assert_eq!(orphans, 1);

        // The gone file should be removed from the store
        assert!(store.get_file("notes/gone.md").unwrap().is_none());
        // The existing file should still be there
        assert!(store.get_file("notes/existing.md").unwrap().is_some());
    }

    #[test]
    fn test_compute_content_hash() {
        let h1 = compute_content_hash("hello");
        let h2 = compute_content_hash("hello");
        let h3 = compute_content_hash("world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }
}
