use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_full::{new_debouncer, DebouncedEvent};
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::embedder::Embedder;
use crate::indexer;
use crate::placement;
use crate::profile::VaultProfile;
use crate::store::Store;

/// Start the file watcher and consumer. Returns a thread handle for the producer
/// and a shutdown sender. On startup, runs a reconciliation index to catch any
/// changes that occurred while the server was down, then begins watching for
/// real-time file changes.
pub fn start_watcher(
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Embedder>>,
    vault_path: Arc<PathBuf>,
    profile: Arc<Option<VaultProfile>>,
    config: Config,
    exclude: Vec<String>,
) -> anyhow::Result<(std::thread::JoinHandle<()>, oneshot::Sender<()>)> {
    let (tx, rx) = mpsc::channel::<Vec<WatchEvent>>(64);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Start producer (begins buffering events immediately)
    let producer_handle = start_producer(
        vault_path.as_ref().clone(),
        exclude,
        tx,
        shutdown_rx,
    );

    // Spawn consumer task
    let store_clone = store.clone();
    let embedder_clone = embedder.clone();
    let vault_clone = vault_path.clone();
    let profile_clone = profile.clone();
    let config_clone = config.clone();
    tokio::spawn(async move {
        // Startup reconciliation: run index to catch changes since last shutdown
        {
            let store_lock = store_clone.lock().await;
            let mut embedder_lock = embedder_clone.lock().await;
            if let Err(e) = crate::indexer::run_index_shared(
                &vault_clone, &config_clone, &store_lock, &mut embedder_lock, false,
            ) {
                tracing::warn!("Startup reconciliation failed: {:#}", e);
            }
        }

        // Then consume events
        run_consumer(rx, store_clone, embedder_clone, vault_clone, profile_clone, config_clone).await;
    });

    Ok((producer_handle, shutdown_tx))
}

/// Events sent from the watcher producer to the consumer.
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// File content was modified or a new file was created.
    Changed(PathBuf),
    /// File was deleted.
    Deleted(PathBuf),
    /// File was moved/renamed (detected via content hash or inode tracking).
    Moved { from: PathBuf, to: PathBuf },
    /// macOS FSEvents buffer overflow — full rescan needed.
    FullRescan,
}

/// Start the producer thread. Returns thread handle.
/// The producer watches the vault, debounces events, and sends batches to tx.
pub fn start_producer(
    vault_path: PathBuf,
    exclude: Vec<String>,
    tx: mpsc::Sender<Vec<WatchEvent>>,
    mut shutdown_rx: oneshot::Receiver<()>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // Create std channel for debouncer events
        let (debouncer_tx, debouncer_rx) = std::sync::mpsc::channel();

        let mut debouncer = match new_debouncer(Duration::from_secs(2), None, debouncer_tx) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("Failed to create file watcher: {}", e);
                return;
            }
        };

        if let Err(e) = debouncer.watch(&vault_path, RecursiveMode::Recursive) {
            tracing::error!("Failed to watch {:?}: {}", vault_path, e);
            return;
        }

        tracing::info!("File watcher started for {:?}", vault_path);

        loop {
            // Check shutdown (non-blocking)
            if shutdown_rx.try_recv().is_ok() {
                tracing::info!("Watcher shutting down");
                break;
            }

            match debouncer_rx.recv_timeout(Duration::from_millis(500)) {
                Ok(Ok(events)) => {
                    let watch_events =
                        process_debounced_events(&events, &vault_path, &exclude);
                    if !watch_events.is_empty()
                        && tx.blocking_send(watch_events).is_err()
                    {
                        tracing::info!("Consumer gone, watcher exiting");
                        break;
                    }
                }
                Ok(Err(errors)) => {
                    for e in errors {
                        tracing::warn!("Watcher error: {:?}", e);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    })
}

/// Convert `DebouncedEvent`s to `WatchEvent`s, filtering to `.md` files.
fn process_debounced_events(
    events: &[DebouncedEvent],
    vault_path: &Path,
    exclude: &[String],
) -> Vec<WatchEvent> {
    let mut result = Vec::new();

    for debounced in events {
        let event = &debounced.event; // Access the inner notify::Event

        let paths: Vec<&PathBuf> = event
            .paths
            .iter()
            .filter(|p| p.extension().map(|e| e == "md").unwrap_or(false))
            .filter(|p| !is_excluded(p, vault_path, exclude))
            .collect();

        if paths.is_empty() {
            continue;
        }

        use notify::EventKind;
        match &event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in paths {
                    result.push(WatchEvent::Changed(path.clone()));
                }
            }
            EventKind::Remove(_) => {
                for path in paths {
                    result.push(WatchEvent::Deleted(path.clone()));
                }
            }
            EventKind::Other => {
                result.push(WatchEvent::FullRescan);
            }
            _ => {}
        }
    }

    result
}

/// Check if a path should be excluded (mirrors `walk_vault` logic in `indexer.rs`).
fn is_excluded(path: &Path, vault_path: &Path, exclude: &[String]) -> bool {
    let rel = path.strip_prefix(vault_path).unwrap_or(path);
    let rel_str = rel.to_string_lossy();
    exclude.iter().any(|pattern| {
        if pattern.ends_with('/') {
            let dir_name = pattern.trim_end_matches('/');
            rel_str.split('/').any(|component| component == dir_name)
        } else {
            rel_str.contains(pattern.as_str())
        }
    })
}

/// Detect file moves by matching `Deleted` + `Changed` pairs via content hash.
///
/// When a file is moved, the OS reports a delete at the old path and a create at
/// the new path. We match these by comparing the stored content hash (for the
/// deleted file) against the on-disk content hash (for the new file). Matched
/// pairs are replaced with `Moved { from, to }` events.
fn detect_moves(events: &mut Vec<WatchEvent>, store: &Store, vault_path: &Path) {
    // Collect deletion paths and their stored content hashes.
    let mut deletion_hashes: HashMap<String, PathBuf> = HashMap::new();
    for event in events.iter() {
        if let WatchEvent::Deleted(path) = event {
            let rel = path
                .strip_prefix(vault_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            if let Ok(Some(record)) = store.get_file(&rel) {
                deletion_hashes.insert(record.content_hash.clone(), path.clone());
            }
        }
    }

    if deletion_hashes.is_empty() {
        return;
    }

    // Collect creation paths (Changed events for files NOT already in store = new files).
    let mut creation_hashes: HashMap<String, PathBuf> = HashMap::new();
    for event in events.iter() {
        if let WatchEvent::Changed(path) = event {
            let rel = path
                .strip_prefix(vault_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            // Only consider files not already in the store (truly new files).
            if store.get_file(&rel).ok().flatten().is_none()
                && let Ok(hash) = indexer::compute_file_hash(path)
            {
                creation_hashes.insert(hash, path.clone());
            }
        }
    }

    // Match deletions to creations by content hash.
    let mut moves: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (hash, del_path) in &deletion_hashes {
        if let Some(create_path) = creation_hashes.get(hash) {
            moves.push((del_path.clone(), create_path.clone()));
        }
    }

    if moves.is_empty() {
        return;
    }

    // Replace matched pairs with Moved events.
    let move_from_set: std::collections::HashSet<PathBuf> =
        moves.iter().map(|(from, _)| from.clone()).collect();
    let move_to_set: std::collections::HashSet<PathBuf> =
        moves.iter().map(|(_, to)| to.clone()).collect();

    events.retain(|event| match event {
        WatchEvent::Deleted(p) => !move_from_set.contains(p),
        WatchEvent::Changed(p) => !move_to_set.contains(p),
        _ => true,
    });

    for (from, to) in moves {
        tracing::info!(from = %from.display(), to = %to.display(), "detected file move");
        events.push(WatchEvent::Moved { from, to });
    }
}

/// Consumer async task that processes batches of watch events.
///
/// Two-pass processing:
/// - Pass 1: Apply mutations (index/remove/rename files)
/// - Pass 2: Rebuild edges for affected files
pub async fn run_consumer(
    mut rx: mpsc::Receiver<Vec<WatchEvent>>,
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Embedder>>,
    vault_path: Arc<PathBuf>,
    _profile: Arc<Option<VaultProfile>>,
    config: Config,
) {
    tracing::info!("Watcher consumer started");

    while let Some(mut events) = rx.recv().await {
        tracing::info!(count = events.len(), "processing event batch");

        // Move detection (needs store lock briefly)
        {
            let store_guard = store.lock().await;
            detect_moves(&mut events, &store_guard, &vault_path);
        }

        let mut affected_file_ids: Vec<i64> = Vec::new();
        let mut had_full_rescan = false;

        // Pass 1: mutations (one event at a time)
        for event in &events {
            match event {
                WatchEvent::Changed(path) => {
                    let rel = path
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    let content = match std::fs::read_to_string(path) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(path = %path.display(), error = %e, "failed to read changed file, skipping");
                            continue;
                        }
                    };

                    let content_hash = match indexer::compute_file_hash(path) {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::warn!(path = %path.display(), error = %e, "failed to hash changed file, skipping");
                            continue;
                        }
                    };

                    let store_guard = store.lock().await;
                    let mut embedder_guard = embedder.lock().await;
                    match indexer::index_file(
                        &rel,
                        &content,
                        &content_hash,
                        &store_guard,
                        &mut embedder_guard,
                        &vault_path,
                        &config,
                    ) {
                        Ok(result) => {
                            tracing::info!(
                                path = %rel,
                                file_id = result.file_id,
                                chunks = result.total_chunks,
                                "indexed changed file"
                            );
                            affected_file_ids.push(result.file_id);
                        }
                        Err(e) => {
                            tracing::warn!(path = %rel, error = %e, "failed to index changed file");
                        }
                    }
                    drop(embedder_guard);
                    drop(store_guard);
                }

                WatchEvent::Deleted(path) => {
                    let rel = path
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    let store_guard = store.lock().await;
                    match indexer::remove_file(&rel, &store_guard) {
                        Ok(()) => {
                            tracing::info!(path = %rel, "removed deleted file from index");
                        }
                        Err(e) => {
                            tracing::warn!(path = %rel, error = %e, "failed to remove deleted file");
                        }
                    }
                    drop(store_guard);
                }

                WatchEvent::Moved { from, to } => {
                    let old_rel = from
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(from)
                        .to_string_lossy()
                        .to_string();
                    let new_rel = to
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(to)
                        .to_string_lossy()
                        .to_string();

                    let store_guard = store.lock().await;
                    match indexer::rename_file(&old_rel, &new_rel, &store_guard) {
                        Ok(()) => {
                            tracing::info!(from = %old_rel, to = %new_rel, "renamed file in index");
                            // Track the file_id for edge rebuild
                            if let Ok(Some(record)) = store_guard.get_file(&new_rel) {
                                affected_file_ids.push(record.id);
                            }

                            // Placement correction detection
                            if let Ok(content) = std::fs::read_to_string(to) {
                                let actual_folder = std::path::Path::new(&new_rel)
                                    .parent()
                                    .map(|p| p.to_string_lossy().to_string())
                                    .unwrap_or_default();

                                match placement::detect_correction_from_frontmatter(
                                    &content,
                                    &actual_folder,
                                ) {
                                    Some(correction) => {
                                        tracing::info!(
                                            file = %new_rel,
                                            suggested = %correction.suggested_folder,
                                            actual = %correction.actual_folder,
                                            "placement correction detected"
                                        );

                                        // Compute mean vector from file chunks
                                        if let Ok(Some(file)) = store_guard.get_file(&new_rel)
                                            && let Ok(vectors) = store_guard.get_chunk_vectors_for_file(file.id)
                                            && !vectors.is_empty()
                                        {
                                            let dim = vectors[0].len();
                                            let mut mean = vec![0.0f32; dim];
                                            for v in &vectors {
                                                for (i, val) in v.iter().enumerate() {
                                                    mean[i] += val;
                                                }
                                            }
                                            let n = vectors.len() as f32;
                                            for val in &mut mean {
                                                *val /= n;
                                            }

                                            // Adjust centroids: boost actual, decay suggested
                                            if let Err(e) = store_guard.adjust_folder_centroid(
                                                &correction.actual_folder,
                                                &mean,
                                                true,
                                            ) {
                                                tracing::warn!(error = %e, "failed to adjust actual folder centroid");
                                            }
                                            if let Err(e) = store_guard.adjust_folder_centroid(
                                                &correction.suggested_folder,
                                                &mean,
                                                false,
                                            ) {
                                                tracing::warn!(error = %e, "failed to adjust suggested folder centroid");
                                            }
                                        }

                                        // Log the correction
                                        if let Err(e) = store_guard.insert_placement_correction(
                                            &new_rel,
                                            &correction.suggested_folder,
                                            &correction.actual_folder,
                                        ) {
                                            tracing::warn!(error = %e, "failed to log placement correction");
                                        }

                                        // Strip placement frontmatter and write atomically
                                        let stripped = placement::strip_placement_frontmatter(&content);
                                        if stripped != content {
                                            let tmp = to.with_extension("md.tmp");
                                            if let Err(e) = std::fs::write(&tmp, &stripped)
                                                .and_then(|_| std::fs::rename(&tmp, to))
                                            {
                                                tracing::warn!(error = %e, "failed to strip placement frontmatter");
                                                let _ = std::fs::remove_file(&tmp);
                                            }
                                        }
                                    }
                                    None => {
                                        // Check if it's a confirmation (suggested == actual) — just strip
                                        let has_suggested = content.contains("suggested_folder:");
                                        if has_suggested {
                                            let stripped = placement::strip_placement_frontmatter(&content);
                                            if stripped != content {
                                                let tmp = to.with_extension("md.tmp");
                                                if let Err(e) = std::fs::write(&tmp, &stripped)
                                                    .and_then(|_| std::fs::rename(&tmp, to))
                                                {
                                                    tracing::warn!(error = %e, "failed to strip placement frontmatter on confirmation");
                                                    let _ = std::fs::remove_file(&tmp);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(from = %old_rel, to = %new_rel, error = %e, "failed to rename file");
                        }
                    }
                    drop(store_guard);
                }

                WatchEvent::FullRescan => {
                    tracing::info!("performing full rescan");
                    let store_guard = store.lock().await;
                    let mut embedder_guard = embedder.lock().await;
                    match indexer::run_index_shared(
                        &vault_path,
                        &config,
                        &store_guard,
                        &mut embedder_guard,
                        false,
                    ) {
                        Ok(result) => {
                            tracing::info!(
                                new = result.new_files,
                                updated = result.updated_files,
                                deleted = result.deleted_files,
                                chunks = result.total_chunks,
                                duration_secs = result.duration.as_secs_f64(),
                                "full rescan complete"
                            );
                            had_full_rescan = true;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "full rescan failed");
                        }
                    }
                    drop(embedder_guard);
                    drop(store_guard);
                }
            }
        }

        // Pass 2: edge rebuild for affected files (skip if full rescan already rebuilt everything)
        if !had_full_rescan && !affected_file_ids.is_empty() {
            tracing::info!(count = affected_file_ids.len(), "rebuilding edges for affected files");
            let store_guard = store.lock().await;
            for file_id in &affected_file_ids {
                // Delete old edges first
                if let Err(e) = store_guard.delete_edges_for_file(*file_id) {
                    tracing::warn!(file_id, error = %e, "failed to delete old edges");
                    continue;
                }

                if let Ok(Some(file)) = store_guard.get_file_by_id(*file_id) {
                    let content = std::fs::read_to_string(vault_path.join(&file.path))
                        .unwrap_or_default();
                    if let Err(e) =
                        indexer::build_edges_for_file(&store_guard, *file_id, &content)
                    {
                        tracing::warn!(
                            file_id,
                            path = %file.path,
                            error = %e,
                            "failed to rebuild edges"
                        );
                    }
                }
            }
            drop(store_guard);
        }

        tracing::info!("batch processing complete");
    }

    tracing::info!("Watcher consumer shutting down (channel closed)");
}
