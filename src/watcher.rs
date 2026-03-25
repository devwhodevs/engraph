use std::path::{Path, PathBuf};
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_full::{new_debouncer, DebouncedEvent};
use tokio::sync::mpsc;
use tokio::sync::oneshot;

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
                    if !watch_events.is_empty() {
                        if tx.blocking_send(watch_events).is_err() {
                            tracing::info!("Consumer gone, watcher exiting");
                            break;
                        }
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
