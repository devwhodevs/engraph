use std::path::PathBuf;

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
