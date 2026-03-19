use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::collections::HashSet;
use std::path::Path;

/// A record representing an indexed file.
#[derive(Debug, Clone)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub content_hash: String,
    pub mtime: i64,
    pub tags: Vec<String>,
    pub indexed_at: String,
}

/// A record representing a chunk of a file.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub id: i64,
    pub file_id: i64,
    pub heading: String,
    pub snippet: String,
    pub vector_id: u64,
    pub token_count: i64,
}

/// Summary statistics for the store.
#[derive(Debug)]
pub struct StoreStats {
    pub file_count: usize,
    pub chunk_count: usize,
    pub tombstone_count: usize,
    pub last_indexed_at: Option<String>,
    pub vault_path: Option<String>,
}

const SCHEMA: &str = r#"
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    mtime        INTEGER NOT NULL,
    tags         TEXT NOT NULL DEFAULT '[]',
    indexed_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    heading     TEXT NOT NULL,
    snippet     TEXT NOT NULL,
    vector_id   INTEGER UNIQUE NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tombstones (
    id         INTEGER PRIMARY KEY,
    vector_id  INTEGER UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
"#;

pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open a store backed by a file on disk.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        self.conn
            .execute_batch(SCHEMA)
            .context("failed to initialize schema")?;
        Ok(())
    }

    // ── Meta ────────────────────────────────────────────────────

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query_map(params![key], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(val) => Ok(Some(val?)),
            None => Ok(None),
        }
    }

    // ── Files ───────────────────────────────────────────────────

    pub fn insert_file(&self, path: &str, hash: &str, mtime: i64, tags: &[String]) -> Result<i64> {
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let now = chrono_now();
        self.conn.execute(
            "INSERT INTO files (path, content_hash, mtime, tags, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(path) DO UPDATE SET
                content_hash = excluded.content_hash,
                mtime        = excluded.mtime,
                tags         = excluded.tags,
                indexed_at   = excluded.indexed_at",
            params![path, hash, mtime, tags_json, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_file(&self, path: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at FROM files WHERE path = ?1",
        )?;
        let mut rows = stmt.query_map(params![path], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    pub fn get_all_files(&self) -> Result<Vec<FileRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, path, content_hash, mtime, tags, indexed_at FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
            })
        })?;
        let mut files = Vec::new();
        for row in rows {
            files.push(row?);
        }
        Ok(files)
    }

    pub fn delete_file(&self, file_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        Ok(())
    }

    // ── Chunks ──────────────────────────────────────────────────

    pub fn insert_chunk(
        &self,
        file_id: i64,
        heading: &str,
        snippet: &str,
        vector_id: u64,
        token_count: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chunks (file_id, heading, snippet, vector_id, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![file_id, heading, snippet, vector_id as i64, token_count],
        )?;
        Ok(())
    }

    pub fn get_chunks_by_file(&self, file_id: i64) -> Result<Vec<ChunkRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_id, heading, snippet, vector_id, token_count
             FROM chunks WHERE file_id = ?1",
        )?;
        let rows = stmt.query_map(params![file_id], |row| {
            Ok(ChunkRecord {
                id: row.get(0)?,
                file_id: row.get(1)?,
                heading: row.get(2)?,
                snippet: row.get(3)?,
                vector_id: row.get::<_, i64>(4)? as u64,
                token_count: row.get(5)?,
            })
        })?;
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row?);
        }
        Ok(chunks)
    }

    pub fn get_chunk_by_vector_id(&self, vector_id: u64) -> Result<Option<ChunkRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_id, heading, snippet, vector_id, token_count
             FROM chunks WHERE vector_id = ?1",
        )?;
        let mut rows = stmt.query_map(params![vector_id as i64], |row| {
            Ok(ChunkRecord {
                id: row.get(0)?,
                file_id: row.get(1)?,
                heading: row.get(2)?,
                snippet: row.get(3)?,
                vector_id: row.get::<_, i64>(4)? as u64,
                token_count: row.get(5)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    // ── Tombstones ──────────────────────────────────────────────

    pub fn add_tombstones(&self, vector_ids: &[u64]) -> Result<()> {
        let now = chrono_now();
        let mut stmt = self.conn.prepare(
            "INSERT OR IGNORE INTO tombstones (vector_id, created_at) VALUES (?1, ?2)",
        )?;
        for &vid in vector_ids {
            stmt.execute(params![vid as i64, now])?;
        }
        Ok(())
    }

    pub fn get_tombstones(&self) -> Result<HashSet<u64>> {
        let mut stmt = self.conn.prepare("SELECT vector_id FROM tombstones")?;
        let rows = stmt.query_map([], |row| Ok(row.get::<_, i64>(0)? as u64))?;
        let mut set = HashSet::new();
        for row in rows {
            set.insert(row?);
        }
        Ok(set)
    }

    pub fn tombstone_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM tombstones", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn clear_tombstones(&self) -> Result<()> {
        self.conn.execute("DELETE FROM tombstones", [])?;
        Ok(())
    }

    // ── Stats ───────────────────────────────────────────────────

    pub fn stats(&self) -> Result<StoreStats> {
        let file_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;
        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        let tombstone_count = self.tombstone_count()?;
        let last_indexed_at = self.get_meta("last_indexed_at")?;
        let vault_path = self.get_meta("vault_path")?;
        Ok(StoreStats {
            file_count: file_count as usize,
            chunk_count: chunk_count as usize,
            tombstone_count,
            last_indexed_at,
            vault_path,
        })
    }

    /// Look up a file's path by its row ID.
    pub fn get_file_path_by_id(&self, file_id: i64) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM files WHERE id = ?1")?;
        let mut rows = stmt.query_map(params![file_id], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(val) => Ok(Some(val?)),
            None => Ok(None),
        }
    }

    /// Return vector_ids for all chunks belonging to a file.
    /// Useful for tombstoning before re-indexing a changed file.
    pub fn get_vector_ids_for_file(&self, file_id: i64) -> Result<Vec<u64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id FROM chunks WHERE file_id = ?1")?;
        let rows = stmt.query_map(params![file_id], |row| Ok(row.get::<_, i64>(0)? as u64))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }
}

fn parse_tags(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

fn chrono_now() -> String {
    // Simple ISO-8601-ish timestamp without pulling in chrono crate.
    // Uses the system time formatted via std.
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    // Return seconds as a string; good enough for ordering.
    // A later task can swap in proper chrono formatting.
    format!("{}", duration.as_secs())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_schema() {
        let store = Store::open_memory().unwrap();
        // Verify all four tables exist by querying sqlite_master.
        let tables: Vec<String> = {
            let mut stmt = store
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .unwrap();
            let rows = stmt.query_map([], |row| row.get(0)).unwrap();
            rows.filter_map(|r| r.ok()).collect()
        };
        assert!(tables.contains(&"meta".to_string()));
        assert!(tables.contains(&"files".to_string()));
        assert!(tables.contains(&"chunks".to_string()));
        assert!(tables.contains(&"tombstones".to_string()));
    }

    #[test]
    fn test_insert_and_get_file() {
        let store = Store::open_memory().unwrap();
        let tags = vec!["rust".to_string(), "programming".to_string()];
        let file_id = store
            .insert_file("notes/test.md", "abc123", 1700000000, &tags)
            .unwrap();
        assert!(file_id > 0);

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.path, "notes/test.md");
        assert_eq!(rec.content_hash, "abc123");
        assert_eq!(rec.mtime, 1700000000);
        assert_eq!(rec.tags, tags);
    }

    #[test]
    fn test_insert_and_get_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("notes/chunk_test.md", "hash1", 100, &[])
            .unwrap();

        store
            .insert_chunk(file_id, "Heading 1", "Some text here", 1, 42)
            .unwrap();
        store
            .insert_chunk(file_id, "Heading 2", "More text", 2, 30)
            .unwrap();

        let chunks = store.get_chunks_by_file(file_id).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "Heading 1");
        assert_eq!(chunks[0].vector_id, 1);
        assert_eq!(chunks[0].token_count, 42);
        assert_eq!(chunks[1].snippet, "More text");

        let chunk = store.get_chunk_by_vector_id(2).unwrap().unwrap();
        assert_eq!(chunk.heading, "Heading 2");
    }

    #[test]
    fn test_delete_file_cascades_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("notes/del.md", "hash", 100, &[])
            .unwrap();
        store
            .insert_chunk(file_id, "H", "snippet", 10, 5)
            .unwrap();
        store
            .insert_chunk(file_id, "H2", "snippet2", 11, 6)
            .unwrap();

        assert_eq!(store.get_chunks_by_file(file_id).unwrap().len(), 2);

        store.delete_file(file_id).unwrap();

        assert!(store.get_file("notes/del.md").unwrap().is_none());
        assert_eq!(store.get_chunks_by_file(file_id).unwrap().len(), 0);
    }

    #[test]
    fn test_tombstone_lifecycle() {
        let store = Store::open_memory().unwrap();

        assert_eq!(store.tombstone_count().unwrap(), 0);
        assert!(store.get_tombstones().unwrap().is_empty());

        store.add_tombstones(&[100, 200, 300]).unwrap();
        assert_eq!(store.tombstone_count().unwrap(), 3);

        let ts = store.get_tombstones().unwrap();
        assert!(ts.contains(&100));
        assert!(ts.contains(&200));
        assert!(ts.contains(&300));

        // Duplicate insert should be ignored.
        store.add_tombstones(&[200, 400]).unwrap();
        assert_eq!(store.tombstone_count().unwrap(), 4);

        store.clear_tombstones().unwrap();
        assert_eq!(store.tombstone_count().unwrap(), 0);
    }

    #[test]
    fn test_file_hash_changed() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("notes/change.md", "old_hash", 100, &["tag1".to_string()])
            .unwrap();
        store
            .insert_chunk(file_id, "H", "text", 50, 10)
            .unwrap();
        store
            .insert_chunk(file_id, "H2", "text2", 51, 12)
            .unwrap();

        // Simulate detecting hash change: collect old vector_ids for tombstoning.
        let old_vector_ids = store.get_vector_ids_for_file(file_id).unwrap();
        assert_eq!(old_vector_ids.len(), 2);
        assert!(old_vector_ids.contains(&50));
        assert!(old_vector_ids.contains(&51));

        // Tombstone old vectors, delete file (cascades chunks), re-insert.
        store.add_tombstones(&old_vector_ids).unwrap();
        store.delete_file(file_id).unwrap();

        let new_file_id = store
            .insert_file("notes/change.md", "new_hash", 200, &["tag1".to_string()])
            .unwrap();
        store
            .insert_chunk(new_file_id, "H", "new text", 60, 15)
            .unwrap();

        let rec = store.get_file("notes/change.md").unwrap().unwrap();
        assert_eq!(rec.content_hash, "new_hash");

        let chunks = store.get_chunks_by_file(new_file_id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].vector_id, 60);

        // Old vectors are tombstoned.
        let ts = store.get_tombstones().unwrap();
        assert!(ts.contains(&50));
        assert!(ts.contains(&51));
    }

    #[test]
    fn test_vault_path_storage() {
        let store = Store::open_memory().unwrap();

        assert!(store.get_meta("vault_path").unwrap().is_none());

        store
            .set_meta("vault_path", "/home/user/vault")
            .unwrap();
        assert_eq!(
            store.get_meta("vault_path").unwrap().unwrap(),
            "/home/user/vault"
        );

        // Update the value.
        store
            .set_meta("vault_path", "/other/vault")
            .unwrap();
        assert_eq!(
            store.get_meta("vault_path").unwrap().unwrap(),
            "/other/vault"
        );

        // Verify stats reflects it.
        let st = store.stats().unwrap();
        assert_eq!(st.vault_path.unwrap(), "/other/vault");
    }
}
