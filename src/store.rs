use anyhow::{Context, Result};
use rusqlite::{Connection, params};
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
    pub docid: Option<String>,
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

/// A single result from an FTS5 full-text search.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub file_id: i64,
    pub chunk_seq: i64,
    pub score: f64,
    pub snippet: String,
}

/// Statistics about edges in the graph.
#[derive(Debug)]
pub struct EdgeStats {
    pub total_edges: usize,
    pub wikilink_count: usize,
    pub mention_count: usize,
    pub connected_file_count: usize,
    pub isolated_file_count: usize,
}

/// Summary statistics for the store.
#[derive(Debug)]
pub struct StoreStats {
    pub file_count: usize,
    pub chunk_count: usize,
    pub tombstone_count: usize,
    pub last_indexed_at: Option<String>,
    pub vault_path: Option<String>,
    pub edge_count: Option<usize>,
    pub wikilink_count: Option<usize>,
    pub mention_count: Option<usize>,
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
    indexed_at   TEXT NOT NULL,
    docid        TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    heading     TEXT NOT NULL,
    snippet     TEXT NOT NULL,
    vector_id   INTEGER UNIQUE NOT NULL,
    token_count INTEGER NOT NULL,
    vector      BLOB
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
        crate::vecstore::init_sqlite_vec();
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_memory() -> Result<Self> {
        crate::vecstore::init_sqlite_vec();
        let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        self.conn
            .execute_batch(SCHEMA)
            .context("failed to initialize schema")?;
        self.migrate()?;
        self.ensure_fts_table()?;
        crate::vecstore::init_vec_table(&self.conn)?;
        self.migrate_vectors_to_vec0()?;
        Ok(())
    }

    /// One-time migration: copy BLOB vectors from `chunks.vector` into the vec0 virtual table.
    /// Safe to call on every startup — skips if vec0 is already populated or no BLOBs exist.
    pub fn migrate_vectors_to_vec0(&self) -> Result<()> {
        let vec_count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks_vec", [], |row| row.get(0))
            .unwrap_or(0);
        let blob_count: i64 = self
            .conn
            .query_row(
                "SELECT count(*) FROM chunks WHERE vector IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if vec_count == 0 && blob_count > 0 {
            tracing::info!(blob_count, "migrating BLOB vectors to vec0");
            let mut stmt = self
                .conn
                .prepare("SELECT vector_id, vector FROM chunks WHERE vector IS NOT NULL")?;
            let rows: Vec<(i64, Vec<u8>)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .filter_map(|r| r.ok())
                .collect();

            for (vid, blob) in &rows {
                self.conn.execute(
                    "INSERT OR IGNORE INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
                    rusqlite::params![vid, blob],
                )?;
            }
            tracing::info!(migrated = rows.len(), "BLOB vector migration complete");
        }

        Ok(())
    }

    /// Run migrations for existing databases that may be missing newer columns.
    fn migrate(&self) -> Result<()> {
        // Check if docid column exists on files table.
        let has_docid: bool = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(files)")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
            let mut found = false;
            for row in rows {
                if row.as_deref() == Ok("docid") {
                    found = true;
                    break;
                }
            }
            found
        };
        if !has_docid {
            self.conn
                .execute_batch("ALTER TABLE files ADD COLUMN docid TEXT;")?;
        }
        // Always ensure the index exists (safe for both fresh and migrated DBs).
        self.conn
            .execute_batch("CREATE INDEX IF NOT EXISTS idx_files_docid ON files(docid);")?;

        // Check if edges table exists.
        let has_edges: bool = {
            let mut stmt = self
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")?;
            let mut rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.next().is_some()
        };
        if !has_edges {
            self.conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS edges (
                    id         INTEGER PRIMARY KEY,
                    from_file  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    to_file    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    edge_type  TEXT NOT NULL,
                    UNIQUE(from_file, to_file, edge_type)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_file);
                CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_file);
                CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);",
            )?;
        }

        // Folder centroids table
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS folder_centroids (
                folder     TEXT PRIMARY KEY,
                centroid   BLOB NOT NULL,
                file_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )?;

        // Tag registry table
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tag_registry (
                name        TEXT PRIMARY KEY,
                usage_count INTEGER NOT NULL DEFAULT 0,
                last_used   TEXT,
                created_by  TEXT NOT NULL DEFAULT 'indexer'
            );",
        )?;

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
        let mut stmt = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query_map(params![key], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(val) => Ok(Some(val?)),
            None => Ok(None),
        }
    }

    // ── Files ───────────────────────────────────────────────────

    pub fn insert_file(
        &self,
        path: &str,
        hash: &str,
        mtime: i64,
        tags: &[String],
        docid: &str,
    ) -> Result<i64> {
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let now = chrono_now();
        self.conn.execute(
            "INSERT INTO files (path, content_hash, mtime, tags, indexed_at, docid)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(path) DO UPDATE SET
                content_hash = excluded.content_hash,
                mtime        = excluded.mtime,
                tags         = excluded.tags,
                indexed_at   = excluded.indexed_at,
                docid        = excluded.docid",
            params![path, hash, mtime, tags_json, now, docid],
        )?;
        let file_id: i64 = self.conn.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![path],
            |row| row.get(0),
        )?;
        Ok(file_id)
    }

    pub fn get_file(&self, path: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid FROM files WHERE path = ?1",
        )?;
        let mut rows = stmt.query_map(params![path], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
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
            .prepare("SELECT id, path, content_hash, mtime, tags, indexed_at, docid FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
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

    /// Insert a chunk with its embedding vector stored as a BLOB.
    pub fn insert_chunk_with_vector(
        &self,
        file_id: i64,
        heading: &str,
        snippet: &str,
        vector_id: u64,
        token_count: i64,
        vector: &[f32],
    ) -> Result<()> {
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO chunks (file_id, heading, snippet, vector_id, token_count, vector)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                file_id,
                heading,
                snippet,
                vector_id as i64,
                token_count,
                vector_bytes
            ],
        )?;
        Ok(())
    }

    /// Get all stored vectors with their IDs.
    /// Returns (vector_id, vector) pairs.
    pub fn get_all_vectors(&self) -> Result<Vec<(u64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id, vector FROM chunks WHERE vector IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            let vid: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let vector: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((vid as u64, vector))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
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
        let mut stmt = self
            .conn
            .prepare("INSERT OR IGNORE INTO tombstones (vector_id, created_at) VALUES (?1, ?2)")?;
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

    // ── Edges ──────────────────────────────────────────────────

    /// Insert an edge. Uses INSERT OR IGNORE for the UNIQUE constraint.
    pub fn insert_edge(&self, from_file: i64, to_file: i64, edge_type: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (from_file, to_file, edge_type) VALUES (?1, ?2, ?3)",
            params![from_file, to_file, edge_type],
        )?;
        Ok(())
    }

    /// Delete all edges involving a file (both directions: from_file OR to_file).
    pub fn delete_edges_for_file(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM edges WHERE from_file = ?1 OR to_file = ?1",
            params![file_id],
        )?;
        Ok(())
    }

    /// Clear all edges (used during --rebuild).
    pub fn clear_edges(&self) -> Result<()> {
        self.conn.execute("DELETE FROM edges", [])?;
        Ok(())
    }

    /// Get outgoing edges, optionally filtered by type.
    pub fn get_outgoing(
        &self,
        file_id: i64,
        edge_type: Option<&str>,
    ) -> Result<Vec<(i64, String)>> {
        let mut results = Vec::new();
        match edge_type {
            Some(et) => {
                let mut stmt = self.conn.prepare(
                    "SELECT to_file, edge_type FROM edges WHERE from_file = ?1 AND edge_type = ?2",
                )?;
                let rows = stmt.query_map(params![file_id, et], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
            None => {
                let mut stmt = self
                    .conn
                    .prepare("SELECT to_file, edge_type FROM edges WHERE from_file = ?1")?;
                let rows = stmt.query_map(params![file_id], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
        }
        Ok(results)
    }

    /// Get incoming edges, optionally filtered by type.
    pub fn get_incoming(
        &self,
        file_id: i64,
        edge_type: Option<&str>,
    ) -> Result<Vec<(i64, String)>> {
        let mut results = Vec::new();
        match edge_type {
            Some(et) => {
                let mut stmt = self.conn.prepare(
                    "SELECT from_file, edge_type FROM edges WHERE to_file = ?1 AND edge_type = ?2",
                )?;
                let rows = stmt.query_map(params![file_id, et], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
            None => {
                let mut stmt = self
                    .conn
                    .prepare("SELECT from_file, edge_type FROM edges WHERE to_file = ?1")?;
                let rows = stmt.query_map(params![file_id], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
        }
        Ok(results)
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
        let (edge_count, wikilink_count, mention_count) = match self.get_edge_stats() {
            Ok(es) => (
                Some(es.total_edges),
                Some(es.wikilink_count),
                Some(es.mention_count),
            ),
            Err(_) => (None, None, None),
        };
        Ok(StoreStats {
            file_count: file_count as usize,
            chunk_count: chunk_count as usize,
            tombstone_count,
            last_indexed_at,
            vault_path,
            edge_count,
            wikilink_count,
            mention_count,
        })
    }

    /// Look up a file's path by its row ID.
    pub fn get_file_path_by_id(&self, file_id: i64) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files WHERE id = ?1")?;
        let mut rows = stmt.query_map(params![file_id], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(val) => Ok(Some(val?)),
            None => Ok(None),
        }
    }

    /// Look up a file record by its row ID.
    pub fn get_file_by_id(&self, file_id: i64) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid FROM files WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![file_id], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    /// Look up a file by its 6-character docid.
    pub fn get_file_by_docid(&self, docid: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid FROM files WHERE docid = ?1",
        )?;
        let mut rows = stmt.query_map(params![docid], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    // ── FTS5 ──────────────────────────────────────────────────

    /// Ensure the FTS5 virtual table exists. Called during init.
    pub fn ensure_fts_table(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                file_id UNINDEXED,
                chunk_seq UNINDEXED
            );",
            )
            .context("failed to create FTS5 virtual table")?;
        Ok(())
    }

    /// Insert a chunk's text into the FTS5 table.
    pub fn insert_fts_chunk(&self, file_id: i64, chunk_seq: i64, text: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chunks_fts (content, file_id, chunk_seq) VALUES (?1, ?2, ?3)",
            params![text, file_id, chunk_seq],
        )?;
        Ok(())
    }

    /// Delete all FTS5 entries for a file.
    pub fn delete_fts_chunks_for_file(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE file_id = ?1",
            params![file_id],
        )?;
        Ok(())
    }

    /// Search the FTS5 index. Returns results ranked by BM25 score.
    /// BM25 in SQLite returns negative values (more negative = better match),
    /// so we negate them to get positive scores where higher = better.
    ///
    /// The query is wrapped in double quotes so that FTS5 treats it as a
    /// phrase/literal rather than interpreting operators like `-`.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>> {
        // Escape any double quotes in the query, then wrap in double quotes
        // so FTS5 treats hyphens etc. as literal characters.
        let escaped = query.replace('"', "\"\"");
        let fts_query = format!("\"{}\"", escaped);

        let mut stmt = self.conn.prepare(
            "SELECT file_id, chunk_seq, bm25(chunks_fts) as score,
                    snippet(chunks_fts, 0, '<b>', '</b>', '...', 64)
             FROM chunks_fts
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
            Ok(FtsResult {
                file_id: row.get(0)?,
                chunk_seq: row.get(1)?,
                score: {
                    let raw: f64 = row.get(2)?;
                    -raw // negate: SQLite BM25 returns negative, more negative = better
                },
                snippet: row.get(3)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
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

    // ── Graph helpers ────────────────────────────────────────────

    /// Get neighbor file IDs within N hops via wikilinks.
    /// Uses Rust-side BFS, not recursive SQL CTE.
    pub fn get_neighbors(&self, file_id: i64, depth: usize) -> Result<Vec<(i64, usize)>> {
        use std::collections::VecDeque;
        let mut visited = HashSet::new();
        visited.insert(file_id);
        let mut queue = VecDeque::new();
        let mut results = Vec::new();
        queue.push_back((file_id, 0usize));
        while let Some((current, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }
            let outgoing = self.get_outgoing(current, Some("wikilink"))?;
            for (neighbor_id, _) in outgoing {
                if visited.insert(neighbor_id) {
                    let hop = current_depth + 1;
                    results.push((neighbor_id, hop));
                    queue.push_back((neighbor_id, hop));
                }
            }
        }
        Ok(results)
    }

    /// Find files that share at least one tag with the given file.
    pub fn get_shared_tags_files(&self, file_id: i64, limit: usize) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT f2.id
             FROM files f1
             JOIN files f2 ON f2.id != f1.id
             WHERE f1.id = ?1
             AND EXISTS (
                 SELECT 1 FROM json_each(f1.tags) t1
                 JOIN json_each(f2.tags) t2 ON t1.value = t2.value
             )
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![file_id, limit as i64], |row| row.get::<_, i64>(0))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Check if a file's FTS5 content contains a term. Escapes for FTS5.
    pub fn file_contains_term(&self, file_id: i64, term: &str) -> Result<bool> {
        let escaped = term.replace('"', "\"\"");
        let query = format!("\"{}\"", escaped);
        let result: Result<i64, _> = self.conn.query_row(
            "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ?1 AND file_id = ?2 LIMIT 1",
            params![query, file_id],
            |row| row.get(0),
        );
        Ok(result.is_ok())
    }

    /// Get the best (highest token_count) chunk for a file.
    pub fn get_best_chunk_for_file(&self, file_id: i64) -> Result<Option<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT heading, snippet FROM chunks WHERE file_id = ?1 ORDER BY token_count DESC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(params![file_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Get statistics about edges in the graph.
    pub fn get_edge_stats(&self) -> Result<EdgeStats> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        let wikilinks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'wikilink'",
            [],
            |r| r.get(0),
        )?;
        let mentions: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'mention'",
            [],
            |r| r.get(0),
        )?;
        let connected: i64 = self.conn.query_row(
            "SELECT COUNT(DISTINCT id) FROM files WHERE id IN \
             (SELECT from_file FROM edges UNION SELECT to_file FROM edges)",
            [],
            |r| r.get(0),
        )?;
        let total_files: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        Ok(EdgeStats {
            total_edges: total as usize,
            wikilink_count: wikilinks as usize,
            mention_count: mentions as usize,
            connected_file_count: connected as usize,
            isolated_file_count: (total_files - connected) as usize,
        })
    }

    /// List files filtered by folder prefix and/or tags (AND logic).
    pub fn list_files(
        &self,
        folder: Option<&str>,
        tags: &[String],
        limit: usize,
    ) -> Result<Vec<FileRecord>> {
        let mut sql = String::from(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid FROM files WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(f) = folder {
            sql.push_str(" AND path LIKE ?");
            param_values.push(Box::new(format!("{}%", f)));
        }
        for tag in tags {
            sql.push_str(" AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)");
            param_values.push(Box::new(tag.clone()));
        }
        sql.push_str(" ORDER BY indexed_at DESC LIMIT ?");
        param_values.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(param_values.iter()), |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Top-level folder grouping with note counts.
    pub fn folder_note_counts(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT CASE WHEN instr(path, '/') > 0
                    THEN substr(path, 1, instr(path, '/') - 1)
                    ELSE '(root)'
                    END AS folder,
                    COUNT(*) as cnt
             FROM files GROUP BY folder ORDER BY cnt DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Tag frequency aggregation via json_each.
    pub fn top_tags(&self, limit: usize) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT value, COUNT(*) as cnt
             FROM files, json_each(files.tags)
             GROUP BY value ORDER BY cnt DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Most recently indexed files.
    pub fn recent_files(&self, limit: usize) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid
             FROM files ORDER BY indexed_at DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Total edges (both directions) for a given file.
    pub fn edge_count_for_file(&self, file_id: i64) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE from_file = ?1 OR to_file = ?1",
            params![file_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get edge counts for multiple files in a single query.
    pub fn edge_counts_for_files(
        &self,
        file_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, usize>> {
        use std::collections::HashMap;
        if file_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: Vec<String> = file_ids.iter().map(|_| "?".to_string()).collect();
        let ph = placeholders.join(",");
        let sql = format!(
            "SELECT fid, COUNT(*) FROM (
                SELECT from_file AS fid FROM edges WHERE from_file IN ({ph})
                UNION ALL
                SELECT to_file AS fid FROM edges WHERE to_file IN ({ph})
            ) GROUP BY fid"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = file_ids
            .iter()
            .chain(file_ids.iter())
            .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut map = HashMap::new();
        for row in rows {
            let (id, count) = row?;
            map.insert(id, count);
        }
        Ok(map)
    }

    /// Find a file by case-insensitive basename match. Returns first match (shortest path).
    pub fn find_file_by_basename(&self, basename: &str) -> Result<Option<FileRecord>> {
        let target = if basename.ends_with(".md") {
            basename.to_string()
        } else {
            format!("{}.md", basename)
        };
        // Try exact path first
        if let Some(f) = self.get_file(&target)? {
            return Ok(Some(f));
        }
        // Basename match via SQL
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid FROM files
             WHERE lower(path) LIKE '%/' || lower(?1) OR lower(path) = lower(?1)
             ORDER BY length(path) ASC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(params![target], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    // ── Vec (sqlite-vec) ────────────────────────────────────────

    pub fn insert_vec(&self, vector_id: u64, embedding: &[f32]) -> Result<()> {
        crate::vecstore::insert_vec(&self.conn, vector_id, embedding)
    }

    pub fn delete_vec(&self, vector_id: u64) -> Result<()> {
        crate::vecstore::delete_vec(&self.conn, vector_id)
    }

    pub fn search_vec(
        &self,
        query: &[f32],
        k: usize,
        tombstones: &std::collections::HashSet<u64>,
    ) -> Result<Vec<(u64, f32)>> {
        crate::vecstore::search_vec(&self.conn, query, k, tombstones)
    }

    pub fn clear_vec(&self) -> Result<()> {
        crate::vecstore::clear_vec(&self.conn)
    }

    // ── Transactions ────────────────────────────────────────────

    pub fn begin_transaction(&self) -> Result<()> {
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        Ok(())
    }

    pub fn commit(&self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    pub fn rollback(&self) -> Result<()> {
        self.conn.execute_batch("ROLLBACK")?;
        Ok(())
    }

    // ── Folder centroids ─────────────────────────────────────────

    pub fn upsert_folder_centroid(
        &self,
        folder: &str,
        centroid: &[f32],
        file_count: usize,
    ) -> Result<()> {
        let blob: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO folder_centroids (folder, centroid, file_count, updated_at)
             VALUES (?1, ?2, ?3, datetime('now'))
             ON CONFLICT(folder) DO UPDATE SET centroid = ?2, file_count = ?3, updated_at = datetime('now')",
            params![folder, blob, file_count as i64],
        )?;
        Ok(())
    }

    pub fn get_folder_centroids(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT folder, centroid FROM folder_centroids")?;
        let rows = stmt.query_map([], |row| {
            let folder: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let centroid: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((folder, centroid))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Helpers ─────────────────────────────────────────────────

    pub fn next_vector_id(&self) -> Result<u64> {
        let max: Option<i64> = self
            .conn
            .query_row("SELECT MAX(vector_id) FROM chunks", [], |row| row.get(0))
            .ok()
            .flatten();
        Ok(max.map_or(0, |m| m as u64 + 1))
    }

    // ── Tags ────────────────────────────────────────────────────

    /// Borrow the underlying connection (for modules that need direct access).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn resolve_tag(&self, proposed: &str) -> Result<crate::tags::TagResolution> {
        crate::tags::resolve_tag(&self.conn, proposed)
    }

    pub fn resolve_tags(&self, proposed: &[String]) -> Result<Vec<String>> {
        crate::tags::resolve_tags(&self.conn, proposed)
    }

    pub fn register_tag(&self, name: &str, created_by: &str) -> Result<()> {
        crate::tags::register_tag(&self.conn, name, created_by)
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
    use crate::docid::generate_docid;

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
        let docid = generate_docid("notes/test.md");
        let file_id = store
            .insert_file("notes/test.md", "abc123", 1700000000, &tags, &docid)
            .unwrap();
        assert!(file_id > 0);

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.path, "notes/test.md");
        assert_eq!(rec.content_hash, "abc123");
        assert_eq!(rec.mtime, 1700000000);
        assert_eq!(rec.tags, tags);
        assert_eq!(rec.docid.unwrap(), docid);
    }

    #[test]
    fn test_insert_and_get_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/chunk_test.md",
                "hash1",
                100,
                &[],
                &generate_docid("notes/chunk_test.md"),
            )
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
            .insert_file(
                "notes/del.md",
                "hash",
                100,
                &[],
                &generate_docid("notes/del.md"),
            )
            .unwrap();
        store.insert_chunk(file_id, "H", "snippet", 10, 5).unwrap();
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
        let docid = generate_docid("notes/change.md");
        let file_id = store
            .insert_file(
                "notes/change.md",
                "old_hash",
                100,
                &["tag1".to_string()],
                &docid,
            )
            .unwrap();
        store.insert_chunk(file_id, "H", "text", 50, 10).unwrap();
        store.insert_chunk(file_id, "H2", "text2", 51, 12).unwrap();

        // Simulate detecting hash change: collect old vector_ids for tombstoning.
        let old_vector_ids = store.get_vector_ids_for_file(file_id).unwrap();
        assert_eq!(old_vector_ids.len(), 2);
        assert!(old_vector_ids.contains(&50));
        assert!(old_vector_ids.contains(&51));

        // Tombstone old vectors, delete file (cascades chunks), re-insert.
        store.add_tombstones(&old_vector_ids).unwrap();
        store.delete_file(file_id).unwrap();

        let new_file_id = store
            .insert_file(
                "notes/change.md",
                "new_hash",
                200,
                &["tag1".to_string()],
                &docid,
            )
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

        store.set_meta("vault_path", "/home/user/vault").unwrap();
        assert_eq!(
            store.get_meta("vault_path").unwrap().unwrap(),
            "/home/user/vault"
        );

        // Update the value.
        store.set_meta("vault_path", "/other/vault").unwrap();
        assert_eq!(
            store.get_meta("vault_path").unwrap().unwrap(),
            "/other/vault"
        );

        // Verify stats reflects it.
        let st = store.stats().unwrap();
        assert_eq!(st.vault_path.unwrap(), "/other/vault");
    }

    #[test]
    fn test_get_file_by_docid() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/findme.md");
        store
            .insert_file("notes/findme.md", "hash", 100, &[], &docid)
            .unwrap();

        let rec = store.get_file_by_docid(&docid).unwrap().unwrap();
        assert_eq!(rec.path, "notes/findme.md");
        assert_eq!(rec.docid.unwrap(), docid);

        // Non-existent docid returns None.
        assert!(store.get_file_by_docid("ffffff").unwrap().is_none());
    }

    // ── Edge tests ─────────────────────────────────────────────

    /// Helper: create two files and return their IDs.
    fn setup_two_files(store: &Store) -> (i64, i64) {
        let a = store
            .insert_file("notes/a.md", "ha", 100, &[], &generate_docid("notes/a.md"))
            .unwrap();
        let b = store
            .insert_file("notes/b.md", "hb", 100, &[], &generate_docid("notes/b.md"))
            .unwrap();
        (a, b)
    }

    #[test]
    fn test_insert_and_get_edges() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);

        store.insert_edge(a, b, "wikilink").unwrap();

        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (b, "wikilink".to_string()));

        let inc = store.get_incoming(b, None).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0], (a, "wikilink".to_string()));

        // No edges in the other direction.
        assert!(store.get_outgoing(b, None).unwrap().is_empty());
        assert!(store.get_incoming(a, None).unwrap().is_empty());
    }

    #[test]
    fn test_delete_edges_for_file_both_directions() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        let c = store
            .insert_file("notes/c.md", "hc", 100, &[], &generate_docid("notes/c.md"))
            .unwrap();

        // a -> b, c -> a
        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(c, a, "mention").unwrap();

        // Delete edges for file a — should remove both.
        store.delete_edges_for_file(a).unwrap();

        assert!(store.get_outgoing(a, None).unwrap().is_empty());
        assert!(store.get_incoming(a, None).unwrap().is_empty());
        assert!(store.get_incoming(b, None).unwrap().is_empty());
        assert!(store.get_outgoing(c, None).unwrap().is_empty());
    }

    #[test]
    fn test_edge_cascade_on_file_delete() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        let c = store
            .insert_file("notes/c.md", "hc", 100, &[], &generate_docid("notes/c.md"))
            .unwrap();

        // a -> b, b -> c
        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(b, c, "mention").unwrap();

        // Delete file b — CASCADE should remove both edges.
        store.delete_file(b).unwrap();

        assert!(store.get_outgoing(a, None).unwrap().is_empty());
        assert!(store.get_incoming(c, None).unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_edge_ignored() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(a, b, "wikilink").unwrap(); // duplicate

        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 1);

        // Same pair with different type is NOT a duplicate.
        store.insert_edge(a, b, "mention").unwrap();
        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_get_outgoing_filtered_by_type() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        let c = store
            .insert_file("notes/c.md", "hc", 100, &[], &generate_docid("notes/c.md"))
            .unwrap();

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(a, c, "mention").unwrap();

        let wikilinks = store.get_outgoing(a, Some("wikilink")).unwrap();
        assert_eq!(wikilinks.len(), 1);
        assert_eq!(wikilinks[0].0, b);

        let mentions = store.get_outgoing(a, Some("mention")).unwrap();
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].0, c);

        // Incoming filtered.
        let inc = store.get_incoming(b, Some("wikilink")).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].0, a);

        let inc = store.get_incoming(b, Some("mention")).unwrap();
        assert!(inc.is_empty());
    }

    // ── Graph helper tests ─────────────────────────────────────

    #[test]
    fn test_get_neighbors_depth_1() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("n/f1.md", "h1", 100, &[], &generate_docid("n/f1.md"))
            .unwrap();
        let f2 = store
            .insert_file("n/f2.md", "h2", 100, &[], &generate_docid("n/f2.md"))
            .unwrap();
        let f3 = store
            .insert_file("n/f3.md", "h3", 100, &[], &generate_docid("n/f3.md"))
            .unwrap();

        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f1, f3, "wikilink").unwrap();

        let neighbors = store.get_neighbors(f1, 1).unwrap();
        assert_eq!(neighbors.len(), 2);

        let ids: Vec<i64> = neighbors.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&f2));
        assert!(ids.contains(&f3));

        // All at depth 1.
        for (_, d) in &neighbors {
            assert_eq!(*d, 1);
        }
    }

    #[test]
    fn test_get_neighbors_depth_2() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("n/f1.md", "h1", 100, &[], &generate_docid("n/f1.md"))
            .unwrap();
        let f2 = store
            .insert_file("n/f2.md", "h2", 100, &[], &generate_docid("n/f2.md"))
            .unwrap();
        let f3 = store
            .insert_file("n/f3.md", "h3", 100, &[], &generate_docid("n/f3.md"))
            .unwrap();
        let f4 = store
            .insert_file("n/f4.md", "h4", 100, &[], &generate_docid("n/f4.md"))
            .unwrap();

        // f1 -> f2 -> f3 -> f4
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f3, "wikilink").unwrap();
        store.insert_edge(f3, f4, "wikilink").unwrap();

        let neighbors = store.get_neighbors(f1, 2).unwrap();
        assert_eq!(neighbors.len(), 2);

        // f2 at depth 1, f3 at depth 2, f4 NOT included.
        let map: std::collections::HashMap<i64, usize> = neighbors.into_iter().collect();
        assert_eq!(map[&f2], 1);
        assert_eq!(map[&f3], 2);
        assert!(!map.contains_key(&f4));
    }

    #[test]
    fn test_get_shared_tags_files() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/f1.md",
                "h1",
                100,
                &["rust".to_string(), "cli".to_string()],
                &generate_docid("n/f1.md"),
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "n/f2.md",
                "h2",
                100,
                &["rust".to_string(), "web".to_string()],
                &generate_docid("n/f2.md"),
            )
            .unwrap();
        let _f3 = store
            .insert_file(
                "n/f3.md",
                "h3",
                100,
                &["python".to_string()],
                &generate_docid("n/f3.md"),
            )
            .unwrap();

        let shared = store.get_shared_tags_files(f1, 10).unwrap();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0], f2);
    }

    #[test]
    fn test_file_contains_term() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("n/fts.md", "h1", 100, &[], &generate_docid("n/fts.md"))
            .unwrap();

        store
            .insert_fts_chunk(f1, 0, "BRE-2579 delivery date extension")
            .unwrap();

        assert!(store.file_contains_term(f1, "delivery").unwrap());
        assert!(store.file_contains_term(f1, "extension").unwrap());
        assert!(!store.file_contains_term(f1, "checkout").unwrap());
    }

    #[test]
    fn test_get_best_chunk_for_file() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("n/best.md", "h1", 100, &[], &generate_docid("n/best.md"))
            .unwrap();

        store
            .insert_chunk(f1, "Small heading", "small snippet", 1, 10)
            .unwrap();
        store
            .insert_chunk(f1, "Big heading", "big snippet", 2, 100)
            .unwrap();

        let best = store.get_best_chunk_for_file(f1).unwrap().unwrap();
        assert_eq!(best.0, "Big heading");
        assert_eq!(best.1, "big snippet");
    }

    #[test]
    fn test_get_edge_stats() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("n/a.md", "ha", 100, &[], &generate_docid("n/a.md"))
            .unwrap();
        let b = store
            .insert_file("n/b.md", "hb", 100, &[], &generate_docid("n/b.md"))
            .unwrap();
        let c = store
            .insert_file("n/c.md", "hc", 100, &[], &generate_docid("n/c.md"))
            .unwrap();
        // d is isolated (no edges).
        let _d = store
            .insert_file("n/d.md", "hd", 100, &[], &generate_docid("n/d.md"))
            .unwrap();

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(a, c, "wikilink").unwrap();
        store.insert_edge(b, c, "mention").unwrap();

        let stats = store.get_edge_stats().unwrap();
        assert_eq!(stats.total_edges, 3);
        assert_eq!(stats.wikilink_count, 2);
        assert_eq!(stats.mention_count, 1);
        assert_eq!(stats.connected_file_count, 3); // a, b, c
        assert_eq!(stats.isolated_file_count, 1); // d
    }

    #[test]
    fn test_list_files_no_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, &["rust".into()], "aaa111")
            .unwrap();
        store
            .insert_file("02-Areas/b.md", "h2", 200, &["health".into()], "bbb222")
            .unwrap();
        store
            .insert_file(
                "01-Projects/c.md",
                "h3",
                300,
                &["rust".into(), "cli".into()],
                "ccc333",
            )
            .unwrap();
        let files = store.list_files(None, &[], 20).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_list_files_folder_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, &[], "aaa111")
            .unwrap();
        store
            .insert_file("02-Areas/b.md", "h2", 200, &[], "bbb222")
            .unwrap();
        let files = store.list_files(Some("01-Projects"), &[], 20).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "01-Projects/a.md");
    }

    #[test]
    fn test_list_files_tag_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("a.md", "h1", 100, &["rust".into(), "cli".into()], "aaa111")
            .unwrap();
        store
            .insert_file("b.md", "h2", 200, &["rust".into()], "bbb222")
            .unwrap();
        store
            .insert_file("c.md", "h3", 300, &["python".into()], "ccc333")
            .unwrap();
        let files = store.list_files(None, &["rust".into()], 20).unwrap();
        assert_eq!(files.len(), 2);
        let files = store
            .list_files(None, &["rust".into(), "cli".into()], 20)
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "a.md");
    }

    #[test]
    fn test_folder_note_counts() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, &[], "a1")
            .unwrap();
        store
            .insert_file("01-Projects/b.md", "h2", 100, &[], "b2")
            .unwrap();
        store
            .insert_file("02-Areas/c.md", "h3", 100, &[], "c3")
            .unwrap();
        store.insert_file("root.md", "h4", 100, &[], "d4").unwrap();
        let counts = store.folder_note_counts().unwrap();
        assert!(counts.iter().any(|(f, c)| f == "01-Projects" && *c == 2));
        assert!(counts.iter().any(|(f, c)| f == "02-Areas" && *c == 1));
        assert!(counts.iter().any(|(f, c)| f == "(root)" && *c == 1));
    }

    #[test]
    fn test_top_tags() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("a.md", "h1", 100, &["rust".into(), "cli".into()], "a1")
            .unwrap();
        store
            .insert_file("b.md", "h2", 100, &["rust".into(), "web".into()], "b2")
            .unwrap();
        store
            .insert_file("c.md", "h3", 100, &["rust".into()], "c3")
            .unwrap();
        let tags = store.top_tags(10).unwrap();
        assert_eq!(tags[0].0, "rust");
        assert_eq!(tags[0].1, 3);
    }

    #[test]
    fn test_recent_files() {
        let store = Store::open_memory().unwrap();
        store.insert_file("old.md", "h1", 100, &[], "a1").unwrap();
        store.insert_file("new.md", "h2", 200, &[], "b2").unwrap();
        let recent = store.recent_files(1).unwrap();
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_edge_count_for_file() {
        let store = Store::open_memory().unwrap();
        let f1 = store.insert_file("a.md", "h1", 100, &[], "a1").unwrap();
        let f2 = store.insert_file("b.md", "h2", 100, &[], "b2").unwrap();
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f1, "wikilink").unwrap();
        assert_eq!(store.edge_count_for_file(f1).unwrap(), 2);
        assert_eq!(store.edge_count_for_file(f2).unwrap(), 2);
    }

    #[test]
    fn test_find_file_by_basename() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/Work/note.md", "h1", 100, &[], "aaa111")
            .unwrap();
        store
            .insert_file("root.md", "h2", 100, &[], "bbb222")
            .unwrap();

        let found = store.find_file_by_basename("note").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().path, "01-Projects/Work/note.md");

        let found = store.find_file_by_basename("note.md").unwrap();
        assert!(found.is_some());

        let found = store.find_file_by_basename("nonexistent").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_edge_counts_for_files() {
        let store = Store::open_memory().unwrap();
        let f1 = store.insert_file("a.md", "h1", 100, &[], "a1").unwrap();
        let f2 = store.insert_file("b.md", "h2", 100, &[], "b2").unwrap();
        let f3 = store.insert_file("c.md", "h3", 100, &[], "c3").unwrap();
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f1, "wikilink").unwrap();
        store.insert_edge(f1, f3, "wikilink").unwrap();
        let counts = store.edge_counts_for_files(&[f1, f2, f3]).unwrap();
        assert_eq!(*counts.get(&f1).unwrap(), 3);
        assert_eq!(*counts.get(&f2).unwrap(), 2);
        assert_eq!(*counts.get(&f3).unwrap(), 1);
        // Empty input returns empty map
        let empty = store.edge_counts_for_files(&[]).unwrap();
        assert!(empty.is_empty());
    }

    // ── Vec integration tests ───────────────────────────────────

    #[test]
    fn test_store_has_vec_table() {
        let store = Store::open_memory().unwrap();
        let count: i64 = store
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_store_vec_roundtrip() {
        let store = Store::open_memory().unwrap();
        let vector: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        store.insert_vec(0, &vector).unwrap();

        let results = store
            .search_vec(&vector, 1, &std::collections::HashSet::new())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_migrate_vectors_to_vec0() {
        let store = Store::open_memory().unwrap();
        // Insert a file + chunk with a vector BLOB.
        let file_id = store
            .insert_file("test.md", "hash123", 0, &[], "abc123")
            .unwrap();
        let vector: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        store
            .insert_chunk_with_vector(file_id, "heading", "snippet", 0, 100, &vector)
            .unwrap();

        // Clear vec0 to simulate a pre-migration state, then re-run the migration.
        store.clear_vec().unwrap();
        store.migrate_vectors_to_vec0().unwrap();

        // Verify vec0 is now populated.
        let results = store
            .search_vec(&vector, 1, &std::collections::HashSet::new())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_store_transaction() {
        let store = Store::open_memory().unwrap();
        store.begin_transaction().unwrap();
        store.set_meta("test_key", "test_value").unwrap();
        store.commit().unwrap();
        assert_eq!(
            store.get_meta("test_key").unwrap(),
            Some("test_value".into())
        );
    }

    #[test]
    fn test_next_vector_id_empty() {
        let store = Store::open_memory().unwrap();
        assert_eq!(store.next_vector_id().unwrap(), 0);
    }
}
