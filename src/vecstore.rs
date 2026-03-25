use std::collections::HashSet;

use anyhow::Result;
use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;

/// Register the sqlite-vec extension as an auto-extension.
///
/// Must be called **before** opening any `Connection` that needs vec0 tables.
/// Safe to call multiple times (idempotent).
pub fn init_sqlite_vec() {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(
            #[allow(clippy::missing_transmute_annotations)]
            std::mem::transmute(sqlite3_vec_init as *const ()),
        ));
    }
}

/// Create the `chunks_vec` virtual table if it doesn't already exist.
pub fn init_vec_table(conn: &Connection, dim: usize) -> Result<()> {
    conn.execute(
        &format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[{dim}] distance_metric=cosine
            )"
        ),
        [],
    )?;
    Ok(())
}

/// Insert a vector with the given ID.
pub fn insert_vec(conn: &Connection, vector_id: u64, embedding: &[f32]) -> Result<()> {
    use zerocopy::AsBytes;
    conn.execute(
        "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
        rusqlite::params![vector_id as i64, embedding.as_bytes()],
    )?;
    Ok(())
}

/// Delete a vector by its ID.
pub fn delete_vec(conn: &Connection, vector_id: u64) -> Result<()> {
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid = ?",
        rusqlite::params![vector_id as i64],
    )?;
    Ok(())
}

/// Search for the `k` nearest neighbors of `query`, excluding `tombstones`.
///
/// Returns `(vector_id, distance)` pairs sorted by ascending distance.
/// Cosine distance: 0.0 = identical, 2.0 = opposite.
pub fn search_vec(
    conn: &Connection,
    query: &[f32],
    k: usize,
    tombstones: &HashSet<u64>,
) -> Result<Vec<(u64, f32)>> {
    use zerocopy::AsBytes;

    // Request extra results to compensate for tombstone filtering.
    let fetch_k = if tombstones.is_empty() { k } else { k * 2 };

    let mut stmt = conn.prepare(
        "SELECT rowid, distance
         FROM chunks_vec
         WHERE embedding MATCH ?1
           AND k = ?2",
    )?;

    let rows = stmt.query_map(rusqlite::params![query.as_bytes(), fetch_k as i64], |row| {
        let id: i64 = row.get(0)?;
        let dist: f32 = row.get(1)?;
        Ok((id as u64, dist))
    })?;

    let mut results: Vec<(u64, f32)> = Vec::new();
    for row in rows {
        let (id, dist) = row?;
        if tombstones.contains(&id) {
            continue;
        }
        results.push((id, dist));
        if results.len() == k {
            break;
        }
    }

    Ok(results)
}

/// Delete all vectors from the table.
pub fn clear_vec(conn: &Connection) -> Result<()> {
    conn.execute("DELETE FROM chunks_vec", [])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_conn() -> Connection {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_vec_table(&conn, 384).unwrap();
        conn
    }

    fn random_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..dim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_init_vec_table() {
        let conn = setup_conn();
        // Verify the table exists by querying sqlite_master.
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "chunks_vec table should exist");
    }

    #[test]
    fn test_insert_and_search() {
        let conn = setup_conn();
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| random_vector(i, 384)).collect();

        for (i, v) in vectors.iter().enumerate() {
            insert_vec(&conn, i as u64, v).unwrap();
        }

        let results = search_vec(&conn, &vectors[0], 5, &HashSet::new()).unwrap();
        assert!(!results.is_empty(), "search returned no results");
        assert_eq!(
            results[0].0, 0,
            "expected the query vector itself to be the top result"
        );
        assert!(
            results[0].1 < 0.01,
            "distance to self should be near zero, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_search_with_tombstones() {
        let conn = setup_conn();
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| random_vector(i + 100, 384)).collect();

        for (i, v) in vectors.iter().enumerate() {
            insert_vec(&conn, i as u64, v).unwrap();
        }

        let mut tombstones = HashSet::new();
        tombstones.insert(0u64);

        let results = search_vec(&conn, &vectors[0], 5, &tombstones).unwrap();
        for (id, _) in &results {
            assert_ne!(*id, 0, "tombstoned ID should not appear in results");
        }
    }

    #[test]
    fn test_delete_vec() {
        let conn = setup_conn();
        insert_vec(&conn, 1, &random_vector(42, 384)).unwrap();

        let count_before: i64 = conn
            .query_row("SELECT count(*) FROM chunks_vec", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count_before, 1);

        delete_vec(&conn, 1).unwrap();

        let count_after: i64 = conn
            .query_row("SELECT count(*) FROM chunks_vec", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count_after, 0);
    }

    #[test]
    fn test_empty_search() {
        let conn = setup_conn();
        let query = random_vector(999, 384);
        let results = search_vec(&conn, &query, 5, &HashSet::new()).unwrap();
        assert!(results.is_empty(), "empty table should return no results");
    }

    #[test]
    fn test_init_vec_table_custom_dim() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_vec_table(&conn, 256).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Insert and search with 256-dim vector
        let vec256: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        insert_vec(&conn, 1, &vec256).unwrap();
        let results = search_vec(&conn, &vec256, 1, &HashSet::new()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
