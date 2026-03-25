// FTS5 search support.
//
// The `FtsResult` struct and `fts_search` method live on `Store` (in store.rs)
// since the store owns the database connection. We re-export `FtsResult` here
// so downstream code can import it from either location.

pub use crate::store::FtsResult;

#[cfg(test)]
mod tests {
    use crate::docid::generate_docid;
    use crate::store::Store;

    fn setup_store() -> Store {
        let store = Store::open_memory().unwrap();
        store.ensure_fts_table().unwrap();
        store
    }

    #[test]
    fn test_fts_exact_match() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/ticket.md",
                "hash1",
                100,
                &[],
                &generate_docid("notes/ticket.md"),
                None,
            )
            .unwrap();

        store
            .insert_fts_chunk(file_id, 0, "BRE-2579 delivery date extension for checkout")
            .unwrap();

        let results = store.fts_search("BRE-2579", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_id, file_id);
        assert_eq!(results[0].chunk_seq, 0);
        assert!(
            results[0].score > 0.0,
            "score should be positive (negated BM25)"
        );
    }

    #[test]
    fn test_fts_no_match() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/note.md",
                "hash1",
                100,
                &[],
                &generate_docid("notes/note.md"),
                None,
            )
            .unwrap();

        store
            .insert_fts_chunk(file_id, 0, "Rust programming language guide")
            .unwrap();

        let results = store.fts_search("kubernetes", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fts_multiple_results() {
        let store = setup_store();

        let file_id1 = store
            .insert_file(
                "notes/a.md",
                "h1",
                100,
                &[],
                &generate_docid("notes/a.md"),
                None,
            )
            .unwrap();
        let file_id2 = store
            .insert_file(
                "notes/b.md",
                "h2",
                100,
                &[],
                &generate_docid("notes/b.md"),
                None,
            )
            .unwrap();
        let file_id3 = store
            .insert_file(
                "notes/c.md",
                "h3",
                100,
                &[],
                &generate_docid("notes/c.md"),
                None,
            )
            .unwrap();

        // Chunk with "delivery" appearing multiple times should rank higher.
        store
            .insert_fts_chunk(
                file_id1,
                0,
                "delivery date delivery schedule delivery tracking",
            )
            .unwrap();
        store
            .insert_fts_chunk(file_id2, 0, "delivery date for the checkout page")
            .unwrap();
        store
            .insert_fts_chunk(file_id3, 0, "unrelated content about Rust and WebAssembly")
            .unwrap();

        let results = store.fts_search("delivery", 10).unwrap();
        assert_eq!(results.len(), 2, "only 2 chunks mention 'delivery'");

        // Results should be sorted by score descending.
        assert!(
            results[0].score >= results[1].score,
            "results should be ranked by relevance"
        );
    }

    #[test]
    fn test_fts_delete_chunks_for_file() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/del.md",
                "hash1",
                100,
                &[],
                &generate_docid("notes/del.md"),
                None,
            )
            .unwrap();

        store
            .insert_fts_chunk(file_id, 0, "first chunk content")
            .unwrap();
        store
            .insert_fts_chunk(file_id, 1, "second chunk content")
            .unwrap();

        // Verify they exist.
        let results = store.fts_search("chunk", 10).unwrap();
        assert_eq!(results.len(), 2);

        // Delete and verify gone.
        store.delete_fts_chunks_for_file(file_id).unwrap();
        let results = store.fts_search("chunk", 10).unwrap();
        assert_eq!(results.len(), 0);
    }
}
