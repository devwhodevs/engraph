//! Deterministic, model-free search regression battery.
//!
//! The old integration suites exercised deleted HNSW/Embedder modules. This
//! fixture uses the current Store + MockLlm path and repeats the same queries
//! to catch ranking nondeterminism without downloading a GGUF model.

use engraph::docid::generate_docid;
use engraph::llm::MockLlm;
use engraph::search::search_internal;
use engraph::store::Store;

fn fixture_store() -> Store {
    let store = Store::open_memory().unwrap();
    let embedder = MockLlm::new(256);
    let documents = [
        (
            "notes/alpha.md",
            "retrieval ranking architecture",
            "retrieval ranking",
        ),
        (
            "notes/beta.md",
            "retrieval ranking operations",
            "retrieval ranking",
        ),
        ("notes/gamma.md", "unrelated gardening notes", "unrelated"),
    ];

    for (path, snippet, vector_seed) in documents {
        let file_id = store
            .insert_file(path, path, 0, &[], &generate_docid(path), None, None)
            .unwrap();
        let vector_id = store.next_vector_id().unwrap();
        let vector = embedder.hash_to_vector(vector_seed);
        store
            .insert_chunk_with_vector(file_id, "# Fixture", snippet, vector_id, 3, &vector)
            .unwrap();
        store.insert_vec(vector_id, &vector).unwrap();
        store.insert_fts_chunk(file_id, 0, snippet).unwrap();
    }
    store
}

#[test]
fn golden_search_battery_is_stable() {
    let store = fixture_store();
    let mut embedder = MockLlm::new(256);

    for _ in 0..10 {
        let retrieval = search_internal("retrieval ranking", 2, &store, &mut embedder).unwrap();
        let retrieval_paths: Vec<&str> = retrieval
            .results
            .iter()
            .map(|result| result.file_path.as_str())
            .collect();
        assert_eq!(
            retrieval_paths,
            vec!["notes/alpha.md", "notes/beta.md"],
            "retrieval golden result changed"
        );

        let unrelated = search_internal("unrelated", 1, &store, &mut embedder).unwrap();
        assert_eq!(
            unrelated.results[0].file_path, "notes/gamma.md",
            "unrelated golden result changed"
        );
    }
}
