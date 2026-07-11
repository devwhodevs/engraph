//! Model-free integration tests for indexing and search.

use std::path::{Path, PathBuf};

use engraph::config::Config;
use engraph::indexer::run_index_shared;
use engraph::llm::MockLlm;
use engraph::search::search_internal;
use engraph::store::Store;
use tempfile::TempDir;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn copy_fixtures_to(dest: &Path) {
    std::fs::create_dir_all(dest).unwrap();
    for entry in std::fs::read_dir(fixtures_dir()).unwrap() {
        let entry = entry.unwrap();
        copy_entry(&entry.path(), &dest.join(entry.file_name()));
    }
}

fn copy_entry(source: &Path, destination: &Path) {
    if source.is_dir() {
        std::fs::create_dir_all(destination).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            copy_entry(&entry.path(), &destination.join(entry.file_name()));
        }
    } else {
        std::fs::copy(source, destination).unwrap();
    }
}

struct Harness {
    _data_dir: TempDir,
    vault_dir: TempDir,
    store: Store,
    embedder: MockLlm,
    config: Config,
}

impl Harness {
    fn new() -> Self {
        let data_dir = TempDir::new().unwrap();
        let vault_dir = TempDir::new().unwrap();
        copy_fixtures_to(vault_dir.path());

        Self {
            store: Store::open(&data_dir.path().join("engraph.db")).unwrap(),
            embedder: MockLlm::new(256),
            config: Config::default(),
            _data_dir: data_dir,
            vault_dir,
        }
    }

    fn index(&mut self) -> engraph::indexer::IndexResult {
        run_index_shared(
            self.vault_dir.path(),
            &self.config,
            &self.store,
            &mut self.embedder,
            false,
            None,
        )
        .unwrap()
    }
}

#[test]
fn full_index_is_searchable() {
    let mut harness = Harness::new();
    let indexed = harness.index();

    assert_eq!(indexed.new_files, 4);
    assert_eq!(harness.store.stats().unwrap().file_count, 4);

    let results = search_internal(
        "Rust error handling",
        5,
        &harness.store,
        &mut harness.embedder,
    )
    .unwrap();
    assert!(
        results
            .results
            .iter()
            .any(|result| result.file_path == "note1.md")
    );
}

#[test]
fn incremental_index_tracks_add_change_and_delete() {
    let mut harness = Harness::new();
    harness.index();

    std::fs::write(
        harness.vault_dir.path().join("note4.md"),
        "# Kubernetes Pods\n\nPods are deployable units.\n",
    )
    .unwrap();
    std::fs::write(
        harness.vault_dir.path().join("note2.md"),
        "# Python Basics\n\nPython has dynamic typing and pattern matching.\n",
    )
    .unwrap();
    std::fs::remove_file(harness.vault_dir.path().join("note3.md")).unwrap();

    let indexed = harness.index();
    assert_eq!(indexed.new_files, 1);
    assert_eq!(indexed.updated_files, 1);
    assert_eq!(harness.store.stats().unwrap().file_count, 4);
    assert!(harness.store.get_file("note4.md").unwrap().is_some());
    assert!(harness.store.get_file("note3.md").unwrap().is_none());

    let results =
        search_internal("Kubernetes Pods", 5, &harness.store, &mut harness.embedder).unwrap();
    assert!(
        results
            .results
            .iter()
            .any(|result| result.file_path == "note4.md")
    );
}
