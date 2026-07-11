//! Model-free integration tests for the write pipeline.

use engraph::llm::MockLlm;
use engraph::search::search_internal;
use engraph::store::Store;
use engraph::writer::{AppendInput, CreateNoteInput, append_to_note, create_note};

struct Harness {
    vault_dir: tempfile::TempDir,
    store: Store,
    embedder: MockLlm,
}

impl Harness {
    fn new() -> Self {
        let vault_dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(vault_dir.path().join("00-Inbox")).unwrap();

        Self {
            store: Store::open_memory().unwrap(),
            embedder: MockLlm::new(256),
            vault_dir,
        }
    }

    fn create_note(&mut self, title: &str, body: &str) -> engraph::writer::WriteResult {
        create_note(
            CreateNoteInput {
                content: format!("# {title}\n\n{body}"),
                filename: Some(title.to_string()),
                type_hint: None,
                tags: vec!["engraph".to_string()],
                folder: Some("00-Inbox".to_string()),
                created_by: "integration-test".to_string(),
                auto_link: Some(false),
            },
            &self.store,
            &mut self.embedder,
            self.vault_dir.path(),
            None,
        )
        .unwrap()
    }
}

#[test]
fn create_note_is_immediately_searchable() {
    let mut harness = Harness::new();
    let created = harness.create_note(
        "RRF Tuning",
        "Reciprocal rank fusion uses a stable rank constant.",
    );

    assert!(harness.vault_dir.path().join(&created.path).is_file());
    let results = search_internal(
        "reciprocal rank fusion",
        5,
        &harness.store,
        &mut harness.embedder,
    )
    .unwrap();
    assert!(
        results
            .results
            .iter()
            .any(|result| result.file_path == created.path)
    );
}

#[test]
fn append_updates_the_search_index() {
    let mut harness = Harness::new();
    let created = harness.create_note("Meeting Notes", "Discussed the roadmap.");

    append_to_note(
        AppendInput {
            file: created.path.clone(),
            content: "## Action Items\n\nShip sqlite vector migration by Friday.".to_string(),
            modified_by: "integration-test".to_string(),
        },
        &harness.store,
        &mut harness.embedder,
        harness.vault_dir.path(),
    )
    .unwrap();

    let results = search_internal(
        "sqlite vector migration",
        5,
        &harness.store,
        &mut harness.embedder,
    )
    .unwrap();
    assert!(
        results
            .results
            .iter()
            .any(|result| result.file_path == created.path)
    );
}

#[test]
fn append_rejects_an_mtime_conflict() {
    let mut harness = Harness::new();
    let created = harness.create_note("Conflict Test", "Original content.");
    let record = harness.store.get_file(&created.path).unwrap().unwrap();
    harness
        .store
        .conn()
        .execute(
            "UPDATE files SET mtime = ?1 WHERE id = ?2",
            rusqlite::params![record.mtime - 1, record.id],
        )
        .unwrap();

    let result = append_to_note(
        AppendInput {
            file: created.path,
            content: "Unexpected append".to_string(),
            modified_by: "integration-test".to_string(),
        },
        &harness.store,
        &mut harness.embedder,
        harness.vault_dir.path(),
    );

    assert!(result.unwrap_err().to_string().contains("mtime conflict"));
}
