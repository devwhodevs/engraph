//! Integration tests for the write pipeline.
//! Run with: cargo test --test write_pipeline -- --ignored

use std::path::Path;

use engraph::embedder::Embedder;
use engraph::store::Store;
use engraph::vecstore;
use engraph::writer::{AppendInput, CreateNoteInput, append_to_note, create_note};

fn setup(vault_dir: &Path) -> (Store, Embedder) {
    // Register sqlite-vec extension
    vecstore::init_sqlite_vec();

    // Create minimal vault structure
    std::fs::create_dir_all(vault_dir.join("00-Inbox")).unwrap();
    std::fs::create_dir_all(vault_dir.join("03-Resources/People")).unwrap();
    std::fs::write(
        vault_dir.join("03-Resources/People/Steve Barbera.md"),
        "# Steve Barbera\n\nRole: VP Engineering\n",
    )
    .unwrap();

    // Open store and set vault path
    let data_dir = tempfile::TempDir::new().unwrap();
    let db_path = data_dir.path().join("engraph.db");
    let store = Store::open(&db_path).unwrap();
    store
        .set_meta("vault_path", &vault_dir.to_string_lossy())
        .unwrap();

    // Index the existing file so it's in the store
    let docid = engraph::docid::generate_docid("03-Resources/People/Steve Barbera.md");
    store
        .insert_file(
            "03-Resources/People/Steve Barbera.md",
            "hash1",
            0,
            &[],
            &docid,
        )
        .unwrap();

    // Load embedder
    let models_dir = engraph::config::Config::data_dir().unwrap().join("models");
    let embedder = Embedder::new(&models_dir).unwrap();

    (store, embedder)
}

#[test]
#[ignore] // requires model download
fn test_create_note_is_immediately_searchable() {
    let vault_dir = tempfile::TempDir::new().unwrap();
    let (store, mut embedder) = setup(vault_dir.path());

    let input = CreateNoteInput {
        content:
            "# RRF Tuning Notes\n\nWe tested reciprocal rank fusion with k=60 and got good results."
                .into(),
        filename: Some("RRF Tuning".into()),
        type_hint: None,
        tags: vec!["engraph".into(), "search".into()],
        folder: Some("00-Inbox".into()),
        created_by: "test".into(),
    };

    let result = create_note(input, &store, &mut embedder, vault_dir.path(), None).unwrap();
    assert!(result.path.starts_with("00-Inbox/"));
    assert!(result.path.ends_with(".md"));
    assert!(!result.docid.is_empty());

    // Verify the file exists on disk
    assert!(vault_dir.path().join(&result.path).exists());

    // Verify it's immediately searchable via sqlite-vec
    let search =
        engraph::search::search_internal("reciprocal rank fusion", 5, &store, &mut embedder)
            .unwrap();
    assert!(
        !search.results.is_empty(),
        "created note should be searchable immediately"
    );
    assert!(
        search.results.iter().any(|r| r.file_path == result.path),
        "created note should appear in search results"
    );
}

#[test]
#[ignore]
fn test_append_updates_index() {
    let vault_dir = tempfile::TempDir::new().unwrap();
    let (store, mut embedder) = setup(vault_dir.path());

    // Create a note first
    let input = CreateNoteInput {
        content: "# Meeting Notes\n\nDiscussed the roadmap for Q2.".into(),
        filename: Some("Meeting 2026-03-25".into()),
        type_hint: None,
        tags: vec![],
        folder: Some("00-Inbox".into()),
        created_by: "test".into(),
    };
    let created = create_note(input, &store, &mut embedder, vault_dir.path(), None).unwrap();

    // Append new content
    let append_input = AppendInput {
        file: created.path.clone(),
        content: "## Action Items\n\n- Ship sqlite-vec migration by Friday\n- Review PR #42".into(),
        modified_by: "test".into(),
    };
    let _appended = append_to_note(append_input, &store, &mut embedder, vault_dir.path()).unwrap();

    // Verify appended content is searchable
    let search =
        engraph::search::search_internal("sqlite-vec migration", 5, &store, &mut embedder).unwrap();
    assert!(
        search.results.iter().any(|r| r.file_path == created.path),
        "appended content should be searchable"
    );
}

#[test]
#[ignore]
fn test_conflict_detection() {
    let vault_dir = tempfile::TempDir::new().unwrap();
    let (store, mut embedder) = setup(vault_dir.path());

    let input = CreateNoteInput {
        content: "# Test Note\n\nOriginal content.".into(),
        filename: Some("conflict-test".into()),
        type_hint: None,
        tags: vec![],
        folder: Some("00-Inbox".into()),
        created_by: "test".into(),
    };
    let created = create_note(input, &store, &mut embedder, vault_dir.path(), None).unwrap();

    // Modify file externally (simulates Obsidian edit)
    let abs_path = vault_dir.path().join(&created.path);
    // Wait a moment so mtime changes
    std::thread::sleep(std::time::Duration::from_millis(1100));
    std::fs::write(&abs_path, "# Modified externally\n\nNew content.").unwrap();

    // Attempt append — should fail with conflict
    let append_input = AppendInput {
        file: created.path,
        content: "appended content".into(),
        modified_by: "test".into(),
    };
    let result = append_to_note(append_input, &store, &mut embedder, vault_dir.path());
    assert!(result.is_err(), "should detect mtime conflict");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("mtime conflict") || err_msg.contains("CONFLICT"),
        "error should mention conflict, got: {}",
        err_msg
    );
}
