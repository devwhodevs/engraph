use crate::config::HttpConfig;

/// Build the OpenAPI 3.1.0 specification for all HTTP endpoints.
pub fn build_openapi_spec(server_url: &str) -> serde_json::Value {
    let mut paths = serde_json::Map::new();

    // Read endpoints
    paths.insert("/api/health-check".into(), build_health_check());
    paths.insert("/api/search".into(), build_search());
    paths.insert("/api/read/{file}".into(), build_read());
    paths.insert("/api/read-section".into(), build_read_section());
    paths.insert("/api/list".into(), build_list());
    paths.insert("/api/vault-map".into(), build_vault_map());
    paths.insert("/api/who/{name}".into(), build_who());
    paths.insert("/api/project/{name}".into(), build_project());
    paths.insert("/api/context".into(), build_context());
    paths.insert("/api/health".into(), build_health());

    // Write endpoints
    paths.insert("/api/create".into(), build_create());
    paths.insert("/api/append".into(), build_append());
    paths.insert("/api/edit".into(), build_edit());
    paths.insert("/api/rewrite".into(), build_rewrite());
    paths.insert("/api/edit-frontmatter".into(), build_edit_frontmatter());
    paths.insert("/api/move".into(), build_move());
    paths.insert("/api/archive".into(), build_archive());
    paths.insert("/api/unarchive".into(), build_unarchive());
    paths.insert("/api/update-metadata".into(), build_update_metadata());
    paths.insert("/api/delete".into(), build_delete());

    // Migration endpoints
    paths.insert("/api/migrate/preview".into(), build_migrate_preview());
    paths.insert("/api/migrate/apply".into(), build_migrate_apply());
    paths.insert("/api/migrate/undo".into(), build_migrate_undo());

    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "engraph",
            "version": "1.5.0",
            "description": "AI-powered semantic search and management API for Obsidian vaults."
        },
        "servers": [{ "url": server_url }],
        "security": [{ "bearerAuth": [] }],
        "components": {
            "schemas": {},
            "securitySchemes": {
                "bearerAuth": { "type": "http", "scheme": "bearer" }
            }
        },
        "paths": paths
    })
}

// ---------------------------------------------------------------------------
// Path builders — each returns one path item to keep macro recursion shallow
// ---------------------------------------------------------------------------

fn build_health_check() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "healthCheck",
            "summary": "Simple liveness check. Returns 'ok' when the server is running.",
            "responses": {
                "200": { "description": "Server is alive" }
            }
        }
    })
}

fn build_search() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "searchVault",
            "summary": "Hybrid semantic + full-text search across the vault.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "Search query text" },
                        "top_n": { "type": "integer", "description": "Number of results (default 10)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Search results with scores and snippets" } }
        }
    })
}

fn build_read() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "readNote",
            "summary": "Read a note's full content with metadata and graph connections.",
            "parameters": [{
                "name": "file", "in": "path", "required": true,
                "description": "File path, basename, or #docid",
                "schema": { "type": "string" }
            }],
            "responses": { "200": { "description": "Note content with metadata" } }
        }
    })
}

fn build_read_section() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "readSection",
            "summary": "Read a specific section of a note by heading name.",
            "parameters": [
                { "name": "file", "in": "query", "required": true, "description": "File path, basename, or #docid", "schema": { "type": "string" } },
                { "name": "heading", "in": "query", "required": true, "description": "Section heading (case-insensitive)", "schema": { "type": "string" } }
            ],
            "responses": { "200": { "description": "Section content" } }
        }
    })
}

fn build_list() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "listNotes",
            "summary": "List notes filtered by folder, tags, creator, or limit.",
            "parameters": [
                { "name": "folder", "in": "query", "required": false, "description": "Folder path prefix filter", "schema": { "type": "string" } },
                { "name": "tags", "in": "query", "required": false, "description": "Comma-separated tags", "schema": { "type": "string" } },
                { "name": "created_by", "in": "query", "required": false, "description": "Agent filter", "schema": { "type": "string" } },
                { "name": "limit", "in": "query", "required": false, "description": "Max results (default 20)", "schema": { "type": "integer" } }
            ],
            "responses": { "200": { "description": "Array of note summaries" } }
        }
    })
}

fn build_vault_map() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getVaultMap",
            "summary": "Get vault structure overview with folder tree, tag cloud, and statistics.",
            "responses": { "200": { "description": "Vault structure map" } }
        }
    })
}

fn build_who() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getWho",
            "summary": "Get a person context bundle with note, related notes, and interaction history.",
            "parameters": [{
                "name": "name", "in": "path", "required": true,
                "description": "Person name (matches filename in People folder)",
                "schema": { "type": "string" }
            }],
            "responses": { "200": { "description": "Person context bundle" } }
        }
    })
}

fn build_project() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getProject",
            "summary": "Get a project context bundle with project note, related files, and graph connections.",
            "parameters": [{
                "name": "name", "in": "path", "required": true,
                "description": "Project name (matches filename)",
                "schema": { "type": "string" }
            }],
            "responses": { "200": { "description": "Project context bundle" } }
        }
    })
}

fn build_context() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "getContext",
            "summary": "Get rich topic context with semantic search, graph expansion, and budget-aware trimming.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["topic"],
                    "properties": {
                        "topic": { "type": "string", "description": "Topic or question" },
                        "budget": { "type": "integer", "description": "Character budget (default 32000)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Context bundle with notes and metadata" } }
        }
    })
}

fn build_health() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getHealth",
            "summary": "Get vault health report with orphans, broken links, stale notes, and inbox status.",
            "responses": { "200": { "description": "Vault health report" } }
        }
    })
}

fn build_create() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "createNote",
            "summary": "Create a new note with automatic placement and frontmatter generation.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": { "type": "string", "description": "Note content (markdown)" },
                        "filename": { "type": "string", "description": "Filename without .md" },
                        "type_hint": { "type": "string", "description": "Type hint for placement" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags to apply" },
                        "folder": { "type": "string", "description": "Explicit folder (skips auto-placement)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Created note path and metadata" } }
        }
    })
}

fn build_append() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "appendToNote",
            "summary": "Append content to the end of an existing note.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "content"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "content": { "type": "string", "description": "Content to append" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Updated note path and metadata" } }
        }
    })
}

fn build_edit() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "editNote",
            "summary": "Edit a specific section of a note by heading. Supports replace, prepend, append modes.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "heading", "content"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "heading": { "type": "string", "description": "Section heading (case-insensitive)" },
                        "content": { "type": "string", "description": "Content to add or replace" },
                        "mode": { "type": "string", "description": "'replace', 'prepend', or 'append' (default)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Updated note path and metadata" } }
        }
    })
}

fn build_rewrite() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "rewriteNote",
            "summary": "Rewrite a note's body content. Preserves frontmatter by default.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "content"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "content": { "type": "string", "description": "New body content" },
                        "preserve_frontmatter": { "type": "boolean", "description": "Preserve frontmatter (default true)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Updated note path and metadata" } }
        }
    })
}

fn build_edit_frontmatter() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "editFrontmatter",
            "summary": "Edit a note's frontmatter with structured operations (set, remove, add_tag, etc.).",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "operations"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "operations": {
                            "type": "array",
                            "description": "Frontmatter operations",
                            "items": { "type": "object", "properties": {
                                "op": { "type": "string", "description": "set/remove/add_tag/remove_tag/add_alias/remove_alias" },
                                "key": { "type": "string", "description": "Property key (for set/remove)" },
                                "value": { "type": "string", "description": "Value" }
                            }}
                        }
                    }
                }}}
            },
            "responses": { "200": { "description": "Updated note path and metadata" } }
        }
    })
}

fn build_move() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "moveNote",
            "summary": "Move a note to a different folder within the vault.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "new_folder"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "new_folder": { "type": "string", "description": "Destination folder path" }
                    }
                }}}
            },
            "responses": { "200": { "description": "New note path" } }
        }
    })
}

fn build_archive() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "archiveNote",
            "summary": "Archive a note (soft delete). Moves to archive folder and removes from index.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Archived note path" } }
        }
    })
}

fn build_unarchive() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "unarchiveNote",
            "summary": "Restore an archived note to its original location and re-index it.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "Archived note path" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Restored note path" } }
        }
    })
}

fn build_update_metadata() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "updateMetadata",
            "summary": "Update a note's tags and aliases in bulk.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "New tags (replaces existing)" },
                        "aliases": { "type": "array", "items": { "type": "string" }, "description": "New aliases (replaces existing)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Updated note metadata" } }
        }
    })
}

fn build_delete() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "deleteNote",
            "summary": "Delete a note. Supports soft (archive) and hard (permanent) modes.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "mode": { "type": "string", "description": "'soft' (default) or 'hard'" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Deletion confirmation" } }
        }
    })
}

fn build_migrate_preview() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "migratePreview",
            "summary": "Generate a PARA migration preview. Classifies notes and suggests folder moves.",
            "requestBody": {
                "content": { "application/json": { "schema": { "type": "object", "properties": {} } } }
            },
            "responses": { "200": { "description": "Migration preview with proposed moves" } }
        }
    })
}

fn build_migrate_apply() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "migrateApply",
            "summary": "Apply a previously generated migration preview. Moves notes to suggested PARA folders.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["preview"],
                    "properties": {
                        "preview": { "type": "object", "description": "Migration preview from migratePreview" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Migration result with moved files count" } }
        }
    })
}

fn build_migrate_undo() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "migrateUndo",
            "summary": "Undo the last applied migration, restoring notes to original locations.",
            "requestBody": {
                "content": { "application/json": { "schema": { "type": "object", "properties": {} } } }
            },
            "responses": { "200": { "description": "Undo result with restored files count" } }
        }
    })
}

/// Build the ChatGPT plugin manifest (ai-plugin.json).
pub fn build_plugin_manifest(config: &HttpConfig, server_url: &str) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "v1",
        "name_for_human": config.plugin.name.as_deref().unwrap_or("engraph"),
        "name_for_model": "engraph",
        "description_for_human": config.plugin.description.as_deref()
            .unwrap_or("Search and manage your Obsidian vault with AI-powered hybrid search."),
        "description_for_model": "Access an Obsidian knowledge vault. Use search to find notes by content or time. read for full content. who/project for context bundles. Write tools create, edit, and organize notes.",
        "auth": {
            "type": "service_http",
            "authorization_type": "bearer",
            "verification_tokens": {}
        },
        "api": {
            "type": "openapi",
            "url": format!("{}/openapi.json", server_url)
        },
        "logo_url": "",
        "contact_email": config.plugin.contact_email.as_deref().unwrap_or(""),
        "legal_info_url": ""
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_spec_structure() {
        let spec = build_openapi_spec("http://localhost:3000");
        assert_eq!(spec["openapi"], "3.1.0");
        assert!(spec["paths"].as_object().unwrap().len() >= 20);
        assert_eq!(spec["servers"][0]["url"], "http://localhost:3000");
    }

    #[test]
    fn test_openapi_has_security() {
        let spec = build_openapi_spec("http://localhost:3000");
        assert!(spec["components"]["securitySchemes"]["bearerAuth"].is_object());
    }

    #[test]
    fn test_plugin_manifest() {
        let config = crate::config::HttpConfig::default();
        let manifest = build_plugin_manifest(&config, "https://vault.example.com");
        assert_eq!(manifest["schema_version"], "v1");
        assert_eq!(manifest["name_for_model"], "engraph");
        assert!(
            manifest["api"]["url"]
                .as_str()
                .unwrap()
                .contains("openapi.json")
        );
    }

    #[test]
    fn test_openapi_has_all_operation_ids() {
        let spec = build_openapi_spec("http://localhost:3000");
        let paths = spec["paths"].as_object().unwrap();
        let mut op_ids: Vec<String> = Vec::new();
        for (_path, methods) in paths {
            for (_method, details) in methods.as_object().unwrap() {
                if let Some(id) = details.get("operationId").and_then(|v| v.as_str()) {
                    op_ids.push(id.to_string());
                }
            }
        }
        let expected = vec![
            "healthCheck",
            "searchVault",
            "readNote",
            "readSection",
            "listNotes",
            "getVaultMap",
            "getWho",
            "getProject",
            "getContext",
            "getHealth",
            "createNote",
            "appendToNote",
            "editNote",
            "rewriteNote",
            "editFrontmatter",
            "moveNote",
            "archiveNote",
            "unarchiveNote",
            "updateMetadata",
            "deleteNote",
            "migratePreview",
            "migrateApply",
            "migrateUndo",
        ];
        for id in &expected {
            assert!(
                op_ids.contains(&id.to_string()),
                "Missing operationId: {id}"
            );
        }
    }

    #[test]
    fn test_openapi_server_url_passed_through() {
        let spec = build_openapi_spec("https://my-tunnel.example.com");
        assert_eq!(spec["servers"][0]["url"], "https://my-tunnel.example.com");
    }

    #[test]
    fn test_plugin_manifest_custom_config() {
        let mut config = crate::config::HttpConfig::default();
        config.plugin.name = Some("my-vault".into());
        config.plugin.contact_email = Some("test@example.com".into());
        let manifest = build_plugin_manifest(&config, "https://example.com");
        assert_eq!(manifest["name_for_human"], "my-vault");
        assert_eq!(manifest["contact_email"], "test@example.com");
    }
}
