use anyhow::Result;
use rusqlite::{Connection, params};
use strsim::levenshtein;

/// Result of resolving a proposed tag against the registry.
#[derive(Debug, Clone, PartialEq)]
pub enum TagResolution {
    /// Exact case-insensitive match found.
    Exact(String),
    /// Fuzzy match: Levenshtein distance ≤ 2.
    Fuzzy {
        proposed: String,
        resolved: String,
        distance: usize,
    },
    /// Proposed tag extends an existing tag via `/` hierarchy.
    Extension(String),
    /// No match — this is a brand-new tag.
    New(String),
}

/// Resolve a single proposed tag against the registry.
///
/// Resolution tiers (priority order):
/// 1. Exact match (case-insensitive)
/// 2. Fuzzy match (Levenshtein distance ≤ 2, pick closest)
/// 3. Prefix extension (proposed starts with `existing_tag/`)
/// 4. New tag
pub fn resolve_tag(conn: &Connection, proposed: &str) -> Result<TagResolution> {
    let lower = proposed.to_lowercase();

    // Tier 1: Exact case-insensitive match.
    let exact: Option<String> = conn
        .prepare("SELECT name FROM tag_registry WHERE LOWER(name) = ?1")?
        .query_map(params![lower], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .next();

    if let Some(name) = exact {
        return Ok(TagResolution::Exact(name));
    }

    // Load all registered tags for fuzzy + prefix checks.
    let all_tags: Vec<String> = conn
        .prepare("SELECT name FROM tag_registry")?
        .query_map([], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    // Tier 2: Fuzzy match — Levenshtein distance ≤ 2.
    let mut best: Option<(String, usize)> = None;
    for tag in &all_tags {
        let dist = levenshtein(&lower, &tag.to_lowercase());
        if dist > 0 && dist <= 2 && (best.is_none() || dist < best.as_ref().unwrap().1) {
            best = Some((tag.clone(), dist));
        }
    }
    if let Some((resolved, distance)) = best {
        return Ok(TagResolution::Fuzzy {
            proposed: proposed.to_string(),
            resolved,
            distance,
        });
    }

    // Tier 3: Prefix extension — proposed starts with `existing_tag/`.
    for tag in &all_tags {
        if lower.starts_with(&format!("{}/", tag.to_lowercase())) {
            return Ok(TagResolution::Extension(proposed.to_string()));
        }
    }

    // Tier 4: New tag.
    Ok(TagResolution::New(proposed.to_string()))
}

/// Register (upsert) a tag: increment usage_count if it exists, insert otherwise.
pub fn register_tag(conn: &Connection, name: &str, created_by: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO tag_registry (name, usage_count, last_used, created_by)
         VALUES (?1, 1, datetime('now'), ?2)
         ON CONFLICT(name) DO UPDATE SET
             usage_count = usage_count + 1,
             last_used = datetime('now')",
        params![name, created_by],
    )?;
    Ok(())
}

/// Resolve a list of proposed tags, returning the final tag names.
///
/// - Exact / Fuzzy matches map to the resolved name.
/// - Extension / New tags pass through as-is.
pub fn resolve_tags(conn: &Connection, proposed: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(proposed.len());
    for tag in proposed {
        let resolved = resolve_tag(conn, tag)?;
        let name = match resolved {
            TagResolution::Exact(name) => name,
            TagResolution::Fuzzy { resolved, .. } => resolved,
            TagResolution::Extension(name) => name,
            TagResolution::New(name) => name,
        };
        out.push(name);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn setup_store() -> Store {
        let store = Store::open_memory().unwrap();
        let conn = store.conn();
        // Seed tags with varying usage counts.
        for (name, count) in [
            ("domaine", 15),
            ("scentbird", 10),
            ("engraph", 8),
            ("work", 20),
            ("work/domaine", 5),
        ] {
            conn.execute(
                "INSERT INTO tag_registry (name, usage_count, last_used, created_by)
                 VALUES (?1, ?2, datetime('now'), 'test')",
                params![name, count],
            )
            .unwrap();
        }
        store
    }

    #[test]
    fn test_exact_match() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "domaine").unwrap();
        assert_eq!(res, TagResolution::Exact("domaine".to_string()));
    }

    #[test]
    fn test_exact_match_case_insensitive() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "Domaine").unwrap();
        assert_eq!(res, TagResolution::Exact("domaine".to_string()));
    }

    #[test]
    fn test_fuzzy_match() {
        let store = setup_store();
        // "doamine" is Levenshtein distance 2 from "domaine" (transposition).
        let res = resolve_tag(store.conn(), "doamine").unwrap();
        match res {
            TagResolution::Fuzzy {
                proposed,
                resolved,
                distance,
            } => {
                assert_eq!(proposed, "doamine");
                assert_eq!(resolved, "domaine");
                assert!(distance <= 2);
            }
            other => panic!("expected Fuzzy, got {other:?}"),
        }
    }

    #[test]
    fn test_hierarchy_extension() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "work/domaine/bre").unwrap();
        assert_eq!(
            res,
            TagResolution::Extension("work/domaine/bre".to_string())
        );
    }

    #[test]
    fn test_new_tag() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "completely-new").unwrap();
        assert_eq!(res, TagResolution::New("completely-new".to_string()));
    }

    #[test]
    fn test_register_tag() {
        let store = setup_store();
        register_tag(store.conn(), "new-tag", "test").unwrap();
        let count: i64 = store
            .conn()
            .query_row(
                "SELECT usage_count FROM tag_registry WHERE name = 'new-tag'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Register again — count should increment.
        register_tag(store.conn(), "new-tag", "test").unwrap();
        let count: i64 = store
            .conn()
            .query_row(
                "SELECT usage_count FROM tag_registry WHERE name = 'new-tag'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_resolve_tags() {
        let store = setup_store();
        let input = vec![
            "domaine".to_string(),
            "doamine".to_string(),
            "completely-new".to_string(),
        ];
        let resolved = resolve_tags(store.conn(), &input).unwrap();
        assert_eq!(resolved, vec!["domaine", "domaine", "completely-new"]);
    }
}
