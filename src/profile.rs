use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// How the vault organizes its notes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructureMethod {
    Flat,
    Folders,
    Para,
    Custom,
}

/// What kind of vault this is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VaultType {
    Obsidian,
    Logseq,
    Plain,
    Custom,
}

/// Complete vault profile, persisted to `vault.toml`.
#[derive(Debug, Serialize, Deserialize)]
pub struct VaultProfile {
    pub vault_path: PathBuf,
    pub vault_type: VaultType,
    pub structure: StructureDetection,
    pub stats: VaultStats,
}

/// Detected folder structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct StructureDetection {
    pub method: StructureMethod,
    pub folders: FolderMap,
}

/// Known folder roles mapped to their detected names.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FolderMap {
    pub inbox: Option<String>,
    pub projects: Option<String>,
    pub areas: Option<String>,
    pub resources: Option<String>,
    pub archive: Option<String>,
    pub templates: Option<String>,
    pub daily: Option<String>,
    pub people: Option<String>,
}

/// Aggregate vault statistics.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VaultStats {
    pub total_files: usize,
    pub files_with_frontmatter: usize,
    pub wikilink_count: usize,
    pub unique_tags: usize,
    pub folder_depth: usize,
    pub folder_count: usize,
}

// ---------------------------------------------------------------------------
// Content-based role detection
// ---------------------------------------------------------------------------

/// Check whether a markdown file's frontmatter looks like a person note.
/// Returns true if it has a tag containing "person" or "people", OR has a "role" key.
fn is_person_like(text: &str) -> bool {
    // Find frontmatter block.
    let fm = if text.starts_with("---\n") {
        text.get(4..)
            .and_then(|rest| rest.find("\n---").map(|end| &rest[..end]))
    } else if text.starts_with("---\r\n") {
        text.get(5..)
            .and_then(|rest| rest.find("\n---").map(|end| &rest[..end]))
    } else {
        None
    };

    let Some(fm) = fm else {
        return false;
    };

    let mut has_person_tag = false;
    let mut in_tags_block = false;

    for line in fm.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("role:") {
            return true;
        }

        if trimmed.starts_with("tags:") {
            let after = trimmed.strip_prefix("tags:").unwrap().trim();
            if after.is_empty() {
                in_tags_block = true;
                continue;
            }
            // Inline list: tags: [person, ...] or tags: person, ...
            let after = after.trim_start_matches('[').trim_end_matches(']');
            for tag in after.split(',') {
                let t = tag
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .trim_matches('#')
                    .to_ascii_lowercase();
                if t == "person" || t == "people" {
                    has_person_tag = true;
                }
            }
            if has_person_tag {
                return true;
            }
            in_tags_block = false;
            continue;
        }

        if in_tags_block {
            if trimmed.starts_with("- ") {
                let t = trimmed
                    .strip_prefix("- ")
                    .unwrap()
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .trim_matches('#')
                    .to_ascii_lowercase();
                if t == "person" || t == "people" {
                    return true;
                }
            } else if !trimmed.is_empty() {
                in_tags_block = false;
            }
        }
    }

    false
}

/// Check whether a filename looks like a date note (YYYY-MM-DD.md).
fn is_date_filename(name: &str) -> bool {
    // Must match exactly: YYYY-MM-DD.md (13 chars: 4+1+2+1+2+3)
    let bytes = name.as_bytes();
    if bytes.len() != 13 {
        return false;
    }
    if &name[4..5] != "-" || &name[7..8] != "-" || &name[10..] != ".md" {
        return false;
    }
    bytes[..4].iter().all(|b| b.is_ascii_digit())
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
        && bytes[8..10].iter().all(|b| b.is_ascii_digit())
}

/// Scan top-level subdirectories and return the one (with trailing slash) where
/// 60%+ of the `.md` files have person-like frontmatter. Returns `None` if no
/// folder qualifies.
pub fn detect_people_folder(root: &Path) -> Result<Option<String>> {
    let entries = std::fs::read_dir(root)
        .with_context(|| format!("cannot read directory {}", root.display()))?;

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }

        let dir = entry.path();
        let mut total = 0usize;
        let mut person_like = 0usize;

        let inner = std::fs::read_dir(&dir)
            .with_context(|| format!("cannot read directory {}", dir.display()))?;
        for inner_entry in inner {
            let inner_entry = inner_entry?;
            if !inner_entry.file_type()?.is_file() {
                continue;
            }
            let fname = inner_entry.file_name();
            let fname_str = fname.to_string_lossy();
            if !fname_str.ends_with(".md") {
                continue;
            }
            total += 1;
            let text = std::fs::read_to_string(inner_entry.path()).unwrap_or_default();
            if is_person_like(&text) {
                person_like += 1;
            }
        }

        if total > 0 && person_like * 100 / total >= 60 {
            return Ok(Some(format!("{}/", name_str)));
        }
    }

    Ok(None)
}

/// Scan top-level subdirectories and return the one (with trailing slash) where
/// 60%+ of the `.md` filenames match the YYYY-MM-DD pattern. Returns `None` if
/// no folder qualifies.
pub fn detect_daily_folder(root: &Path) -> Result<Option<String>> {
    let entries = std::fs::read_dir(root)
        .with_context(|| format!("cannot read directory {}", root.display()))?;

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }

        let dir = entry.path();
        let mut total = 0usize;
        let mut date_like = 0usize;

        let inner = std::fs::read_dir(&dir)
            .with_context(|| format!("cannot read directory {}", dir.display()))?;
        for inner_entry in inner {
            let inner_entry = inner_entry?;
            if !inner_entry.file_type()?.is_file() {
                continue;
            }
            let fname = inner_entry.file_name();
            let fname_str = fname.to_string_lossy();
            if !fname_str.ends_with(".md") {
                continue;
            }
            total += 1;
            if is_date_filename(&fname_str) {
                date_like += 1;
            }
        }

        if total > 0 && date_like * 100 / total >= 60 {
            return Ok(Some(format!("{}/", name_str)));
        }
    }

    Ok(None)
}

/// Find the archive folder by looking for well-known names (case-insensitive):
/// "archive", "_archive", ".archive", or folders matching PARA-style patterns
/// like "04-Archive".
pub fn detect_archive_folder(root: &Path) -> Result<Option<String>> {
    let archive_names: &[&str] = &["archive", "_archive", ".archive"];

    let entries = std::fs::read_dir(root)
        .with_context(|| format!("cannot read directory {}", root.display()))?;

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Strip leading digits and separators for PARA-style matching.
        let stripped = name_str
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches(['-', '_', ' ']);

        let lower = stripped.to_ascii_lowercase();
        if archive_names.contains(&lower.as_str()) {
            return Ok(Some(format!("{}/", name_str)));
        }
    }

    Ok(None)
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

/// Map a folder name (case-insensitive, ignoring leading number prefixes like `00-`)
/// to a PARA role.
fn para_role(name: &str) -> Option<&'static str> {
    // Strip optional leading digits and separator (e.g. "00-", "01-").
    let stripped = name
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .trim_start_matches(['-', '_', ' ']);

    match stripped.to_ascii_lowercase().as_str() {
        "inbox" => Some("inbox"),
        "projects" => Some("projects"),
        "areas" => Some("areas"),
        "resources" => Some("resources"),
        "archive" => Some("archive"),
        "templates" => Some("templates"),
        "daily" => Some("daily"),
        "people" => Some("people"),
        _ => None,
    }
}

/// Detect vault structure by checking for PARA-style numbered folders.
///
/// - If at least 3 of the 4 core PARA folders (inbox, projects, areas, resources) exist -> Para
/// - If there are subdirectories but no PARA pattern -> Folders
/// - If mostly flat .md files -> Flat
pub fn detect_structure(path: &Path) -> Result<StructureDetection> {
    let mut folders = FolderMap::default();
    let mut para_hits = 0u32;
    let mut dir_count = 0usize;

    let entries = std::fs::read_dir(path)
        .with_context(|| format!("cannot read directory {}", path.display()))?;

    for entry in entries {
        let entry = entry?;
        let ft = entry.file_type()?;
        if !ft.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden directories.
        if name_str.starts_with('.') {
            continue;
        }

        dir_count += 1;

        if let Some(role) = para_role(&name_str) {
            let folder_name = name_str.to_string();
            match role {
                "inbox" => {
                    folders.inbox = Some(folder_name);
                    para_hits += 1;
                }
                "projects" => {
                    folders.projects = Some(folder_name);
                    para_hits += 1;
                }
                "areas" => {
                    folders.areas = Some(folder_name);
                    para_hits += 1;
                }
                "resources" => {
                    folders.resources = Some(folder_name);
                    para_hits += 1;
                }
                "archive" => {
                    folders.archive = Some(folder_name);
                }
                "templates" => {
                    folders.templates = Some(folder_name);
                }
                "daily" => {
                    folders.daily = Some(folder_name);
                }
                "people" => {
                    folders.people = Some(folder_name);
                }
                _ => {}
            }
        }
    }

    let method = if para_hits >= 3 {
        StructureMethod::Para
    } else if dir_count > 0 {
        StructureMethod::Folders
    } else {
        StructureMethod::Flat
    };

    // For non-PARA vaults, try content-based detection for roles not yet filled.
    if method != StructureMethod::Para {
        if folders.people.is_none() {
            folders.people = detect_people_folder(path)
                .ok()
                .flatten()
                .map(|s| s.trim_end_matches('/').to_string());
        }
        if folders.daily.is_none() {
            folders.daily = detect_daily_folder(path)
                .ok()
                .flatten()
                .map(|s| s.trim_end_matches('/').to_string());
        }
        if folders.archive.is_none() {
            folders.archive = detect_archive_folder(path)
                .ok()
                .flatten()
                .map(|s| s.trim_end_matches('/').to_string());
        }
    }

    Ok(StructureDetection { method, folders })
}

/// Count wikilinks (`[[...]]`) in a string. Handles nested brackets conservatively.
fn count_wikilinks(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0usize;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            // Find closing ]].
            if let Some(rest) = text.get(i + 2..)
                && let Some(close) = rest.find("]]")
            {
                // Only count if the content is non-empty and doesn't span lines.
                let inner = &rest[..close];
                if !inner.is_empty() && !inner.contains('\n') {
                    count += 1;
                }
                i += 2 + close + 2;
                continue;
            }
        }
        i += 1;
    }
    count
}

/// Check whether a file starts with YAML frontmatter (`---` on the first line).
fn has_frontmatter(text: &str) -> bool {
    text.starts_with("---\n") || text.starts_with("---\r\n")
}

/// Extract tags from YAML frontmatter. Handles both list and inline formats:
/// ```yaml
/// tags:
///   - foo
///   - bar
/// ```
/// and
/// ```yaml
/// tags: [foo, bar]
/// ```
fn extract_tags(text: &str) -> Vec<String> {
    // Find frontmatter block.
    let fm = if text.starts_with("---\n") {
        text.get(4..)
            .and_then(|rest| rest.find("\n---").map(|end| &rest[..end]))
    } else if text.starts_with("---\r\n") {
        text.get(5..)
            .and_then(|rest| rest.find("\n---").map(|end| &rest[..end]))
    } else {
        None
    };

    let Some(fm) = fm else {
        return Vec::new();
    };

    let mut tags = Vec::new();
    let mut in_tags_block = false;

    for line in fm.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("tags:") {
            let after = trimmed.strip_prefix("tags:").unwrap().trim();
            if after.is_empty() {
                // Multi-line list follows.
                in_tags_block = true;
                continue;
            }
            // Inline list: tags: [a, b] or tags: a, b
            let after = after.trim_start_matches('[').trim_end_matches(']');
            for tag in after.split(',') {
                let t = tag
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .trim_matches('#');
                if !t.is_empty() {
                    tags.push(t.to_string());
                }
            }
            return tags;
        }

        if in_tags_block {
            if trimmed.starts_with("- ") {
                let t = trimmed
                    .strip_prefix("- ")
                    .unwrap()
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .trim_matches('#');
                if !t.is_empty() {
                    tags.push(t.to_string());
                }
            } else if !trimmed.is_empty() {
                // End of tags block (new key).
                break;
            }
        }
    }

    tags
}

/// Walk all `.md` files under `path` recursively.
fn walk_md_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    walk_md_recursive(path, &mut files)?;
    Ok(files)
}

fn walk_md_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let ft = entry.file_type()?;
        let path = entry.path();

        if ft.is_dir() {
            // Skip hidden directories.
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            walk_md_recursive(&path, out)?;
        } else if ft.is_file()
            && let Some(ext) = path.extension()
            && ext == "md"
        {
            out.push(path);
        }
    }

    Ok(())
}

/// Count distinct folders and maximum depth relative to `root`.
fn folder_stats(root: &Path) -> Result<(usize, usize)> {
    let mut count = 0usize;
    let mut max_depth = 0usize;
    folder_stats_recursive(root, root, &mut count, &mut max_depth)?;
    Ok((count, max_depth))
}

fn folder_stats_recursive(
    dir: &Path,
    root: &Path,
    count: &mut usize,
    max_depth: &mut usize,
) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        *count += 1;
        let depth = entry
            .path()
            .strip_prefix(root)
            .map(|p| p.components().count())
            .unwrap_or(0);
        if depth > *max_depth {
            *max_depth = depth;
        }
        folder_stats_recursive(&entry.path(), root, count, max_depth)?;
    }

    Ok(())
}

/// Scan vault files for statistics.
pub fn scan_vault_stats(path: &Path) -> Result<VaultStats> {
    let md_files = walk_md_files(path)?;
    let mut all_tags = std::collections::HashSet::new();
    let mut files_with_frontmatter = 0;
    let mut wikilink_count = 0;

    for file in &md_files {
        let text = std::fs::read_to_string(file).unwrap_or_default();
        if has_frontmatter(&text) {
            files_with_frontmatter += 1;
        }
        wikilink_count += count_wikilinks(&text);
        for tag in extract_tags(&text) {
            all_tags.insert(tag);
        }
    }

    let (fc, fd) = folder_stats(path)?;

    Ok(VaultStats {
        total_files: md_files.len(),
        files_with_frontmatter,
        wikilink_count,
        unique_tags: all_tags.len(),
        folder_count: fc,
        folder_depth: fd,
    })
}

/// Detect vault type based on marker files/directories.
pub fn detect_vault_type(path: &Path) -> VaultType {
    if path.join(".obsidian").is_dir() {
        VaultType::Obsidian
    } else if path.join(".logseq").is_dir() {
        VaultType::Logseq
    } else {
        VaultType::Plain
    }
}

/// Write vault profile to `vault.toml` in the given directory.
pub fn write_vault_toml(profile: &VaultProfile, config_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(config_dir)
        .with_context(|| format!("cannot create directory {}", config_dir.display()))?;

    let toml_str = toml::to_string_pretty(profile).context("failed to serialize vault profile")?;
    let dest = config_dir.join("vault.toml");
    std::fs::write(&dest, toml_str)
        .with_context(|| format!("failed to write {}", dest.display()))?;

    Ok(())
}

/// Load vault profile from `vault.toml` in the given directory.
/// Returns `Ok(None)` if the file does not exist.
pub fn load_vault_toml(config_dir: &Path) -> Result<Option<VaultProfile>> {
    let path = config_dir.join("vault.toml");
    if !path.exists() {
        return Ok(None);
    }

    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let profile: VaultProfile =
        toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))?;

    Ok(Some(profile))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_detect_para_structure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create PARA-style numbered folders.
        std::fs::create_dir(root.join("00-Inbox")).unwrap();
        std::fs::create_dir(root.join("01-Projects")).unwrap();
        std::fs::create_dir(root.join("02-Areas")).unwrap();
        std::fs::create_dir(root.join("03-Resources")).unwrap();
        std::fs::create_dir(root.join("04-Archive")).unwrap();
        std::fs::create_dir(root.join("05-Templates")).unwrap();
        std::fs::create_dir(root.join("07-Daily")).unwrap();

        let result = detect_structure(root).unwrap();
        assert_eq!(result.method, StructureMethod::Para);
        assert_eq!(result.folders.inbox.as_deref(), Some("00-Inbox"));
        assert_eq!(result.folders.projects.as_deref(), Some("01-Projects"));
        assert_eq!(result.folders.areas.as_deref(), Some("02-Areas"));
        assert_eq!(result.folders.resources.as_deref(), Some("03-Resources"));
        assert_eq!(result.folders.archive.as_deref(), Some("04-Archive"));
        assert_eq!(result.folders.templates.as_deref(), Some("05-Templates"));
        assert_eq!(result.folders.daily.as_deref(), Some("07-Daily"));
    }

    #[test]
    fn test_detect_flat_structure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create only .md files, no subdirectories.
        std::fs::write(root.join("note1.md"), "Hello").unwrap();
        std::fs::write(root.join("note2.md"), "World").unwrap();

        let result = detect_structure(root).unwrap();
        assert_eq!(result.method, StructureMethod::Flat);
    }

    #[test]
    fn test_detect_folders_structure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create non-PARA subdirectories.
        std::fs::create_dir(root.join("notes")).unwrap();
        std::fs::create_dir(root.join("journal")).unwrap();
        std::fs::create_dir(root.join("references")).unwrap();

        let result = detect_structure(root).unwrap();
        assert_eq!(result.method, StructureMethod::Folders);
    }

    #[test]
    fn test_detect_wikilinks() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        std::fs::write(
            root.join("a.md"),
            "See [[Note One]] and [[Note Two]] for details.",
        )
        .unwrap();
        std::fs::write(root.join("b.md"), "No links here.").unwrap();
        std::fs::write(root.join("c.md"), "Link to [[Note One|alias]] only.").unwrap();

        let stats = scan_vault_stats(root).unwrap();
        assert_eq!(stats.total_files, 3);
        assert_eq!(stats.wikilink_count, 3);
    }

    #[test]
    fn test_detect_frontmatter() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        std::fs::write(
            root.join("with_fm.md"),
            "---\ntitle: Test\ntags:\n  - foo\n---\nContent here.",
        )
        .unwrap();
        std::fs::write(root.join("without_fm.md"), "Just some text.").unwrap();
        std::fs::write(
            root.join("also_fm.md"),
            "---\ntags: [bar, baz]\n---\nMore content.",
        )
        .unwrap();

        let stats = scan_vault_stats(root).unwrap();
        assert_eq!(stats.total_files, 3);
        assert_eq!(stats.files_with_frontmatter, 2);
        assert_eq!(stats.unique_tags, 3); // foo, bar, baz
    }

    #[test]
    fn test_detect_vault_type_obsidian() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join(".obsidian")).unwrap();

        assert_eq!(detect_vault_type(root), VaultType::Obsidian);
    }

    #[test]
    fn test_detect_vault_type_logseq() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join(".logseq")).unwrap();

        assert_eq!(detect_vault_type(root), VaultType::Logseq);
    }

    #[test]
    fn test_detect_vault_type_plain() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(detect_vault_type(tmp.path()), VaultType::Plain);
    }

    #[test]
    fn test_write_and_load_vault_toml() {
        let tmp = TempDir::new().unwrap();
        let config_dir = tmp.path();

        let profile = VaultProfile {
            vault_path: PathBuf::from("/test/vault"),
            vault_type: VaultType::Obsidian,
            structure: StructureDetection {
                method: StructureMethod::Para,
                folders: FolderMap {
                    inbox: Some("00-Inbox".to_string()),
                    projects: Some("01-Projects".to_string()),
                    areas: Some("02-Areas".to_string()),
                    resources: Some("03-Resources".to_string()),
                    archive: Some("04-Archive".to_string()),
                    templates: Some("05-Templates".to_string()),
                    daily: Some("07-Daily".to_string()),
                    people: None,
                },
            },
            stats: VaultStats {
                total_files: 100,
                files_with_frontmatter: 80,
                wikilink_count: 500,
                unique_tags: 25,
                folder_depth: 3,
                folder_count: 10,
            },
        };

        write_vault_toml(&profile, config_dir).unwrap();

        // Verify the file was created.
        assert!(config_dir.join("vault.toml").exists());

        // Load it back.
        let loaded = load_vault_toml(config_dir).unwrap().unwrap();
        assert_eq!(loaded.vault_path, PathBuf::from("/test/vault"));
        assert_eq!(loaded.vault_type, VaultType::Obsidian);
        assert_eq!(loaded.structure.method, StructureMethod::Para);
        assert_eq!(loaded.structure.folders.inbox.as_deref(), Some("00-Inbox"));
        assert_eq!(loaded.stats.total_files, 100);
        assert_eq!(loaded.stats.wikilink_count, 500);
        assert_eq!(loaded.stats.unique_tags, 25);
    }

    #[test]
    fn test_load_vault_toml_missing_file() {
        let tmp = TempDir::new().unwrap();
        let result = load_vault_toml(tmp.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_count_wikilinks_empty() {
        assert_eq!(count_wikilinks(""), 0);
        assert_eq!(count_wikilinks("no links here"), 0);
    }

    #[test]
    fn test_count_wikilinks_multiple() {
        assert_eq!(count_wikilinks("[[a]] text [[b|alias]] more [[c]]"), 3);
    }

    #[test]
    fn test_extract_tags_inline_list() {
        let text = "---\ntags: [foo, bar, baz]\n---\ncontent";
        assert_eq!(extract_tags(text), vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn test_extract_tags_multiline() {
        let text = "---\ntags:\n  - alpha\n  - beta\n---\ncontent";
        assert_eq!(extract_tags(text), vec!["alpha", "beta"]);
    }

    #[test]
    fn test_extract_tags_no_frontmatter() {
        let text = "just some text";
        assert!(extract_tags(text).is_empty());
    }

    #[test]
    fn test_folder_stats_depth() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("a/b/c")).unwrap();
        std::fs::create_dir(root.join("d")).unwrap();

        let (count, depth) = folder_stats(root).unwrap();
        assert_eq!(count, 4); // a, a/b, a/b/c, d
        assert_eq!(depth, 3); // a/b/c is depth 3
    }

    #[test]
    fn test_detect_people_folder_from_content() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("contacts")).unwrap();
        // 3 out of 4 files have person-like frontmatter
        for name in &["alice.md", "bob.md", "charlie.md"] {
            std::fs::write(
                root.join("contacts").join(name),
                "---\ntags:\n  - person\nrole: Engineer\n---\n",
            )
            .unwrap();
        }
        std::fs::write(root.join("contacts/readme.md"), "# Contacts\n").unwrap();

        let detected = detect_people_folder(root).unwrap();
        assert_eq!(detected.as_deref(), Some("contacts/"));
    }

    #[test]
    fn test_detect_daily_folder_from_filenames() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("journal")).unwrap();
        for date in &["2026-03-24.md", "2026-03-25.md", "2026-03-26.md"] {
            std::fs::write(root.join("journal").join(date), "# Daily\n").unwrap();
        }
        let detected = detect_daily_folder(root).unwrap();
        assert_eq!(detected.as_deref(), Some("journal/"));
    }
}
