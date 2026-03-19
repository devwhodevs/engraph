use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

/// Application configuration, loaded from `~/.engraph/config.toml` with CLI overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Path to the Obsidian vault to index.
    pub vault_path: Option<PathBuf>,
    /// Number of results to return from search.
    pub top_n: usize,
    /// Glob patterns to exclude from indexing.
    pub exclude: Vec<String>,
    /// Number of files to process per embedding batch.
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vault_path: None,
            top_n: 5,
            exclude: vec![],
            batch_size: 64,
        }
    }
}

impl Config {
    /// Canonical data directory: `~/.engraph/`.
    pub fn data_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("could not determine home directory")?;
        Ok(home.join(".engraph"))
    }

    /// Load config from `~/.engraph/config.toml`, falling back to defaults.
    pub fn load() -> Result<Self> {
        let config_path = Self::data_dir()?.join("config.toml");

        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?;
            let config: Config = toml::from_str(&contents)
                .with_context(|| format!("failed to parse {}", config_path.display()))?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Merge CLI-provided values over the loaded config.
    pub fn merge_vault_path(&mut self, path: Option<PathBuf>) {
        if path.is_some() {
            self.vault_path = path;
        }
    }

    /// Merge CLI-provided top_n over the loaded config.
    pub fn merge_top_n(&mut self, n: Option<usize>) {
        if let Some(n) = n {
            self.top_n = n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_values() {
        let cfg = Config::default();
        assert_eq!(cfg.top_n, 5);
        assert_eq!(cfg.batch_size, 64);
        assert!(cfg.exclude.is_empty());
        assert!(cfg.vault_path.is_none());
    }

    #[test]
    fn data_dir_ends_with_engraph() {
        let dir = Config::data_dir().unwrap();
        assert!(dir.ends_with(".engraph"));
    }

    #[test]
    fn parse_config_toml() {
        let toml_str = r#"
vault_path = "/tmp/vault"
top_n = 10
exclude = ["*.canvas", ".obsidian"]
batch_size = 128
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.vault_path.unwrap(), PathBuf::from("/tmp/vault"));
        assert_eq!(cfg.top_n, 10);
        assert_eq!(cfg.exclude, vec!["*.canvas", ".obsidian"]);
        assert_eq!(cfg.batch_size, 128);
    }

    #[test]
    fn parse_partial_config_uses_defaults() {
        let toml_str = r#"top_n = 20"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.top_n, 20);
        assert_eq!(cfg.batch_size, 64); // default
        assert!(cfg.vault_path.is_none());
    }

    #[test]
    fn merge_overrides_when_present() {
        let mut cfg = Config::default();
        cfg.merge_vault_path(Some(PathBuf::from("/my/vault")));
        cfg.merge_top_n(Some(42));
        assert_eq!(cfg.vault_path.unwrap(), PathBuf::from("/my/vault"));
        assert_eq!(cfg.top_n, 42);
    }

    #[test]
    fn merge_preserves_when_none() {
        let mut cfg = Config::default();
        cfg.top_n = 10;
        cfg.merge_top_n(None);
        assert_eq!(cfg.top_n, 10);
    }

    #[test]
    fn load_from_nonexistent_file_returns_defaults() {
        // Config::load() reads from ~/.engraph/config.toml.
        // If it doesn't exist, defaults are fine. We test the parsing path
        // separately above. This just ensures load() doesn't panic.
        let cfg = Config::load().unwrap();
        assert_eq!(cfg.batch_size, 64);
    }
}
