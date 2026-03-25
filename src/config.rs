use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Model override configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Override embedding model URI (e.g., "hf:repo/file.gguf").
    pub embed: Option<String>,
    /// Override reranker model URI.
    pub rerank: Option<String>,
    /// Override expansion/orchestrator model URI.
    pub expand: Option<String>,
}

/// Application configuration, loaded from `~/.engraph/config.toml` with CLI overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Whether intelligence features are enabled. None = not yet configured.
    pub intelligence: Option<bool>,
    /// Model override URIs.
    pub models: ModelConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vault_path: None,
            top_n: 5,
            exclude: vec![".obsidian/".to_string()],
            batch_size: 64,
            intelligence: None,
            models: ModelConfig::default(),
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

    /// Load vault profile from `~/.engraph/vault.toml`, if it exists.
    pub fn load_vault_profile() -> Result<Option<crate::profile::VaultProfile>> {
        let dir = Self::data_dir()?;
        crate::profile::load_vault_toml(&dir)
    }

    /// Whether intelligence is enabled (defaults to false if not configured).
    pub fn intelligence_enabled(&self) -> bool {
        self.intelligence.unwrap_or(false)
    }

    /// Save config to a specific path.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Load config from a specific path.
    pub fn load_from(path: &Path) -> Result<Self> {
        let contents =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: Config =
            toml::from_str(&contents).with_context(|| format!("parsing {}", path.display()))?;
        Ok(config)
    }

    /// Save to the default config path (`~/.engraph/config.toml`).
    pub fn save(&self) -> Result<()> {
        let path = Self::data_dir()?.join("config.toml");
        std::fs::create_dir_all(path.parent().unwrap())?;
        self.save_to(&path)
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
        assert_eq!(cfg.exclude, vec![".obsidian/"]);
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

    #[test]
    fn parse_intelligence_config() {
        let toml_str = r#"
intelligence = true

[models]
embed = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
rerank = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.intelligence, Some(true));
        assert!(cfg.models.embed.is_some());
        assert!(cfg.models.rerank.is_some());
        assert!(cfg.models.expand.is_none());
    }

    #[test]
    fn intelligence_defaults_to_none() {
        let cfg = Config::default();
        assert!(cfg.intelligence.is_none());
        assert!(cfg.models.embed.is_none());
    }

    #[test]
    fn intelligence_false_disables_features() {
        let toml_str = r#"intelligence = false"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.intelligence, Some(false));
        assert!(!cfg.intelligence_enabled());
    }

    #[test]
    fn test_config_roundtrip_with_intelligence() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");

        let mut cfg = Config::default();
        cfg.intelligence = Some(true);
        cfg.models.embed = Some("hf:custom/model/embed.gguf".into());

        cfg.save_to(&config_path).unwrap();

        let loaded = Config::load_from(&config_path).unwrap();
        assert_eq!(loaded.intelligence, Some(true));
        assert_eq!(
            loaded.models.embed,
            Some("hf:custom/model/embed.gguf".into())
        );
    }
}
