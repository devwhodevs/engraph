use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Trait for embedding backends. Any model that can embed text implements this.
pub trait ModelBackend {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>>;
    fn token_count(&self, text: &str) -> usize;
    fn dim(&self) -> usize;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    Onnx,
    Gguf,
    File,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub format: ModelFormat,
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    pub name: String,
    pub format: ModelFormat,
    pub url: String,
    pub sha256: String,
    pub dim: usize,
    pub description: String,
}

pub struct ModelRegistry {
    pub entries: Vec<ModelRegistryEntry>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            entries: vec![ModelRegistryEntry {
                name: "onnx:all-MiniLM-L6-v2".to_string(),
                format: ModelFormat::Onnx,
                url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx".to_string(),
                sha256: "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452".to_string(),
                dim: 384,
                description: "Lightweight general-purpose sentence embeddings".to_string(),
            }],
        }
    }
}

impl ModelRegistry {
    pub fn get(&self, name: &str) -> Option<&ModelRegistryEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}

pub fn parse_model_spec(spec: &str) -> ModelSpec {
    if let Some(path) = spec.strip_prefix("file:") {
        return ModelSpec {
            format: ModelFormat::File,
            name: spec.to_string(),
            path: path.to_string(),
        };
    }
    if let Some((format_str, name)) = spec.split_once(':') {
        let format = match format_str {
            "onnx" => ModelFormat::Onnx,
            "gguf" => ModelFormat::Gguf,
            _ => ModelFormat::Onnx,
        };
        ModelSpec {
            format,
            name: name.to_string(),
            path: String::new(),
        }
    } else {
        ModelSpec {
            format: ModelFormat::Onnx,
            name: spec.to_string(),
            path: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_default() {
        let registry = ModelRegistry::default();
        assert_eq!(registry.entries.len(), 1);
        let entry = &registry.entries[0];
        assert_eq!(entry.name, "onnx:all-MiniLM-L6-v2");
        assert_eq!(entry.dim, 384);
        assert_eq!(entry.format, ModelFormat::Onnx);
    }

    #[test]
    fn test_parse_model_spec_onnx() {
        let spec = parse_model_spec("onnx:all-MiniLM-L6-v2");
        assert_eq!(spec.format, ModelFormat::Onnx);
        assert_eq!(spec.name, "all-MiniLM-L6-v2");
        assert!(spec.path.is_empty());
    }

    #[test]
    fn test_parse_model_spec_file() {
        let spec = parse_model_spec("file:/path/to/model.onnx");
        assert_eq!(spec.format, ModelFormat::File);
        assert_eq!(spec.name, "file:/path/to/model.onnx");
        assert_eq!(spec.path, "/path/to/model.onnx");
    }

    #[test]
    fn test_parse_model_spec_bare() {
        let spec = parse_model_spec("my-custom-model");
        assert_eq!(spec.format, ModelFormat::Onnx);
        assert_eq!(spec.name, "my-custom-model");
        assert!(spec.path.is_empty());
    }

    #[test]
    fn test_registry_get_existing() {
        let registry = ModelRegistry::default();
        let entry = registry.get("onnx:all-MiniLM-L6-v2");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().dim, 384);
    }

    #[test]
    fn test_registry_get_missing() {
        let registry = ModelRegistry::default();
        assert!(registry.get("nonexistent-model").is_none());
    }
}
