use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use hnsw_rs::anndists::dist::distances::DistCosine;
use hnsw_rs::api::AnnT;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::hnswio::HnswIo;

const EMBEDDING_DIM: usize = 384;
const MAX_NB_CONNECTION: usize = 16;
const MAX_LAYER: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const FILE_BASENAME: &str = "engraph";

/// Wrapper around the HNSW index for vector similarity search.
pub struct HnswIndex {
    inner: Hnsw<'static, f32, DistCosine>,
    next_id: u64,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(max_elements: usize) -> Self {
        let inner = Hnsw::new(
            MAX_NB_CONNECTION,
            max_elements,
            MAX_LAYER,
            EF_CONSTRUCTION,
            DistCosine,
        );
        Self { inner, next_id: 0 }
    }

    /// Load an HNSW index from the given directory.
    ///
    /// The `HnswIo` is leaked to satisfy the `'static` lifetime on the inner `Hnsw`.
    /// This is acceptable because we only load an index once for the lifetime of the process.
    pub fn load(dir: &Path) -> Result<Self> {
        let hnsw_io = Box::new(HnswIo::new(dir, FILE_BASENAME));
        let hnsw_io: &'static mut HnswIo = Box::leak(hnsw_io);
        let inner: Hnsw<'static, f32, DistCosine> = hnsw_io
            .load_hnsw()
            .context("failed to load HNSW index from disk")?;

        let nb_point = inner.get_nb_point();
        let next_id = nb_point as u64;

        Ok(Self { inner, next_id })
    }

    /// Insert a single vector and return its assigned vector ID.
    pub fn insert(&mut self, vector: &[f32]) -> u64 {
        assert_eq!(
            vector.len(),
            EMBEDDING_DIM,
            "vector dimension mismatch: expected {EMBEDDING_DIM}, got {}",
            vector.len()
        );
        let id = self.next_id;
        self.inner.insert((vector, id as usize));
        self.next_id += 1;
        id
    }

    /// Insert a vector with a specific ID (used when rebuilding from stored vectors).
    pub fn insert_with_id(&mut self, vector: &[f32], id: u64) {
        assert_eq!(
            vector.len(),
            EMBEDDING_DIM,
            "vector dimension mismatch: expected {EMBEDDING_DIM}, got {}",
            vector.len()
        );
        self.inner.insert((vector, id as usize));
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }

    /// Insert a batch of vectors and return their assigned vector IDs.
    pub fn insert_batch(&mut self, vectors: &[Vec<f32>]) -> Vec<u64> {
        vectors.iter().map(|v| self.insert(v)).collect()
    }

    /// Search for the k nearest neighbors of the query vector.
    ///
    /// Returns `(vector_id, score)` pairs sorted by ascending distance,
    /// excluding any IDs in `tombstones`. Requests `k * 2` results from
    /// the underlying index for tombstone headroom.
    pub fn search(&self, query: &[f32], k: usize, tombstones: &HashSet<u64>) -> Vec<(u64, f32)> {
        if self.inner.get_nb_point() == 0 {
            return Vec::new();
        }

        let ef_search = (k * 2).max(EF_CONSTRUCTION);
        let neighbours = self.inner.search(query, k * 2, ef_search);

        let mut results: Vec<(u64, f32)> = neighbours
            .into_iter()
            .filter(|n| !tombstones.contains(&(n.d_id as u64)))
            .map(|n| (n.d_id as u64, n.distance))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Save the index to the given directory.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).context("failed to create HNSW save directory")?;
        self.inner
            .file_dump(dir, FILE_BASENAME)
            .context("failed to dump HNSW index to disk")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn random_vector(seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random using a linear congruential generator.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..EMBEDDING_DIM)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Normalize to [-1, 1]
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(100);
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| random_vector(i)).collect();
        let ids = index.insert_batch(&vectors);
        assert_eq!(ids.len(), 10);

        // Search for the first vector — it should be the top result.
        let results = index.search(&vectors[0], 5, &HashSet::new());
        assert!(!results.is_empty(), "search returned no results");
        assert_eq!(
            results[0].0, ids[0],
            "expected the query vector itself to be the top result"
        );
        // Distance to itself should be ~0 for cosine.
        assert!(
            results[0].1 < 0.01,
            "distance to self should be near zero, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_search_with_tombstones() {
        let mut index = HnswIndex::new(100);
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| random_vector(i + 100)).collect();
        let ids = index.insert_batch(&vectors);

        // Tombstone the first vector.
        let mut tombstones = HashSet::new();
        tombstones.insert(ids[0]);

        let results = index.search(&vectors[0], 5, &tombstones);
        for (id, _score) in &results {
            assert_ne!(*id, ids[0], "tombstoned ID should not appear in results");
        }
    }

    #[test]
    fn test_save_and_load() {
        let tmpdir = TempDir::new().unwrap();
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| random_vector(i + 200)).collect();

        // Build and save.
        {
            let mut index = HnswIndex::new(100);
            index.insert_batch(&vectors);
            index.save(tmpdir.path()).unwrap();
        }

        // Load and search.
        let index = HnswIndex::load(tmpdir.path()).unwrap();
        let results = index.search(&vectors[0], 3, &HashSet::new());
        assert!(
            !results.is_empty(),
            "search after reload returned no results"
        );
        assert_eq!(
            results[0].0, 0,
            "expected vector 0 to be the top result after reload"
        );
    }

    #[test]
    fn test_empty_index_search() {
        let index = HnswIndex::new(100);
        let query = random_vector(999);
        let results = index.search(&query, 5, &HashSet::new());
        assert!(results.is_empty(), "empty index should return no results");
    }
}
