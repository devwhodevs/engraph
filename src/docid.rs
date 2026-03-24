use sha2::{Digest, Sha256};

/// Generate a 6-character hex docid from a file path.
/// Deterministic: same path always produces same docid.
pub fn generate_docid(path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.as_bytes());
    let hash = hasher.finalize();
    format!("{:02x}{:02x}{:02x}", hash[0], hash[1], hash[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_docid_length_and_hex() {
        let docid = generate_docid("notes/test.md");
        assert_eq!(docid.len(), 6, "docid should be 6 characters");
        assert!(
            docid.chars().all(|c| c.is_ascii_hexdigit()),
            "docid should be all hex chars, got: {}",
            docid
        );
    }

    #[test]
    fn test_docid_deterministic() {
        let a = generate_docid("notes/test.md");
        let b = generate_docid("notes/test.md");
        assert_eq!(a, b, "same path must produce same docid");
    }

    #[test]
    fn test_docid_unique() {
        let a = generate_docid("notes/a.md");
        let b = generate_docid("notes/b.md");
        assert_ne!(a, b, "different paths should produce different docids");
    }
}
