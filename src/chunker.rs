/// Represents a single semantic chunk extracted from a markdown file.
pub struct Chunk {
    /// The `## ` heading line, if any.
    pub heading: Option<String>,
    /// Full chunk text (without frontmatter).
    pub text: String,
    /// First 200 chars of `text`, truncated with `"..."` if needed.
    pub snippet: String,
}

/// Result of parsing a markdown file.
pub struct ParsedMarkdown {
    /// Tags extracted from YAML frontmatter.
    pub tags: Vec<String>,
    /// Semantic chunks split on `## ` headings.
    pub chunks: Vec<Chunk>,
}

/// Parse markdown content into frontmatter tags and heading-based chunks.
///
/// 1. Strip YAML frontmatter (between `---` at start), parse `tags` if present.
/// 2. Split on lines starting with `## ` — deeper headings stay in parent chunk.
/// 3. Content before first `## ` becomes a chunk with `heading: None`.
/// 4. Skip empty (whitespace-only) chunks.
/// 5. Snippet = first 200 chars of text, `"..."` appended if truncated.
pub fn chunk_markdown(content: &str) -> ParsedMarkdown {
    let (tags, body) = parse_frontmatter(content);

    let mut chunks = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in body.lines() {
        if line.starts_with("## ") {
            // Flush previous chunk
            flush_chunk(&mut chunks, current_heading.take(), &current_lines);
            current_heading = Some(line.to_string());
            current_lines.clear();
        } else {
            current_lines.push(line);
        }
    }
    // Flush last chunk
    flush_chunk(&mut chunks, current_heading, &current_lines);

    ParsedMarkdown { tags, chunks }
}

fn flush_chunk(chunks: &mut Vec<Chunk>, heading: Option<String>, lines: &[&str]) {
    let text = lines.join("\n").trim().to_string();
    if text.is_empty() && heading.is_none() {
        return;
    }
    // Build full text including the heading line
    let full_text = match &heading {
        Some(h) => {
            if text.is_empty() {
                h.clone()
            } else {
                format!("{h}\n{text}")
            }
        }
        None => text.clone(),
    };
    if full_text.trim().is_empty() {
        return;
    }
    let snippet = make_snippet(&full_text);
    chunks.push(Chunk {
        heading,
        text: full_text,
        snippet,
    });
}

fn make_snippet(text: &str) -> String {
    if text.len() > 200 {
        let truncated: String = text.chars().take(200).collect();
        format!("{truncated}...")
    } else {
        text.to_string()
    }
}

/// Parse YAML frontmatter and return (tags, body_without_frontmatter).
fn parse_frontmatter(content: &str) -> (Vec<String>, &str) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (Vec::new(), content);
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    let after_first = after_first.trim_start_matches(|c: char| c == '-'); // handle "----"
    let after_first = after_first.strip_prefix('\n').unwrap_or(after_first);

    if let Some(end_pos) = after_first.find("\n---") {
        let yaml_block = &after_first[..end_pos];
        let body_start = end_pos + 4; // skip "\n---"
        let body = after_first[body_start..]
            .strip_prefix('\n')
            .unwrap_or(&after_first[body_start..]);
        let tags = parse_tags_from_yaml(yaml_block);
        (tags, body)
    } else {
        (Vec::new(), content)
    }
}

/// Parse `tags` field from a YAML block. Supports:
/// - `tags: [a, b, c]`
/// - `tags:\n  - a\n  - b`
fn parse_tags_from_yaml(yaml: &str) -> Vec<String> {
    let lines: Vec<&str> = yaml.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("tags:") {
            let after_colon = trimmed.strip_prefix("tags:").unwrap().trim();
            // Inline list: tags: [a, b]
            if after_colon.starts_with('[') {
                let inner = after_colon
                    .trim_start_matches('[')
                    .trim_end_matches(']');
                return inner
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            // If there's content after colon on same line (single tag)
            if !after_colon.is_empty() {
                return vec![after_colon.to_string()];
            }
            // Block list: tags:\n  - a\n  - b
            let mut tags = Vec::new();
            for subsequent in &lines[i + 1..] {
                let st = subsequent.trim();
                if st.starts_with("- ") {
                    tags.push(st.strip_prefix("- ").unwrap().trim().to_string());
                } else if st.is_empty() {
                    continue;
                } else {
                    break;
                }
            }
            return tags;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_by_headings() {
        let md = "## A\nContent A\n## B\nContent B\n";
        let parsed = chunk_markdown(md);
        assert_eq!(parsed.chunks.len(), 2);
        assert_eq!(parsed.chunks[0].heading.as_deref(), Some("## A"));
        assert_eq!(parsed.chunks[1].heading.as_deref(), Some("## B"));
        assert!(parsed.chunks[0].text.contains("Content A"));
        assert!(parsed.chunks[1].text.contains("Content B"));
    }

    #[test]
    fn test_no_headings_single_chunk() {
        let md = "Just some plain text\nwith multiple lines.";
        let parsed = chunk_markdown(md);
        assert_eq!(parsed.chunks.len(), 1);
        assert!(parsed.chunks[0].heading.is_none());
        assert!(parsed.chunks[0].text.contains("Just some plain text"));
    }

    #[test]
    fn test_frontmatter_excluded() {
        let md = "---\ntags: [a]\n---\n# Title\nBody";
        let parsed = chunk_markdown(md);
        assert_eq!(parsed.chunks.len(), 1);
        assert!(!parsed.chunks[0].text.contains("tags"));
        assert!(!parsed.chunks[0].text.contains("---"));
        assert!(parsed.chunks[0].text.contains("Body"));
    }

    #[test]
    fn test_nested_headings_stay_together() {
        let md = "## Parent\nParent content\n### Child\nChild content\n";
        let parsed = chunk_markdown(md);
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.chunks[0].heading.as_deref(), Some("## Parent"));
        assert!(parsed.chunks[0].text.contains("### Child"));
        assert!(parsed.chunks[0].text.contains("Child content"));
    }

    #[test]
    fn test_snippet_truncation() {
        let long_text = "a".repeat(300);
        let md = format!("## Heading\n{long_text}");
        let parsed = chunk_markdown(&md);
        assert_eq!(parsed.chunks.len(), 1);
        assert!(parsed.chunks[0].snippet.ends_with("..."));
        // 200 chars + "..." = 203
        assert_eq!(parsed.chunks[0].snippet.len(), 203);
    }

    #[test]
    fn test_empty_file() {
        let parsed = chunk_markdown("");
        assert!(parsed.chunks.is_empty());
    }

    #[test]
    fn test_parse_frontmatter_tags() {
        let md = "---\ntags: [rust, cli, search]\n---\n# Hello\nWorld";
        let parsed = chunk_markdown(md);
        assert_eq!(parsed.tags, vec!["rust", "cli", "search"]);
    }
}
