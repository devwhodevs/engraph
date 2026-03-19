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

/// Split oversized chunks into sub-chunks that fit within `max_tokens`.
///
/// - `token_count` counts tokens in a string (closure for testability).
/// - Chunks under `max_tokens` pass through unchanged.
/// - Over-sized chunks are split on sentence boundaries (`. ` or `\n`).
/// - Each sub-chunk after the first includes `overlap_tokens` worth of trailing
///   text from the previous sub-chunk.
/// - Subsequent sub-chunks get ` (cont.)` appended to the parent heading.
pub fn split_oversized_chunks(
    chunks: Vec<Chunk>,
    token_count: &dyn Fn(&str) -> usize,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<Chunk> {
    let mut result = Vec::new();
    for chunk in chunks {
        if token_count(&chunk.text) <= max_tokens {
            result.push(chunk);
            continue;
        }
        // Split text into sentences on `. ` or `\n`
        let sentences = split_sentences(&chunk.text);
        let mut sub_chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for sentence in &sentences {
            let candidate = if current.is_empty() {
                sentence.to_string()
            } else {
                format!("{current}{sentence}")
            };
            if !current.is_empty() && token_count(&candidate) > max_tokens {
                // Flush current sub-chunk
                sub_chunks.push(current.clone());
                // Build overlap prefix from the end of the previous sub-chunk
                let overlap = build_overlap(&current, token_count, overlap_tokens);
                current = format!("{overlap}{sentence}");
            } else {
                current = candidate;
            }
        }
        if !current.trim().is_empty() {
            sub_chunks.push(current);
        }

        // Convert sub-chunks into Chunk structs
        for (i, sub_text) in sub_chunks.into_iter().enumerate() {
            let heading = if i == 0 {
                chunk.heading.clone()
            } else {
                chunk.heading.as_ref().map(|h| format!("{h} (cont.)"))
            };
            let full_text = match &heading {
                Some(h) => format!("{h}\n{}", sub_text.trim()),
                None => sub_text.trim().to_string(),
            };
            let snippet = make_snippet(&full_text);
            result.push(Chunk {
                heading,
                text: full_text,
                snippet,
            });
        }
    }
    result
}

/// Split text into sentence-like segments, preserving delimiters.
/// Splits on `. ` (sentence end) and `\n` (line break).
fn split_sentences(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        current.push(chars[i]);
        if chars[i] == '\n' {
            segments.push(current.clone());
            current.clear();
        } else if chars[i] == '.' && i + 1 < chars.len() && chars[i + 1] == ' ' {
            current.push(' ');
            i += 1; // consume the space
            segments.push(current.clone());
            current.clear();
        }
        i += 1;
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Build an overlap string from the end of `text` that is approximately
/// `overlap_tokens` tokens long, measured by `token_count`.
fn build_overlap(text: &str, token_count: &dyn Fn(&str) -> usize, overlap_tokens: usize) -> String {
    if overlap_tokens == 0 {
        return String::new();
    }
    // Work backwards through words to build overlap
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut overlap = String::new();
    for &word in words.iter().rev() {
        let candidate = if overlap.is_empty() {
            word.to_string()
        } else {
            format!("{word} {overlap}")
        };
        if token_count(&candidate) > overlap_tokens {
            break;
        }
        overlap = candidate;
    }
    if overlap.is_empty() {
        overlap
    } else {
        format!("{overlap} ")
    }
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

    #[test]
    fn test_long_chunk_split() {
        // Generate ~600 words of text with sentence boundaries
        let sentences: Vec<String> = (0..60)
            .map(|i| format!("This is sentence number {} with several words to pad it out.", i))
            .collect();
        let long_text = sentences.join(" ");
        let word_count = long_text.split_whitespace().count();
        assert!(word_count > 512, "Test text must exceed 512 tokens (words); got {word_count}");

        let chunk = Chunk {
            heading: Some("## Long Section".to_string()),
            text: format!("## Long Section\n{long_text}"),
            snippet: make_snippet(&format!("## Long Section\n{long_text}")),
        };

        let token_fn = |s: &str| s.split_whitespace().count();
        let result = split_oversized_chunks(vec![chunk], &token_fn, 512, 50);

        assert!(result.len() >= 2, "Expected at least 2 sub-chunks, got {}", result.len());
        // First chunk keeps original heading
        assert_eq!(result[0].heading.as_deref(), Some("## Long Section"));
        // Subsequent chunks get (cont.)
        assert_eq!(result[1].heading.as_deref(), Some("## Long Section (cont.)"));
        // All sub-chunks should be within token limit
        for c in &result {
            let tokens = token_fn(&c.text);
            assert!(tokens <= 512, "Sub-chunk has {tokens} tokens, exceeds 512");
        }
        // Snippets should be regenerated
        for c in &result {
            assert!(!c.snippet.is_empty());
        }
    }

    #[test]
    fn test_short_chunk_no_split() {
        let chunk = Chunk {
            heading: Some("## Short".to_string()),
            text: "## Short\nJust a few words here.".to_string(),
            snippet: "## Short\nJust a few words here.".to_string(),
        };

        let token_fn = |s: &str| s.split_whitespace().count();
        let result = split_oversized_chunks(vec![chunk], &token_fn, 512, 50);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].heading.as_deref(), Some("## Short"));
        assert_eq!(result[0].text, "## Short\nJust a few words here.");
    }
}
