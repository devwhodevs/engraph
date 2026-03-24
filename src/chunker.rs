/// Represents a single semantic chunk extracted from a markdown file.
pub struct Chunk {
    /// The heading line (any `#` level), if any.
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
    /// Semantic chunks produced by smart break-point scoring.
    pub chunks: Vec<Chunk>,
}

/// A scored candidate position where a chunk boundary could be placed.
pub struct BreakPoint {
    pub byte_offset: usize,
    pub line_number: usize,
    pub score: u32,
    pub inside_code_fence: bool,
}

/// Scan content line by line and assign break-point scores.
///
/// Scoring rules:
/// - `# ` heading: 100
/// - `## ` heading: 90
/// - `### ` heading: 80
/// - `#### ` heading: 70
/// - `##### ` heading: 60
/// - `###### ` heading: 50
/// - `---`/`***`/`___` (thematic breaks): 60
/// - Code fence boundaries (`` ``` ``): 80
/// - Empty lines: 20
/// - List items (`- `, `* `, digit prefix): 5
/// - Other non-empty lines: 1 (excluded from results)
pub fn find_break_points(content: &str) -> Vec<BreakPoint> {
    let mut break_points = Vec::new();
    let mut inside_code_fence = false;
    let mut byte_offset = 0;

    for (line_number, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        let score = if trimmed.starts_with("```") {
            // Toggle fence state; the fence boundary itself is NOT "inside"
            inside_code_fence = !inside_code_fence;
            // Mark as not inside — fence boundaries are valid break points
            let bp_inside = false;
            break_points.push(BreakPoint {
                byte_offset,
                line_number,
                score: 80,
                inside_code_fence: bp_inside,
            });
            byte_offset += line.len() + if byte_offset + line.len() < content.len() { 1 } else { 0 };
            continue;
        } else if inside_code_fence {
            // Lines inside code fences: push with inside_code_fence = true
            // so callers can inspect the field; smart_chunk filters them out.
            break_points.push(BreakPoint {
                byte_offset,
                line_number,
                score: 1,
                inside_code_fence: true,
            });
            byte_offset += line.len() + if byte_offset + line.len() < content.len() { 1 } else { 0 };
            continue;
        } else if trimmed.starts_with("# ") && !trimmed.starts_with("## ") {
            100
        } else if trimmed.starts_with("## ") && !trimmed.starts_with("### ") {
            90
        } else if trimmed.starts_with("### ") && !trimmed.starts_with("#### ") {
            80
        } else if trimmed.starts_with("#### ") && !trimmed.starts_with("##### ") {
            70
        } else if trimmed.starts_with("##### ") && !trimmed.starts_with("###### ") {
            60
        } else if trimmed.starts_with("###### ") {
            50
        } else if is_thematic_break(trimmed) {
            60
        } else if trimmed.is_empty() {
            20
        } else if is_list_item(trimmed) {
            5
        } else {
            1
        };

        if score > 1 {
            break_points.push(BreakPoint {
                byte_offset,
                line_number,
                score,
                inside_code_fence,
            });
        }

        byte_offset += line.len() + if byte_offset + line.len() < content.len() { 1 } else { 0 };
    }

    break_points
}

/// Check if a line is a thematic break (`---`, `***`, `___` with 3+ chars, optional spaces).
fn is_thematic_break(trimmed: &str) -> bool {
    if trimmed.len() < 3 {
        return false;
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let first = chars[0];
    if first != '-' && first != '*' && first != '_' {
        return false;
    }
    chars.iter().all(|&c| c == first || c == ' ')
        && chars.iter().filter(|&&c| c == first).count() >= 3
}

/// Check if a line starts as a list item.
fn is_list_item(trimmed: &str) -> bool {
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        return true;
    }
    // Check for ordered list: digit(s) followed by `. ` or `) `
    let mut chars = trimmed.chars();
    if let Some(first) = chars.next()
        && first.is_ascii_digit() {
            for c in chars {
                if c.is_ascii_digit() {
                    continue;
                }
                if c == '.' || c == ')' {
                    return true;
                }
                break;
            }
        }
    false
}

/// Approximate token count: ~4 chars per token.
fn approx_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Extract the first heading line from text (any `#` level).
fn extract_heading(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') && trimmed.contains(' ') {
            return Some(line.to_string());
        }
    }
    None
}

/// Smart chunk splitting using scored break points.
///
/// - `target_tokens`: desired chunk size in approximate tokens (~4 chars/token)
/// - `overlap_pct`: percentage of target_tokens to overlap between chunks (e.g. 15 = 15%)
///
/// Never splits inside code fences. Finds the best break point near the token
/// target using a weighted score that considers both inherent score and distance.
pub fn smart_chunk(content: &str, target_tokens: usize, overlap_pct: usize) -> Vec<Chunk> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let break_points = find_break_points(content);
    let target_chars = target_tokens * 4;
    let overlap_chars = (target_chars * overlap_pct) / 100;

    // If the content fits in one chunk, return it as-is
    if approx_tokens(content) <= target_tokens {
        let heading = extract_heading(content);
        let snippet = make_snippet(content.trim());
        return vec![Chunk {
            heading,
            text: content.trim().to_string(),
            snippet,
        }];
    }

    let mut chunks = Vec::new();
    let mut start_offset = 0;

    while start_offset < content.len() {
        let remaining = &content[start_offset..];
        if remaining.trim().is_empty() {
            break;
        }

        // If remaining content fits in one chunk, take it all
        if approx_tokens(remaining) <= target_tokens {
            let text = remaining.trim().to_string();
            if !text.is_empty() {
                let heading = extract_heading(&text);
                let snippet = make_snippet(&text);
                chunks.push(Chunk {
                    heading,
                    text,
                    snippet,
                });
            }
            break;
        }

        // Find the ideal cut point: target_chars from start_offset
        let ideal_end = start_offset + target_chars;

        // Find the best break point near ideal_end
        // Filter to break points that are:
        // 1. After start_offset
        // 2. Not inside code fences
        // 3. Within a reasonable range of ideal_end
        let best_bp = break_points
            .iter()
            .filter(|bp| {
                bp.byte_offset > start_offset
                    && !bp.inside_code_fence
                    && bp.byte_offset <= start_offset + target_chars * 2
            })
            .max_by(|a, b| {
                let score_a = weighted_score(a, ideal_end);
                let score_b = weighted_score(b, ideal_end);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let cut_offset = match best_bp {
            Some(bp) => bp.byte_offset,
            None => {
                // No good break point found; cut at target
                let cut = (start_offset + target_chars).min(content.len());
                // Try to find a newline near the cut
                let fallback = if let Some(nl) = content[start_offset..cut.min(content.len())]
                    .rfind('\n')
                    .map(|p| start_offset + p + 1)
                {
                    if nl > start_offset {
                        nl
                    } else {
                        cut
                    }
                } else {
                    cut
                };
                // Guard: always advance by at least one byte to prevent infinite loops
                fallback.max(start_offset + 1).min(content.len())
            }
        };

        let chunk_text = content[start_offset..cut_offset].trim().to_string();
        if !chunk_text.is_empty() {
            let heading = extract_heading(&chunk_text);
            let snippet = make_snippet(&chunk_text);
            chunks.push(Chunk {
                heading,
                text: chunk_text,
                snippet,
            });
        }

        // Move start forward, applying overlap
        if cut_offset >= content.len() {
            break;
        }
        start_offset = if overlap_chars > 0 && cut_offset > overlap_chars {
            (cut_offset - overlap_chars).max(start_offset + 1)
        } else {
            cut_offset
        };
    }

    chunks
}

/// Compute a weighted score that balances break-point quality with proximity to target.
fn weighted_score(bp: &BreakPoint, ideal_offset: usize) -> f64 {
    let distance = (bp.byte_offset as f64 - ideal_offset as f64).abs();
    // Normalize distance: closer to ideal = higher score multiplier
    // At distance 0, multiplier = 1.0; at distance = ideal_offset, multiplier ~= 0
    let distance_factor = 1.0 / (1.0 + distance / 500.0);
    bp.score as f64 * distance_factor
}

/// Parse markdown content into frontmatter tags and smart-chunked pieces.
///
/// 1. Strip YAML frontmatter (between `---` at start), parse `tags` if present.
/// 2. Run `smart_chunk` on the body with target 512 tokens, 15% overlap.
/// 3. Return `ParsedMarkdown { tags, chunks }`.
pub fn chunk_markdown(content: &str) -> ParsedMarkdown {
    let (tags, body) = parse_frontmatter(content);

    let chunks = smart_chunk(body, 512, 15);

    ParsedMarkdown { tags, chunks }
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
    let after_first = after_first.trim_start_matches('-'); // handle "----"
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
                let inner = after_colon.trim_start_matches('[').trim_end_matches(']');
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

    // ── Break-point detection tests ──────────────────────────────────────

    #[test]
    fn test_find_break_points() {
        let content = "# Title\n\nSome text\n\n## Section\nContent\n### Sub\nMore\n\n---\n";
        let bps = find_break_points(content);

        // Collect (line_number, score) pairs for easy assertion
        let pairs: Vec<(usize, u32)> = bps.iter().map(|bp| (bp.line_number, bp.score)).collect();

        // # Title -> 100
        assert!(pairs.contains(&(0, 100)), "Expected # heading at line 0 with score 100, got: {:?}", pairs);
        // empty line -> 20
        assert!(pairs.contains(&(1, 20)), "Expected empty line at line 1 with score 20");
        // empty line -> 20
        assert!(pairs.contains(&(3, 20)), "Expected empty line at line 3 with score 20");
        // ## Section -> 90
        assert!(pairs.contains(&(4, 90)), "Expected ## heading at line 4 with score 90");
        // ### Sub -> 80
        assert!(pairs.contains(&(6, 80)), "Expected ### heading at line 6 with score 80");
        // empty line -> 20
        assert!(pairs.contains(&(8, 20)), "Expected empty line at line 8 with score 20");
        // --- -> 60
        assert!(pairs.contains(&(9, 60)), "Expected thematic break at line 9 with score 60");

        // "Some text", "Content", "More" have score 1 and should NOT appear
        // (only lines inside code fences get score 1 in results)
        for bp in &bps {
            assert!(bp.score > 1 || bp.inside_code_fence,
                "Non-fence break points should not include lines with score <= 1");
        }
    }

    #[test]
    fn test_find_break_points_code_fence() {
        let content = "Before\n\n```rust\nlet x = 1;\nlet y = 2;\n```\n\nAfter\n";
        let bps = find_break_points(content);

        // The opening ``` should be a break point with score 80, NOT inside fence
        let opening = bps.iter().find(|bp| bp.line_number == 2).unwrap();
        assert_eq!(opening.score, 80);
        assert!(!opening.inside_code_fence, "Opening fence should not be marked as inside");

        // The closing ``` should be a break point with score 80, NOT inside fence
        // (it toggles the fence off)
        let closing = bps.iter().find(|bp| bp.line_number == 5).unwrap();
        assert_eq!(closing.score, 80);
        assert!(!closing.inside_code_fence, "Closing fence should not be marked as inside");

        // Lines inside the fence (let x = 1; let y = 2;) SHOULD appear with inside_code_fence = true
        let inside_bps: Vec<&BreakPoint> = bps.iter().filter(|bp| bp.line_number == 3 || bp.line_number == 4).collect();
        assert_eq!(inside_bps.len(), 2, "Expected 2 break points inside code fence");
        for bp in &inside_bps {
            assert!(bp.inside_code_fence, "Line {} inside fence should have inside_code_fence=true", bp.line_number);
            assert_eq!(bp.score, 1, "Line {} inside fence should have score 1", bp.line_number);
        }
    }

    #[test]
    fn test_find_break_points_list_items() {
        let content = "- item one\n* item two\n1. numbered\nplain text\n";
        let bps = find_break_points(content);
        let pairs: Vec<(usize, u32)> = bps.iter().map(|bp| (bp.line_number, bp.score)).collect();
        assert!(pairs.contains(&(0, 5)), "Expected list item at line 0 with score 5");
        assert!(pairs.contains(&(1, 5)), "Expected list item at line 1 with score 5");
        assert!(pairs.contains(&(2, 5)), "Expected numbered list item at line 2 with score 5");
        // "plain text" has score 1, should NOT appear
        assert!(!bps.iter().any(|bp| bp.line_number == 3), "Plain text should not be a break point");
    }

    // ── Smart chunk tests ────────────────────────────────────────────────

    #[test]
    fn test_smart_chunk_single() {
        // Short content should produce a single chunk
        let content = "# Hello\nSome short content here.";
        let chunks = smart_chunk(content, 512, 15);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Hello"));
        assert!(chunks[0].text.contains("short content"));
    }

    #[test]
    fn test_smart_chunk_splits_large_content() {
        // Build content larger than 512 tokens (~2048 chars)
        let mut content = String::new();
        content.push_str("# Introduction\n\n");
        for i in 0..30 {
            content.push_str(&format!(
                "## Section {}\nThis is paragraph {} with enough text to take up space. \
                 We need each section to have meaningful content so the chunker has \
                 good break points to choose from.\n\n",
                i, i
            ));
        }

        let chunks = smart_chunk(&content, 512, 15);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks for large content, got {}",
            chunks.len()
        );

        // Each chunk should have a snippet
        for c in &chunks {
            assert!(!c.snippet.is_empty());
        }
    }

    #[test]
    fn test_smart_chunk_empty() {
        let chunks = smart_chunk("", 512, 15);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_smart_chunk_whitespace_only() {
        let chunks = smart_chunk("   \n\n  \n", 512, 15);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_code_fence_protection() {
        // Content with a code block that should NOT be split
        let mut content = String::new();
        content.push_str("# Before Code\nSome intro text.\n\n");
        content.push_str("```python\n");
        for i in 0..50 {
            content.push_str(&format!("x_{} = compute_value({})\n", i, i));
        }
        content.push_str("```\n\n");
        content.push_str("# After Code\nSome conclusion.\n");

        let bps = find_break_points(&content);
        // Verify no break points inside the code fence are eligible (not inside_code_fence)
        let fence_start_line = 3; // ```python
        let fence_end_line = fence_start_line + 51; // ``` closing

        for bp in &bps {
            if bp.line_number > fence_start_line && bp.line_number < fence_end_line {
                // These should either not exist or be marked inside_code_fence
                assert!(
                    bp.inside_code_fence || bp.score <= 1,
                    "Break point at line {} (score {}) should be inside code fence or excluded",
                    bp.line_number,
                    bp.score
                );
            }
        }
    }

    // ── Existing tests (updated for smart chunking) ──────────────────────

    #[test]
    fn test_chunk_by_headings() {
        let md = "## A\nContent A\n\n## B\nContent B\n";
        let parsed = chunk_markdown(md);
        // Smart chunking with small content should keep it as one chunk
        // since total tokens < 512
        assert!(parsed.chunks.len() >= 1);
        // The content should all be present
        let all_text: String = parsed.chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>().join(" ");
        assert!(all_text.contains("Content A"));
        assert!(all_text.contains("Content B"));
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
        assert!(!parsed.chunks[0].text.contains("---\ntags"));
        assert!(parsed.chunks[0].text.contains("Body"));
    }

    #[test]
    fn test_snippet_truncation() {
        let long_text = "a".repeat(300);
        let md = format!("## Heading\n{long_text}");
        let parsed = chunk_markdown(&md);
        assert!(!parsed.chunks.is_empty());
        // At least one chunk should have a truncated snippet
        let has_truncated = parsed.chunks.iter().any(|c| c.snippet.ends_with("..."));
        assert!(has_truncated, "Expected at least one snippet to be truncated");
        // Verify truncation length
        for c in &parsed.chunks {
            if c.snippet.ends_with("...") {
                assert_eq!(c.snippet.len(), 203);
            }
        }
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
            .map(|i| {
                format!(
                    "This is sentence number {} with several words to pad it out.",
                    i
                )
            })
            .collect();
        let long_text = sentences.join(" ");
        let word_count = long_text.split_whitespace().count();
        assert!(
            word_count > 512,
            "Test text must exceed 512 tokens (words); got {word_count}"
        );

        let chunk = Chunk {
            heading: Some("## Long Section".to_string()),
            text: format!("## Long Section\n{long_text}"),
            snippet: make_snippet(&format!("## Long Section\n{long_text}")),
        };

        let token_fn = |s: &str| s.split_whitespace().count();
        let result = split_oversized_chunks(vec![chunk], &token_fn, 512, 50);

        assert!(
            result.len() >= 2,
            "Expected at least 2 sub-chunks, got {}",
            result.len()
        );
        // First chunk keeps original heading
        assert_eq!(result[0].heading.as_deref(), Some("## Long Section"));
        // Subsequent chunks get (cont.)
        assert_eq!(
            result[1].heading.as_deref(),
            Some("## Long Section (cont.)")
        );
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

    #[test]
    fn test_extract_heading() {
        assert_eq!(
            extract_heading("# Title\nBody text"),
            Some("# Title".to_string())
        );
        assert_eq!(
            extract_heading("## Sub\nBody"),
            Some("## Sub".to_string())
        );
        assert_eq!(extract_heading("No heading here"), None);
        assert_eq!(
            extract_heading("Some text\n### Deep heading\nMore"),
            Some("### Deep heading".to_string())
        );
    }

    #[test]
    fn test_thematic_break_detection() {
        assert!(is_thematic_break("---"));
        assert!(is_thematic_break("***"));
        assert!(is_thematic_break("___"));
        assert!(is_thematic_break("- - -"));
        assert!(is_thematic_break("----"));
        assert!(!is_thematic_break("--"));
        assert!(!is_thematic_break("abc"));
    }

    #[test]
    fn test_list_item_detection() {
        assert!(is_list_item("- item"));
        assert!(is_list_item("* item"));
        assert!(is_list_item("1. item"));
        assert!(is_list_item("10. item"));
        assert!(!is_list_item("plain text"));
        assert!(!is_list_item(""));
    }
}
