#[derive(Debug, Clone)]
pub struct HeadingInfo {
    pub line: usize,
    pub level: u8,
    pub text: String,
}

pub fn parse_headings(content: &str) -> Vec<HeadingInfo> {
    let mut headings = Vec::new();
    let mut in_code_block = false;
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            let hashes = rest.chars().take_while(|&c| c == '#').count();
            let level = 1 + hashes as u8;
            let after_hashes = &rest[hashes..];
            if level <= 6 && (after_hashes.is_empty() || after_hashes.starts_with(' ')) {
                let text = after_hashes.trim().trim_end_matches('#').trim();
                headings.push(HeadingInfo {
                    line: i,
                    level,
                    text: text.to_string(),
                });
            }
        }
    }
    headings
}

#[derive(Debug, Clone)]
pub struct Section {
    pub heading: HeadingInfo,
    pub body_start: usize,
    pub body_end: usize,
    pub content: String,
}

pub fn find_section(content: &str, heading_text: &str) -> Option<Section> {
    let headings = parse_headings(content);
    let target = heading_text.trim().to_lowercase();
    let lines: Vec<&str> = content.lines().collect();

    let idx = headings.iter().position(|h| h.text.to_lowercase() == target)?;
    let h = &headings[idx];
    let body_start = h.line + 1;
    let body_end = headings[idx + 1..]
        .iter()
        .find(|next| next.level <= h.level)
        .map(|next| next.line)
        .unwrap_or(lines.len());

    let content_str = lines[body_start..body_end].join("\n");
    Some(Section {
        heading: HeadingInfo {
            line: h.line,
            level: h.level,
            text: h.text.clone(),
        },
        body_start,
        body_end,
        content: content_str,
    })
}

pub fn split_frontmatter(content: &str) -> (Option<String>, String) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.first().map(|l| l.trim()) != Some("---") {
        return (None, content.to_string());
    }
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim() == "---" {
            let fm = lines[1..i].join("\n");
            let body = lines[i + 1..].join("\n");
            return (Some(fm), body);
        }
    }
    (None, content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_headings_basic() {
        let content = "# Title\n\nSome text\n\n## Section A\n\nContent\n\n## Section B\n";
        let headings = parse_headings(content);
        assert_eq!(headings.len(), 3);
        assert_eq!(headings[0].level, 1);
        assert_eq!(headings[0].text, "Title");
        assert_eq!(headings[1].level, 2);
        assert_eq!(headings[1].text, "Section A");
    }

    #[test]
    fn test_parse_headings_ignores_code_blocks() {
        let content = "# Real\n\n```\n# Not a heading\n```\n\n## Also Real\n";
        let headings = parse_headings(content);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].text, "Real");
        assert_eq!(headings[1].text, "Also Real");
    }

    #[test]
    fn test_parse_headings_strips_trailing_hashes() {
        let content = "## Heading ##\n";
        let headings = parse_headings(content);
        assert_eq!(headings[0].text, "Heading");
    }

    #[test]
    fn test_find_section_basic() {
        let content = "# Title\n\n## Interactions\n\nEntry 1\nEntry 2\n\n## Links\n\nSome links\n";
        let section = find_section(content, "Interactions").unwrap();
        assert_eq!(section.heading.text, "Interactions");
        assert!(section.content.contains("Entry 1"));
        assert!(!section.content.contains("Some links"));
    }

    #[test]
    fn test_find_section_case_insensitive() {
        let content = "## My Section\n\nContent\n";
        assert!(find_section(content, "my section").is_some());
    }

    #[test]
    fn test_find_section_with_subsections() {
        let content = "# Title\n\n## Interactions\n\nEntry\n\n### Sub-detail\n\nMore\n\n## Links\n\nSome links\n";
        let section = find_section(content, "Interactions").unwrap();
        assert!(section.content.contains("Entry"));
        assert!(section.content.contains("Sub-detail"));
        assert!(!section.content.contains("Some links"));
    }

    #[test]
    fn test_find_section_not_found() {
        let content = "## Existing\n\nContent\n";
        assert!(find_section(content, "Missing").is_none());
    }

    #[test]
    fn test_split_frontmatter_valid() {
        let content = "---\ntitle: Test\ntags:\n  - foo\n---\n\n# Body\n";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_some());
        assert!(fm.unwrap().contains("title: Test"));
        assert!(body.contains("# Body"));
    }

    #[test]
    fn test_split_frontmatter_none() {
        let content = "# No frontmatter\n\nJust content\n";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_none());
        assert!(body.contains("No frontmatter"));
    }

    #[test]
    fn test_parse_headings_ignores_inline_tags() {
        let content = "# Title\n\nSome text with #tag and #another-tag\n\n## Real Section\n";
        let headings = parse_headings(content);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].text, "Title");
        assert_eq!(headings[1].text, "Real Section");
    }
}
