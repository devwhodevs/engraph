use std::collections::HashSet;

/// Extract unique wikilink targets from text.
/// Handles [[Target]], [[Target|Display]], [[Target#Heading]].
/// Skips embeds (![[...]]).
pub fn extract_wikilink_targets(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    let mut i = 0;

    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            // Check for embed prefix (! before [[)
            let is_embed = i > 0 && bytes[i - 1] == b'!';
            if let Some(rest) = text.get(i + 2..)
                && let Some(close) = rest.find("]]")
            {
                let inner = &rest[..close];
                if !is_embed && !inner.is_empty() && !inner.contains('\n') {
                    // Strip heading: [[Note#Section]] → "Note"
                    let target = inner.split('#').next().unwrap_or(inner);
                    // Strip display: [[Note|Display]] → "Note"
                    let target = target.split('|').next().unwrap_or(target);
                    let target = target.trim().to_string();
                    if !target.is_empty() && seen.insert(target.clone()) {
                        targets.push(target);
                    }
                }
                i += 2 + close + 2;
                continue;
            }
        }
        i += 1;
    }
    targets
}

/// Extract query terms for relevance filtering.
/// Splits on whitespace, lowercases, drops terms shorter than 3 chars.
pub fn extract_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 3)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_wikilink_targets() {
        let text =
            "See [[Note One]] and [[Note Two|display]] for details. Also [[Note One]] again.";
        let targets = extract_wikilink_targets(text);
        assert!(targets.contains(&"Note One".to_string()));
        assert!(targets.contains(&"Note Two".to_string()));
        assert_eq!(targets.len(), 2); // deduplicated
    }

    #[test]
    fn test_extract_wikilinks_with_headings() {
        let text = "Link to [[Note#Section]] here.";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Note"]);
    }

    #[test]
    fn test_extract_wikilinks_empty() {
        assert!(extract_wikilink_targets("no links here").is_empty());
        assert!(extract_wikilink_targets("").is_empty());
    }

    #[test]
    fn test_extract_wikilinks_skip_embeds() {
        let text = "![[embedded image.png]] and [[real link]]";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["real link"]);
    }

    #[test]
    fn test_extract_wikilinks_heading_and_display() {
        let text = "[[Note#Section|Custom Display]]";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Note"]); // strip both heading and display
    }

    #[test]
    fn test_extract_query_terms() {
        let terms = extract_query_terms("BRE-2579 delivery date");
        assert_eq!(terms, vec!["bre-2579", "delivery", "date"]);
    }

    #[test]
    fn test_extract_query_terms_short_words_dropped() {
        let terms = extract_query_terms("a is the big query");
        assert_eq!(terms, vec!["the", "big", "query"]);
    }
}
