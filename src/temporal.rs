//! Temporal date extraction, heuristic range parsing, and proximity scoring.
//!
//! Provides utilities for:
//! - Extracting authoring dates from note frontmatter or filenames
//! - Parsing natural-language temporal expressions into date ranges
//! - Scoring search results by temporal proximity to a target range

use time::macros::format_description;
use time::{Date, Duration, Month, OffsetDateTime, Weekday};

// ── Date extraction ─────────────────────────────────────────────

/// Extract a note's authoring date as a Unix timestamp (start of day UTC).
///
/// Priority: frontmatter `date: YYYY-MM-DD` → filename `YYYY-MM-DD` pattern → None.
pub fn extract_note_date(frontmatter: &str, filename: &str) -> Option<i64> {
    // Try frontmatter first
    if let Some(ts) = extract_date_from_frontmatter(frontmatter) {
        return Some(ts);
    }
    // Fall back to filename pattern
    extract_date_from_filename(filename)
}

/// Parse `date: YYYY-MM-DD` from YAML frontmatter.
fn extract_date_from_frontmatter(frontmatter: &str) -> Option<i64> {
    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("date:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'');
            // Take only the first 10 chars in case of datetime like 2026-03-25T10:00:00
            let date_str = if value.len() >= 10 {
                &value[..10]
            } else {
                value
            };
            if let Some(ts) = parse_iso_date(date_str) {
                return Some(ts);
            }
        }
    }
    None
}

/// Extract YYYY-MM-DD pattern from a filename.
fn extract_date_from_filename(filename: &str) -> Option<i64> {
    // Look for YYYY-MM-DD pattern anywhere in the filename.
    // Only check ASCII char boundaries to avoid panics on multi-byte UTF-8 filenames.
    let bytes = filename.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for i in 0..=bytes.len() - 10 {
        // Skip non-ASCII-start positions to avoid slicing mid-character
        if !filename.is_char_boundary(i) || !filename.is_char_boundary(i + 10) {
            continue;
        }
        let candidate = &filename[i..i + 10];
        if candidate.as_bytes()[4] == b'-'
            && candidate.as_bytes()[7] == b'-'
            && let Some(ts) = parse_iso_date(candidate)
        {
            return Some(ts);
        }
    }
    None
}

/// Parse an ISO date string (YYYY-MM-DD) into a Unix timestamp at start of day UTC.
fn parse_iso_date(s: &str) -> Option<i64> {
    let fmt = format_description!("[year]-[month]-[day]");
    let date = Date::parse(s, &fmt).ok()?;
    Some(date.midnight().assume_utc().unix_timestamp())
}

// ── Temporal scoring ────────────────────────────────────────────

/// Score a file by temporal proximity to a date range.
///
/// - Inside range: 1.0
/// - Outside range: `1.0 / (1.0 + days_away * 0.1)` (smooth decay)
///
/// All timestamps are Unix seconds (UTC).
pub fn temporal_score(note_date: i64, range_start: i64, range_end: i64) -> f64 {
    if note_date >= range_start && note_date <= range_end {
        return 1.0;
    }

    let seconds_away = if note_date < range_start {
        (range_start - note_date) as f64
    } else {
        (note_date - range_end) as f64
    };

    let days_away = seconds_away / 86400.0;
    1.0 / (1.0 + days_away * 0.1)
}

// ── Heuristic date range parsing ────────────────────────────────

/// Scan a natural-language query for temporal keywords and return a date range
/// as `(start_timestamp, end_timestamp)` in Unix seconds UTC.
///
/// Supported patterns:
/// - "today" / "this morning" → today 00:00–23:59:59
/// - "yesterday" → yesterday 00:00–23:59:59
/// - "last week" → previous Monday–Sunday 23:59:59
/// - "this week" → current Monday–Sunday 23:59:59
/// - "last month" → previous month 1st–last day 23:59:59
/// - "this month" → current month 1st–last day 23:59:59
/// - "recent" / "recently" → last 7 days
/// - Month names with optional year: "March 2026", "march"
/// - ISO dates: "2026-03-25" → that day
/// - "January to March" → Jan 1–Mar 31 (current year)
/// - No match → None
pub fn parse_date_range_heuristic(query: &str) -> Option<(i64, i64)> {
    let now = OffsetDateTime::now_utc();
    parse_date_range_heuristic_with_ref(query, now)
}

/// Internal implementation with injectable reference time for testing.
fn parse_date_range_heuristic_with_ref(query: &str, now: OffsetDateTime) -> Option<(i64, i64)> {
    let lower = query.to_lowercase();
    let today = now.date();

    // "today" or "this morning"
    if lower.contains("today") || lower.contains("this morning") {
        return Some(day_range(today));
    }

    // "yesterday"
    if lower.contains("yesterday") {
        let yesterday = today.previous_day()?;
        return Some(day_range(yesterday));
    }

    // "last week" — previous Monday to Sunday
    if lower.contains("last week") {
        let current_monday = monday_of_week(today);
        let prev_monday = current_monday.checked_sub(Duration::weeks(1))?;
        let prev_sunday = prev_monday.checked_add(Duration::days(6))?;
        return Some((start_of_day(prev_monday), end_of_day(prev_sunday)));
    }

    // "this week" — current Monday to Sunday
    if lower.contains("this week") {
        let current_monday = monday_of_week(today);
        let current_sunday = current_monday.checked_add(Duration::days(6))?;
        return Some((start_of_day(current_monday), end_of_day(current_sunday)));
    }

    // "last month" — previous month 1st to last day
    if lower.contains("last month") {
        let (prev_year, prev_month) = prev_month(today.year(), today.month());
        return month_range(prev_year, prev_month);
    }

    // "this month" — current month 1st to last day
    if lower.contains("this month") {
        return month_range(today.year(), today.month());
    }

    // "recent" / "recently" — last 7 days
    if lower.contains("recent") {
        let week_ago = today.checked_sub(Duration::days(6))?;
        return Some((start_of_day(week_ago), end_of_day(today)));
    }

    // ISO date: "2026-03-25"
    if let Some(ts) = find_iso_date_in_query(&lower) {
        return Some(ts);
    }

    // "January to March" / "jan to mar" — month range (current year)
    if let Some(range) = parse_month_to_month(&lower, today.year()) {
        return Some(range);
    }

    // "March 2026" or just "march" — specific month with optional year
    if let Some(range) = parse_month_with_optional_year(&lower, today.year()) {
        return Some(range);
    }

    None
}

// ── JSON date range parsing ─────────────────────────────────────

/// Parse a date range from LLM orchestrator JSON.
///
/// Expected format: `{"date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}`.
/// The end date gets +86399 seconds (end of day).
pub fn parse_date_range_from_json(json: &serde_json::Value) -> Option<(i64, i64)> {
    let range = json.get("date_range")?;
    let start_str = range.get("start")?.as_str()?;
    let end_str = range.get("end")?.as_str()?;

    let start_ts = parse_iso_date(start_str)?;
    let end_ts = parse_iso_date(end_str)? + 86399; // end of day

    Some((start_ts, end_ts))
}

// ── Helpers ─────────────────────────────────────────────────────

/// Return (start_of_day, end_of_day) timestamps for a given date.
fn day_range(date: Date) -> (i64, i64) {
    (start_of_day(date), end_of_day(date))
}

/// Unix timestamp for 00:00:00 UTC of the given date.
fn start_of_day(date: Date) -> i64 {
    date.midnight().assume_utc().unix_timestamp()
}

/// Unix timestamp for 23:59:59 UTC of the given date.
fn end_of_day(date: Date) -> i64 {
    date.with_hms(23, 59, 59)
        .expect("valid HMS")
        .assume_utc()
        .unix_timestamp()
}

/// Find the Monday of the ISO week containing `date`.
fn monday_of_week(date: Date) -> Date {
    let wd = date.weekday();
    let days_since_monday = match wd {
        Weekday::Monday => 0,
        Weekday::Tuesday => 1,
        Weekday::Wednesday => 2,
        Weekday::Thursday => 3,
        Weekday::Friday => 4,
        Weekday::Saturday => 5,
        Weekday::Sunday => 6,
    };
    date.checked_sub(Duration::days(days_since_monday))
        .expect("valid date subtraction")
}

/// Return the previous month and its year.
fn prev_month(year: i32, month: Month) -> (i32, Month) {
    let m = month as u8;
    if m == 1 {
        (year - 1, Month::December)
    } else {
        (year, Month::try_from(m - 1).expect("valid month"))
    }
}

/// Return (start_of_day of 1st, end_of_day of last day) for a given year/month.
fn month_range(year: i32, month: Month) -> Option<(i64, i64)> {
    let first = Date::from_calendar_date(year, month, 1).ok()?;
    let last = last_day_of_month(year, month)?;
    Some((start_of_day(first), end_of_day(last)))
}

/// Get the last day of a given month.
fn last_day_of_month(year: i32, month: Month) -> Option<Date> {
    let m = month as u8;
    if m == 12 {
        Date::from_calendar_date(year + 1, Month::January, 1)
            .ok()?
            .previous_day()
    } else {
        let next_month = Month::try_from(m + 1).ok()?;
        Date::from_calendar_date(year, next_month, 1)
            .ok()?
            .previous_day()
    }
}

/// Find an ISO date (YYYY-MM-DD) in the query and return a day range.
fn find_iso_date_in_query(query: &str) -> Option<(i64, i64)> {
    let bytes = query.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for i in 0..=bytes.len() - 10 {
        let candidate = &query[i..i + 10];
        if candidate.as_bytes()[4] == b'-'
            && candidate.as_bytes()[7] == b'-'
            && let Some(ts) = parse_iso_date(candidate)
        {
            let fmt = format_description!("[year]-[month]-[day]");
            if let Ok(date) = Date::parse(candidate, &fmt) {
                return Some(day_range(date));
            }
            // Fallback: use the parsed timestamp
            return Some((ts, ts + 86399));
        }
    }
    None
}

/// Parse "January to March" style range.
fn parse_month_to_month(query: &str, current_year: i32) -> Option<(i64, i64)> {
    // Look for "MONTH to MONTH" or "MONTH - MONTH"
    let separators = [" to ", " - ", " through "];
    for sep in &separators {
        if let Some(idx) = query.find(sep) {
            let before = query[..idx].trim();
            let after = query[idx + sep.len()..].trim();

            // Extract month name (last word before separator, first word after)
            let start_month = parse_month_name(last_word(before))?;
            let end_month = parse_month_name(first_word(after))?;

            let start = Date::from_calendar_date(current_year, start_month, 1).ok()?;
            let end = last_day_of_month(current_year, end_month)?;
            return Some((start_of_day(start), end_of_day(end)));
        }
    }
    None
}

/// Parse "March 2026" or bare "march" into a month range.
fn parse_month_with_optional_year(query: &str, current_year: i32) -> Option<(i64, i64)> {
    let words: Vec<&str> = query.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if let Some(month) = parse_month_name(word) {
            // Check if next word is a 4-digit year
            let year = if i + 1 < words.len() {
                words[i + 1]
                    .parse::<i32>()
                    .ok()
                    .filter(|&y| (1900..=2100).contains(&y))
                    .unwrap_or(current_year)
            } else {
                current_year
            };
            return month_range(year, month);
        }
    }
    None
}

/// Parse a month name (full or 3-letter abbreviation) into a `time::Month`.
fn parse_month_name(s: &str) -> Option<Month> {
    match s.to_lowercase().as_str() {
        "jan" | "january" => Some(Month::January),
        "feb" | "february" => Some(Month::February),
        "mar" | "march" => Some(Month::March),
        "apr" | "april" => Some(Month::April),
        "may" => Some(Month::May),
        "jun" | "june" => Some(Month::June),
        "jul" | "july" => Some(Month::July),
        "aug" | "august" => Some(Month::August),
        "sep" | "september" => Some(Month::September),
        "oct" | "october" => Some(Month::October),
        "nov" | "november" => Some(Month::November),
        "dec" | "december" => Some(Month::December),
        _ => None,
    }
}

fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

fn last_word(s: &str) -> &str {
    s.split_whitespace().last().unwrap_or("")
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    // ── extract_note_date ───────────────────────────────────────

    #[test]
    fn extract_date_from_frontmatter_yaml() {
        let fm = "---\ntitle: Test\ndate: 2026-03-25\ntags: [work]\n---";
        let ts = extract_note_date(fm, "random-note.md").unwrap();
        assert_eq!(ts, date_ts(2026, 3, 25));
    }

    #[test]
    fn extract_date_from_frontmatter_quoted() {
        let fm = "---\ndate: \"2025-12-31\"\n---";
        let ts = extract_note_date(fm, "note.md").unwrap();
        assert_eq!(ts, date_ts(2025, 12, 31));
    }

    #[test]
    fn extract_date_from_frontmatter_datetime() {
        // date field with full datetime should still extract the date part
        let fm = "---\ndate: 2026-01-15T10:30:00\n---";
        let ts = extract_note_date(fm, "note.md").unwrap();
        assert_eq!(ts, date_ts(2026, 1, 15));
    }

    #[test]
    fn extract_date_from_filename_pattern() {
        let fm = "---\ntitle: Daily Note\n---";
        let ts = extract_note_date(fm, "2026-03-25.md").unwrap();
        assert_eq!(ts, date_ts(2026, 3, 25));
    }

    #[test]
    fn extract_date_from_filename_with_prefix() {
        let fm = "";
        let ts = extract_note_date(fm, "daily-2026-03-25-standup.md").unwrap();
        assert_eq!(ts, date_ts(2026, 3, 25));
    }

    #[test]
    fn extract_date_frontmatter_takes_priority_over_filename() {
        let fm = "---\ndate: 2026-01-01\n---";
        let ts = extract_note_date(fm, "2026-12-31.md").unwrap();
        // Frontmatter date should win
        assert_eq!(ts, date_ts(2026, 1, 1));
    }

    #[test]
    fn extract_date_no_date_returns_none() {
        let fm = "---\ntitle: No Date Here\ntags: [misc]\n---";
        assert!(extract_note_date(fm, "random-note.md").is_none());
    }

    #[test]
    fn extract_date_empty_inputs() {
        assert!(extract_note_date("", "").is_none());
    }

    // ── temporal_score ──────────────────────────────────────────

    #[test]
    fn score_inside_range() {
        let start = date_ts(2026, 3, 20);
        let end = date_ts(2026, 3, 26) + 86399;
        let note = date_ts(2026, 3, 23);
        assert_eq!(temporal_score(note, start, end), 1.0);
    }

    #[test]
    fn score_at_range_boundary() {
        let start = date_ts(2026, 3, 20);
        let end = date_ts(2026, 3, 26) + 86399;
        // Exactly at start
        assert_eq!(temporal_score(start, start, end), 1.0);
        // Exactly at end
        assert_eq!(temporal_score(end, start, end), 1.0);
    }

    #[test]
    fn score_one_day_before_range() {
        let start = date_ts(2026, 3, 20);
        let end = date_ts(2026, 3, 26) + 86399;
        let note = date_ts(2026, 3, 19); // 1 day before start
        let score = temporal_score(note, start, end);
        // 1.0 / (1.0 + 1.0 * 0.1) = 1.0 / 1.1 ≈ 0.909
        let expected = 1.0 / (1.0 + 1.0 * 0.1);
        assert!((score - expected).abs() < 1e-10);
    }

    #[test]
    fn score_one_week_outside_range() {
        let start = date_ts(2026, 3, 20);
        let end = date_ts(2026, 3, 26) + 86399;
        let note = date_ts(2026, 4, 2); // 7 days after end (note is start of Apr 2, end is end of Mar 26)
        let score = temporal_score(note, start, end);
        // days_away ≈ (note_ts - end_ts) / 86400
        let days_away = (note - end) as f64 / 86400.0;
        let expected = 1.0 / (1.0 + days_away * 0.1);
        assert!((score - expected).abs() < 1e-10);
        // Should be significantly less than 1.0
        assert!(score < 0.7);
    }

    #[test]
    fn score_far_outside_range() {
        let start = date_ts(2026, 3, 20);
        let end = date_ts(2026, 3, 26) + 86399;
        let note = date_ts(2025, 1, 1); // ~15 months before
        let score = temporal_score(note, start, end);
        // Should be very low but still positive
        assert!(score > 0.0);
        assert!(score < 0.1);
    }

    // ── parse_date_range_heuristic ──────────────────────────────

    // Reference time: 2026-03-26 14:30:00 UTC (Thursday)
    fn ref_time() -> OffsetDateTime {
        datetime!(2026-03-26 14:30:00 UTC)
    }

    #[test]
    fn heuristic_today() {
        let (start, end) =
            parse_date_range_heuristic_with_ref("what happened today", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 26));
        assert_eq!(end, date_ts(2026, 3, 26) + 86399);
    }

    #[test]
    fn heuristic_this_morning() {
        let (start, end) =
            parse_date_range_heuristic_with_ref("notes from this morning", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 26));
        assert_eq!(end, date_ts(2026, 3, 26) + 86399);
    }

    #[test]
    fn heuristic_yesterday() {
        let (start, end) =
            parse_date_range_heuristic_with_ref("yesterday's standup", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 25));
        assert_eq!(end, date_ts(2026, 3, 25) + 86399);
    }

    #[test]
    fn heuristic_last_week() {
        // 2026-03-26 is Thursday. Last week = Mon Mar 16 – Sun Mar 22
        let (start, end) =
            parse_date_range_heuristic_with_ref("what did I do last week", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 16));
        assert_eq!(end, date_ts(2026, 3, 22) + 86399);
    }

    #[test]
    fn heuristic_this_week() {
        // 2026-03-26 is Thursday. This week = Mon Mar 23 – Sun Mar 29
        let (start, end) =
            parse_date_range_heuristic_with_ref("this week's tasks", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 23));
        assert_eq!(end, date_ts(2026, 3, 29) + 86399);
    }

    #[test]
    fn heuristic_last_month() {
        // Current: March 2026. Last month = Feb 1 – Feb 28, 2026
        let (start, end) =
            parse_date_range_heuristic_with_ref("last month summary", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 2, 1));
        assert_eq!(end, date_ts(2026, 2, 28) + 86399);
    }

    #[test]
    fn heuristic_this_month() {
        let (start, end) = parse_date_range_heuristic_with_ref("this month", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 1));
        assert_eq!(end, date_ts(2026, 3, 31) + 86399);
    }

    #[test]
    fn heuristic_recent() {
        // "recent" = last 7 days: Mar 20 – Mar 26
        let (start, end) = parse_date_range_heuristic_with_ref("recent notes", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 20));
        assert_eq!(end, date_ts(2026, 3, 26) + 86399);
    }

    #[test]
    fn heuristic_recently() {
        let result = parse_date_range_heuristic_with_ref("what I recently worked on", ref_time());
        assert!(result.is_some());
    }

    #[test]
    fn heuristic_iso_date() {
        let (start, end) =
            parse_date_range_heuristic_with_ref("notes from 2026-03-25", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 25));
        assert_eq!(end, date_ts(2026, 3, 25) + 86399);
    }

    #[test]
    fn heuristic_month_name_with_year() {
        let (start, end) =
            parse_date_range_heuristic_with_ref("notes from March 2026", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 3, 1));
        assert_eq!(end, date_ts(2026, 3, 31) + 86399);
    }

    #[test]
    fn heuristic_month_name_bare() {
        // Bare month name uses current year
        let (start, end) =
            parse_date_range_heuristic_with_ref("february notes", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 2, 1));
        assert_eq!(end, date_ts(2026, 2, 28) + 86399);
    }

    #[test]
    fn heuristic_month_to_month() {
        let (start, end) =
            parse_date_range_heuristic_with_ref("january to march", ref_time()).unwrap();
        assert_eq!(start, date_ts(2026, 1, 1));
        assert_eq!(end, date_ts(2026, 3, 31) + 86399);
    }

    #[test]
    fn heuristic_no_temporal_match() {
        assert!(parse_date_range_heuristic_with_ref("how does RRF work", ref_time()).is_none());
    }

    #[test]
    fn heuristic_no_temporal_match_empty() {
        assert!(parse_date_range_heuristic_with_ref("", ref_time()).is_none());
    }

    // ── parse_date_range_from_json ──────────────────────────────

    #[test]
    fn json_valid_range() {
        let json: serde_json::Value = serde_json::json!({
            "date_range": {
                "start": "2026-03-19",
                "end": "2026-03-25"
            }
        });
        let (start, end) = parse_date_range_from_json(&json).unwrap();
        assert_eq!(start, date_ts(2026, 3, 19));
        assert_eq!(end, date_ts(2026, 3, 25) + 86399);
    }

    #[test]
    fn json_missing_date_range() {
        let json: serde_json::Value = serde_json::json!({"query": "test"});
        assert!(parse_date_range_from_json(&json).is_none());
    }

    #[test]
    fn json_missing_start() {
        let json: serde_json::Value = serde_json::json!({
            "date_range": {
                "end": "2026-03-25"
            }
        });
        assert!(parse_date_range_from_json(&json).is_none());
    }

    #[test]
    fn json_missing_end() {
        let json: serde_json::Value = serde_json::json!({
            "date_range": {
                "start": "2026-03-19"
            }
        });
        assert!(parse_date_range_from_json(&json).is_none());
    }

    #[test]
    fn json_invalid_date_format() {
        let json: serde_json::Value = serde_json::json!({
            "date_range": {
                "start": "not-a-date",
                "end": "2026-03-25"
            }
        });
        assert!(parse_date_range_from_json(&json).is_none());
    }

    // ── Test helpers ────────────────────────────────────────────

    /// Helper to get Unix timestamp for start of day UTC.
    fn date_ts(year: i32, month: u8, day: u8) -> i64 {
        Date::from_calendar_date(year, Month::try_from(month).unwrap(), day)
            .unwrap()
            .midnight()
            .assume_utc()
            .unix_timestamp()
    }
}
