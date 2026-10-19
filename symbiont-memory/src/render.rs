// SPDX-License-Identifier: MPL-2.0
//! Turning the memory into prompt text, within a token budget.
//!
//! Rendering is a pure function of the memory state and the request: the same
//! state renders byte for byte the same text. That matters twice. A server
//! with a prefix cache reuses a prompt prefix only up to the first byte that
//! differs, and a replay of a recorded run must see what the live run saw.

use std::fmt::Write as _;

use getset::CopyGetters;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    Archive,
    Entry,
    EntrySeq,
};

pub(crate) const EXPECT_WRITE: &str = "writing to a String cannot fail";

/// A rough token count of `text`: one token per four bytes, rounded up.
///
/// A real tokenizer depends on the model. The budget only has to keep the
/// memory from crowding out the rest of the prompt, and the four byte rule
/// overestimates code and English slightly, which errs on the safe side.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// How much of the archive goes into a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct RenderConfig {
    /// The best entries shown with their report.
    elites: usize,
    /// How many of the [`Self::elites`], best first, also show their source.
    /// Sources are the largest part of the memory; one full program is
    /// usually enough for the model to build on.
    elites_with_source: usize,
    /// The newest entries shown with their report and no source, after the
    /// elites. They tell the lane what was just tried, good or bad.
    recent: usize,
    /// The longest source shown, in bytes. A longer one is cut at a line
    /// boundary and marked as cut.
    max_source_bytes: usize,
    /// The token budget of the archive section, see [`estimate_tokens`].
    archive_tokens: usize,
}

impl RenderConfig {
    /// A configuration with the given limits.
    #[must_use]
    pub fn new(
        elites: usize,
        elites_with_source: usize,
        recent: usize,
        max_source_bytes: usize,
        archive_tokens: usize,
    ) -> Self {
        Self {
            elites,
            elites_with_source: elites_with_source.min(elites),
            recent,
            max_source_bytes,
            archive_tokens,
        }
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self::new(3, 1, 5, 6_000, 3_000)
    }
}

/// Render the best and the newest entries of `archive`, leaving out the
/// `exclude`d ids (the host shows a lane's parents itself).
///
/// Blocks are added in priority order (elites, then recent attempts) for as
/// long as they fit into [`RenderConfig::archive_tokens`]. An elite whose
/// source does not fit is shown without it. Returns the text and the entries
/// it shows.
pub(crate) fn render_archive<I: PartialEq + Clone>(
    archive: &Archive<I>,
    exclude: &[I],
    config: &RenderConfig,
) -> (String, Vec<EntrySeq>) {
    let mut text = String::new();
    let mut shown = Vec::with_capacity(config.elites + config.recent);
    let mut used = 0;
    let budget = config.archive_tokens;
    let visible = |e: &&Entry<I>| !exclude.contains(e.id());

    let elites = Vec::from_iter(
        archive
            .ranked()
            .into_iter()
            .filter(visible)
            .take(config.elites),
    );
    let mut elite_blocks = Vec::with_capacity(elites.len());
    for (rank, entry) in elites.iter().enumerate() {
        let with_source = (rank < config.elites_with_source)
            .then(|| elite_block(entry, Some(config.max_source_bytes)));
        let block = match with_source {
            Some(block) if used + estimate_tokens(&block) <= budget => block,
            _ => elite_block(entry, None),
        };
        if used + estimate_tokens(&block) > budget {
            break;
        }
        used += estimate_tokens(&block);
        shown.push(entry.seq());
        elite_blocks.push(block);
    }

    let recent = Vec::from_iter(
        archive
            .recent()
            .filter(visible)
            .filter(|e| !shown.contains(&e.seq()))
            .take(config.recent),
    );
    let mut recent_blocks = Vec::with_capacity(recent.len());
    for entry in recent {
        let block = recent_block(entry);
        if used + estimate_tokens(&block) > budget {
            break;
        }
        used += estimate_tokens(&block);
        shown.push(entry.seq());
        recent_blocks.push(block);
    }

    if !elite_blocks.is_empty() {
        text.push_str("## Best attempts so far\n\n");
        text.extend(elite_blocks);
    }
    if !recent_blocks.is_empty() {
        text.push_str("## Most recent attempts\n\n");
        text.extend(recent_blocks);
    }
    assert!(shown.len() <= config.elites + config.recent);
    if shown.is_empty() {
        assert!(text.is_empty());
    }
    (text, shown)
}

fn elite_block<I>(entry: &Entry<I>, source: Option<usize>) -> String {
    let mut block = String::new();
    writeln!(block, "### {} (score {})", entry.seq(), entry.score().get()).expect(EXPECT_WRITE);
    push_report(&mut block, entry.report());
    if let Some(max_bytes) = source {
        writeln!(
            block,
            "```rust\n{}\n```",
            truncate_lines(entry.source().trim_end(), max_bytes)
        )
        .expect(EXPECT_WRITE);
    }
    block.push('\n');
    block
}

fn recent_block<I>(entry: &Entry<I>) -> String {
    let mut block = String::new();
    writeln!(block, "### {} (score {})", entry.seq(), entry.score().get()).expect(EXPECT_WRITE);
    push_report(&mut block, entry.report());
    block.push('\n');
    block
}

fn push_report(block: &mut String, report: &str) {
    let report = report.trim();
    if !report.is_empty() {
        block.push_str(report);
        block.push('\n');
    }
}

/// `text` cut to at most `max_bytes` at a line boundary, with a marker line
/// if anything was cut.
pub(crate) fn truncate_lines(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut out = String::with_capacity(max_bytes + 32);
    for line in text.lines() {
        if out.len() + line.len() + 1 > max_bytes {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("// ... (cut)");
    assert!(out.len() <= max_bytes + "// ... (cut)".len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchiveConfig,
        Evaluated,
        Score,
    };

    fn archive(scores: &[f64]) -> Archive<u64> {
        let mut archive = Archive::new(ArchiveConfig::default());
        for (id, &score) in scores.iter().enumerate() {
            let id = u64::try_from(id).expect("small");
            let _ = archive.commit(
                Evaluated::builder()
                    .id(id)
                    .source(format!("fn f() -> u64 {{ {id} }}"))
                    .score(Score::new(score).expect("finite"))
                    .report(format!("report of {id}"))
                    .build(),
            );
        }
        archive
    }

    #[test]
    fn estimate_rounds_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn empty_archive_renders_nothing() {
        let (text, shown) = render_archive(&archive(&[]), &[], &RenderConfig::default());
        assert!(text.is_empty());
        assert!(shown.is_empty());
    }

    #[test]
    fn elites_then_recent_without_repeats() {
        let archive = archive(&[1.0, 5.0, 3.0, 0.0]);
        let config = RenderConfig::new(2, 1, 5, 6_000, 3_000);
        let (text, shown) = render_archive(&archive, &[], &config);
        // Elites #1 and #2, then the remaining entries newest first.
        assert_eq!(
            shown,
            vec![
                EntrySeq::from_raw(1),
                EntrySeq::from_raw(2),
                EntrySeq::from_raw(3),
                EntrySeq::from_raw(0)
            ]
        );
        assert!(text.starts_with("## Best attempts so far\n\n### #1 (score 5)"));
        assert!(text.contains("fn f() -> u64 { 1 }"));
        // Only the best elite shows its source.
        assert!(!text.contains("fn f() -> u64 { 2 }"));
        assert!(text.contains("## Most recent attempts"));
    }

    #[test]
    fn excluded_ids_are_not_shown() {
        let archive = archive(&[1.0, 5.0]);
        let (text, shown) = render_archive(&archive, &[1], &RenderConfig::default());
        assert_eq!(shown, vec![EntrySeq::from_raw(0)]);
        assert!(!text.contains("report of 1"));
    }

    #[test]
    fn budget_drops_source_then_blocks() {
        let archive = archive(&[1.0, 5.0, 3.0]);
        let tiny = RenderConfig::new(3, 3, 0, 6_000, 12);
        let (text, shown) = render_archive(&archive, &[], &tiny);
        assert_eq!(shown, vec![EntrySeq::from_raw(1)]);
        assert!(!text.contains("```rust"));
        let zero = RenderConfig::new(3, 3, 3, 6_000, 0);
        assert!(render_archive(&archive, &[], &zero).1.is_empty());
    }

    #[test]
    fn truncate_cuts_at_line_boundaries() {
        assert_eq!(truncate_lines("ab\ncd", 10), "ab\ncd");
        assert_eq!(truncate_lines("ab\ncd\nef", 6), "ab\ncd\n// ... (cut)");
        assert_eq!(truncate_lines("abcdef", 3), "// ... (cut)");
    }

    #[test]
    fn rendering_is_deterministic() {
        let archive = archive(&[2.0, 2.0, 1.0]);
        let a = render_archive(&archive, &[], &RenderConfig::default());
        let b = render_archive(&archive, &[], &RenderConfig::default());
        assert_eq!(a, b);
    }
}
