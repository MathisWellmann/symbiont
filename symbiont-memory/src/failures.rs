// SPDX-License-Identifier: MPL-2.0
//! Recurring failures, counted from evolution traces and runtime panics.
//!
//! Weak models repeat the same mistakes across lanes: a method the host type
//! does not have, a type mismatch in the same expression, the same index out
//! of bounds. Every rejected attempt is already in the lane's
//! [`EvolutionTrace`], so counting them needs no model call. A failure that
//! more than one lane hit becomes a "known pitfall" in later prompts.

use std::fmt::Write as _;

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};
use symbiont::{
    EvolutionTrace,
    LadderEvent,
};
use tracing::debug;

use crate::render::EXPECT_WRITE;

/// The longest signature kept, in bytes. Long enough for a rustc headline
/// with two type names, short enough that a pitfall list stays a list.
const MAX_SIGNATURE_BYTES: usize = 200;

/// How the pitfall list is kept and shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CopyGetters)]
// Missing fields take their defaults, so a config file names only what it
// changes. Deserializing skips the constructors: check with
// `MemoryConfig::validate`.
#[serde(default)]
#[getset(get_copy = "pub")]
pub struct FailureConfig {
    /// How many lanes must hit a failure before it is shown. One hit can be
    /// a fluke of one lane; two is a pattern.
    min_occurrences: u32,
    /// The most pitfalls shown.
    max_shown: usize,
    /// The most distinct failures tracked. The rarest, least recently seen
    /// one is dropped when a new one arrives.
    max_tracked: usize,
}

impl FailureConfig {
    /// A configuration with the given limits. `max_tracked` is raised to
    /// `max_shown` if it is smaller.
    #[must_use]
    pub fn new(min_occurrences: u32, max_shown: usize, max_tracked: usize) -> Self {
        Self {
            min_occurrences: min_occurrences.max(1),
            max_shown,
            max_tracked: max_tracked.max(max_shown).max(1),
        }
    }
}

impl Default for FailureConfig {
    fn default() -> Self {
        Self::new(2, 5, 256)
    }
}

/// One recurring failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Pitfall {
    /// The normalized headline, e.g. ``error[E0599]: no method named `foo`
    /// found for struct `Bar` in the current scope``. Numbers that stand alone
    /// are replaced by `N`, so `index 5` and `index 7` count as one failure.
    #[getset(get = "pub")]
    signature: String,
    /// The failure kind: a `kind` label of the symbiont self-heal ladder
    /// (`compile`, `parse`, ...) or `panic`.
    #[getset(get = "pub")]
    kind: String,
    /// How many lanes (or panics) hit it.
    #[getset(get_copy = "pub")]
    count: u32,
    /// The commit at which it was first seen.
    #[getset(get_copy = "pub")]
    first_seen: u64,
    /// The commit at which it was last seen.
    #[getset(get_copy = "pub")]
    last_seen: u64,
}

/// The recurring failures of all lanes so far.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failures {
    /// In first-seen order.
    pitfalls: Vec<Pitfall>,
    /// One tick per commit call: the clock of `first_seen` and `last_seen`.
    clock: u64,
    config: FailureConfig,
}

impl Failures {
    /// No failures yet.
    #[must_use]
    pub fn new(config: FailureConfig) -> Self {
        Self {
            pitfalls: Vec::new(),
            clock: 0,
            config,
        }
    }

    /// Count the rejected attempts of one lane.
    ///
    /// A failure counts once per lane, however often the lane repeated it:
    /// the question is how many lanes run into it, and a lane that echoes its
    /// own broken code ten times is one lane. Transient inference errors,
    /// context resets and the terminal verdict are not about the code and do
    /// not count.
    pub fn commit_trace(&mut self, trace: &EvolutionTrace) {
        self.commit_ladder(trace.attempts().iter().map(|attempt| attempt.ladder()));
    }

    /// [`Self::commit_trace`] on the ladder events of one lane.
    pub fn commit_ladder<'a>(&mut self, ladder: impl IntoIterator<Item = &'a LadderEvent>) {
        let mut seen = Vec::<(String, &str)>::new();
        for event in ladder {
            let LadderEvent::SelfHeal {
                kind, diagnostics, ..
            } = event
            else {
                continue;
            };
            for signature in signatures(kind, diagnostics) {
                if !seen.iter().any(|(s, _)| *s == signature) {
                    seen.push((signature, kind.as_str()));
                }
            }
        }
        self.clock += 1;
        for (signature, kind) in seen {
            self.count(signature, kind);
        }
    }

    /// Count a panic the host caught while it evaluated a candidate, e.g.
    /// from `RevisionFn::take_panic`.
    pub fn commit_panic(&mut self, message: &str) {
        self.clock += 1;
        let headline = message
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or_default();
        let signature = normalize(&format!("panic: {headline}"));
        self.count(signature, "panic");
    }

    fn count(&mut self, signature: String, kind: &str) {
        assert!(!signature.is_empty());
        assert!(signature.len() <= MAX_SIGNATURE_BYTES + "...".len());
        if let Some(pitfall) = self.pitfalls.iter_mut().find(|p| p.signature == signature) {
            pitfall.count += 1;
            pitfall.last_seen = self.clock;
            if pitfall.count == self.config.min_occurrences {
                debug!(
                    kind = pitfall.kind.as_str(),
                    count = pitfall.count,
                    signature = pitfall.signature.as_str(),
                    "memory: failure recurred, now a known pitfall candidate"
                );
            }
            return;
        }
        if self.pitfalls.len() >= self.config.max_tracked {
            let (idx, _) = self
                .pitfalls
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| (p.count, p.last_seen))
                .expect("max_tracked is positive, so the list is not empty");
            self.pitfalls.remove(idx);
        }
        self.pitfalls.push(Pitfall {
            signature,
            kind: kind.to_string(),
            count: 1,
            first_seen: self.clock,
            last_seen: self.clock,
        });
        assert!(self.pitfalls.len() <= self.config.max_tracked);
    }

    /// The pitfalls a prompt shows: the most frequent ones that reached
    /// [`FailureConfig::min_occurrences`], listed in first-seen order.
    ///
    /// First-seen order rather than frequency order keeps the rendered list
    /// stable: a count going up does not reorder it, and a new pitfall is
    /// appended, so the cached prompt prefix before it survives.
    #[must_use]
    pub fn shown(&self) -> Vec<&Pitfall> {
        let mut frequent = Vec::from_iter(
            self.pitfalls
                .iter()
                .filter(|p| p.count >= self.config.min_occurrences),
        );
        frequent.sort_by(|a, b| b.count.cmp(&a.count).then(a.first_seen.cmp(&b.first_seen)));
        frequent.truncate(self.config.max_shown);
        frequent.sort_by_key(|p| p.first_seen);
        frequent
    }

    /// The "known pitfalls" prompt section, empty if nothing recurred yet.
    #[must_use]
    pub fn render(&self) -> String {
        let shown = self.shown();
        if shown.is_empty() {
            return String::new();
        }
        let mut text = String::from(
            "## Known pitfalls\n\nEarlier attempts by other lanes failed like this. Avoid it.\n\n",
        );
        for pitfall in shown {
            writeln!(text, "- {}", pitfall.signature).expect(EXPECT_WRITE);
        }
        text.push('\n');
        text
    }

    /// Every tracked failure, in first-seen order.
    #[must_use]
    pub fn pitfalls(&self) -> &[Pitfall] {
        &self.pitfalls
    }
}

/// The failure signatures in the diagnostics of one self-heal step.
///
/// A compile failure yields one signature per rustc `error` headline; any
/// other kind yields its first non-empty line, prefixed with the kind.
fn signatures(kind: &str, diagnostics: &str) -> Vec<String> {
    if kind == "compile" {
        let mut headlines = Vec::from_iter(
            diagnostics
                .lines()
                .map(str::trim)
                .filter(|l| l.starts_with("error[") || l.starts_with("error:"))
                .map(normalize),
        );
        headlines.dedup();
        if !headlines.is_empty() {
            return headlines;
        }
    }
    let first = diagnostics.lines().map(str::trim).find(|l| !l.is_empty());
    match first {
        Some(line) => vec![normalize(&format!("{kind}: {line}"))],
        None => vec![kind.to_string()],
    }
}

/// Collapse whitespace, replace numbers that stand alone by `N` and cut to
/// [`MAX_SIGNATURE_BYTES`].
///
/// "Stand alone" means not glued to a letter or `_`: `index 5` becomes
/// `index N`, while `u32` and `E0599` stay as they are.
fn normalize(line: &str) -> String {
    let mut out = String::with_capacity(line.len().min(MAX_SIGNATURE_BYTES + 3));
    for word in line.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        let mut prev_ident = false;
        let mut in_number = false;
        for c in word.chars() {
            if c.is_ascii_digit() && !prev_ident {
                if !in_number {
                    out.push('N');
                    in_number = true;
                }
                continue;
            }
            // `1.5` and `1_000` stay one number.
            if in_number && (c == '.' || c == '_') {
                continue;
            }
            in_number = false;
            prev_ident = c.is_alphanumeric() || c == '_';
            out.push(c);
        }
    }
    if out.len() > MAX_SIGNATURE_BYTES {
        let mut cut = MAX_SIGNATURE_BYTES;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn self_heal(kind: &str, diagnostics: &str) -> LadderEvent {
        LadderEvent::SelfHeal {
            kind: kind.to_string(),
            diagnostics: diagnostics.to_string(),
            api_hints: Vec::new(),
        }
    }

    const NO_METHOD: &str = "[E1] replaces `step_by` on line 4\n\
        error[E0599]: no method named `step_by` found for struct `GameState` in the current scope\n\
        --> line 4:11\n";

    #[test]
    fn normalize_replaces_free_numbers_only() {
        assert_eq!(
            normalize("index out of bounds: the len is 3 but the index is 17"),
            "index out of bounds: the len is N but the index is N"
        );
        assert_eq!(
            normalize("expected `u32`, found `f64`"),
            "expected `u32`, found `f64`"
        );
        assert_eq!(normalize("error[E0599]:  a\tb"), "error[E0599]: a b");
        assert_eq!(normalize("value 1.5 and 1_000"), "value N and N");
        let long = "x".repeat(500);
        assert_eq!(normalize(&long).len(), MAX_SIGNATURE_BYTES + 3);
        let wide = "é".repeat(200);
        assert!(normalize(&wide).ends_with("..."));
    }

    #[test]
    fn compile_failures_yield_rustc_headlines() {
        let both = format!("{NO_METHOD}error[E0308]: mismatched types\n");
        assert_eq!(
            signatures("compile", &both),
            vec![
                "error[E0599]: no method named `step_by` found for struct `GameState` in the current scope",
                "error[E0308]: mismatched types",
            ]
        );
        assert_eq!(
            signatures("parse", "\n expected `;`\nmore"),
            vec!["parse: expected `;`"]
        );
        assert_eq!(signatures("no_rust_code", ""), vec!["no_rust_code"]);
        // A compile failure without a headline falls back to the first line.
        assert_eq!(
            signatures("compile", "linker failed"),
            vec!["compile: linker failed"]
        );
    }

    #[test]
    fn a_failure_counts_once_per_lane() {
        let mut failures = Failures::new(FailureConfig::default());
        let lane = [
            self_heal("compile", NO_METHOD),
            self_heal("compile", NO_METHOD),
            LadderEvent::Terminal {
                reason: "max retries".to_string(),
            },
        ];
        failures.commit_ladder(&lane);
        assert_eq!(failures.pitfalls().len(), 1);
        assert_eq!(failures.pitfalls()[0].count(), 1);
        assert!(failures.shown().is_empty(), "one lane is not a pattern");
        assert!(failures.render().is_empty());

        failures.commit_ladder(&lane[..1]);
        assert_eq!(failures.pitfalls()[0].count(), 2);
        assert_eq!(failures.pitfalls()[0].last_seen(), 2);
        let text = failures.render();
        assert!(text.starts_with("## Known pitfalls"));
        assert!(text.contains("- error[E0599]: no method named `step_by`"));
    }

    #[test]
    fn non_code_events_are_ignored() {
        let mut failures = Failures::new(FailureConfig::default());
        failures.commit_ladder(&[
            LadderEvent::TransientRetry {
                backoff: std::time::Duration::from_secs(1),
                cause: "429".to_string(),
            },
            LadderEvent::ContextReset {
                messages_dropped: 3,
                brief: "too long".to_string(),
            },
        ]);
        assert!(failures.pitfalls().is_empty());
    }

    #[test]
    fn panics_are_normalized_and_counted() {
        let mut failures = Failures::new(FailureConfig::new(2, 5, 8));
        failures.commit_panic("index out of bounds: the len is 3 but the index is 5\nbacktrace");
        failures.commit_panic("index out of bounds: the len is 4 but the index is 9");
        let shown = failures.shown();
        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].kind(), "panic");
        assert_eq!(
            shown[0].signature(),
            "panic: index out of bounds: the len is N but the index is N"
        );
    }

    #[test]
    fn shown_keeps_first_seen_order_and_cap() {
        let mut failures = Failures::new(FailureConfig::new(1, 2, 8));
        failures.commit_panic("a");
        failures.commit_panic("b");
        failures.commit_panic("c");
        failures.commit_panic("c");
        failures.commit_panic("b");
        failures.commit_panic("b");
        // b (3) and c (2) are the most frequent; listed in first-seen order.
        let shown = Vec::from_iter(failures.shown().into_iter().map(|p| p.signature().as_str()));
        assert_eq!(shown, vec!["panic: b", "panic: c"]);
    }

    #[test]
    fn tracking_evicts_the_rarest_stalest_failure() {
        let mut failures = Failures::new(FailureConfig::new(1, 1, 2));
        failures.commit_panic("a");
        failures.commit_panic("a");
        failures.commit_panic("b");
        failures.commit_panic("c");
        let tracked = Vec::from_iter(failures.pitfalls().iter().map(|p| p.signature().as_str()));
        assert_eq!(tracked, vec!["panic: a", "panic: c"]);
    }
}
