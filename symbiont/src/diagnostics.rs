// SPDX-License-Identifier: MPL-2.0
//! Structured compiler diagnostics of a failed candidate build.
//!
//! The compiler runs with `--message-format=json`, so every diagnostic
//! arrives as a record with its spans, its error code and rustc's own
//! `rendered` text. This module keeps the parts that the repair loop acts
//! on:
//!
//! - **spans** in the candidate, as byte ranges. The candidate is the
//!   byte-for-byte prefix of `lib.rs` (see [`crate::layout`]), so a span in
//!   `src/lib.rs` indexes the candidate string directly. A span that falls
//!   outside the candidate (in the harness glue) is dropped from the span
//!   list but the diagnostic itself is kept.
//! - **suggestions** rustc attaches to a diagnostic, with their
//!   applicability. `MachineApplicable` ones are what an auto-fix pass may
//!   apply without asking the model.
//! - the **rendered** text, which is what the model reads. It is rustc's
//!   text with the location remapped: `src/lib.rs:6:24` becomes `6:24`.
//!
//! Warnings are absent by construction: the generated crate compiles with
//! `--cap-lints allow`.

use std::{
    fmt::Write as _,
    ops::Range,
};

use serde::{
    Deserialize,
    Serialize,
};

use crate::EXPECT_WRITE;

/// The file name rustc reports for the crate root of the generated dylib.
const LIB_RS: &str = "src/lib.rs";

/// One `error` diagnostic of a failed build, located in the candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// The error code, e.g. `E0308`. Absent for errors without a code
    /// (parse errors, `aborting due to ..`).
    pub code: Option<String>,
    /// The one-line message, e.g. `mismatched types`.
    pub message: String,
    /// The spans in the candidate. The primary span comes first.
    pub spans: Vec<DiagnosticSpan>,
    /// Suggestions rustc attached to the diagnostic.
    pub suggestions: Vec<Suggestion>,
    /// rustc's rendered text, with `src/lib.rs:` locations remapped to the
    /// candidate's own line numbers.
    pub rendered: String,
}

/// A location in the candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSpan {
    /// Byte range in the candidate string.
    pub bytes: Range<usize>,
    /// 1-based line of the first byte.
    pub line_start: usize,
    /// 1-based column (in chars) of the first byte.
    pub column_start: usize,
    /// 1-based line of the last byte.
    pub line_end: usize,
    /// 1-based column (in chars) one past the last byte.
    pub column_end: usize,
    /// The span rustc marks as the location of the error.
    pub is_primary: bool,
    /// rustc's label for the span, e.g. `expected \`usize\`, found \`&str\``.
    pub label: Option<String>,
}

/// A replacement rustc proposes for one span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Suggestion {
    /// The child message that carries the suggestion, e.g.
    /// `consider borrowing here`.
    pub message: String,
    /// The span to replace.
    pub span: DiagnosticSpan,
    /// The text that replaces the span.
    pub replacement: String,
    /// How sure rustc is.
    pub applicability: Applicability,
}

/// rustc's confidence in a suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Applicability {
    /// Applying the suggestion fixes the error without changing the
    /// meaning of the program. `rustfix` and `cargo fix` apply these.
    MachineApplicable,
    /// The suggestion is probably right but may change behaviour.
    MaybeIncorrect,
    /// The suggestion contains placeholders the user has to fill in.
    HasPlaceholders,
    /// rustc does not know.
    Unspecified,
}

/// Parse the `--message-format=json` output of a failed `cargo rustc` into
/// the error diagnostics located in `candidate`.
///
/// `stdout` is cargo's JSON stream, one message per line. Every message
/// whose `reason` is not `compiler-message` and every compiler message
/// below `error` level is skipped, as is the trailing `aborting due to N
/// previous errors` summary, which carries no location and only repeats
/// the count.
pub(crate) fn parse_cargo_json(stdout: &str, candidate: &str) -> Vec<Diagnostic> {
    let candidate_len = candidate.len();
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<CargoMessage>(line).ok())
        .filter(|message| message.reason == "compiler-message")
        .filter_map(|message| message.message)
        .filter(|message| message.level == "error")
        .filter(|message| !is_summary(&message.message))
        .map(|message| locate(message, candidate_len))
        .collect()
}

/// `aborting due to 2 previous errors` and the `For more information about
/// this error` trailer. Neither says anything the model can act on.
fn is_summary(message: &str) -> bool {
    message.starts_with("aborting due to") || message.starts_with("For more information about")
}

/// Turn a raw rustc message into a [`Diagnostic`] whose spans index the
/// candidate.
fn locate(message: RustcMessage, candidate_len: usize) -> Diagnostic {
    let mut spans: Vec<DiagnosticSpan> = message
        .spans
        .iter()
        .filter_map(|span| in_candidate(span, candidate_len))
        .collect();
    // The primary span first, so `spans[0]` is where the error is.
    spans.sort_by_key(|span| !span.is_primary);

    let suggestions = message
        .children
        .iter()
        .flat_map(|child| {
            child.spans.iter().filter_map(|span| {
                let replacement = span.suggested_replacement.clone()?;
                Some(Suggestion {
                    message: child.message.clone(),
                    span: in_candidate(span, candidate_len)?,
                    replacement,
                    applicability: span
                        .suggestion_applicability
                        .unwrap_or(Applicability::Unspecified),
                })
            })
        })
        .collect();

    Diagnostic {
        code: message.code.map(|code| code.code),
        message: message.message,
        spans,
        suggestions,
        rendered: remap_rendered(message.rendered.as_deref().unwrap_or_default()),
    }
}

/// The span as a location in the candidate, or `None` if it is in another
/// file or in the harness glue that follows the candidate.
fn in_candidate(span: &RustcSpan, candidate_len: usize) -> Option<DiagnosticSpan> {
    if span.file_name != LIB_RS || span.byte_end > candidate_len {
        return None;
    }
    Some(DiagnosticSpan {
        bytes: span.byte_start..span.byte_end,
        line_start: span.line_start,
        column_start: span.column_start,
        line_end: span.line_end,
        column_end: span.column_end,
        is_primary: span.is_primary,
        label: span.label.clone(),
    })
}

/// Strip the file name from every `--> src/lib.rs:LINE:COL` location, so the
/// text reads as a location in the code block the model wrote. The line
/// numbers need no arithmetic: the candidate starts at line 1 of `lib.rs`.
fn remap_rendered(rendered: &str) -> String {
    let needle = format!("--> {LIB_RS}:");
    rendered.replace(&needle, "--> line ")
}

/// Render the diagnostics for the model: every rendered error, numbered
/// `E1..En` so a repair can refer to one by its number.
pub(crate) fn render_for_prompt(diagnostics: &[Diagnostic], out: &mut String) {
    for (idx, diagnostic) in diagnostics.iter().enumerate() {
        writeln!(out, "[E{}]", idx + 1).expect(EXPECT_WRITE);
        out.push_str(diagnostic.rendered.trim_end());
        out.push('\n');
    }
}

// ---------------------------------------------------------------------------
// The wire format of `cargo --message-format=json`. Only the fields this
// module reads; everything else is ignored by serde.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CargoMessage {
    reason: String,
    #[serde(default)]
    message: Option<RustcMessage>,
}

#[derive(Deserialize)]
struct RustcMessage {
    message: String,
    #[serde(default)]
    code: Option<RustcCode>,
    level: String,
    #[serde(default)]
    spans: Vec<RustcSpan>,
    #[serde(default)]
    children: Vec<RustcMessage>,
    #[serde(default)]
    rendered: Option<String>,
}

#[derive(Deserialize)]
struct RustcCode {
    code: String,
}

#[derive(Deserialize)]
struct RustcSpan {
    file_name: String,
    byte_start: usize,
    byte_end: usize,
    line_start: usize,
    line_end: usize,
    column_start: usize,
    column_end: usize,
    is_primary: bool,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    suggested_replacement: Option<String>,
    #[serde(default)]
    suggestion_applicability: Option<Applicability>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One cargo JSON line with one E0308 whose primary span is inside the
    /// candidate, a secondary span, and a `MachineApplicable` suggestion.
    const E0308: &str = r#"{"reason":"compiler-message","package_id":"x","manifest_path":"x","target":{"name":"symbiont_evolvable"},"message":{"rendered":"error[E0308]: mismatched types\n --> src/lib.rs:3:24\n  |\n3 |     let wrong: usize = \"not a usize\";\n  |                -----   ^^^^^^^^^^^^^ expected `usize`, found `&str`\n  |                |\n  |                expected due to this\n\n","$message_type":"diagnostic","children":[{"children":[],"code":null,"level":"help","message":"try this","rendered":null,"spans":[{"byte_end":79,"byte_start":66,"column_end":37,"column_start":24,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":3,"line_start":3,"suggested_replacement":"1","suggestion_applicability":"MachineApplicable","text":[]}]}],"code":{"code":"E0308","explanation":"x"},"level":"error","message":"mismatched types","spans":[{"byte_end":79,"byte_start":66,"column_end":37,"column_start":24,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":"expected `usize`, found `&str`","line_end":3,"line_start":3,"suggested_replacement":null,"suggestion_applicability":null,"text":[]},{"byte_end":63,"byte_start":58,"column_end":21,"column_start":16,"expansion":null,"file_name":"src/lib.rs","is_primary":false,"label":"expected due to this","line_end":3,"line_start":3,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

    const ABORTING: &str = r#"{"reason":"compiler-message","package_id":"x","manifest_path":"x","target":{"name":"symbiont_evolvable"},"message":{"rendered":"error: aborting due to 1 previous error\n\n","$message_type":"diagnostic","children":[],"code":null,"level":"error","message":"aborting due to 1 previous error","spans":[]}}"#;

    const FAILURE_NOTE: &str = r#"{"reason":"compiler-message","package_id":"x","manifest_path":"x","target":{"name":"symbiont_evolvable"},"message":{"rendered":"For more information about this error, try `rustc --explain E0308`.\n","$message_type":"diagnostic","children":[],"code":null,"level":"failure-note","message":"For more information about this error, try `rustc --explain E0308`.","spans":[]}}"#;

    const BUILD_FINISHED: &str = r#"{"reason":"build-finished","success":false}"#;

    const WARNING: &str = r#"{"reason":"compiler-message","package_id":"x","manifest_path":"x","target":{"name":"symbiont_evolvable"},"message":{"rendered":"warning: unused variable\n","$message_type":"diagnostic","children":[],"code":null,"level":"warning","message":"unused variable: `scale`","spans":[]}}"#;

    /// The candidate the E0308 fixture above is located in (spans at bytes
    /// 58..63 and 66..79).
    const CANDIDATE: &str =
        "fn f(counter: &mut usize) {\n    let x = 1;\n    let wrong: usize = \"not a usize\";\n}";

    #[test]
    fn errors_are_kept_and_noise_is_dropped() {
        let stdout = [
            E0308,
            WARNING,
            ABORTING,
            FAILURE_NOTE,
            BUILD_FINISHED,
            "not json",
        ]
        .join("\n");
        let diagnostics = parse_cargo_json(&stdout, CANDIDATE);
        assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.code.as_deref(), Some("E0308"));
        assert_eq!(diagnostic.message, "mismatched types");
    }

    #[test]
    fn spans_index_the_candidate_with_the_primary_first() {
        let diagnostics = parse_cargo_json(E0308, CANDIDATE);
        let spans = &diagnostics[0].spans;
        assert_eq!(spans.len(), 2);
        assert!(spans[0].is_primary);
        assert_eq!(&CANDIDATE[spans[0].bytes.clone()], "\"not a usize\"");
        assert_eq!((spans[0].line_start, spans[0].column_start), (3, 24));
        assert_eq!(&CANDIDATE[spans[1].bytes.clone()], "usize");
        assert_eq!(spans[1].label.as_deref(), Some("expected due to this"));
    }

    #[test]
    fn suggestions_carry_their_applicability() {
        let diagnostics = parse_cargo_json(E0308, CANDIDATE);
        let suggestions = &diagnostics[0].suggestions;
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].message, "try this");
        assert_eq!(suggestions[0].replacement, "1");
        assert_eq!(
            suggestions[0].applicability,
            Applicability::MachineApplicable
        );
        assert_eq!(
            &CANDIDATE[suggestions[0].span.bytes.clone()],
            "\"not a usize\""
        );
    }

    /// The candidate ends at byte 50 here, so both spans (58..63 and
    /// 66..79) reach into the glue and are dropped. The diagnostic stays.
    #[test]
    fn spans_in_the_harness_glue_are_dropped() {
        let short = &CANDIDATE[..50];
        let diagnostics = parse_cargo_json(E0308, short);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].spans.is_empty());
        assert!(diagnostics[0].suggestions.is_empty());
    }

    #[test]
    fn rendered_text_drops_the_file_name() {
        let diagnostics = parse_cargo_json(E0308, CANDIDATE);
        let rendered = &diagnostics[0].rendered;
        assert!(rendered.contains("--> line 3:24"), "{rendered}");
        assert!(!rendered.contains("src/lib.rs"), "{rendered}");
        assert!(rendered.contains("expected `usize`, found `&str`"));
    }

    #[test]
    fn prompt_rendering_numbers_the_errors() {
        let diagnostics = parse_cargo_json(&[E0308, E0308].join("\n"), CANDIDATE);
        let mut out = String::new();
        render_for_prompt(&diagnostics, &mut out);
        assert!(
            out.starts_with("[E1]\nerror[E0308]: mismatched types\n"),
            "{out}"
        );
        assert!(out.contains("\n[E2]\nerror[E0308]"), "{out}");
    }

    #[test]
    fn diagnostics_round_trip_through_serde() {
        let diagnostics = parse_cargo_json(E0308, CANDIDATE);
        let json = serde_json::to_string(&diagnostics).expect("serializes");
        let back: Vec<Diagnostic> = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, diagnostics);
    }
}
