// SPDX-License-Identifier: MPL-2.0
//! A host's rewrite of a candidate's source before it is compiled.
//!
//! The agent writes one text; the host may need the compiler to see another,
//! for example named parameter reads rewritten to index reads. The revision
//! keeps the agent's text: that is what [`crate::Runtime::revision_code`]
//! returns, what the agent edits and what a byte-identical candidate is
//! matched against. Only the dylib is built from the transformed text.
//!
//! The transform must keep the number of lines, so that the compiler's line
//! numbers point at the agent's lines. Diagnostics are moved back onto the
//! agent's text where a line is the same in both; spans and suggestions on a
//! line the transform changed are dropped, because a byte there has no
//! counterpart in the agent's text.

use std::{
    ops::Range,
    sync::Arc,
};

use crate::{
    Diagnostic,
    DiagnosticSpan,
    Error,
};

/// A rewrite of the candidate's source before compilation: the text to
/// compile, or a message for the agent, which sees it like a compile error.
/// See [`crate::Runtime::set_source_transform`].
pub type SourceTransform = Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

/// The text `transform` makes of `source`, or the compile failure the agent
/// sees when it refuses.
///
/// # Panics
///
/// If the transform changes the number of lines: line numbers in compiler
/// output would no longer point at the agent's code.
pub(crate) fn apply(transform: &SourceTransform, source: &str) -> crate::Result<String> {
    match transform(source) {
        Ok(compiled) => {
            assert_eq!(
                compiled.lines().count(),
                source.lines().count(),
                "a source transform must keep the number of lines"
            );
            Ok(compiled)
        }
        Err(message) => Err(Error::CompilationFailed {
            code: source.to_string(),
            err: message,
            diagnostics: Vec::new(),
        }),
    }
}

/// `error`, raised by compiling `compiled`, restated for `source`, the text
/// `compiled` was transformed from: the failed code is `source`, and every
/// span lies in `source` or is dropped.
pub(crate) fn onto_source(error: Error, compiled: &str, source: &str) -> Error {
    if compiled == source {
        return error;
    }
    let Error::CompilationFailed {
        err, diagnostics, ..
    } = error
    else {
        return error;
    };
    let lines = Lines::new(compiled, source);
    let diagnostics = diagnostics
        .into_iter()
        .map(|mut diagnostic| {
            diagnostic.spans.retain_mut(|span| lines.relocate(span));
            diagnostic
                .suggestions
                .retain_mut(|suggestion| lines.relocate(&mut suggestion.span));
            diagnostic
        })
        .collect::<Vec<Diagnostic>>();
    Error::CompilationFailed {
        code: source.to_string(),
        err,
        diagnostics,
    }
}

/// The lines of the compiled text and of the source, side by side.
struct Lines<'a> {
    compiled: Vec<(usize, &'a str)>,
    source: Vec<(usize, &'a str)>,
}

impl<'a> Lines<'a> {
    fn new(compiled: &'a str, source: &'a str) -> Self {
        Self {
            compiled: starts(compiled),
            source: starts(source),
        }
    }

    /// Moves `span` from the compiled text onto the source; `false` if a
    /// line it covers differs between the two.
    fn relocate(&self, span: &mut DiagnosticSpan) -> bool {
        let Some(lines) = span
            .line_start
            .checked_sub(1)
            .zip(span.line_end.checked_sub(1))
        else {
            return false;
        };
        let same = (lines.0..=lines.1).all(|line| {
            matches!(
                (self.compiled.get(line), self.source.get(line)),
                (Some((_, a)), Some((_, b))) if a == b
            )
        });
        if !same {
            return false;
        }
        let shift = |byte: usize, line: usize| byte - self.compiled[line].0 + self.source[line].0;
        span.bytes = Range {
            start: shift(span.bytes.start, lines.0),
            end: shift(span.bytes.end, lines.1),
        };
        true
    }
}

/// The byte offset and text of every line, without its line break.
fn starts(text: &str) -> Vec<(usize, &str)> {
    let mut at = 0;
    text.split_inclusive('\n')
        .map(|line| {
            let start = at;
            at += line.len();
            (start, line.trim_end_matches(['\n', '\r']))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Applicability,
        Suggestion,
    };

    fn span(text: &str, bytes: Range<usize>, line: usize) -> DiagnosticSpan {
        DiagnosticSpan {
            line_start: line,
            column_start: 1,
            line_end: line,
            column_end: 1,
            is_primary: true,
            label: None,
            text: text[bytes.clone()].to_string(),
            bytes,
        }
    }

    fn failure(spans: Vec<DiagnosticSpan>, suggestion: DiagnosticSpan) -> Error {
        Error::CompilationFailed {
            code: String::new(),
            err: "rendered".to_string(),
            diagnostics: vec![Diagnostic {
                code: Some("E0308".to_string()),
                message: "mismatched types".to_string(),
                spans,
                suggestions: vec![Suggestion {
                    message: "fix".to_string(),
                    span: suggestion,
                    replacement: "y".to_string(),
                    applicability: Applicability::MachineApplicable,
                }],
                rendered: "rendered".to_string(),
            }],
        }
    }

    const SOURCE: &str = "let a = p.f(\"name\", 1.0);\nlet b = x;\n";
    const COMPILED: &str = "let a = p.f_at(0);\nlet b = x;\n";

    #[test]
    fn spans_on_unchanged_lines_move_onto_the_source() {
        let x = COMPILED.find('x').expect("x is in the text");
        let error = failure(
            vec![span(COMPILED, x..x + 1, 2)],
            span(COMPILED, x..x + 1, 2),
        );
        let Error::CompilationFailed {
            code, diagnostics, ..
        } = onto_source(error, COMPILED, SOURCE)
        else {
            panic!("still a compile failure")
        };
        assert_eq!(code, SOURCE);
        let moved = diagnostics[0].spans[0].bytes.clone();
        assert_eq!(&SOURCE[moved.clone()], "x");
        assert_eq!(diagnostics[0].suggestions[0].span.bytes, moved);
    }

    #[test]
    fn spans_on_changed_lines_are_dropped() {
        let at = COMPILED.find("f_at").expect("f_at is in the text");
        let error = failure(
            vec![span(COMPILED, at..at + 4, 1)],
            span(COMPILED, at..at + 4, 1),
        );
        let Error::CompilationFailed { diagnostics, .. } = onto_source(error, COMPILED, SOURCE)
        else {
            panic!("still a compile failure")
        };
        assert!(diagnostics[0].spans.is_empty());
        assert!(diagnostics[0].suggestions.is_empty());
        assert_eq!(diagnostics[0].rendered, "rendered", "the message stays");
    }

    #[test]
    fn an_identity_transform_changes_nothing() {
        let error = failure(Vec::new(), span(SOURCE, 0..3, 1));
        let same = onto_source(error, SOURCE, SOURCE);
        assert!(matches!(same, Error::CompilationFailed { code, .. } if code.is_empty()));
    }

    #[test]
    fn a_refusal_is_a_compile_failure_of_the_source() {
        let refuse: SourceTransform = Arc::new(|_| Err("hole `x`: bad".to_string()));
        let Err(Error::CompilationFailed {
            code,
            err,
            diagnostics,
        }) = apply(&refuse, SOURCE)
        else {
            panic!("a refusal is a compile failure")
        };
        assert_eq!(code, SOURCE);
        assert_eq!(err, "hole `x`: bad");
        assert!(diagnostics.is_empty());
        let rewrite: SourceTransform = Arc::new(|s| Ok(s.replace("x", "z")));
        assert_eq!(
            apply(&rewrite, SOURCE).expect("a rewrite applies"),
            SOURCE.replace('x', "z")
        );
    }

    #[test]
    #[should_panic(expected = "keep the number of lines")]
    fn a_transform_must_keep_the_lines() {
        let join: SourceTransform = Arc::new(|s| Ok(s.replace('\n', " ")));
        let _ = apply(&join, SOURCE);
    }
}
