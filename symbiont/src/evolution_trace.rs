// SPDX-License-Identifier: MPL-2.0
//! The [`EvolutionTrace`]: the full agent trajectory of one evolution lane.
//!
//! Where [`crate::EvolveFailure`] records the individual rejections that fed
//! backpressure to the agent, a trace records the *whole* lane: every prompt
//! and nudge, every assistant turn and tool exchange, every recovery decision
//! the harness took, the per-request token breakdown, the per-stage timings,
//! and the final outcome. It is what a host persists to reconstruct, offline,
//! why a lane ended the way it did.
//!
//! The transcript is stored **once** per lane. Each attempt records the range
//! of [`EvolutionTrace::history`] it produced rather than its own copy, so
//! memory stays linear in the transcript instead of quadratic in the attempt
//! count.
//!
//! The system prompt is deliberately absent: it is byte-identical across every
//! attempt of a lane and every lane of a batch, and it embeds the generated
//! host-API documentation. Call [`crate::system_prompt`] once and store it
//! beside the traces.

use std::{
    fmt::Write as _,
    ops::Range,
    time::Duration,
};

use rig_agent::agent::CompletionCall;
use rig_core::{
    completion::Usage,
    message::{
        AssistantContent,
        Message,
        UserContent,
    },
};

use crate::revision::Revision;

/// The full agent trajectory of one lane (or of a single-prompt
/// [`crate::Runtime::evolve`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvolutionTrace {
    /// Lane index; `0` for single-prompt [`crate::Runtime::evolve`].
    pub lane: usize,
    /// The prompt the lane started with, before any corrective nudge.
    pub base_prompt: String,
    /// The lane's complete ordered transcript: every user turn (the base
    /// prompt and each nudge), every assistant turn, every tool call and
    /// result.
    ///
    /// Owned once. [`AttemptTrace`]s index into it via
    /// [`RunTrace::produced`]. A context or repeat reset clears the lane's
    /// working history but not this: the transcript keeps everything that was
    /// ever exchanged, which is the point of a trace.
    pub history: Vec<Message>,
    /// One entry per lane iteration, in order. An iteration whose inference
    /// call failed outright is present too, with [`AttemptTrace::run`] set to
    /// `None`.
    pub attempts: Vec<AttemptTrace>,
    /// How the lane ended.
    pub outcome: TraceOutcome,
    /// Wall time of the whole lane.
    pub duration: Duration,
}

/// One iteration of a lane's self-healing ladder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttemptTrace {
    /// Position in the lane timeline. Dense: always `0..attempts.len()`.
    pub seq: usize,
    /// The lane's self-healing attempt counter at this iteration; the same
    /// numbering as [`crate::EvolveFailure::attempt`] and the `attempt` metric
    /// label.
    ///
    /// **Not unique across entries.** A transient HTTP retry deliberately does
    /// not consume the attempt budget, so consecutive entries can carry the
    /// same value. Index by [`Self::seq`]; report this.
    pub attempt: usize,
    /// This iteration's user-prompt text: the base prompt, or the corrective
    /// nudge built from the previous failure.
    pub prompt: String,
    /// The agent run, when there was one.
    ///
    /// `None` when [`crate::EvolutionAgent::run`] itself returned an error \u2014 a
    /// transient HTTP failure or a context-size overflow \u2014 in which case no
    /// messages, no usage and no completion calls exist for this iteration.
    pub run: Option<RunTrace>,
    /// How far this iteration got through the pipeline, and how long each
    /// stage took.
    pub stages: StageTimings,
    /// What the harness did in response to this iteration.
    pub ladder: LadderEvent,
    /// Wall time of this iteration.
    pub duration: Duration,
}

/// The parts of an iteration that exist only once the agent run succeeded.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunTrace {
    /// Range into [`EvolutionTrace::history`] of the messages this run
    /// produced: its own user-prompt turn plus every assistant turn and tool
    /// exchange that followed.
    ///
    /// The request this run *sent* was `history[..produced.start + 1]` plus
    /// the system prompt.
    pub produced: Range<usize>,
    /// The run's final assistant text.
    pub response: String,
    /// Aggregate token usage for this run.
    pub usage: Usage,
    /// One entry per HTTP completion request. Each entry's `raw` wire body is
    /// cleared by the blanket [`crate::EvolutionAgent`] implementation;
    /// `usage`, `finish_reason` and the provider ids are kept.
    pub completion_calls: Vec<CompletionCall>,
}

/// Per-attempt mirror of the
/// [`PIPELINE_STAGE_DURATION`](crate::observability::PIPELINE_STAGE_DURATION)
/// histogram.
///
/// A field is `None` when the attempt failed before reaching that stage, which
/// is what makes "attempt 3 spent ninety seconds compiling, then failed"
/// recoverable from the trace rather than only from aggregate metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageTimings {
    /// Time spent in the agent run: inference plus any tool-calling turns.
    pub llm: Option<Duration>,
    /// Time spent parsing the Rust code block and validating signatures.
    pub parse_validate: Option<Duration>,
    /// The build stage, once parsing and validation passed.
    pub build: Option<BuildRecord>,
}

/// What the build stage did with a validated candidate.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BuildRecord {
    /// The candidate was compiled into a dylib and loaded.
    Built {
        /// Time queued on the single build slot before the work began.
        slot_wait: Duration,
        /// Time spent in `cargo`.
        compile: Duration,
        /// Time spent copying, `dlopen`ing and resolving symbols.
        load: Duration,
    },
    /// The candidate was byte-identical to an already-registered revision, so
    /// no build was spent and the existing revision was reused. There is no
    /// compile or load duration to report.
    Deduped {
        /// Time queued on the build slot before the check ran.
        slot_wait: Duration,
        /// The revision that was reused.
        revision: Revision,
    },
}

/// One step of the harness's reaction ladder. Every [`AttemptTrace`] has
/// exactly one, so the sequence of these *is* the lane's recovery path:
/// `SelfHeal → SelfHeal → Terminal` versus
/// `TransientRetry → SelfHeal → Registered`.
// Tagged `event` rather than `kind`: `SelfHeal` carries its own `kind` field,
// mirroring `EvolveFailure::kind`, and serde forbids a variant field that
// collides with the internal tag.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LadderEvent {
    /// This attempt produced a valid implementation, which was registered.
    /// The lane ends here.
    Registered {
        /// The revision the implementation was registered under.
        revision: Revision,
    },
    /// The failure was fed back to the agent as a corrective nudge.
    SelfHeal {
        /// Failure kind; the same labels as [`crate::EvolveFailure::kind`]:
        /// `no_rust_code`, `parse`, `max_turns`, `signature`, `unsafe`,
        /// `forbidden` or `compile`.
        ///
        /// A `String` rather than the `&'static str` the producing side has,
        /// so that a persisted trace can be deserialized without borrowing
        /// from the input.
        kind: String,
        /// The diagnostics quoted back to the agent.
        diagnostics: String,
    },
    /// A transient inference error. The same prompt is retried after
    /// `backoff`, and the attempt budget is not consumed.
    TransientRetry {
        /// How long the lane slept before retrying.
        backoff: Duration,
        /// The error that triggered the retry.
        cause: String,
    },
    /// The request exceeded the model's context window. The lane's working
    /// history was discarded and it restarted from the base prompt.
    ContextReset {
        /// How many chat messages were dropped from the working history.
        /// They remain in [`EvolutionTrace::history`].
        messages_dropped: usize,
        /// The one-line error summary carried into the restart prompt.
        brief: String,
    },
    /// The agent echoed its previously rejected code verbatim. The lane's
    /// working history was discarded likewise, with a do-not-repeat
    /// instruction that does not quote the rejected code.
    RepeatReset {
        /// How many chat messages were dropped from the working history.
        messages_dropped: usize,
        /// The one-line error summary carried into the restart prompt.
        brief: String,
    },
    /// This attempt's failure ended the lane.
    Terminal {
        /// The final error.
        reason: String,
    },
}

/// How a lane ended.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceOutcome {
    /// A revision was registered.
    Registered {
        /// The revision the lane produced.
        revision: Revision,
    },
    /// The lane gave up.
    Failed {
        /// The final error.
        reason: String,
    },
}

impl EvolutionTrace {
    /// Start a trace for `lane`, beginning from `base_prompt`.
    pub(crate) fn new(lane: usize, base_prompt: String) -> Self {
        Self {
            lane,
            base_prompt,
            history: Vec::new(),
            attempts: Vec::new(),
            outcome: TraceOutcome::Failed {
                // Replaced when the lane ends; a lane that panics mid-flight
                // never yields its trace at all.
                reason: "lane did not finish".to_string(),
            },
            duration: Duration::ZERO,
        }
    }

    /// Append an attempt, assigning its [`AttemptTrace::seq`].
    pub(crate) fn push_attempt(
        &mut self,
        attempt: usize,
        prompt: String,
        run: Option<RunTrace>,
        stages: StageTimings,
        ladder: LadderEvent,
        duration: Duration,
    ) {
        self.attempts.push(AttemptTrace {
            seq: self.attempts.len(),
            attempt,
            prompt,
            run,
            stages,
            ladder,
            duration,
        });
    }

    /// Total token usage across every attempt that reached the model.
    #[must_use]
    pub fn usage(&self) -> Usage {
        self.attempts
            .iter()
            .filter_map(|attempt| attempt.run.as_ref())
            .fold(Usage::new(), |mut total, run| {
                total += run.usage;
                total
            })
    }

    /// The number of HTTP completion requests the lane made.
    #[must_use]
    pub fn completion_calls(&self) -> usize {
        self.attempts
            .iter()
            .filter_map(|attempt| attempt.run.as_ref())
            .map(|run| run.completion_calls.len())
            .sum()
    }

    /// A human-readable transcript: one block per attempt, with the prompt,
    /// the tool exchanges, the response, the ladder decision, the token usage
    /// and the stage timings.
    ///
    /// For machine consumption use [`Self::to_json_pretty`] or
    /// [`Self::write_jsonl`].
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "lane {} — {} attempt(s), {:?}",
            self.lane,
            self.attempts.len(),
            self.duration,
        );
        let _ = writeln!(out, "base prompt:\n{}\n", self.base_prompt);

        for attempt in &self.attempts {
            let _ = writeln!(
                out,
                "── attempt {} (seq {}, {:?}) ──",
                attempt.attempt, attempt.seq, attempt.duration,
            );
            let _ = writeln!(out, "prompt:\n{}", attempt.prompt);

            if let Some(run) = &attempt.run {
                for message in self
                    .history
                    .get(run.produced.clone())
                    .unwrap_or_default()
                    .iter()
                {
                    render_tool_activity(&mut out, message);
                }
                let _ = writeln!(out, "response:\n{}", run.response);
                let _ = writeln!(
                    out,
                    "usage: {} in / {} out over {} request(s)",
                    run.usage.input_tokens,
                    run.usage.output_tokens,
                    run.completion_calls.len(),
                );
            } else {
                let _ = writeln!(out, "(no agent run: the inference call itself failed)");
            }

            render_stages(&mut out, &attempt.stages);
            let _ = writeln!(out, "ladder: {}\n", render_ladder(&attempt.ladder));
        }

        let _ = match &self.outcome {
            TraceOutcome::Registered { revision } => {
                writeln!(out, "outcome: registered revision {revision}")
            }
            TraceOutcome::Failed { reason } => writeln!(out, "outcome: failed — {reason}"),
        };
        out
    }

    /// Write the trace as JSON Lines.
    ///
    /// The first line is a `trace` header carrying the lane metadata, the full
    /// transcript and the outcome; each following line is one `attempt` object
    /// referencing the transcript by index range. Appending several traces to
    /// one file therefore stays valid JSONL, in the shape session logs of
    /// other agent harnesses use.
    ///
    /// # Errors
    ///
    /// Propagates write failures from `w`, and serialization failures (which
    /// indicate a bug, since every field is plain data).
    pub fn write_jsonl<W: std::io::Write>(&self, mut w: W) -> Result<(), std::io::Error> {
        let header = serde_json::json!({
            "type": "trace",
            "lane": self.lane,
            "base_prompt": self.base_prompt,
            "history": self.history,
            "outcome": self.outcome,
            "duration": self.duration,
        });
        writeln!(w, "{}", serde_json::to_string(&header)?)?;

        for attempt in &self.attempts {
            let mut line = serde_json::to_value(attempt).map_err(std::io::Error::other)?;
            if let Some(object) = line.as_object_mut() {
                object.insert("type".to_string(), serde_json::json!("attempt"));
                object.insert("lane".to_string(), serde_json::json!(self.lane));
            }
            writeln!(w, "{}", serde_json::to_string(&line)?)?;
        }
        Ok(())
    }

    /// The whole trace as one pretty-printed JSON document.
    #[must_use]
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self)
            .expect("an `EvolutionTrace` is plain data and always serializes")
    }
}

/// Render the tool calls and tool results carried by one transcript message.
fn render_tool_activity(out: &mut String, message: &Message) {
    match message {
        Message::Assistant { content, .. } => {
            for item in content.iter() {
                if let AssistantContent::ToolCall(call) = item {
                    let _ = writeln!(
                        out,
                        "  → tool call {}({})",
                        call.function.name, call.function.arguments,
                    );
                }
            }
        }
        Message::User { content } => {
            for item in content.iter() {
                if let UserContent::ToolResult(result) = item {
                    let _ = writeln!(out, "  ← tool result from {}", result.name);
                }
            }
        }
        // The preamble travels as the agent's own configuration, not as a
        // transcript turn; see the module docs on why it is not traced.
        Message::System { .. } => {}
    }
}

/// Render the stage breakdown of one attempt.
fn render_stages(out: &mut String, stages: &StageTimings) {
    let mut parts: Vec<String> = Vec::new();
    if let Some(llm) = stages.llm {
        parts.push(format!("llm {llm:?}"));
    }
    if let Some(parse_validate) = stages.parse_validate {
        parts.push(format!("parse/validate {parse_validate:?}"));
    }
    match &stages.build {
        Some(BuildRecord::Built {
            slot_wait,
            compile,
            load,
        }) => {
            parts.push(format!(
                "build slot {slot_wait:?}, compile {compile:?}, load {load:?}"
            ));
        }
        Some(BuildRecord::Deduped {
            slot_wait,
            revision,
        }) => parts.push(format!(
            "build slot {slot_wait:?}, deduped onto revision {revision}"
        )),
        None => {}
    }
    if !parts.is_empty() {
        let _ = writeln!(out, "stages: {}", parts.join(", "));
    }
}

/// One-line summary of a ladder decision.
fn render_ladder(ladder: &LadderEvent) -> String {
    match ladder {
        LadderEvent::Registered { revision } => format!("registered revision {revision}"),
        LadderEvent::SelfHeal { kind, diagnostics } => {
            format!("self-heal ({kind}): {}", first_line(diagnostics))
        }

        LadderEvent::TransientRetry { backoff, cause } => {
            format!("transient retry in {backoff:?}: {}", first_line(cause))
        }
        LadderEvent::ContextReset {
            messages_dropped, ..
        } => format!("context reset, dropped {messages_dropped} message(s)"),
        LadderEvent::RepeatReset {
            messages_dropped, ..
        } => format!("repeat reset, dropped {messages_dropped} message(s)"),
        LadderEvent::Terminal { reason } => format!("terminal: {}", first_line(reason)),
    }
}

/// The first line of `text`, for one-line summaries of multi-line diagnostics.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_with(attempts: Vec<AttemptTrace>, outcome: TraceOutcome) -> EvolutionTrace {
        EvolutionTrace {
            lane: 0,
            base_prompt: "write a sort".to_string(),
            history: Vec::new(),
            attempts,
            outcome,
            duration: Duration::from_secs(1),
        }
    }

    fn attempt(seq: usize, attempt: usize, run: Option<RunTrace>) -> AttemptTrace {
        AttemptTrace {
            seq,
            attempt,
            prompt: "p".to_string(),
            run,
            stages: StageTimings::default(),
            ladder: LadderEvent::Terminal {
                reason: "done".to_string(),
            },
            duration: Duration::from_millis(1),
        }
    }

    fn run_with(usage: Usage, calls: usize) -> RunTrace {
        RunTrace {
            produced: 0..1,
            response: "r".to_string(),
            usage,
            completion_calls: (0..calls)
                .map(|index| CompletionCall::new(index, usage))
                .collect(),
        }
    }

    fn usage_of(input: u64, output: u64) -> Usage {
        let mut usage = Usage::new();
        usage.input_tokens = input;
        usage.output_tokens = output;
        usage.total_tokens = input + output;
        usage
    }

    /// A lane's usage is the sum over the attempts that reached the model;
    /// attempts whose inference call failed contribute nothing.
    #[test]
    fn usage_sums_over_runs_only() {
        let trace = trace_with(
            vec![
                attempt(0, 1, Some(run_with(usage_of(10, 5), 1))),
                attempt(1, 1, None),
                attempt(2, 2, Some(run_with(usage_of(3, 2), 2))),
            ],
            TraceOutcome::Failed {
                reason: "gave up".to_string(),
            },
        );

        assert_eq!(trace.usage().input_tokens, 13);
        assert_eq!(trace.usage().output_tokens, 7);
        assert_eq!(trace.completion_calls(), 3);
    }

    /// `push_attempt` assigns a dense `seq` even when `attempt` repeats, which
    /// it does across a transient retry.
    #[test]
    fn seq_is_dense_while_attempt_may_repeat() {
        let mut trace = EvolutionTrace::new(2, "base".to_string());
        for attempt_number in [1, 1, 2] {
            trace.push_attempt(
                attempt_number,
                "p".to_string(),
                None,
                StageTimings::default(),
                LadderEvent::TransientRetry {
                    backoff: Duration::from_secs(1),
                    cause: "503".to_string(),
                },
                Duration::ZERO,
            );
        }

        let seqs: Vec<usize> = trace.attempts.iter().map(|a| a.seq).collect();
        let attempts: Vec<usize> = trace.attempts.iter().map(|a| a.attempt).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        assert_eq!(attempts, vec![1, 1, 2]);
    }

    /// The trace round-trips through serde, so a persisted trace can be read
    /// back for offline analysis.
    #[test]
    fn serde_round_trip() {
        let mut trace = trace_with(
            vec![AttemptTrace {
                stages: StageTimings {
                    llm: Some(Duration::from_millis(120)),
                    parse_validate: Some(Duration::from_micros(90)),
                    build: Some(BuildRecord::Deduped {
                        slot_wait: Duration::from_millis(3),
                        revision: Revision::new(7),
                    }),
                },
                ladder: LadderEvent::SelfHeal {
                    kind: "compile".to_string(),
                    diagnostics: "E0277\nsecond line".to_string(),
                },
                ..attempt(0, 1, Some(run_with(usage_of(1, 1), 1)))
            }],
            TraceOutcome::Registered {
                revision: Revision::new(7),
            },
        );
        trace.history = vec![Message::user("hi")];

        let json = trace.to_json_pretty();
        let back: EvolutionTrace =
            serde_json::from_str(&json).expect("a serialized trace deserializes");

        assert_eq!(back.attempts.len(), 1);
        assert_eq!(back.attempts[0].stages, trace.attempts[0].stages);
        assert_eq!(back.history.len(), 1);
        assert!(matches!(
            back.outcome,
            TraceOutcome::Registered { revision } if revision == Revision::new(7)
        ));
    }

    /// JSONL is one header line plus one line per attempt, each tagged so a
    /// concatenated multi-lane file stays interpretable.
    #[test]
    fn jsonl_is_header_plus_one_line_per_attempt() {
        let trace = trace_with(
            vec![attempt(0, 1, None), attempt(1, 2, None)],
            TraceOutcome::Failed {
                reason: "gave up".to_string(),
            },
        );

        let mut buffer = Vec::new();
        trace.write_jsonl(&mut buffer).expect("writing to a Vec");
        let text = String::from_utf8(buffer).expect("valid utf-8");
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines.len(), 3);
        for line in &lines {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("each line is its own JSON document");
            assert_eq!(value["lane"], 0);
        }
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(lines[0]).unwrap()["type"],
            "trace"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(lines[2]).unwrap()["seq"],
            1
        );
    }

    /// `render` reports an attempt that never reached the model rather than
    /// silently showing an empty block.
    #[test]
    fn render_marks_attempts_without_a_run() {
        let trace = trace_with(
            vec![AttemptTrace {
                ladder: LadderEvent::TransientRetry {
                    backoff: Duration::from_secs(2),
                    cause: "503 Service Unavailable\nretry later".to_string(),
                },
                ..attempt(0, 1, None)
            }],
            TraceOutcome::Failed {
                reason: "gave up".to_string(),
            },
        );

        let rendered = trace.render();
        assert!(rendered.contains("no agent run"));
        assert!(rendered.contains("transient retry"));
        // Multi-line causes collapse to their first line in the summary.
        assert!(!rendered.contains("retry later"));
    }
}
