// SPDX-License-Identifier: MPL-2.0
//! The [`EvolutionTrace`]: the full agent trajectory of one evolution lane.
//!
//! [`crate::EvolveFailure`] records only the rejections that fed backpressure
//! to the agent. A trace records the whole lane. It holds every prompt and
//! nudge, every assistant turn and tool exchange, and every recovery decision.
//! It also holds the per-request token breakdown, the per-stage timings and
//! the final outcome. A host persists a trace to find offline why a lane ended
//! the way it did.
//!
//! The transcript is stored **once** per lane. Each attempt records the range
//! of [`EvolutionTrace::history`] that it produced instead of its own copy.
//! As a result, memory stays linear in the transcript, not quadratic in the
//! attempt count.

use std::{
    fmt::Write as _,
    ops::Range,
    time::Duration,
};

use getset::{
    CopyGetters,
    Getters,
    MutGetters,
    Setters,
};
use rig_agent::agent::CompletionCall;
use rig_core::{
    completion::Usage,
    message::Message,
};
use serde::{
    Deserialize,
    Serialize,
};
use typed_builder::TypedBuilder;

use crate::{
    AppliedFix,
    EXPECT_WRITE,
    Lane,
    revision::Revision,
};

/// The full agent trajectory of one lane (or of a single-prompt
/// [`crate::Runtime::evolve`]).
#[derive(Debug, Clone, Serialize, Deserialize, Getters, CopyGetters, Setters)]
pub struct EvolutionTrace {
    /// Provider route id, for the header's call config.
    #[serde(default)]
    #[getset(get = "pub(super)")]
    provider: String,

    /// The model the Agent uses.
    #[serde(default)]
    #[getset(get = "pub(super)")]
    model: String,

    /// Lane index. It is `0` for single-prompt [`crate::Runtime::evolve`].
    #[getset(get_copy = "pub")]
    lane: Lane,

    /// The system prompt of the agent.
    #[serde(default)]
    #[getset(get = "pub")]
    system_prompt: String,

    /// The prompt the lane started with, before any corrective nudge.
    #[getset(get = "pub")]
    base_prompt: String,

    /// The lane's complete ordered transcript: every user turn (the base
    /// prompt and each nudge), every assistant turn, every tool call and
    /// result.
    ///
    /// Owned once. Each [`AttemptTrace`] indexes into it through
    /// [`RunTrace::produced`]. A context or repeat reset stops sending the
    /// earlier messages to the model, but does not remove them from here. The
    /// transcript keeps everything the lane exchanged.
    #[getset(get = "pub", set = "pub(crate)")]
    history: Vec<Message>,

    /// One entry per lane iteration, in order. An iteration whose inference
    /// call failed outright is present too, with [`AttemptTrace::run`] set to
    /// `None`.
    #[getset(get = "pub")]
    attempts: Vec<AttemptTrace>,

    /// How the lane ended.
    #[getset(get = "pub", set = "pub(crate)")]
    outcome: TraceOutcome,

    /// Wall time of the whole lane.
    #[getset(get_copy = "pub", set = "pub(crate)")]
    duration: Duration,
}

/// One iteration of a lane's self-healing ladder.
#[derive(Debug, Clone, Serialize, Deserialize, Getters, CopyGetters)]
pub struct AttemptTrace {
    /// Position in the lane timeline. Dense: always `0..attempts.len()`.
    #[getset(get_copy = "pub")]
    seq: usize,

    /// The lane's self-healing attempt counter at this iteration. It uses the
    /// same numbering as [`crate::EvolveFailure::attempt`] and the `attempt`
    /// metric label.
    ///
    /// **Not unique across entries.** A transient HTTP retry does not consume
    /// the attempt budget by design. Two entries in sequence can thus carry
    /// the same value. Index by [`Self::seq`]. Report this field.
    #[getset(get_copy = "pub")]
    attempt: usize,

    /// This iteration's user-prompt text: the base prompt, or the corrective
    /// nudge built from the previous failure.
    #[getset(get = "pub")]
    prompt: String,

    /// The agent run, when there was one.
    ///
    /// It is `None` when [`crate::EvolutionAgent::run`] returned an error: a
    /// transient HTTP failure, or a context-size overflow. Such an iteration
    /// has no messages, no usage and no completion calls.
    #[getset(get = "pub")]
    run: Option<RunTrace>,

    /// How far this iteration got through the pipeline, and how long each
    /// stage took.
    #[getset(get = "pub")]
    stages: StageTimings,

    /// What the harness did in response to this iteration.
    #[getset(get = "pub")]
    ladder: LadderEvent,

    /// Wall time of this iteration.
    #[getset(get_copy = "pub")]
    duration: Duration,
}

/// The parts of an iteration that exist only once the agent run succeeded.
#[derive(Debug, Clone, Serialize, Deserialize, Getters, TypedBuilder)]
pub struct RunTrace {
    /// Range into [`EvolutionTrace::history`] of the messages this run
    /// produced: its own user-prompt turn plus every assistant turn and tool
    /// exchange that followed.
    ///
    /// The request this run *sent* was `history[..produced.start + 1]` plus
    /// the system prompt.
    #[getset(get = "pub")]
    produced: Range<usize>,

    /// The run's final assistant text.
    #[getset(get = "pub")]
    response: String,

    /// Aggregate token usage for this run.
    #[getset(get = "pub")]
    usage: Usage,

    /// One entry per HTTP completion request. The blanket
    /// [`crate::EvolutionAgent`] implementation clears the `raw` wire body of
    /// each entry. It keeps `usage`, `finish_reason` and the provider ids.
    #[getset(get = "pub")]
    completion_calls: Vec<CompletionCall>,
}

/// Per-attempt mirror of the
/// [`PIPELINE_STAGE_DURATION`](crate::observability::PIPELINE_STAGE_DURATION)
/// histogram.
///
/// A field is `None` when the attempt failed before it got to that stage.
/// These timings make a statement such as "attempt 3 compiled for 90 seconds,
/// then failed" recoverable from one trace. The metrics give only aggregates.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Getters, Setters, MutGetters,
)]
pub struct StageTimings {
    /// Time spent in the agent run: inference and any tool-calling turns.
    #[getset(get = "pub", set = "pub(crate)")]
    llm: Option<Duration>,

    /// Time spent to parse the Rust code block and validate the signatures.
    #[getset(get = "pub", set = "pub(crate)")]
    parse_validate: Option<Duration>,

    /// The build stage. It runs after the parse and validate stage passes.
    #[getset(get = "pub", set = "pub(crate)", get_mut = "pub(crate)")]
    build: Option<BuildRecord>,

    /// How the response related to the previous candidate: the edits it
    /// applied, if it was an edit (see [`crate::edit`]). `None` for a whole
    /// code block, and for an attempt that never got that far. Absent from
    /// traces written before symbiont 0.36.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[getset(get = "pub", set = "pub(crate)")]
    edits: Option<EditRecord>,
}

impl StageTimings {
    /// Show the stage breakdown of one attempt.
    pub(crate) fn render_stages(&self, out: &mut String) {
        let mut parts: Vec<String> = Vec::new();
        if let Some(llm) = self.llm {
            parts.push(format!("llm {llm:?}"));
        }
        if let Some(parse_validate) = self.parse_validate {
            parts.push(format!("parse/validate {parse_validate:?}"));
        }
        if let Some(edits) = &self.edits {
            parts.push(format!(
                "edits {} ({} anchor(s), {} hunk(s), {} item(s))",
                edits.total(),
                edits.anchors,
                edits.hunks,
                edits.items
            ));
        }
        match &self.build {
            Some(BuildRecord::Built {
                slot_wait,
                compile,
                load,
                autofixes,
            }) => {
                parts.push(format!(
                    "build slot {slot_wait:?}, compile {compile:?}, load {load:?}"
                ));
                if !autofixes.is_empty() {
                    parts.push(format!("{} autofix(es)", autofixes.len()));
                }
            }
            Some(BuildRecord::Deduped {
                slot_wait,
                revision,
                autofixes,
            }) => {
                parts.push(format!(
                    "build slot {slot_wait:?}, deduped onto revision {revision}"
                ));
                if !autofixes.is_empty() {
                    parts.push(format!("{} autofix(es)", autofixes.len()));
                }
            }
            None => {}
        }
        if !parts.is_empty() {
            writeln!(out, "stages: {}", parts.join(", ")).expect(EXPECT_WRITE);
        }
    }
}

/// The edits a response applied to the previous candidate, by form (see
/// [`crate::edit`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditRecord {
    /// `E<n> => text` replacements of a reported error's span.
    pub anchors: usize,
    /// `SEARCH`/`REPLACE` hunks.
    pub hunks: usize,
    /// Top-level items replaced by name from a code block.
    pub items: usize,
}

impl EditRecord {
    /// Every edit the response applied.
    #[must_use]
    pub fn total(&self) -> usize {
        self.anchors + self.hunks + self.items
    }
}

/// What the build stage did with a validated candidate.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BuildRecord {
    /// The runtime compiled the candidate into a dylib and loaded it.
    Built {
        /// Time in the queue for the single build slot, before the work began.
        slot_wait: Duration,
        /// Time spent in `cargo`.
        compile: Duration,
        /// Time spent to copy the dylib, load it, and resolve its symbols.
        load: Duration,
        /// Compiler suggestions the runtime applied to the candidate before
        /// the build that this record times (see
        /// [`crate::Runtime`]'s autofix pass). Empty when the candidate
        /// compiled, or failed, as written. Absent from traces written before
        /// symbiont 0.36.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        autofixes: Vec<AppliedFix>,
    },
    /// The candidate was byte-identical to a registered revision. The runtime
    /// reused that revision and spent no build. There is no compile duration
    /// and no load duration to report.
    Deduped {
        /// Time in the queue for the build slot, before the check ran.
        slot_wait: Duration,
        /// The revision that the runtime reused.
        revision: Revision,
        /// Compiler suggestions applied before the candidate turned out to be
        /// registered already: the text as written failed, the patched text
        /// was a known revision. Empty when the candidate matched as written.
        /// Absent from traces written before symbiont 0.36.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        autofixes: Vec<AppliedFix>,
    },
}

/// One step of the harness's reaction ladder.
///
/// Every [`AttemptTrace`] carries exactly one. The sequence of these events
/// gives the recovery path of the lane, such as `SelfHeal → SelfHeal →
/// Terminal`, or `TransientRetry → SelfHeal → Registered`.
// Tagged `event` and not `kind`: `SelfHeal` carries its own `kind` field, which
// mirrors `EvolveFailure::kind`. Serde forbids a variant field that collides
// with the internal tag.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LadderEvent {
    /// The attempt produced a valid implementation and the runtime registered
    /// it. The lane ends here.
    Registered {
        /// The revision that the runtime registered the implementation under.
        revision: Revision,
    },
    /// The harness fed the failure back to the agent as a corrective nudge.
    SelfHeal {
        /// Failure kind. It uses the same labels as
        /// [`crate::EvolveFailure::kind`]: `no_rust_code`, `parse`,
        /// `max_turns`, `signature`, `unsafe`, `forbidden`, `compile` or
        /// `edit`.
        ///
        /// A `String`, and not the `&'static str` that the producing side
        /// holds. A persisted trace must deserialize without a borrow from the
        /// input.
        kind: String,
        /// The diagnostics that the harness quoted back to the agent.
        diagnostics: String,
        /// The host types whose definitions the nudge attached, because the
        /// compiler reported a method, field or name they do not have (see
        /// `api_hints`). Empty for every other failure kind, and for a compile
        /// failure that named no host type. Absent from traces written before
        /// symbiont 0.36.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        api_hints: Vec<String>,
    },
    /// A transient inference error. The lane retries the same prompt after
    /// `backoff`. This does not consume the attempt budget.
    TransientRetry {
        /// The time the lane waited before it retried.
        backoff: Duration,
        /// The error that caused the retry.
        cause: String,
    },
    /// The request exceeded the model's context window. The lane restarted
    /// from the base prompt.
    ContextReset {
        /// The count of chat messages that the lane stopped sending. They stay
        /// in [`EvolutionTrace::history`].
        messages_dropped: usize,
        /// The one-line error summary that the restart prompt carries.
        brief: String,
    },
    /// The agent repeated its rejected code word for word. The lane restarted
    /// from the base prompt with a do-not-repeat instruction. That instruction
    /// does not quote the rejected code.
    RepeatReset {
        /// The count of chat messages that the lane stopped sending. They stay
        /// in [`EvolutionTrace::history`].
        messages_dropped: usize,
        /// The one-line error summary that the restart prompt carries.
        brief: String,
    },
    /// The failure of this attempt ended the lane.
    Terminal {
        /// The final error.
        reason: String,
    },
}

impl LadderEvent {
    /// A one-line summary of a ladder decision.
    pub(crate) fn render_ladder(&self) -> String {
        use LadderEvent::*;
        match self {
            Registered { revision } => format!("registered revision {revision}"),
            SelfHeal {
                kind,
                diagnostics,
                api_hints,
            } if api_hints.is_empty() => {
                format!("self-heal ({kind}): {}", first_line(diagnostics))
            }
            SelfHeal {
                kind,
                diagnostics,
                api_hints,
            } => format!(
                "self-heal ({kind}, docs attached for {}): {}",
                api_hints.join(", "),
                first_line(diagnostics)
            ),
            TransientRetry { backoff, cause } => {
                format!("transient retry in {backoff:?}: {}", first_line(cause))
            }
            ContextReset {
                messages_dropped, ..
            } => format!("context reset, dropped {messages_dropped} message(s)"),
            RepeatReset {
                messages_dropped, ..
            } => format!("repeat reset, dropped {messages_dropped} message(s)"),
            Terminal { reason } => format!("terminal: {}", first_line(reason)),
        }
    }
}

/// How a lane ended.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceOutcome {
    /// The runtime registered a revision.
    Registered {
        /// The revision that the lane produced.
        revision: Revision,
    },
    /// The lane gave up.
    Failed {
        /// The final error.
        reason: String,
    },
}

impl EvolutionTrace {
    /// Start a trace for `lane`, which begins from `base_prompt`.
    pub(crate) fn new(
        provider: String,
        model: String,
        lane: Lane,
        system_prompt: String,
        base_prompt: String,
    ) -> Self {
        Self {
            provider,
            model,
            lane,
            system_prompt,
            base_prompt,
            history: Vec::new(),
            attempts: Vec::new(),
            outcome: TraceOutcome::Failed {
                // The lane replaces this when it ends. A lane that panics in
                // flight never gives up its trace at all.
                reason: "lane did not finish".to_string(),
            },
            duration: Duration::ZERO,
        }
    }

    /// A trace for a failure that occurred outside any lane. Such a failure
    /// has no trajectory to report.
    pub(crate) fn empty() -> Self {
        Self {
            outcome: TraceOutcome::Failed {
                reason: "failed before the lane started".to_string(),
            },
            ..Self::new(
                String::new(),
                String::new(),
                Lane::from(0),
                String::new(),
                String::new(),
            )
        }
    }

    /// Append an attempt and assign its [`AttemptTrace::seq`].
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

    /// The total token usage of every attempt that got to the model.
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

    /// The count of HTTP completion requests that the lane made.
    #[must_use]
    pub fn completion_calls(&self) -> usize {
        self.attempts
            .iter()
            .filter_map(|attempt| attempt.run.as_ref())
            .map(|run| run.completion_calls.len())
            .sum()
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
            provider: "sglang".to_string(),
            model: "Qwen/Qwen3.8-27B-FP8".to_string(),
            lane: Lane::from(0),
            system_prompt: String::new(),
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

    /// The usage of a lane is the sum of the attempts that got to the model.
    /// An attempt whose inference call failed adds nothing.
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

    /// The trace round-trips through serde, so a persisted trace can be read
    /// back for offline analysis. A trace persisted before the `provider`,
    /// `model` and `system_prompt` fields existed (0.33/0.34) deserializes
    /// too, with the fields defaulting to empty.
    #[test]
    fn serde_round_trip() {
        let mut trace = trace_with(
            vec![attempt(0, 1, Some(run_with(usage_of(1, 1), 1)))],
            TraceOutcome::Registered {
                revision: Revision::new(7),
            },
        );
        trace.history = vec![Message::user("hi")];

        let json = serde_json::to_string(&trace).expect("a trace serializes");
        let back: EvolutionTrace =
            serde_json::from_str(&json).expect("a serialized trace deserializes");

        assert_eq!(back.provider(), trace.provider());
        assert_eq!(back.model(), trace.model());
        assert_eq!(back.system_prompt(), trace.system_prompt());
        assert_eq!(back.attempts.len(), 1);
        assert_eq!(back.history.len(), 1);
        assert!(matches!(
            back.outcome(),
            TraceOutcome::Registered { revision } if revision == &Revision::new(7)
        ));

        // Strip the fields to reproduce a 0.33/0.34 persisted trace.
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        let object = value.as_object_mut().expect("a trace is an object");
        assert!(object.remove("provider").is_some());
        assert!(object.remove("model").is_some());
        assert!(object.remove("system_prompt").is_some());

        let old: EvolutionTrace = serde_json::from_value(value).expect("an old trace deserializes");
        assert_eq!(old.provider(), "");
        assert_eq!(old.model(), "");
        assert_eq!(old.system_prompt(), "");
    }

    /// The repair-round records added in 0.36 (`edits`, `autofixes`,
    /// `api_hints`) are absent from a 0.35 trace and must read back as empty.
    /// The other way round, a trace without any repair activity serializes
    /// without them, so the persisted shape of a plain lane did not change.
    #[test]
    fn repair_records_are_optional_in_the_wire_shape() {
        let stages_0_35 = serde_json::json!({
            "llm": { "secs": 1, "nanos": 0 },
            "parse_validate": { "secs": 0, "nanos": 500 },
            "build": { "kind": "built", "slot_wait": { "secs": 0, "nanos": 0 },
                       "compile": { "secs": 3, "nanos": 0 }, "load": { "secs": 0, "nanos": 10 } }
        });
        let stages: StageTimings =
            serde_json::from_value(stages_0_35).expect("a 0.35 stages record deserializes");
        assert!(stages.edits().is_none());
        assert!(matches!(
            stages.build(),
            Some(BuildRecord::Built { autofixes, .. }) if autofixes.is_empty()
        ));
        let back = serde_json::to_value(&stages).expect("serializes");
        assert!(back.get("edits").is_none(), "{back}");
        assert!(back["build"].get("autofixes").is_none(), "{back}");

        let heal_0_35 = serde_json::json!({
            "event": "self_heal", "kind": "compile", "diagnostics": "error[E0308]"
        });
        let heal: LadderEvent =
            serde_json::from_value(heal_0_35).expect("a 0.35 self-heal deserializes");
        assert!(matches!(&heal, LadderEvent::SelfHeal { api_hints, .. } if api_hints.is_empty()));
        assert!(
            serde_json::to_value(&heal)
                .expect("serializes")
                .get("api_hints")
                .is_none()
        );

        // With repair activity, every record is on the wire.
        let mut stages = StageTimings::default();
        stages.set_edits(Some(EditRecord {
            anchors: 2,
            hunks: 1,
            items: 0,
        }));
        let value = serde_json::to_value(&stages).expect("serializes");
        assert_eq!(value["edits"]["anchors"], 2);
        assert_eq!(value["edits"]["hunks"], 1);
        let heal = LadderEvent::SelfHeal {
            kind: "compile".to_string(),
            diagnostics: String::new(),
            api_hints: vec!["Account".to_string()],
        };
        assert_eq!(
            serde_json::to_value(&heal).expect("serializes")["api_hints"][0],
            "Account"
        );
        assert!(heal.render_ladder().contains("docs attached for Account"));
    }

    /// `push_attempt` assigns a dense `seq` even when `attempt` repeats. The
    /// `attempt` counter repeats across a transient retry.
    #[test]
    fn seq_is_dense_while_attempt_may_repeat() {
        let mut trace = EvolutionTrace::new(
            "sglang".to_string(),
            "Qwen/Qwen3.8-27B-FP8".to_string(),
            Lane::from(2),
            "system".to_string(),
            "base".to_string(),
        );
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
}
