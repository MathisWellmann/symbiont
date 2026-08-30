// SPDX-License-Identifier: MPL-2.0
//! Build the [`LogLine`] records of one harness session out of an
//! [`EvolutionTrace`].
//!
//! Every record is a typed value of [`crate::dsh::types`], so the compiler,
//! and not a reviewer, checks the export against the format. [`Log`] owns the
//! three cursors a session log has to keep dense and monotonic: the event
//! `seq`, the wall clock, and the message-id counter.

use std::time::Duration;

use getset::CopyGetters;
use rig_core::message::{
    AssistantContent,
    Message as RigMessage,
    ReasoningContent,
    ToolResultContent,
    UserContent,
};

use crate::{
    AttemptTrace,
    DshSession,
    EvolutionTrace,
    LadderEvent,
    TraceOutcome,
    dsh::{
        millis_of,
        token_usage,
        types::{
            AssistantMessageData,
            ContentBlock,
            ContextForm,
            EpochHeader,
            Event,
            LlmCallConfig,
            LlmFailure,
            LogLine,
            Message,
            MessageSource,
            RequestContextData,
            RequestHeaderData,
            RequestHeaderReason,
            Role,
            SessionHeaderLine,
            SessionTitleData,
            StepStartData,
            TitleSource,
            TokenUsage,
            ToolCallData,
            ToolResultData,
            TurnEndData,
            TurnEndReason,
            TurnStartData,
        },
    },
};

/// The plugin name that every harness note of this crate is attributed to.
const PLUGIN: &str = "symbiont";

/// The append-only session log under construction.
#[derive(CopyGetters)]
pub(super) struct Log {
    /// The records emitted so far, in log order.
    lines: Vec<LogLine>,
    /// Next event `seq`. The harness requires it dense from zero: it rejects a
    /// log whose event `seq` differs from the event's index.
    seq: u64,
    /// Timestamp cursor, in Unix epoch milliseconds.
    #[getset(get_copy = "pub(super)")]
    time_ms: u64,
    /// Counter behind the synthetic message ids.
    next_id: u64,
}

impl Log {
    pub(super) fn new(time_ms: u64) -> Self {
        Log {
            lines: Vec::new(),
            seq: 0,
            time_ms,
            next_id: 0,
        }
    }

    /// The finished log, in order.
    pub(super) fn into_lines(self) -> Vec<LogLine> {
        self.lines
    }

    /// Write the `session` header record: the log's first line.
    pub(super) fn header(&mut self, trace: &EvolutionTrace, session: &DshSession<'_>) {
        self.lines.push(LogLine::Session(SessionHeaderLine {
            version: 0,
            id: session.resolved_id(trace),
            created_at: self.time_ms,
            cwd: session.cwd().map(ToString::to_string),
            parent_session: None,
            seed_length: None,
            origin: None,
            delegation_depth: 0,
            agent_preset: Some("standard".to_string()),
        }));
    }

    /// Write one attempt as one turn.
    pub(super) fn attempt(
        &mut self,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
        attempt: &AttemptTrace,
        turn: u64,
    ) {
        let attempt_start = self.time_ms;
        // The model time of the attempt, split evenly over the turns it took.
        // The trace times the agent run as a whole, not each turn inside it,
        // so an even split is the most this can honestly claim.
        let step_llm = attempt
            .stages()
            .llm()
            .map(|llm| llm / assistant_turns(trace, attempt).max(1));

        let event = self.event(TurnStartData { turn });
        self.lines.push(LogLine::TurnStart(event));

        let mut step = 1;
        self.step_start(turn, step);

        // The header is log-only and must sit inside an open turn. It does not
        // change across a lane, so the first turn is the only one to carry it.
        if turn == 1 {
            self.request_header(trace, session);
        }

        let messages = attempt
            .run()
            .as_ref()
            .and_then(|run| trace.history().get(run.produced().clone()))
            .unwrap_or_default();

        if messages.is_empty() {
            // Either the inference call itself failed, so the attempt produced
            // no transcript at all, or the trace carries no history. Show the
            // prompt that was sent regardless.
            self.user_text(attempt.prompt());
        } else {
            let mut step_has_assistant = false;
            let mut call_index = 0;
            for message in messages {
                match message {
                    // The system prompt travels as agent configuration, so a
                    // `System` turn never reaches a lane transcript.
                    RigMessage::System { .. } => {}
                    RigMessage::User { content } => {
                        for item in content.iter() {
                            match item {
                                UserContent::ToolResult(result) => self.tool_result(
                                    turn,
                                    step,
                                    result.call.as_str(),
                                    result.content.iter().map(tool_result_block).collect(),
                                ),
                                other => self.user_text(&user_text(other)),
                            }
                        }
                    }
                    RigMessage::Assistant { content, .. } => {
                        // One step is one model call plus the tool executions
                        // it asked for. A second assistant turn opens the next
                        // step: the harness clears the step's outstanding tool
                        // calls at `step/end`.
                        if step_has_assistant {
                            self.step_end(turn, step);
                            step += 1;
                            self.step_start(turn, step);
                        }
                        step_has_assistant = true;

                        // The harness measures model time as `step/start` to
                        // `assistant/message`, so the wait for this turn lands
                        // between them.
                        if let Some(llm) = step_llm {
                            self.advance(llm);
                        }

                        let blocks: Vec<ContentBlock> =
                            content.iter().map(assistant_block).collect();
                        // One completion call per assistant turn, in order, so
                        // the step's own token accounting travels with the
                        // message the harness folds it out of.
                        let usage = attempt
                            .run()
                            .as_ref()
                            .and_then(|run| run.completion_calls().get(call_index))
                            .and_then(|call| token_usage(&call.usage));
                        call_index += 1;
                        self.assistant_message(turn, step, trace, session, blocks, usage);

                        for item in content.iter() {
                            if let AssistantContent::ToolCall(call) = item {
                                let event = self.event(ToolCallData {
                                    turn,
                                    step,
                                    call_id: call.id.to_string(),
                                    name: call.function.name.clone(),
                                    arguments: call.function.arguments.to_string(),
                                });
                                self.lines.push(LogLine::ToolCall(event));
                            }
                        }
                    }
                }
            }
        }

        // Everything after the model answered — parse, validate, build — is
        // wall time of this attempt that no event of its own reports, so the
        // turn closes on the attempt's measured duration.
        self.advance_to(attempt_start.saturating_add(millis_of(attempt.duration())));

        self.notice(&attempt_notice(attempt));
        self.step_end(turn, step);
        self.turn_end(turn, turn_end_reason(attempt.ladder()));
    }

    /// Write a closing turn that carries the lane's outcome.
    pub(super) fn outcome_turn(
        &mut self,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
        turn: u64,
    ) {
        let event = self.event(TurnStartData { turn });
        self.lines.push(LogLine::TurnStart(event));
        self.step_start(turn, 1);
        if turn == 1 {
            // A trace with no attempt at all still needs its header.
            self.request_header(trace, session);
        }
        self.session_title(trace);
        self.notice(&outcome_notice(trace));
        self.step_end(turn, 1);
        self.turn_end(turn, TurnEndReason::Completed);
    }

    /// Open a step.
    fn step_start(&mut self, turn: u64, step: u64) {
        let event = self.event(StepStartData { turn, step });
        self.lines.push(LogLine::StepStart(event));
    }

    /// Close a step.
    fn step_end(&mut self, turn: u64, step: u64) {
        let event = self.event(StepStartData { turn, step });
        self.lines.push(LogLine::StepEnd(event));
    }

    /// Close a turn.
    fn turn_end(&mut self, turn: u64, reason: TurnEndReason) {
        let event = self.event(TurnEndData { turn, reason });
        self.lines.push(LogLine::TurnEnd(event));
    }

    /// Name the session.
    ///
    /// Without this event the web UI has no title to show and falls back to
    /// the basename of the session's working directory, so every lane of one
    /// example renders under the same heading.
    ///
    /// Written once, at the end, because the title reports the outcome. The
    /// durable title is the latest snapshot, so one event settles it.
    fn session_title(&mut self, trace: &EvolutionTrace) {
        let event = self.event(SessionTitleData {
            title: session_title(trace),
            // `dsh-session-title`'s invariant: `messageSeqs` is empty if and
            // only if the source is `user`. A title symbiont chose is a
            // deliberate name, not one derived from a message, so it cites
            // none.
            message_seqs: Vec::new(),
            source: TitleSource::User,
        });
        self.lines.push(LogLine::SessionTitle(event));
    }

    /// Write the request header and, when known, the route's context window.
    fn request_header(&mut self, trace: &EvolutionTrace, session: &DshSession<'_>) {
        let event = self.event(RequestHeaderData {
            header: EpochHeader {
                config: LlmCallConfig {
                    provider: trace.provider().to_string(),
                    model: session.model().to_string(),
                    reasoning_effort: None,
                    temperature: None,
                    max_tokens: None,
                    stop: None,
                },
                adapter_defaults: None,
                system: Some(trace.system_prompt().to_string()),
                // The tool schemas live on the caller's agent, not in the
                // trace, so the header claims none rather than an empty set.
                tools: None,
            },
            reason: RequestHeaderReason::Initial,
        });
        self.lines.push(LogLine::RequestHeader(event));

        if let Some(window) = session.context_window() {
            let event = self.event(RequestContextData {
                provider: trace.provider().to_string(),
                model: session.model().to_string(),
                context_window: Some(window),
            });
            self.lines.push(LogLine::RequestContext(event));
        }
    }

    /// Write a plain user turn.
    fn user_text(&mut self, text: &str) {
        let message = self.message(
            Role::User,
            vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            MessageSource::user(),
        );
        let event = self.event(message).on_surface();
        self.lines.push(LogLine::UserMessage(event));
    }

    /// Write a one-line harness note: what symbiont did, not what the model
    /// said. The harness collapses a `notice` to its `summary` until the
    /// reader expands the row.
    fn notice(&mut self, text: &str) {
        let source = MessageSource::plugin(
            PLUGIN,
            ContextForm::Notice {
                summary: summary_line(text),
            },
        );
        let message = self.note(
            vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            source,
        );
        let event = self.event(message).on_surface();
        self.lines.push(LogLine::UserMessage(event));
    }

    /// Write one assembled assistant turn.
    fn assistant_message(
        &mut self,
        turn: u64,
        step: u64,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
        content: Vec<ContentBlock>,
        usage: Option<TokenUsage>,
    ) {
        let message = self.message(
            Role::Assistant,
            content,
            MessageSource::model(&trace.provider(), session.model()),
        );
        // `usage` is absent, not zeroed, when the provider reported no
        // accounting: the harness reads a missing `usage` as "unreported" and
        // a present one as a measurement.
        let event = self
            .event(AssistantMessageData {
                turn,
                step,
                message,
                usage,
            })
            .on_surface();
        self.lines.push(LogLine::AssistantMessage(event));
    }

    /// Write one tool result, answering the call `call_id`.
    fn tool_result(&mut self, turn: u64, step: u64, call_id: &str, content: Vec<ContentBlock>) {
        let message = self.message(
            Role::User,
            vec![ContentBlock::ToolResult {
                tool_call_id: call_id.to_string(),
                content,
                is_error: Some(false),
            }],
            MessageSource::tool(call_id),
        );
        let event = self
            .event(ToolResultData {
                turn,
                step,
                message,
                error: None,
                meta: None,
            })
            .on_surface();
        self.lines.push(LogLine::ToolResult(event));
    }

    /// One message with a freshly minted id.
    fn message(
        &mut self,
        role: Role,
        content: Vec<ContentBlock>,
        source: MessageSource,
    ) -> Message {
        Message {
            id: self.mint_id("msg"),
            role,
            content,
            source,
        }
    }

    /// One harness note. Its id is minted from a separate prefix, so a reader
    /// can tell a note from a message the lane actually exchanged.
    fn note(&mut self, content: Vec<ContentBlock>, source: MessageSource) -> Message {
        Message {
            id: self.mint_id("note"),
            role: Role::User,
            content,
            source,
        }
    }

    /// Wrap `data` in an event envelope and advance the `seq` cursor.
    fn event<D>(&mut self, data: D) -> Event<D> {
        // The harness stores `time` signed. A cursor that ever grew past the
        // signed range would be a clock far outside any session's lifetime, so
        // saturating there is the honest reading.
        let time = i64::try_from(self.time_ms).unwrap_or(i64::MAX);
        let event = Event::new(self.seq, time, data);
        self.seq += 1;
        event
    }

    /// Move the clock forward by `duration`, rounded up to the millisecond the
    /// harness records.
    ///
    /// Rounding up rather than down keeps a sub-millisecond stage from folding
    /// to a zero-length one, which would make a step look like it never ran.
    fn advance(&mut self, duration: Duration) {
        let millis = millis_of(duration);
        let millis = if millis == 0 && duration > Duration::ZERO {
            1
        } else {
            millis
        };
        self.time_ms = self.time_ms.saturating_add(millis);
    }

    /// Move the clock to `target`, never backwards.
    ///
    /// A stage that reported more time than the attempt containing it would
    /// otherwise rewind the log, and the harness derives every duration it
    /// shows by subtracting event times.
    pub(super) fn advance_to(&mut self, target: u64) {
        self.time_ms = self.time_ms.max(target);
    }

    /// A fresh message id, unique within the session.
    fn mint_id(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("symbiont-{prefix}-{}", self.next_id)
    }
}

/// How many assistant turns one attempt put into the transcript.
///
/// The model time of the run is divided by this, so a run that answered in
/// three tool-calling turns reports three timed steps instead of one.
fn assistant_turns(trace: &EvolutionTrace, attempt: &AttemptTrace) -> u32 {
    attempt
        .run()
        .as_ref()
        .and_then(|run| trace.history().get(run.produced().clone()))
        .map_or(0, |messages| {
            u32::try_from(
                messages
                    .iter()
                    .filter(|message| matches!(message, RigMessage::Assistant { .. }))
                    .count(),
            )
            .unwrap_or(u32::MAX)
        })
}

/// One assistant content item as a harness content block.
fn assistant_block(content: &AssistantContent) -> ContentBlock {
    use AssistantContent::*;
    match content {
        Text(text) => ContentBlock::Text {
            text: text.text.clone(),
        },
        Reasoning(reasoning) => ContentBlock::Reasoning {
            text: reasoning
                .content
                .iter()
                .filter_map(|item| match item {
                    ReasoningContent::Text { text, .. } => Some(text.as_str()),
                    ReasoningContent::Summary(summary) => Some(summary.as_str()),
                    // Opaque provider payloads carry no readable text.
                    ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => None,
                })
                .collect(),
        },
        ToolCall(call) => ContentBlock::ToolCall {
            id: call.id.to_string(),
            name: call.function.name.clone(),
            // The harness wants the model's raw argument JSON, as a string.
            arguments: call.function.arguments.to_string(),
        },
        // A harness image block references an attachment the harness owns, so
        // an image out of a rig transcript has no faithful counterpart.
        Image(_) => ContentBlock::Text {
            text: "[image omitted]".to_string(),
        },
    }
}

/// One tool-result item as a harness content block.
fn tool_result_block(content: &ToolResultContent) -> ContentBlock {
    use ToolResultContent::*;
    let text = match content {
        Text(text) => text.text.clone(),
        Json { value } => value.to_string(),
        Image(_) => "[image omitted]".to_string(),
    };
    ContentBlock::Text { text }
}

/// The readable text of a non-tool-result user content item.
fn user_text(content: &UserContent) -> String {
    use UserContent::*;
    match content {
        Text(text) => text.text.clone(),
        ToolResult(result) => format!("[tool result from {}]", result.name),
        Image(_) => "[image omitted]".to_string(),
        Audio(_) => "[audio omitted]".to_string(),
        Video(_) => "[video omitted]".to_string(),
        Document(_) => "[document omitted]".to_string(),
    }
}

/// The note that closes one attempt: what the ladder decided, and where the
/// time went.
fn attempt_notice(attempt: &AttemptTrace) -> String {
    let mut out = format!(
        "symbiont attempt {} (seq {}, {:?}): {}\n",
        attempt.attempt(),
        attempt.seq(),
        attempt.duration(),
        attempt.ladder().render_ladder(),
    );
    attempt.stages().render_stages(&mut out);
    out
}

/// The note that closes the lane.
fn outcome_notice(trace: &EvolutionTrace) -> String {
    let usage = trace.usage();
    use TraceOutcome::*;
    let verdict = match trace.outcome() {
        Registered { revision } => format!("registered revision {revision}"),
        Failed { reason } => format!("failed — {reason}"),
    };
    format!(
        "symbiont lane {} finished in {:?}: {verdict}\n{} attempt(s), {} completion call(s), \
         {} input / {} output tokens",
        trace.lane(),
        trace.duration(),
        trace.attempts().len(),
        trace.completion_calls(),
        usage.input_tokens,
        usage.output_tokens,
    )
}

/// Why the turn of `ladder` ended, in the harness's vocabulary.
fn turn_end_reason(ladder: &LadderEvent) -> TurnEndReason {
    use LadderEvent::*;
    let message = match ladder {
        // The model answered; the harness then chose what to do with the
        // answer. That choice is not a failure of the turn.
        Registered { .. } | SelfHeal { .. } | ContextReset { .. } | RepeatReset { .. } => {
            return TurnEndReason::Completed;
        }
        TransientRetry { cause, .. } => cause.clone(),
        Terminal { reason } => reason.clone(),
    };
    TurnEndReason::Error {
        error: LlmFailure {
            message,
            code: "UNKNOWN".to_string(),
            status: None,
            provider_retry_after_ms: None,
            request_id: None,
        },
    }
}

/// The session's heading in the harness UI: which lane this was, and how it
/// ended.
///
/// Leads with the identity rather than the prompt, because the prompt is
/// near-identical across the rounds of one run while the lane index and the
/// outcome are what tell two sessions apart.
fn session_title(trace: &EvolutionTrace) -> String {
    let outcome = match trace.outcome() {
        TraceOutcome::Registered { revision } => format!("rev {revision}"),
        TraceOutcome::Failed { .. } => "failed".to_string(),
    };
    let attempts = trace.attempts().len();
    let plural = if attempts == 1 { "" } else { "s" };

    title_text(&format!(
        "symbiont lane {} · {outcome} · {attempts} attempt{plural}",
        trace.lane(),
    ))
}

/// The collapsed one-line form of a note. The harness bounds a `notice`
/// summary at 120 characters.
fn summary_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    if line.chars().count() <= 120 {
        return line.to_string();
    }
    let head: String = line.chars().take(119).collect();
    format!("{head}…")
}

/// The harness's title budget: 80 UTF-8 bytes.
///
/// A title appended straight to the log never passes through the title
/// service, so it never meets `normalizeSessionTitle`. This applies the same
/// contract at the writing end instead.
const MAX_TITLE_BYTES: usize = 80;

/// Normalize a title the way the harness's title service would: no control
/// characters, whitespace runs collapsed to one space, and truncated to
/// [`MAX_TITLE_BYTES`] on a character boundary.
fn title_text(raw: &str) -> String {
    // A control character becomes a space rather than vanishing, so a title
    // built from a newline-separated source does not run two words together.
    let spaced: String = raw
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let cleaned = spaced.split_whitespace().collect::<Vec<&str>>().join(" ");

    let mut end = cleaned.len().min(MAX_TITLE_BYTES);
    // Never split a code point: walk back to the nearest boundary.
    while end > 0 && !cleaned.is_char_boundary(end) {
        end -= 1;
    }
    cleaned[..end].trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A title never passes through the harness's title service, so this end
    /// has to keep the service's contract: one line, no control characters,
    /// and at most 80 UTF-8 bytes cut on a character boundary.
    #[test]
    fn a_title_is_normalized_and_bounded() {
        assert_eq!(title_text("  two\n\tlines\u{7}here "), "two lines here");

        // 40 two-byte characters: 80 bytes exactly, so nothing is cut.
        let exact = "\u{e9}".repeat(40);
        assert_eq!(title_text(&exact).len(), MAX_TITLE_BYTES);

        // One more character has to go, and the cut must not split it.
        let over = "\u{e9}".repeat(41);
        let bounded = title_text(&over);
        assert!(bounded.len() <= MAX_TITLE_BYTES);
        assert_eq!(bounded.chars().count(), 40);
    }
}
