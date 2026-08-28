use std::{
    io::{
        self,
        Write,
    },
    time::Duration,
};

use getset::CopyGetters;
use rig_core::message::{
    AssistantContent,
    Message,
    ReasoningContent,
    ToolResultContent,
    UserContent,
};
use serde_json::{
    Value,
    json,
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
            AgentPreset,
            DshHeader,
        },
    },
    evolution_trace::{
        render_ladder,
        render_stages,
    },
};

/// The append-only session log under construction.
#[derive(CopyGetters)]
pub(super) struct Log<W: Write> {
    out: W,
    /// Next event `seq`. The harness requires it dense from zero: it rejects a
    /// log whose event `seq` differs from the event's index.
    seq: u64,
    /// Timestamp cursor, in Unix epoch milliseconds.
    #[getset(get_copy = "pub(super)")]
    time_ms: u64,
    /// Counter behind the synthetic message ids.
    next_id: u64,
}

impl<W: Write> Log<W> {
    pub(super) fn new(out: W, time_ms: u64) -> Self {
        Log {
            out,
            seq: 0,
            time_ms,
            next_id: 0,
        }
    }

    /// Write the `session` header record: the log's first line.
    pub(super) fn header(
        &mut self,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
    ) -> io::Result<()> {
        let header = serde_json::to_value(
            DshHeader::builder()
                .id(session.resolved_id(trace))
                .created_at(self.time_ms)
                .delegation_depth(0)
                .agent_preset(AgentPreset::Standard)
                .cwd(session.cwd())
                .build(),
        )?;
        writeln!(self.out, "{header}")
    }

    /// Write one attempt as one turn.
    pub(super) fn attempt(
        &mut self,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
        attempt: &AttemptTrace,
        turn: u64,
    ) -> io::Result<()> {
        let attempt_start = self.time_ms;
        // The model time of the attempt, split evenly over the turns it took.
        // The trace times the agent run as a whole, not each turn inside it,
        // so an even split is the most this can honestly claim.
        let step_llm = attempt
            .stages()
            .llm()
            .map(|llm| llm / assistant_turns(trace, attempt).max(1));

        self.event("turn/start", json!({ "turn": turn }))?;

        let mut step = 1;
        self.event("step/start", json!({ "turn": turn, "step": step }))?;

        // The header is log-only and must sit inside an open turn. It does not
        // change across a lane, so the first turn is the only one to carry it.
        if turn == 1 {
            self.request_header(session)?;
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
            self.user_text(attempt.prompt())?;
        } else {
            let mut step_has_assistant = false;
            let mut call_index = 0;
            for message in messages {
                match message {
                    // The system prompt travels as agent configuration, so a
                    // `System` turn never reaches a lane transcript.
                    Message::System { .. } => {}
                    Message::User { content } => {
                        for item in content.iter() {
                            match item {
                                UserContent::ToolResult(result) => self.tool_result(
                                    turn,
                                    step,
                                    result.call.as_str(),
                                    &result
                                        .content
                                        .iter()
                                        .map(tool_result_block)
                                        .collect::<Vec<_>>(),
                                )?,
                                other => self.user_text(&user_text(other))?,
                            }
                        }
                    }
                    Message::Assistant { content, .. } => {
                        // One step is one model call plus the tool executions
                        // it asked for. A second assistant turn opens the next
                        // step: the harness clears the step's outstanding tool
                        // calls at `step/end`.
                        if step_has_assistant {
                            self.event("step/end", json!({ "turn": turn, "step": step }))?;
                            step += 1;
                            self.event("step/start", json!({ "turn": turn, "step": step }))?;
                        }
                        step_has_assistant = true;

                        // The harness measures model time as `step/start` to
                        // `assistant/message`, so the wait for this turn lands
                        // between them.
                        if let Some(llm) = step_llm {
                            self.advance(llm);
                        }

                        let blocks: Vec<Value> = content.iter().map(assistant_block).collect();
                        // One completion call per assistant turn, in order, so
                        // the step's own token accounting travels with the
                        // message the harness folds it out of.
                        let usage = attempt
                            .run()
                            .as_ref()
                            .and_then(|run| run.completion_calls().get(call_index))
                            .and_then(|call| token_usage(&call.usage));
                        call_index += 1;
                        self.assistant_message(turn, step, session, &blocks, usage)?;

                        for item in content.iter() {
                            if let AssistantContent::ToolCall(call) = item {
                                self.event(
                                    "tool/call",
                                    json!({
                                        "turn": turn,
                                        "step": step,
                                        "callId": call.id.as_str(),
                                        "name": call.function.name,
                                        "arguments": call.function.arguments.to_string(),
                                    }),
                                )?;
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

        self.notice(&attempt_notice(attempt))?;
        self.event("step/end", json!({ "turn": turn, "step": step }))?;
        self.event(
            "turn/end",
            json!({ "turn": turn, "reason": turn_end_reason(attempt.ladder()) }),
        )
    }

    /// Write a closing turn that carries the lane's outcome.
    pub(super) fn outcome_turn(
        &mut self,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
        turn: u64,
    ) -> io::Result<()> {
        self.event("turn/start", json!({ "turn": turn }))?;
        self.event("step/start", json!({ "turn": turn, "step": 1 }))?;
        if turn == 1 {
            // A trace with no attempt at all still needs its header.
            self.request_header(session)?;
        }
        self.session_title(trace)?;
        self.notice(&outcome_notice(trace))?;
        self.event("step/end", json!({ "turn": turn, "step": 1 }))?;
        self.event(
            "turn/end",
            json!({ "turn": turn, "reason": { "kind": "completed" } }),
        )
    }

    /// Name the session.
    ///
    /// Without this event the web UI has no title to show and falls back to
    /// the basename of the session's working directory, so every lane of one
    /// example renders under the same heading.
    ///
    /// Written once, at the end, because the title reports the outcome. The
    /// durable title is the latest snapshot, so one event settles it.
    fn session_title(&mut self, trace: &EvolutionTrace) -> io::Result<()> {
        self.event(
            "session/title",
            json!({
                "title": session_title(trace),
                // `dsh-session-title`'s invariant: `messageSeqs` is empty if
                // and only if the source is `user`. A title symbiont chose is
                // a deliberate name, not one derived from a message, so it
                // cites none.
                "messageSeqs": [],
                "source": { "kind": "user" },
            }),
        )
    }

    /// Write the request header and, when known, the route's context window.
    fn request_header(&mut self, session: &DshSession<'_>) -> io::Result<()> {
        self.event(
            "request/header",
            json!({
                "header": {
                    "config": { "provider": session.provider(), "model": session.model() },
                    "system": session.system_prompt(),
                },
                "reason": "initial",
            }),
        )?;

        if let Some(window) = session.context_window() {
            self.event(
                "request/context",
                json!({
                    "provider": session.provider(),
                    "model": session.model(),
                    "contextWindow": window,
                }),
            )?;
        }
        Ok(())
    }

    /// Write a plain user turn.
    fn user_text(&mut self, text: &str) -> io::Result<()> {
        let id = self.mint_id("msg");
        self.surface_event(
            "user/message",
            json!({
                "id": id,
                "role": "user",
                "content": [{ "type": "text", "text": text }],
                "source": { "kind": "user" },
            }),
        )
    }

    /// Write a one-line harness note: what symbiont did, not what the model
    /// said. The harness collapses a `notice` to its `summary` until the
    /// reader expands the row.
    fn notice(&mut self, text: &str) -> io::Result<()> {
        let id = self.mint_id("note");
        self.surface_event(
            "user/message",
            json!({
                "id": id,
                "role": "user",
                "content": [{ "type": "text", "text": text }],
                "source": {
                    "kind": "plugin",
                    "plugin": "symbiont",
                    "form": "notice",
                    "summary": summary_line(text),
                },
            }),
        )
    }

    /// Write one assembled assistant turn.
    fn assistant_message(
        &mut self,
        turn: u64,
        step: u64,
        session: &DshSession<'_>,
        blocks: &[Value],
        usage: Option<Value>,
    ) -> io::Result<()> {
        let id = self.mint_id("msg");
        let mut data = json!({
            "turn": turn,
            "step": step,
            "message": {
                "id": id,
                "role": "assistant",
                "content": blocks,
                "source": {
                    "kind": "model",
                    "provider": session.provider(),
                    "model": session.model(),
                },
            },
        });
        // Absent, not zeroed, when the provider reported no accounting: the
        // harness reads a missing `usage` as "unreported" and a present one as
        // a measurement.
        if let Some(usage) = usage {
            data["usage"] = usage;
        }
        self.surface_event("assistant/message", data)
    }

    /// Write one tool result, answering the call `call_id`.
    fn tool_result(
        &mut self,
        turn: u64,
        step: u64,
        call_id: &str,
        blocks: &[Value],
    ) -> io::Result<()> {
        let id = self.mint_id("msg");
        self.surface_event(
            "tool/result",
            json!({
                "turn": turn,
                "step": step,
                "message": {
                    "id": id,
                    "role": "user",
                    "content": [{
                        "type": "tool-result",
                        "toolCallId": call_id,
                        "content": blocks,
                        "isError": false,
                    }],
                    "source": { "kind": "tool", "callId": call_id },
                },
            }),
        )
    }

    /// Append one log-only event.
    fn event(&mut self, kind: &str, data: Value) -> io::Result<()> {
        self.write_event(kind, data, false)
    }

    /// Append one event that also lands on the ordered transcript.
    fn surface_event(&mut self, kind: &str, data: Value) -> io::Result<()> {
        self.write_event(kind, data, true)
    }

    /// Serialize one event record and advance the `seq` and time cursors.
    fn write_event(&mut self, kind: &str, data: Value, on_surface: bool) -> io::Result<()> {
        let mut event = json!({
            "type": kind,
            "seq": self.seq,
            "time": self.time_ms,
            "data": data,
        });
        if on_surface {
            event["surfaceOp"] = json!("append");
        }
        self.seq += 1;
        writeln!(self.out, "{event}")
    }

    /// Move the clock forward by `duration`, rounded up to the millisecond the
    /// harness records.
    ///
    /// Rounding up rather than down keeps a sub-millisecond stage from folding
    /// to a zero-length one, which would make a step look like it never ran.
    fn advance(&mut self, duration: Duration) {
        let millis = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
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
                    .filter(|message| matches!(message, Message::Assistant { .. }))
                    .count(),
            )
            .unwrap_or(u32::MAX)
        })
}

/// One assistant content item as a harness content block.
fn assistant_block(content: &AssistantContent) -> Value {
    use AssistantContent::*;
    match content {
        Text(text) => json!({ "type": "text", "text": text.text }),
        Reasoning(reasoning) => {
            let text: String = reasoning
                .content
                .iter()
                .filter_map(|item| match item {
                    ReasoningContent::Text { text, .. } => Some(text.as_str()),
                    ReasoningContent::Summary(summary) => Some(summary.as_str()),
                    // Opaque provider payloads carry no readable text.
                    ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => None,
                })
                .collect();
            json!({ "type": "reasoning", "text": text })
        }
        ToolCall(call) => json!({
            "type": "tool-call",
            "id": call.id.as_str(),
            "name": call.function.name,
            // The harness wants the model's raw argument JSON, as a string.
            "arguments": call.function.arguments.to_string(),
        }),
        // A harness image block references an attachment the harness owns, so
        // an image out of a rig transcript has no faithful counterpart.
        Image(_) => json!({ "type": "text", "text": "[image omitted]" }),
    }
}

/// One tool-result item as a harness content block.
fn tool_result_block(content: &ToolResultContent) -> Value {
    use ToolResultContent::*;
    match content {
        Text(text) => json!({ "type": "text", "text": text.text }),
        Json { value } => json!({ "type": "text", "text": value.to_string() }),
        Image(_) => json!({ "type": "text", "text": "[image omitted]" }),
    }
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
        render_ladder(attempt.ladder()),
    );
    render_stages(&mut out, attempt.stages());
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
fn turn_end_reason(ladder: &LadderEvent) -> Value {
    use LadderEvent::*;
    match ladder {
        // The model answered; the harness then chose what to do with the
        // answer. That choice is not a failure of the turn.
        Registered { .. } | SelfHeal { .. } | ContextReset { .. } | RepeatReset { .. } => {
            json!({ "kind": "completed" })
        }
        TransientRetry { cause, .. } => {
            json!({ "kind": "error", "error": { "message": cause, "code": "UNKNOWN" } })
        }
        Terminal { reason } => {
            json!({ "kind": "error", "error": { "message": reason, "code": "UNKNOWN" } })
        }
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
