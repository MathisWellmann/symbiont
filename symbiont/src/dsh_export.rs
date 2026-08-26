// SPDX-License-Identifier: MPL-2.0
//! Export an [`EvolutionTrace`] as a DeepSeek Harness (`dsh`) session log.
//!
//! The harness stores one session as a JSON Lines file: a `session` header
//! record, then one [`SessionEvent`][ev] record per line. `seq` is dense and
//! zero-based, `time` is Unix epoch milliseconds, and the message-producing
//! events (`user/message`, `assistant/message`, `tool/result`) carry a
//! `surfaceOp` that places them on the ordered transcript the UI shows.
//!
//! [ev]: https://github.com/deepseek-ai/deepseek-harness
//!
//! # The mapping
//!
//! | symbiont                                              | dsh                                  |
//! | ------------------------------------------------------| -------------------------------------|
//! | [`DshSession::system_prompt`]                         | `request/header` → `header.system`   |
//! | [`AttemptTrace`]                                      | one turn (`turn/start` … `turn/end`) |
//! | one assistant message plus the tool calls it made     | one step                             |
//! | [`EvolutionTrace::history`] user text                 | `user/message`                       |
//! | history assistant turn                                | `assistant/message`                  |
//! | history [`AssistantContent::ToolCall`]                | `tool/call`                          |
//! | history [`UserContent::ToolResult`]                   | `tool/result`                        |
//! | [`AttemptTrace::ladder`] and [`AttemptTrace::stages`] | a `notice` user message              |
//! | [`EvolutionTrace::outcome`] | a final `notice` user message                                  |
//!
//! # What the trace does not hold
//!
//! Three fields the harness header wants are not in an [`EvolutionTrace`], so
//! [`DshSession`] takes them from the caller:
//!
//! - the **system prompt**, which the trace omits by design (see the
//!   [`crate::EvolutionTrace`] module docs) — pass [`crate::system_prompt`];
//! - the **provider and model names**, which rig's [`CompletionCall`] does not
//!   record;
//! - an **absolute start time**, because a trace holds only [`Duration`]s.
//!   Events are laid out from [`DshSession::started_at`] forward, one
//!   millisecond apart. The ordering is exact; the individual timestamps are
//!   synthetic.
//!
//! [`CompletionCall`]: crate::CompletionCall
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//!
//! use symbiont::{
//!     DocMode,
//!     DshSession,
//!     EvolutionTrace,
//! };
//!
//! async fn save(trace: &EvolutionTrace) -> Result<(), Box<dyn std::error::Error>> {
//!     let prompt = symbiont::system_prompt(None, DocMode::Inline).await?;
//!     let cwd = std::env::current_dir()?;
//!     let cwd = cwd.to_string_lossy();
//!
//!     let session = DshSession::builder()
//!         .system_prompt(&prompt)
//!         .provider("openrouter")
//!         .model("moonshotai/kimi-k3")
//!         .cwd(&cwd)
//!         .build();
//!
//!     let root = Path::new(env!("HOME")).join(".dsh/sessions");
//!     let path = symbiont::export_dsh_session(trace, &session, &root)?;
//!     println!("wrote {}", path.display());
//!     Ok(())
//! }
//! ```

#[cfg(feature = "dsh-export")]
use std::fmt::Write as _;
use std::{
    io::{
        self,
        Write,
    },
    time::{
        SystemTime,
        UNIX_EPOCH,
    },
};

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
use typed_builder::TypedBuilder;

use crate::evolution_trace::{
    AttemptTrace,
    EvolutionTrace,
    LadderEvent,
    TraceOutcome,
    render_ladder,
    render_stages,
};

/// The on-disk session-format version this exporter writes. The harness
/// refuses any other value outright, before it looks at the header shape.
const SESSION_FORMAT_VERSION: u32 = 0;

/// The metadata a DeepSeek Harness session needs that an [`EvolutionTrace`]
/// does not carry. See the [module docs](self) for why each one is here.
#[derive(Debug, Clone, TypedBuilder)]
pub struct DshSession<'a> {
    /// The rendered system prompt, shown in the session's request header.
    /// Pass what [`crate::system_prompt`] returned for the run.
    system_prompt: &'a str,

    /// Provider route id, for the header's call config.
    #[builder(default = "symbiont")]
    provider: &'a str,

    /// Model id, for the header's call config.
    #[builder(default = "unknown")]
    model: &'a str,

    /// Working directory the run happened in. It decides the project
    /// directory the harness files the session under, and the directory the
    /// picker shows. `None` files the session under `_no-cwd`.
    #[builder(default, setter(strip_option))]
    cwd: Option<&'a str>,

    /// Wall-clock instant the lane started. Event timestamps count up from
    /// here.
    #[builder(default = SystemTime::now())]
    started_at: SystemTime,

    /// Session id. It must be unique inside the project directory. `None`
    /// derives `session-symbiont-lane<lane>-<millis>` from the trace.
    #[builder(default, setter(strip_option))]
    session_id: Option<String>,

    /// The model's advertised context window, for the `request/context`
    /// record. Omitted when `None`.
    #[builder(default, setter(strip_option))]
    context_window: Option<u64>,
}

impl DshSession<'_> {
    /// The session id to file this trace under.
    #[must_use]
    pub fn resolved_id(&self, trace: &EvolutionTrace) -> String {
        self.session_id.clone().unwrap_or_else(|| {
            format!(
                "session-symbiont-lane{}-{}",
                trace.lane(),
                epoch_millis(self.started_at),
            )
        })
    }
}

/// Write `trace` as a DeepSeek Harness session log in plain JSON Lines.
///
/// The harness reads a *zstd-compressed* artifact under its normal
/// configuration, so prefer [`export_dsh_session`], which also puts the file
/// where the picker finds it. Use this one to inspect the records, to pipe
/// them elsewhere, or to compress them yourself.
///
/// # Errors
///
/// Returns the write errors of `out`.
pub fn write_dsh_session<W: Write>(
    trace: &EvolutionTrace,
    session: &DshSession<'_>,
    out: W,
) -> io::Result<()> {
    let mut log = Log {
        out,
        seq: 0,
        time_ms: epoch_millis(session.started_at),
        next_id: 0,
    };

    log.header(trace, session)?;

    let mut turn = 0;
    for attempt in trace.attempts() {
        turn += 1;
        log.attempt(trace, session, attempt, turn)?;
    }

    turn += 1;
    log.outcome_turn(trace, session, turn)
}

/// Write `trace` into `sessions_root` as a zstd session artifact, under the
/// directory layout the harness's JSONL backend expects
/// (`<root>/<project-key>/<session-id>/session.jsonl.zstd`), and return the
/// path written.
///
/// `sessions_root` is `$DSH_HOME/sessions`, which is `~/.dsh/sessions` by
/// default.
///
/// # The artifact is a frame container, not a compressed file
///
/// A `.jsonl.zstd` session is a **concatenation of independently decodable
/// zstd frames**, which is what lets the harness append a batch without
/// rewriting the file. The layout is load-bearing in one specific way: the
/// session picker reads only the *first frame* of every artifact it lists and
/// requires it to decompress to exactly one newline-terminated line — the
/// header. It never decodes the rest.
///
/// So this writes the header as its own frame and the events as a second one.
/// Compressing the whole log as a single frame produces a file that
/// round-trips perfectly through `zstd -d` and still takes `dsh` down at
/// startup with `corrupt Zstandard session log: first frame is not exactly
/// one header line` — the listing walks every session, so one bad artifact
/// stops the app from booting at all.
///
/// # Compression is not optional either
///
/// The backend refuses to load a root that mixes encodings: it walks every
/// session directory and errors out if it finds an artifact with the suffix
/// it is not configured for. Dropping a plain `session.jsonl` into a
/// zstd-configured root breaks the same listing.
///
/// # Errors
///
/// Returns the directory-creation, serialization and write errors.
#[cfg(feature = "dsh-export")]
pub fn export_dsh_session(
    trace: &EvolutionTrace,
    session: &DshSession<'_>,
    sessions_root: &std::path::Path,
) -> io::Result<std::path::PathBuf> {
    let dir = sessions_root
        .join(project_key(session.cwd))
        .join(encode_segment(&session.resolved_id(trace)));
    std::fs::create_dir_all(&dir)?;

    let mut jsonl = Vec::new();
    write_dsh_session(trace, session, &mut jsonl)?;

    // The header is the first line and JSON, so it holds no raw newline: the
    // first `\n` in the stream always ends it.
    let header_end = jsonl
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(jsonl.len(), |index| index + 1);
    let (header, events) = jsonl.split_at(header_end);

    let mut artifact = zstd_frame(header)?;
    if !events.is_empty() {
        artifact.extend_from_slice(&zstd_frame(events)?);
    }

    let path = dir.join("session.jsonl.zstd");
    std::fs::write(&path, artifact)?;
    Ok(path)
}

/// Compress `input` into one complete, checksummed zstd frame — the unit the
/// harness's container is built from.
#[cfg(feature = "dsh-export")]
fn zstd_frame(input: &[u8]) -> io::Result<Vec<u8>> {
    let mut encoder = zstd::stream::raw::Encoder::new(ZSTD_LEVEL)?;
    // The harness compresses every frame with `ZSTD_c_checksumFlag` set, and
    // its frame scanner reads the flag out of each frame header to find the
    // next boundary. Matching it keeps a written frame byte-comparable with a
    // harness-written one.
    encoder.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))?;

    let mut writer = zstd::stream::write::Encoder::with_encoder(Vec::new(), encoder);
    writer.write_all(input)?;
    writer.finish()
}

/// Compression level for a session artifact. Traces are text and write once,
/// so the default level is the right trade.
#[cfg(feature = "dsh-export")]
const ZSTD_LEVEL: i32 = zstd::DEFAULT_COMPRESSION_LEVEL;

/// The harness's project-directory name for `cwd`: separator runs collapse to
/// `-`, anything outside `[A-Za-z0-9._-]` becomes `~XXXX` over UTF-16 code
/// units, and the result is wrapped in `--`.
#[cfg(feature = "dsh-export")]
fn project_key(cwd: Option<&str>) -> String {
    let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) else {
        return "_no-cwd".to_string();
    };

    let mut readable = String::new();
    let mut separator_run = false;
    for unit in cwd.encode_utf16() {
        match char::from_u32(u32::from(unit)) {
            Some('/' | '\\' | ':') => {
                if !separator_run {
                    readable.push('-');
                }
                separator_run = true;
            }
            Some(ch) if is_safe_segment_char(ch) => {
                readable.push(ch);
                separator_run = false;
            }
            _ => {
                let _ = write!(readable, "~{unit:04X}");
                separator_run = false;
            }
        }
    }

    let trimmed: String = readable
        .trim_start_matches('-')
        .encode_utf16()
        .take(251)
        .collect::<Vec<u16>>()
        .iter()
        .filter_map(|unit| char::from_u32(u32::from(*unit)))
        .collect();
    let body = if trimmed.is_empty() { "root" } else { &trimmed };
    format!("--{body}--")
}

/// The harness's injective single-path-segment encoding of a session id.
#[cfg(feature = "dsh-export")]
fn encode_segment(raw: &str) -> String {
    match raw {
        "" => "_".to_string(),
        "." => "~002E".to_string(),
        ".." => "~002E~002E".to_string(),
        _ => raw
            .encode_utf16()
            .map(|unit| match char::from_u32(u32::from(unit)) {
                Some(ch) if is_safe_segment_char(ch) => ch.to_string(),
                _ => format!("~{unit:04X}"),
            })
            .collect(),
    }
}

/// The harness's literal-in-a-path-segment character class. `~` is excluded:
/// it introduces an escape.
#[cfg(feature = "dsh-export")]
fn is_safe_segment_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')
}

/// Milliseconds since the Unix epoch, saturating at `0` for a pre-epoch time.
fn epoch_millis(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

/// The append-only session log under construction.
struct Log<W: Write> {
    out: W,
    /// Next event `seq`. The harness requires it dense from zero: it rejects a
    /// log whose event `seq` differs from the event's index.
    seq: u64,
    /// Timestamp cursor, in Unix epoch milliseconds.
    time_ms: u64,
    /// Counter behind the synthetic message ids.
    next_id: u64,
}

impl<W: Write> Log<W> {
    /// Write the `session` header record: the log's first line.
    fn header(&mut self, trace: &EvolutionTrace, session: &DshSession<'_>) -> io::Result<()> {
        let mut header = json!({
            "type": "session",
            "version": SESSION_FORMAT_VERSION,
            "id": session.resolved_id(trace),
            "createdAt": self.time_ms,
            "delegationDepth": 0,
            "agentPreset": "symbiont",
        });
        if let Some(cwd) = session.cwd {
            header["cwd"] = json!(cwd);
        }
        writeln!(self.out, "{header}")
    }

    /// Write one attempt as one turn.
    fn attempt(
        &mut self,
        trace: &EvolutionTrace,
        session: &DshSession<'_>,
        attempt: &AttemptTrace,
        turn: u64,
    ) -> io::Result<()> {
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
            self.user_text(turn, step, attempt.prompt())?;
        } else {
            let mut step_has_assistant = false;
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
                                other => self.user_text(turn, step, &user_text(other))?,
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

                        let blocks: Vec<Value> = content.iter().map(assistant_block).collect();
                        self.assistant_message(turn, step, session, &blocks)?;

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

        self.notice(turn, step, &attempt_notice(attempt))?;
        self.event("step/end", json!({ "turn": turn, "step": step }))?;
        self.event(
            "turn/end",
            json!({ "turn": turn, "reason": turn_end_reason(attempt.ladder()) }),
        )
    }

    /// Write a closing turn that carries the lane's outcome.
    fn outcome_turn(
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
        self.notice(turn, 1, &outcome_notice(trace))?;
        self.event("step/end", json!({ "turn": turn, "step": 1 }))?;
        self.event(
            "turn/end",
            json!({ "turn": turn, "reason": { "kind": "completed" } }),
        )
    }

    /// Write the request header and, when known, the route's context window.
    fn request_header(&mut self, session: &DshSession<'_>) -> io::Result<()> {
        self.event(
            "request/header",
            json!({
                "header": {
                    "config": { "provider": session.provider, "model": session.model },
                    "system": session.system_prompt,
                },
                "reason": "initial",
            }),
        )?;

        if let Some(window) = session.context_window {
            self.event(
                "request/context",
                json!({
                    "provider": session.provider,
                    "model": session.model,
                    "contextWindow": window,
                }),
            )?;
        }
        Ok(())
    }

    /// Write a plain user turn.
    fn user_text(&mut self, turn: u64, step: u64, text: &str) -> io::Result<()> {
        let id = self.mint_id("msg");
        let _ = (turn, step);
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
    fn notice(&mut self, turn: u64, step: u64, text: &str) -> io::Result<()> {
        let id = self.mint_id("note");
        let _ = (turn, step);
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
    ) -> io::Result<()> {
        let id = self.mint_id("msg");
        self.surface_event(
            "assistant/message",
            json!({
                "turn": turn,
                "step": step,
                "message": {
                    "id": id,
                    "role": "assistant",
                    "content": blocks,
                    "source": {
                        "kind": "model",
                        "provider": session.provider,
                        "model": session.model,
                    },
                },
            }),
        )
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
        self.time_ms += 1;
        writeln!(self.out, "{event}")
    }

    /// A fresh message id, unique within the session.
    fn mint_id(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("symbiont-{prefix}-{}", self.next_id)
    }
}

/// One assistant content item as a harness content block.
fn assistant_block(content: &AssistantContent) -> Value {
    match content {
        AssistantContent::Text(text) => json!({ "type": "text", "text": text.text }),
        AssistantContent::Reasoning(reasoning) => {
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
        AssistantContent::ToolCall(call) => json!({
            "type": "tool-call",
            "id": call.id.as_str(),
            "name": call.function.name,
            // The harness wants the model's raw argument JSON, as a string.
            "arguments": call.function.arguments.to_string(),
        }),
        // A harness image block references an attachment the harness owns, so
        // an image out of a rig transcript has no faithful counterpart.
        AssistantContent::Image(_) => json!({ "type": "text", "text": "[image omitted]" }),
    }
}

/// One tool-result item as a harness content block.
fn tool_result_block(content: &ToolResultContent) -> Value {
    match content {
        ToolResultContent::Text(text) => json!({ "type": "text", "text": text.text }),
        ToolResultContent::Json { value } => json!({ "type": "text", "text": value.to_string() }),
        ToolResultContent::Image(_) => json!({ "type": "text", "text": "[image omitted]" }),
    }
}

/// The readable text of a non-tool-result user content item.
fn user_text(content: &UserContent) -> String {
    match content {
        UserContent::Text(text) => text.text.clone(),
        UserContent::ToolResult(result) => format!("[tool result from {}]", result.name),
        UserContent::Image(_) => "[image omitted]".to_string(),
        UserContent::Audio(_) => "[audio omitted]".to_string(),
        UserContent::Video(_) => "[video omitted]".to_string(),
        UserContent::Document(_) => "[document omitted]".to_string(),
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
    let verdict = match trace.outcome() {
        TraceOutcome::Registered { revision } => format!("registered revision {revision}"),
        TraceOutcome::Failed { reason } => format!("failed — {reason}"),
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
    match ladder {
        // The model answered; the harness then chose what to do with the
        // answer. That choice is not a failure of the turn.
        LadderEvent::Registered { .. }
        | LadderEvent::SelfHeal { .. }
        | LadderEvent::ContextReset { .. }
        | LadderEvent::RepeatReset { .. } => json!({ "kind": "completed" }),
        LadderEvent::TransientRetry { cause, .. } => {
            json!({ "kind": "error", "error": { "message": cause, "code": "UNKNOWN" } })
        }
        LadderEvent::Terminal { reason } => {
            json!({ "kind": "error", "error": { "message": reason, "code": "UNKNOWN" } })
        }
    }
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rig_core::{
        completion::Usage,
        message::{
            AssistantContent,
            Message,
            Text,
            ToolCall,
            ToolCallId,
            ToolFunction,
            ToolResult,
            ToolResultContent,
            UserContent,
        },
    };

    use super::*;
    use crate::{
        evolution_trace::{
            BuildRecord,
            RunTrace,
            StageTimings,
        },
        revision::Revision,
    };

    /// The stage timings of an attempt that got all the way through a build.
    fn built_stages() -> StageTimings {
        let mut stages = StageTimings::default();
        stages.set_llm(Some(Duration::from_millis(900)));
        stages.set_parse_validate(Some(Duration::from_micros(80)));
        stages.set_build(Some(BuildRecord::Built {
            slot_wait: Duration::from_millis(2),
            compile: Duration::from_secs(3),
            load: Duration::from_millis(1),
        }));
        stages
    }

    /// A lane that called a tool, failed to compile, self-healed and then
    /// registered — the shape every field of the exporter has to survive.
    fn sample_trace() -> EvolutionTrace {
        let call_id = ToolCallId::new("call_1").expect("a non-empty id");
        let history = vec![
            Message::user("write a sort"),
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::ToolCall(ToolCall {
                    id: call_id.clone(),
                    provider: None,
                    function: ToolFunction {
                        name: "api_index".to_string(),
                        arguments: json!({ "path": "prelude" }),
                    },
                    signature: None,
                    additional_params: None,
                })],
            },
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    call: call_id,
                    provider: None,
                    name: "api_index".to_string(),
                    content: vec![ToolResultContent::Text(Text::from(
                        "pub fn sort(..)".to_string(),
                    ))],
                })],
            },
            Message::assistant("```rust\nfn sort() {}\n```"),
            Message::user("it did not compile: E0277"),
            Message::assistant("```rust\nfn sort() { /* fixed */ }\n```"),
        ];

        let mut trace = EvolutionTrace::new(3, "write a sort".to_string());
        trace.push_attempt(
            1,
            "write a sort".to_string(),
            Some(
                RunTrace::builder()
                    .produced(0..4)
                    .response("```rust\nfn sort() {}\n```".to_string())
                    .usage(Usage::new())
                    .completion_calls(Vec::new())
                    .build(),
            ),
            StageTimings::default(),
            LadderEvent::SelfHeal {
                kind: "compile".to_string(),
                diagnostics: "E0277: the trait bound is not satisfied".to_string(),
            },
            Duration::from_secs(4),
        );
        trace.push_attempt(
            2,
            "it did not compile: E0277".to_string(),
            Some(
                RunTrace::builder()
                    .produced(4..6)
                    .response("```rust\nfn sort() { /* fixed */ }\n```".to_string())
                    .usage(Usage::new())
                    .completion_calls(Vec::new())
                    .build(),
            ),
            built_stages(),
            LadderEvent::Registered {
                revision: Revision::new(1),
            },
            Duration::from_secs(5),
        );
        trace.set_history(history);
        trace.set_outcome(TraceOutcome::Registered {
            revision: Revision::new(1),
        });
        trace.set_duration(Duration::from_secs(9));
        trace
    }

    fn export(trace: &EvolutionTrace) -> Vec<Value> {
        let session = DshSession::builder()
            .system_prompt("you write rust")
            .provider("openrouter")
            .model("kimi")
            .cwd("/tmp/project")
            .started_at(UNIX_EPOCH + Duration::from_millis(1_700_000_000_000))
            .context_window(131_072)
            .build();

        let mut buffer = Vec::new();
        write_dsh_session(trace, &session, &mut buffer).expect("writing to a Vec");
        String::from_utf8(buffer)
            .expect("valid utf-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is its own JSON document"))
            .collect()
    }

    /// The first line is the `session` header the harness's shape guard
    /// accepts: a numeric `version` it reads, a string `id`, an epoch
    /// `createdAt` and a non-negative `delegationDepth`.
    #[test]
    fn first_line_is_a_loadable_header() {
        let header = &export(&sample_trace())[0];

        assert_eq!(header["type"], "session");
        assert_eq!(header["version"], 0);
        assert!(header["id"].as_str().is_some_and(|id| !id.is_empty()));
        assert_eq!(header["createdAt"], 1_700_000_000_000_u64);
        assert_eq!(header["delegationDepth"], 0);
        assert_eq!(header["cwd"], "/tmp/project");
    }

    /// The harness rejects a log whose event `seq` differs from the event's
    /// index, and it reads `time` as epoch milliseconds.
    #[test]
    fn event_seq_is_dense_and_time_is_monotonic() {
        let lines = export(&sample_trace());
        let mut previous_time = 0;

        for (index, event) in lines.iter().skip(1).enumerate() {
            assert_eq!(event["seq"], index, "seq must equal the event index");
            let time = event["time"].as_u64().expect("an epoch-millisecond time");
            assert!(time > previous_time, "time must not go backwards");
            previous_time = time;
        }
    }

    /// Turns and steps nest, every turn closes, and no step outlives its turn.
    /// The harness's session invariant refuses any other bracketing.
    #[test]
    fn turns_and_steps_are_balanced() {
        let lines = export(&sample_trace());
        let (mut open_turn, mut open_step) = (None, None);
        let (mut next_turn, mut next_step) = (1, 1);

        for event in lines.iter().skip(1) {
            let (turn, step) = (
                event["data"]["turn"].as_u64(),
                event["data"]["step"].as_u64(),
            );
            match event["type"].as_str().expect("a tagged event") {
                "turn/start" => {
                    assert_eq!(open_turn, None, "a turn opened inside another turn");
                    assert_eq!(turn, Some(next_turn));
                    open_turn = turn;
                    next_step = 1;
                }
                "turn/end" => {
                    assert_eq!(open_turn, turn);
                    assert_eq!(open_step, None, "a turn ended with a step still open");
                    open_turn = None;
                    next_turn += 1;
                }
                "step/start" => {
                    assert_eq!(open_turn, turn);
                    assert_eq!(open_step, None, "a step opened inside another step");
                    assert_eq!(step, Some(next_step));
                    open_step = step;
                }
                "step/end" => {
                    assert_eq!((open_turn, open_step), (turn, step));
                    open_step = None;
                    next_step += 1;
                }
                // A step-scoped event must name the open bracket.
                "assistant/message" | "tool/call" | "tool/result" => {
                    assert_eq!((open_turn, open_step), (turn, step));
                }
                // Log-only records must sit inside a turn.
                "request/header" | "request/context" => assert!(open_turn.is_some()),
                _ => {}
            }
        }

        assert_eq!(open_turn, None, "the log left a turn open");
    }

    /// Every `tool/result` answers a `tool/call` of the same step. The harness
    /// clears a step's outstanding calls at `step/end`, so a result that
    /// slipped into the next step would be rejected as uncorrelated.
    #[test]
    fn tool_results_answer_a_call_of_their_own_step() {
        let lines = export(&sample_trace());
        let mut pending: Vec<String> = Vec::new();
        let mut results = 0;

        for event in lines.iter().skip(1) {
            match event["type"].as_str().expect("a tagged event") {
                "tool/call" => pending.push(event["data"]["callId"].to_string()),
                "tool/result" => {
                    let call_id = event["data"]["message"]["source"]["callId"].to_string();
                    assert!(
                        pending.contains(&call_id),
                        "tool/result {call_id} has no tool/call in its step",
                    );
                    // The harness reads `content[0].isError` on every result.
                    assert!(
                        event["data"]["message"]["content"][0]["isError"].is_boolean(),
                        "a tool result must carry its error flag",
                    );
                    results += 1;
                }
                "step/end" => pending.clear(),
                _ => {}
            }
        }

        assert_eq!(results, 1, "the sample lane called exactly one tool");
    }

    /// The system prompt is the reason this exporter takes an argument the
    /// trace does not hold. It has to reach the request header.
    #[test]
    fn the_system_prompt_reaches_the_request_header() {
        let lines = export(&sample_trace());
        let header = lines
            .iter()
            .find(|event| event["type"] == "request/header")
            .expect("a request header");

        assert_eq!(header["data"]["header"]["system"], "you write rust");
        assert_eq!(header["data"]["header"]["config"]["model"], "kimi");
        assert!(
            header.get("surfaceOp").is_none(),
            "a log-only event must not claim a place on the transcript",
        );
    }

    /// The transcript carries the prompt, the assistant turns and the tool
    /// exchange, in the order the lane produced them.
    #[test]
    fn the_transcript_holds_every_message() {
        let lines = export(&sample_trace());
        let surface: Vec<&str> = lines
            .iter()
            .filter(|event| event["surfaceOp"] == "append")
            .map(|event| event["type"].as_str().expect("a tagged event"))
            .collect();

        assert_eq!(
            surface,
            vec![
                // attempt 1: prompt, the tool-calling turn, its result, the answer
                "user/message",
                "assistant/message",
                "tool/result",
                "assistant/message",
                "user/message", // the ladder note
                // attempt 2: the nudge and the fixed answer
                "user/message",
                "assistant/message",
                "user/message", // the ladder note
                // the closing outcome note
                "user/message",
            ],
        );
    }

    /// A tool call travels as a block on its assistant turn *and* as its own
    /// `tool/call` event, with the raw argument JSON as a string on both.
    #[test]
    fn a_tool_call_carries_raw_argument_json() {
        let lines = export(&sample_trace());
        let call = lines
            .iter()
            .find(|event| event["type"] == "tool/call")
            .expect("a tool call");

        assert_eq!(call["data"]["name"], "api_index");
        assert_eq!(call["data"]["arguments"], r#"{"path":"prelude"}"#);
        assert_eq!(call["data"]["callId"], "call_1");

        let block = lines
            .iter()
            .find(|event| event["type"] == "assistant/message")
            .expect("an assistant turn")["data"]["message"]["content"][0]
            .clone();
        assert_eq!(block["type"], "tool-call");
        assert_eq!(block["arguments"], r#"{"path":"prelude"}"#);
    }

    /// The ladder decision and the stage timings are not model output, so they
    /// travel as collapsed harness notes rather than as user turns.
    #[test]
    fn ladder_and_outcome_travel_as_notices() {
        let lines = export(&sample_trace());
        let notes: Vec<&Value> = lines
            .iter()
            .filter(|event| event["data"]["source"]["kind"] == "plugin")
            .collect();

        assert_eq!(notes.len(), 3, "two attempts and the lane outcome");
        for note in &notes {
            assert_eq!(note["data"]["source"]["form"], "notice");
            let summary = note["data"]["source"]["summary"]
                .as_str()
                .expect("a notice carries its one-line account");
            assert!(summary.chars().count() <= 120, "the harness bounds it");
            assert!(!summary.contains('\n'));
        }

        let first = notes[0]["data"]["content"][0]["text"]
            .as_str()
            .expect("note text");
        assert!(first.contains("self-heal (compile)"));

        let second = notes[1]["data"]["content"][0]["text"]
            .as_str()
            .expect("note text");
        assert!(second.contains("registered revision 1"));
        assert!(
            second.contains("compile 3s"),
            "the stage breakdown travels too"
        );

        let last = notes[2]["data"]["content"][0]["text"]
            .as_str()
            .expect("note text");
        assert!(last.contains("lane 3 finished"));
        assert!(last.contains("registered revision 1"));
    }

    /// An attempt whose inference call failed has no transcript. It still gets
    /// a turn, so the retry is visible instead of silently missing.
    #[test]
    fn an_attempt_without_a_run_still_gets_a_turn() {
        let mut trace = EvolutionTrace::new(0, "base".to_string());
        trace.push_attempt(
            1,
            "base".to_string(),
            None,
            StageTimings::default(),
            LadderEvent::TransientRetry {
                backoff: Duration::from_secs(2),
                cause: "503 Service Unavailable".to_string(),
            },
            Duration::from_secs(2),
        );
        trace.set_outcome(TraceOutcome::Failed {
            reason: "gave up".to_string(),
        });

        let lines = export(&trace);
        let end = lines
            .iter()
            .find(|event| event["type"] == "turn/end")
            .expect("the turn closes");

        assert_eq!(end["data"]["reason"]["kind"], "error");
        assert_eq!(
            end["data"]["reason"]["error"]["message"],
            "503 Service Unavailable"
        );
        assert!(
            lines
                .iter()
                .any(|event| event["data"]["content"][0]["text"] == "base"),
            "the prompt that was sent stays visible",
        );
    }

    /// A trace that never reached the model still produces a loadable session:
    /// one turn that reports why.
    #[test]
    fn a_trace_without_attempts_is_still_a_session() {
        let mut trace = EvolutionTrace::new(0, String::new());
        trace.set_outcome(TraceOutcome::Failed {
            reason: "failed before the lane started".to_string(),
        });

        let lines = export(&trace);
        assert_eq!(lines[0]["type"], "session");
        assert!(lines.iter().any(|event| event["type"] == "request/header"));
        assert!(lines.iter().any(|event| event["type"] == "turn/end"));
    }

    /// The artifact is a frame container, and the harness's session listing
    /// decodes only the **first** frame of every session it finds, demanding
    /// exactly one header line from it.
    ///
    /// This is a regression test with teeth: a log compressed as one frame
    /// round-trips perfectly through `zstd -d`, passes every other check here,
    /// and still stops `dsh` from booting, because the listing walks the whole
    /// sessions root. Decoding the whole file is exactly the check that misses
    /// it, so this one decodes the first frame alone.
    #[cfg(feature = "dsh-export")]
    #[test]
    fn the_first_zstd_frame_holds_only_the_header() {
        use std::io::Read as _;

        let root =
            std::env::temp_dir().join(format!("symbiont-dsh-export-frames-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        let session = DshSession::builder()
            .system_prompt("you write rust")
            .cwd("/tmp/project")
            .build();
        let path = export_dsh_session(&sample_trace(), &session, &root).expect("the export writes");
        let artifact = std::fs::read(&path).expect("the artifact is readable");

        let mut first_frame = Vec::new();
        zstd::stream::read::Decoder::new(artifact.as_slice())
            .expect("a zstd stream")
            .single_frame()
            .read_to_end(&mut first_frame)
            .expect("the first frame decodes on its own");
        let first_frame = String::from_utf8(first_frame).expect("valid utf-8");

        assert!(
            first_frame.ends_with('\n') && first_frame.matches('\n').count() == 1,
            "the first frame must be exactly the header line, got {} line(s)",
            first_frame.lines().count(),
        );
        let header: Value =
            serde_json::from_str(first_frame.trim_end()).expect("the header line is JSON");
        assert_eq!(header["type"], "session");

        // The events follow in their own frame, and the container still reads
        // back as one contiguous JSONL stream.
        let whole = zstd::decode_all(artifact.as_slice()).expect("the container decodes");
        let whole = String::from_utf8(whole).expect("valid utf-8");
        assert!(
            whole.lines().count() > 1,
            "the events must follow the header",
        );
        assert!(whole.starts_with(&first_frame), "the header comes first");
        assert!(whole.ends_with('\n'), "every record is newline-terminated");

        std::fs::remove_dir_all(&root).expect("the test cleans up after itself");
    }

    /// The project directory and the session directory follow the harness's
    /// own encoding, or the picker files the session somewhere it never looks.
    #[cfg(feature = "dsh-export")]
    #[test]
    fn paths_match_the_harness_encoding() {
        assert_eq!(
            project_key(Some("/home/m/MathisWellmann/symbiont")),
            "--home-m-MathisWellmann-symbiont--",
        );
        assert_eq!(project_key(None), "_no-cwd");
        assert_eq!(project_key(Some("/")), "--root--");
        // A space is not in the literal class, so it escapes to its UTF-16
        // code unit.
        assert_eq!(
            project_key(Some("/tmp/my project")),
            "--tmp-my~0020project--"
        );
        assert_eq!(encode_segment("session-abc_1.2"), "session-abc_1.2");
        assert_eq!(encode_segment("a/b"), "a~002Fb");
        assert_eq!(encode_segment(".."), "~002E~002E");
    }
}
