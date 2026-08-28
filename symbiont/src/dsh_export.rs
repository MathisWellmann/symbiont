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
//! | [`EvolutionTrace::outcome`]                           | a final `notice` user message        |
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
//!   Events are laid out from [`DshSession::started_at`] forward, spaced by
//!   the trace's own measurements: an attempt spans
//!   [`AttemptTrace::duration`], the model time inside it spans
//!   [`StageTimings::llm`], and the lane spans
//!   [`EvolutionTrace::duration`]. The harness derives every duration it
//!   shows by subtracting two event times, so those come out right; only the
//!   absolute instant is the caller's to supply.
//!
//! [`StageTimings::llm`]: crate::StageTimings::llm
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

use std::{
    fmt::Write as _,
    io::{
        self,
        Write,
    },
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
    time::{
        Duration,
        SystemTime,
        UNIX_EPOCH,
    },
};

use rig_core::{
    completion::Usage,
    message::{
        AssistantContent,
        Message,
        ReasoningContent,
        ToolResultContent,
        UserContent,
    },
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

    /// Session id.
    ///
    /// **It must be unique across the whole sessions root, not merely inside
    /// its project directory.** The harness's session listing collects every
    /// id from every project directory into one set and throws on the first
    /// repeat — at startup, so a repeat stops `dsh` from booting rather than
    /// hiding one session behind another. A fixed id is therefore only safe
    /// while the run's working directory never changes: the same id written
    /// from two directories lands in two project directories and collides.
    ///
    /// `None` derives a unique id, which is what you want unless you are
    /// deliberately overwriting one specific session.
    #[builder(default, setter(strip_option))]
    session_id: Option<String>,

    /// Disambiguator behind a derived [`Self::session_id`], drawn once when
    /// the session is built.
    ///
    /// It is a field rather than something [`Self::resolved_id`] draws per
    /// call because that method has to be **stable**: the artifact's path and
    /// the `id` inside its header both come from it, and the harness rejects
    /// a log whose header id and cwd do not reproduce the path it was found
    /// at.
    #[builder(default = fresh_nonce(), setter(skip))]
    nonce: u64,

    /// The model's advertised context window, for the `request/context`
    /// record. Omitted when `None`.
    #[builder(default, setter(strip_option))]
    context_window: Option<u64>,
}

impl DshSession<'_> {
    /// The session id to file this trace under.
    ///
    /// Stable for a given session and trace: the harness checks that a log's
    /// header id and cwd name the exact path the log was found at, so the
    /// path and the header must be derived from the same answer.
    #[must_use]
    pub fn resolved_id(&self, trace: &EvolutionTrace) -> String {
        self.session_id.clone().unwrap_or_else(|| {
            format!(
                "session-symbiont-lane{}-{}-{:x}",
                trace.lane(),
                epoch_millis(self.started_at),
                self.nonce,
            )
        })
    }
}

/// A fresh disambiguator for a derived session id.
///
/// Mixes the process id, a process-local counter and the wall clock, so ids
/// stay distinct across the lanes of one batch, across repeated runs of one
/// binary, and across two binaries that started in the same millisecond.
fn fresh_nonce() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| u64::from(since.subsec_nanos()));

    (u64::from(std::process::id()) << 40) ^ (nanos << 8) ^ count
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

    let session_start = log.time_ms;
    let mut turn = 0;
    for attempt in trace.attempts() {
        turn += 1;
        log.attempt(trace, session, attempt, turn)?;
    }

    // The lane's own measured duration covers whatever happened between and
    // around the attempts, so the closing turn sits at the end of it.
    log.advance_to(session_start.saturating_add(millis_of(trace.duration())));

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
const ZSTD_LEVEL: i32 = zstd::DEFAULT_COMPRESSION_LEVEL;

/// The harness's project-directory name for `cwd`: separator runs collapse to
/// `-`, anything outside `[A-Za-z0-9._-]` becomes `~XXXX` over UTF-16 code
/// units, and the result is wrapped in `--`.
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
fn is_safe_segment_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')
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

/// A duration as whole milliseconds, saturating rather than wrapping.
fn millis_of(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
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
            "agentPreset": "standard", // Could be standard, code, minimal or cordis.
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
            self.user_text(turn, step, attempt.prompt())?;
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
        self.session_title(trace)?;
        self.notice(turn, 1, &outcome_notice(trace))?;
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
                    "provider": session.provider,
                    "model": session.model,
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
    fn advance_to(&mut self, target: u64) {
        self.time_ms = self.time_ms.max(target);
    }

    /// A fresh message id, unique within the session.
    fn mint_id(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("symbiont-{prefix}-{}", self.next_id)
    }
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

/// One completion call's token accounting as a harness `TokenUsage`, or
/// `None` when the provider reported nothing.
///
/// Rig documents all-zero [`Usage`] as its sentinel for missing provider
/// metrics and does not distinguish that from a genuine all-zero report, so
/// an empty record becomes an absent one rather than a measured zero.
///
/// The cache fields are deliberately dropped. The harness defines its counts
/// as **disjoint** — billed input is `inputTokens + cacheReadTokens +
/// cacheWriteTokens` — while rig leaves it to the provider whether
/// `input_tokens` already contains the cached tokens. Reporting rig's cache
/// counts alongside its input count would double-bill them on the providers
/// that fold them in, so only what is unambiguous travels.
fn token_usage(usage: &Usage) -> Option<Value> {
    if usage.input_tokens == 0 && usage.output_tokens == 0 && usage.reasoning_tokens == 0 {
        return None;
    }

    let mut out = json!({
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
    });
    if usage.reasoning_tokens > 0 {
        out["reasoningTokens"] = json!(usage.reasoning_tokens);
    }
    Some(out)
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

    use rig_agent::agent::CompletionCall;
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
        evolve_info::Lane,
        revision::Revision,
    };

    /// A `Usage` that reports `input` prompt and `output` completion tokens.
    fn usage_of(input: u64, output: u64) -> Usage {
        let mut usage = Usage::new();
        usage.input_tokens = input;
        usage.output_tokens = output;
        usage.total_tokens = input + output;
        usage
    }

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

        let mut trace = EvolutionTrace::new(Lane::from(3), "write a sort".to_string());
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
    /// index, and it reads `time` as epoch milliseconds. Time may stand still
    /// between two events — several records of one step share a millisecond in
    /// a harness-written log too — but it must never run backwards, because
    /// every duration the UI shows is a subtraction of two event times.
    #[test]
    fn event_seq_is_dense_and_time_never_goes_backwards() {
        let lines = export(&sample_trace());
        let mut previous_time = 0;

        for (index, event) in lines.iter().skip(1).enumerate() {
            assert_eq!(event["seq"], index, "seq must equal the event index");
            let time = event["time"].as_u64().expect("an epoch-millisecond time");
            assert!(time >= previous_time, "time must not go backwards");
            previous_time = time;
        }
    }

    /// Event times come from the trace's own measurements, not from a counter.
    ///
    /// They used to be one millisecond apart, which made a lane that ran for
    /// nine seconds render as a handful of milliseconds everywhere in the UI:
    /// `dsh-session-stats` derives model time by subtracting the `step/start`
    /// time from the `assistant/message` time, and the session duration the
    /// same way.
    #[test]
    fn event_times_reproduce_the_traces_own_durations() {
        let lines = export(&sample_trace());
        let events = &lines[1..];
        let time_of = |event: &Value| event["time"].as_u64().expect("a time");

        let start = time_of(&events[0]);
        let end = time_of(events.last().expect("the log is not empty"));
        assert_eq!(
            end - start,
            9_000,
            "the session must span the lane's measured 9s",
        );

        // Attempt 2 reported 900ms in the model over one assistant turn, so
        // that is what the harness must be able to fold out of the pair.
        let step_start = events
            .iter()
            .filter(|event| event["type"] == "step/start" && event["data"]["turn"] == 2)
            .map(time_of)
            .next()
            .expect("the second turn opens a step");
        let answered = events
            .iter()
            .filter(|event| event["type"] == "assistant/message" && event["data"]["turn"] == 2)
            .map(time_of)
            .next()
            .expect("the second turn has an assistant message");
        assert_eq!(answered - step_start, 900, "the llm stage of attempt 2");

        // The first attempt measured 4s, so the second one starts there.
        let turn_two = events
            .iter()
            .find(|event| event["type"] == "turn/start" && event["data"]["turn"] == 2)
            .map(time_of)
            .expect("a second turn");
        assert_eq!(turn_two - start, 4_000, "attempt 1 measured 4s");
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

    /// Each assistant turn carries the token accounting of the completion
    /// call that produced it, in order. The harness folds `usage.outputTokens`
    /// off this event, so without it a session reports no tokens at all.
    #[test]
    fn each_assistant_turn_carries_its_own_token_usage() {
        let mut trace = EvolutionTrace::new(Lane::from(0), "p".to_string());
        trace.set_history(vec![
            Message::user("p"),
            Message::assistant("first"),
            Message::assistant("second"),
        ]);
        trace.push_attempt(
            1,
            "p".to_string(),
            Some(
                RunTrace::builder()
                    .produced(0..3)
                    .response("second".to_string())
                    .usage(Usage::new())
                    .completion_calls(vec![
                        CompletionCall::new(0, usage_of(11, 22)),
                        CompletionCall::new(1, usage_of(33, 44)),
                    ])
                    .build(),
            ),
            StageTimings::default(),
            LadderEvent::Registered {
                revision: Revision::new(1),
            },
            Duration::from_secs(1),
        );
        trace.set_outcome(TraceOutcome::Registered {
            revision: Revision::new(1),
        });

        let lines = export(&trace);
        let usages: Vec<&Value> = lines
            .iter()
            .filter(|event| event["type"] == "assistant/message")
            .map(|event| &event["data"]["usage"])
            .collect();

        assert_eq!(usages.len(), 2, "one per assistant turn");
        assert_eq!(usages[0]["inputTokens"], 11);
        assert_eq!(usages[0]["outputTokens"], 22);
        // The second turn gets the second call's numbers, not the first's.
        assert_eq!(usages[1]["inputTokens"], 33);
        assert_eq!(usages[1]["outputTokens"], 44);
    }

    /// Rig reports all-zero usage when the provider reported nothing, so an
    /// empty record must travel as an absent field rather than a measured
    /// zero. The cache counts are dropped because the harness defines its
    /// counts as disjoint while rig does not, and reporting both would
    /// double-bill a provider that folds cache hits into its input count.
    #[test]
    fn unreported_usage_is_absent_and_cache_counts_are_dropped() {
        assert_eq!(token_usage(&Usage::new()), None);

        let mut usage = usage_of(10, 5);
        usage.reasoning_tokens = 3;
        usage.cached_input_tokens = 7;
        usage.cache_creation_input_tokens = 2;

        let mapped = token_usage(&usage).expect("a reported usage maps");
        assert_eq!(mapped["inputTokens"], 10);
        assert_eq!(mapped["outputTokens"], 5);
        assert_eq!(mapped["reasoningTokens"], 3);
        assert!(mapped.get("cacheReadTokens").is_none());
        assert!(mapped.get("cacheWriteTokens").is_none());
    }

    /// Without a `session/title` the web UI falls back to the basename of the
    /// session's working directory, so every lane of one example renders under
    /// the same heading. The title has to say which lane this was and how it
    /// ended.
    #[test]
    fn the_session_is_titled_by_lane_and_outcome() {
        let lines = export(&sample_trace());
        let title = lines
            .iter()
            .find(|event| event["type"] == "session/title")
            .expect("the session is titled");

        assert_eq!(
            title["data"]["title"],
            "symbiont lane 3 \u{b7} rev 1 \u{b7} 2 attempts"
        );

        // `dsh-session-title`'s invariant: `messageSeqs` is empty if and only
        // if the source is `user`.
        assert_eq!(title["data"]["source"]["kind"], "user");
        assert_eq!(
            title["data"]["messageSeqs"],
            serde_json::json!([]),
            "a `user`-sourced title must cite no message seqs",
        );
        assert!(
            title.get("surfaceOp").is_none(),
            "a title is log-only and must not claim a place on the transcript",
        );
    }

    /// A lane that gave up says so, and one attempt is not "1 attempts".
    #[test]
    fn a_failed_lane_is_titled_as_failed() {
        let mut trace = EvolutionTrace::new(Lane::from(0), "base".to_string());
        trace.push_attempt(
            1,
            "base".to_string(),
            None,
            StageTimings::default(),
            LadderEvent::Terminal {
                reason: "gave up".to_string(),
            },
            Duration::from_secs(1),
        );
        trace.set_outcome(TraceOutcome::Failed {
            reason: "gave up".to_string(),
        });

        let lines = export(&trace);
        let title = lines
            .iter()
            .find(|event| event["type"] == "session/title")
            .expect("the session is titled");
        assert_eq!(
            title["data"]["title"],
            "symbiont lane 0 \u{b7} failed \u{b7} 1 attempt"
        );
    }

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

    /// An attempt whose inference call failed has no transcript. It still gets
    /// a turn, so the retry is visible instead of silently missing.
    #[test]
    fn an_attempt_without_a_run_still_gets_a_turn() {
        let mut trace = EvolutionTrace::new(Lane::from(0), "base".to_string());
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
        let mut trace = EvolutionTrace::new(Lane::from(0), String::new());
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
