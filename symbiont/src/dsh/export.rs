// SPDX-License-Identifier: MPL-2.0
//! Write a projected session out as JSON Lines, or as the harness's own
//! zstd artifact under `~/.dsh/sessions`.
//!
//! The mapping itself, and the metadata a trace does not carry, are in the
//! [module docs](crate::dsh) of `dsh`.

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
        SystemTime,
        UNIX_EPOCH,
    },
};

use getset::{
    CopyGetters,
    Getters,
};
use typed_builder::TypedBuilder;

use crate::{
    dsh::{
        log::Log,
        millis_of,
        types::LogLine,
    },
    evolution_trace::EvolutionTrace,
};

/// The metadata a DeepSeek Harness session needs that an [`EvolutionTrace`]
/// does not carry. See the [module docs](crate::dsh) for why each one is here.
#[derive(Debug, Clone, TypedBuilder, Getters, CopyGetters)]
pub struct DshSession<'a> {
    /// The rendered system prompt, shown in the session's request header.
    /// Pass what [`crate::system_prompt`] returned for the run.
    #[getset(get_copy = "pub(super)")]
    system_prompt: &'a str,

    /// Provider route id, for the header's call config.
    #[builder(default = "symbiont")]
    #[getset(get_copy = "pub(super)")]
    provider: &'a str,

    /// Model id, for the header's call config.
    #[builder(default = "unknown")]
    #[getset(get_copy = "pub(super)")]
    model: &'a str,

    /// Working directory the run happened in. It decides the project
    /// directory the harness files the session under, and the directory the
    /// picker shows. `None` files the session under `_no-cwd`.
    #[builder(default, setter(strip_option))]
    #[getset(get_copy = "pub(super)")]
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
    #[getset(get_copy = "pub(super)")]
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
                "symbiont-lane{}-{}-{:x}",
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

/// Project `trace` onto the DeepSeek Harness session records, in log order.
///
/// The first element is always the [`LogLine::Session`] header. This is the
/// structured form of what [`write_dsh_session`] serializes; take it when you
/// want to inspect, filter or splice the records rather than write them.
///
/// The projection is **lossy on purpose**. The harness has no vocabulary for
/// a [`crate::LadderEvent`], a [`crate::StageTimings`] or a
/// [`crate::TraceOutcome`], so those travel as `notice` messages a human
/// reads. The [`EvolutionTrace`] stays the machine-readable record; see the
/// [module docs](crate::dsh) for the full mapping.
#[must_use]
pub fn dsh_lines(trace: &EvolutionTrace, session: &DshSession<'_>) -> Vec<LogLine> {
    let mut log = Log::new(epoch_millis(session.started_at));

    log.header(trace, session);

    let session_start = log.time_ms();
    let mut turn = 0;
    for attempt in trace.attempts() {
        turn += 1;
        log.attempt(trace, session, attempt, turn);
    }

    // The lane's own measured duration covers whatever happened between and
    // around the attempts, so the closing turn sits at the end of it.
    log.advance_to(session_start.saturating_add(millis_of(trace.duration())));

    turn += 1;
    log.outcome_turn(trace, session, turn);

    log.into_lines()
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
/// Returns the write errors of `out`. Also returns serialization errors,
/// which indicate a bug, because a [`LogLine`] is plain data.
pub fn write_dsh_session<W: Write>(
    trace: &EvolutionTrace,
    session: &DshSession<'_>,
    mut out: W,
) -> io::Result<()> {
    for line in dsh_lines(trace, session) {
        let json = serde_json::to_string(&line).map_err(io::Error::other)?;
        writeln!(out, "{json}")?;
    }
    Ok(())
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

/// Milliseconds since the Unix epoch, saturating at `0` for a pre-epoch time.
fn epoch_millis(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
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
    use serde_json::{
        Value,
        json,
    };

    use super::*;
    use crate::{
        LadderEvent,
        TraceOutcome,
        dsh::tests::usage_of,
        evolution_trace::{
            BuildRecord,
            RunTrace,
            StageTimings,
        },
        evolve_info::Lane,
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

    fn sample_session() -> DshSession<'static> {
        DshSession::builder()
            .system_prompt("you write rust")
            .provider("local")
            .model("kimi")
            .cwd("/tmp/project")
            .started_at(UNIX_EPOCH + Duration::from_millis(1_700_000_000_000))
            .context_window(131_072)
            .build()
    }

    fn export(trace: &EvolutionTrace) -> Vec<Value> {
        let session = sample_session();

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

    /// [`write_dsh_session`] is a serializer over [`dsh_lines`] and nothing
    /// more: line `i` of the file is the serialized form of record `i`.
    ///
    /// Every line also has to decode back into a [`LogLine`] that serializes
    /// to the same bytes. `LogLine` is a closed `type`-tagged enum, so this is
    /// what proves the writer emits no record the reader would reject, and
    /// that no field is dropped on the way back in.
    #[test]
    fn the_written_lines_are_the_typed_records() {
        let trace = sample_trace();
        let session = sample_session();
        let records = dsh_lines(&trace, &session);

        let mut buffer = Vec::new();
        write_dsh_session(&trace, &session, &mut buffer).expect("writing to a Vec");
        let text = String::from_utf8(buffer).expect("valid utf-8");

        assert_eq!(
            text.lines().count(),
            records.len(),
            "one written line per record",
        );
        assert!(!records.is_empty(), "a sample trace produces records");

        for (index, (record, line)) in records.iter().zip(text.lines()).enumerate() {
            let written = serde_json::to_string(record).expect("a record is plain data");
            assert_eq!(written, line, "record {index} is written verbatim");

            let back: LogLine =
                serde_json::from_str(line).expect("every written line decodes as a `LogLine`");
            assert_eq!(
                serde_json::to_string(&back).expect("a record is plain data"),
                line,
                "record {index} round-trips without loss",
            );
        }
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
