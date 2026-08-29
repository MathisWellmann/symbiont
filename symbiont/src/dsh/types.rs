//! Rust/serde mirror of the DeepSeek Harness (DSH) session trajectory format.
//!
//! On disk: `~/.dsh/sessions/<project-key>/<session-dir>/session.jsonl.zstd`
//! where `<project-key>` is the cwd with `/` replaced by `-`.
//! The file is a concatenation of independent, checksummed Zstandard frames:
//! one frame holds the header line only, each later frame holds one append batch of JSONL records.
//! Each JSONL line decodes to a `LogLine`.

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::{
    Map,
    Value,
};

// =============================================================================
// Top level: one JSONL line
// =============================================================================

/// One JSONL record of a session trajectory log.
///
/// Line 1 is always the [`SessionHeaderLine`] (`type: "session"`). Every later
/// line is either a session event (envelope `seq`/`time`/`data`) or a packed
/// chunk row (a storage-side compression of runs of `assistant/chunk` delta
/// events — expand it with [`ChunkRow::expand`] before treating it as an
/// event).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LogLine {
    /// Immutable session header (first line only).
    #[serde(rename = "session")]
    Session(SessionHeaderLine),

    // ---- packed chunk rows (storage-only; NOT session events) ----
    /// Run of ≥3 consecutive `text-delta` chunks of one stream block.
    #[serde(rename = "text-chunks")]
    TextChunks(ChunkRow<TextRunData>),
    /// Run of ≥3 consecutive `reasoning-delta` chunks of one stream block.
    #[serde(rename = "reasoning-chunks")]
    ReasoningChunks(ChunkRow<TextRunData>),
    /// Run of ≥3 consecutive `tool-call-delta` chunks of one stream block.
    #[serde(rename = "tool-call-chunks")]
    ToolCallChunks(ChunkRow<ToolCallRunData>),

    // ---- core session events (dsh-session) ----
    #[serde(rename = "turn/start")]
    /// A turn opened.
    TurnStart(Event<TurnStartData>),
    #[serde(rename = "turn/end")]
    /// A turn closed.
    TurnEnd(Event<TurnEndData>),
    /// `step/end` has the identical payload shape.
    #[serde(rename = "step/start")]
    StepStart(Event<StepStartData>),
    #[serde(rename = "step/end")]
    /// A step closed.
    StepEnd(Event<StepStartData>),
    /// The `data` payload is the whole user-role message.
    #[serde(rename = "user/message")]
    UserMessage(Event<Message>),
    #[serde(rename = "assistant/chunk")]
    /// One raw stream chunk of a live assistant turn.
    AssistantChunk(Event<AssistantChunkData>),
    #[serde(rename = "assistant/message")]
    /// The assembled assistant message of one step.
    AssistantMessage(Event<AssistantMessageData>),
    #[serde(rename = "tool/call")]
    /// The model asked for one tool invocation.
    ToolCall(Event<ToolCallData>),
    #[serde(rename = "tool/result")]
    /// One tool call answered.
    ToolResult(Event<ToolResultData>),
    #[serde(rename = "todo/write")]
    /// A whole-list snapshot of the todo list.
    TodoWrite(Event<TodoWriteData>),
    #[serde(rename = "request/header")]
    /// The full header of the next model request.
    RequestHeader(Event<RequestHeaderData>),
    #[serde(rename = "request/context")]
    /// Route metadata of the next model request.
    RequestContext(Event<RequestContextData>),
    /// Empty payload (`data: {}`); marks the end of seed history.
    #[serde(rename = "session/end-seed")]
    EndSeed(Event<Value>),

    // ---- plugin-merged events (observed on disk, v0) ----
    #[serde(rename = "permission/preset")]
    /// The selected permission preset.
    PermissionPreset(Event<PermissionPresetData>),
    #[serde(rename = "sandbox/mode")]
    /// A session sandbox-mode override.
    SandboxMode(Event<SandboxModeData>),
    #[serde(rename = "approval/policy")]
    /// A session approval-policy override.
    ApprovalPolicy(Event<ApprovalPolicyData>),
    #[serde(rename = "approval/asked")]
    /// An approval question was put to the answerer chain.
    ApprovalAsked(Event<ApprovalAskedData>),
    #[serde(rename = "approval/decided")]
    /// The outcome of an earlier `approval/asked`.
    ApprovalDecided(Event<ApprovalDecidedData>),
    #[serde(rename = "agent/inbox/spliced")]
    /// One mutation of an agent's durable pending-message lists.
    InboxSpliced(Event<InboxSplicedData>),
    #[serde(rename = "session/title")]
    /// A session title snapshot.
    SessionTitle(Event<SessionTitleData>),
    #[serde(rename = "session/title-llm-request")]
    /// The request a title provider was asked to answer.
    TitleLlmRequest(Event<TitleLlmRequestData>),
    #[serde(rename = "compaction/start")]
    /// A compaction run began.
    CompactionStart(Event<CompactionStartData>),
    #[serde(rename = "compaction/end")]
    /// A compaction run ended.
    CompactionEnd(Event<CompactionEndData>),
    #[serde(rename = "compaction/summary")]
    /// The summary a compaction run produced.
    CompactionSummary(Event<CompactionSummaryData>),
    #[serde(rename = "compaction/prune")]
    /// A compaction dropped the events its summary shadows.
    CompactionPrune(Event<CompactionPruneData>),
    #[serde(rename = "command/run")]
    /// A slash command was invoked.
    CommandRun(Event<CommandRunData>),
    #[serde(rename = "command/done")]
    /// A slash command finished.
    CommandDone(Event<CommandDoneData>),
    #[serde(rename = "llm/retry")]
    /// A model call failed and will be retried.
    LlmRetry(Event<LlmRetryData>),
    #[serde(rename = "llm/retry-started")]
    /// The retry announced by the matching `llm/retry` started.
    LlmRetryStarted(Event<LlmRetryStartedData>),
    #[serde(rename = "agent-preset/selected")]
    /// The agent preset the session runs under was chosen.
    AgentPresetSelected(Event<AgentPresetSelectedData>),
    #[serde(rename = "web/deepseek-search-llm-request")]
    /// A provider-side web search was requested.
    WebSearchLlmRequest(Event<WebSearchLlmRequestData>),
}

/// Immutable session header — the first JSONL line of every log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeaderLine {
    /// On-disk format version; `0` for all current logs.
    pub version: u32,
    /// Session id (free-form branded string).
    pub id: String,
    /// Unix epoch milliseconds.
    pub created_at: u64,
    /// Absolute working directory the session was created in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Session this one was forked from (seed lineage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
    /// How many leading events were inherited through a seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_length: Option<u64>,
    /// Only present for subagent children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
    /// Delegation depth: `0` for top-level sessions; REQUIRED on disk.
    pub delegation_depth: u32,
    /// Agent preset id; durable because it decides tools + prompt on resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_preset: Option<String>,
}

/// Coarse origin marker for subagent sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    /// The session is a child of a delegating parent.
    Subagent,
}

// =============================================================================
// Event envelope
// =============================================================================

/// Shared envelope of every session event (all non-header, non-chunk-row lines).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Event<D> {
    /// Monotonic sequence number. Contiguous from 0 across the decoded log
    /// (packed chunk rows expand to `len` members), so for plain events
    /// `events[i].seq == i`.
    pub seq: u64,
    /// Unix epoch milliseconds.
    pub time: i64,
    /// Type-specific payload.
    pub data: D,
    /// How this event entered the ordered surface. Present on surface events
    /// (`user/message`, `assistant/message`, `tool/result`); absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_op: Option<SurfaceOp>,
    /// Seqs of earlier events cited as sources (e.g. the chunk seqs that built
    /// an `assistant/message`). Surface events only; may be an empty array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_seqs: Option<Vec<u64>>,
    /// Written as `true` only on purely informational records a reader may
    /// safely skip when it does not recognize the event type. Absent means the
    /// event is REQUIRED — an unrecognized type without this marker must
    /// reject the log, not silently skip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignorable: Option<bool>,
}

impl<D> Event<D> {
    /// A log-only event: one that carries `data` but does not land on the
    /// ordered transcript.
    pub fn new(seq: u64, time: i64, data: D) -> Self {
        Self {
            seq,
            time,
            data,
            surface_op: None,
            source_event_seqs: None,
            ignorable: None,
        }
    }

    /// Place this event on the tail of the ordered transcript.
    #[must_use]
    pub fn on_surface(mut self) -> Self {
        self.surface_op = Some(SurfaceOp::Append);
        self
    }
}

/// How a surface event entered the ordered surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceOp {
    /// JSON: `"append"` — added to the tail.
    Append,
    /// JSON: `{"op":"replace","start":s,"end":e}` — replaces surface nodes
    /// `start..=end` (both inclusive) with this one. Used by compaction.
    /// Replaces the surface nodes `start..=end`, both inclusive.
    Replace {
        /// First replaced node, inclusive.
        start: u64,
        /// Last replaced node, inclusive.
        end: u64,
    },
}

impl<'de> Deserialize<'de> for SurfaceOp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let v = Value::deserialize(deserializer)?;
        match v {
            Value::String(s) if s == "append" => Ok(SurfaceOp::Append),
            Value::Object(m) => {
                let op = m
                    .get("op")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| D::Error::custom("surfaceOp object missing `op`"))?;
                if op != "replace" {
                    return Err(D::Error::custom(format!("unknown surfaceOp `op` {op:?}")));
                }
                let start = m
                    .get("start")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| D::Error::custom("surfaceOp missing `start`"))?;
                let end = m
                    .get("end")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| D::Error::custom("surfaceOp missing `end`"))?;
                Ok(SurfaceOp::Replace { start, end })
            }
            other => Err(D::Error::custom(format!(
                "unexpected surfaceOp value {other:?}"
            ))),
        }
    }
}

impl Serialize for SurfaceOp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            SurfaceOp::Append => serializer.serialize_str("append"),
            SurfaceOp::Replace { start, end } => {
                #[derive(Serialize)]
                struct R {
                    op: &'static str,
                    start: u64,
                    end: u64,
                }
                R {
                    op: "replace",
                    start: *start,
                    end: *end,
                }
                .serialize(serializer)
            }
        }
    }
}

// =============================================================================
// Packed chunk rows (lossless storage packing of assistant/chunk delta runs)
// =============================================================================

/// One packed storage row: a run of ≥3 consecutive same-block delta chunks.
///
/// Member `k` reconstructs as `seq = seq0 + k`,
/// `time = time0 + sum(dt[0..k])` (a `dt` gap may be negative when the wall
/// clock stepped backwards).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkRow<D> {
    /// Seq of the first member.
    pub seq0: u64,
    /// Time of the first member (epoch ms).
    pub time0: i64,
    /// Run payload.
    pub data: D,
}

/// Payload of `text-chunks` / `reasoning-chunks` rows: one entry per member —
/// never joined, token boundaries are data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextRunData {
    /// The turn the packed run belongs to.
    pub turn: u64,
    /// The step the packed run belongs to.
    pub step: u64,
    /// The stream block index every member shares.
    pub index: u64,
    /// Epoch-ms gaps between consecutive members; `len == members - 1`.
    pub dt: Vec<i64>,
    /// Per-member text deltas, in order.
    pub texts: Vec<String>,
}

/// Payload of `tool-call-chunks` rows: the run-constant call identity plus
/// each member's raw arguments fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRunData {
    /// The turn the packed run belongs to.
    pub turn: u64,
    /// The step the packed run belongs to.
    pub step: u64,
    /// Index of the stream block the run's deltas belong to.
    pub index: u64,
    /// Millisecond gaps between members, so `time = time0 + sum(dt[..k])`.
    pub dt: Vec<i64>,
    /// Provider-issued call id (constant across the run).
    pub id: String,
    /// Present iff every member carried it, with one uniform value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Per-member raw arguments fragments, in order.
    pub(super) args: Vec<String>,
}

// =============================================================================
// Core event payloads (dsh-session `SessionEventMap`)
// =============================================================================

/// `turn/start`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnStartData {
    /// The turn that opened.
    pub turn: u64,
}

/// `turn/end`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnEndData {
    /// The turn that closed.
    pub turn: u64,
    /// Why it closed.
    pub reason: TurnEndReason,
}

/// Why a turn ended (discriminated by `kind`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TurnEndReason {
    /// The model answered and the harness had nothing left to run.
    Completed,
    /// A cancellation request interrupted the live turn.
    Aborted {
        /// What requested the cancellation.
        reason: TurnEndCancelCause,
    },
    /// A hook or policy refused to let the turn run.
    Blocked,
    /// Structured LLM failure (verbatim provider facts, or flattened).
    Error {
        /// The provider or transport facts of the failure.
        error: LlmFailure,
    },
    /// At least one step hit its output-token ceiling.
    MaxTokens,
    /// A persistence backend closed a crash-orphaned turn on reload.
    Interrupted,
}

/// Why an active agent driver was cancelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TurnEndCancelCause {
    /// The person at the keyboard.
    User,
    /// A parent session cancelled its delegate.
    Parent,
    /// A hook cancelled the turn.
    Hook {
        /// What the hook gave as its reason.
        reason: String,
    },
    /// The driver was torn down under the running turn.
    Disposed,
    /// Imports whose original coarse record carried no cause.
    Legacy,
}

/// `step/start` and `step/end` share this payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepStartData {
    /// The turn this step belongs to.
    pub turn: u64,
    /// The step's index within the turn.
    pub step: u64,
}

/// `assistant/chunk` — raw stream chunk, token-level replay fidelity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantChunkData {
    /// The turn this belongs to.
    pub turn: u64,
    /// The step within the turn.
    pub step: u64,
    /// The adapter's raw stream chunk.
    pub chunk: StreamChunk,
}

/// `assistant/message` — assembled assistant message for one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageData {
    /// The turn this belongs to.
    pub turn: u64,
    /// The step within the turn.
    pub step: u64,
    /// The assembled assistant message.
    pub message: Message,
    /// Present when the adapter reported token accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// `tool/call` — the model requested one tool invocation.
/// `arguments` is the raw JSON string exactly as the model produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallData {
    /// The turn this belongs to.
    pub turn: u64,
    /// The step within the turn.
    pub step: u64,
    /// Correlates the call with its `tool/result`.
    pub call_id: String,
    /// Tool name, as registered with the model.
    pub name: String,
    /// The model's raw argument JSON, as a string.
    pub arguments: String,
}

/// `tool/result` — a completed tool call's model-facing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultData {
    /// The turn this belongs to.
    pub turn: u64,
    /// The step within the turn.
    pub step: u64,
    /// Role `user`; content is exactly one `tool-result` block.
    pub message: Message,
    /// Optional internal failure identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolResultError>,
    /// Opaque tool-private presentation payload (JSON-serializable); e.g.
    /// `dsh-tool-fs` carries a result-time contextual diff here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// The identity of a tool failure the harness recognized.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultError {
    /// Error name, for the reader.
    pub name: String,
    /// Stable machine-routing code.
    pub code: String,
}

/// `todo/write` — whole-list snapshot; latest write wins on replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoWriteData {
    /// The whole list, in display order.
    pub todos: Vec<TodoItem>,
}

/// One entry of the todo list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoItem {
    /// What the entry asks for.
    pub content: String,
    /// How far the entry got.
    pub status: TodoStatus,
}

/// How far one todo entry got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not started.
    Pending,
    /// Being worked on right now.
    InProgress,
    /// Done.
    Completed,
}

/// `request/header` — full header for the next request (log-only; the latest
/// snapshot reconstructs the request header).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestHeaderData {
    /// The header the next request will carry.
    pub header: EpochHeader,
    /// Why this snapshot was appended.
    pub reason: RequestHeaderReason,
}

/// Why a `request/header` snapshot was appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestHeaderReason {
    /// The first header of the session.
    Initial,
    /// The session was resumed, so the header is restated.
    Resume,
    /// Some part of the header changed.
    Change,
}

/// Logged request state outside derived history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpochHeader {
    /// Provider, model and sampling scalars.
    pub config: LlmCallConfig,
    /// Effective config fields materialized from the exact adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_defaults: Option<LlmCallConfigAdapterDefaults>,
    /// Rendered system prompt text; absent for a system-less request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Assembled tool schemas; absent for a tool-less request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
}

/// Provider, model, reasoning effort, and sampling scalars of one
/// conversation's requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCallConfig {
    /// Provider route id.
    pub provider: String,
    /// Model id on that route.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Reasoning-effort level, when the route takes one.
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Sampling temperature, when the route takes one.
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Output-token ceiling, when one was set.
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Stop sequences, when any were set.
    pub stop: Option<Vec<String>>,
}

/// Only ever `true` where present (a marker that the adapter supplied the
/// default).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCallConfigAdapterDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// `true` when the adapter, not the caller, set the reasoning effort.
    pub reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// `true` when the adapter, not the caller, set the output ceiling.
    pub max_tokens: Option<bool>,
}

/// JSON-schema description of a tool, as sent to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSchema {
    /// Tool name the model calls.
    pub name: String,
    /// What the tool does, in the model's words.
    pub description: String,
    /// JSON Schema object for the arguments (arbitrary JSON).
    pub parameters: Value,
}

/// `request/context` — route metadata for the next request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestContextData {
    /// Provider route id.
    pub provider: String,
    /// Model id on that route.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Context window of the route, in tokens, when it is known.
    pub context_window: Option<u64>,
}

// =============================================================================
// LLM message primitives (dsh-llm)
// =============================================================================

/// One immutable message shared by delivery, durable history, and model
/// requests.
///
/// Invariants per usage:
/// - `user/message` event: `role == "user"`.
/// - `assistant/message` event: `role == "assistant"`, `source.kind == "model"`.
/// - `tool/result` event: `role == "user"`, `content` is exactly one
///   `tool-result` block, `source.kind == "tool"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    /// Stable identity (free-form branded string).
    pub id: String,
    /// Who the message speaks as.
    pub role: Role,
    /// Exact model-facing blocks.
    pub content: Vec<ContentBlock>,
    /// Producer-supplied provenance + plugin extras.
    pub source: MessageSource,
}

/// Who a message speaks as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Harness instructions.
    System,
    /// The person at the keyboard.
    User,
    /// The model.
    Assistant,
}

/// Where a message (or injected content) came from.
///
/// `kind` answers *who produced this*; the flattened extras carry the
/// kind-specific fields plus plugin-private additions:
/// - `model` → `{ provider, model, replayState? }`
/// - `tool`  → `{ callId }`
/// - `plugin`→ `{ plugin, form?, ... }` where `form` is a
///   [`ContextForm`] (`snapshot` carries `sections`, `notice` a `summary`)
///   plus plugin extras (e.g. `compactionId` from the compact plugin).
/// - `user`  → usually empty; client plugins may add fields such as
///   `rpcId` / `clientTimeZone`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSource {
    /// Who produced this message.
    pub kind: SourceKind,
    /// Kind-specific fields + forward-compatible plugin extras.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl MessageSource {
    /// A message the human (or, here, the harness acting as one) typed.
    #[must_use]
    pub fn user() -> Self {
        Self {
            kind: "user".to_string(),
            extra: Map::new(),
        }
    }

    /// A message the model produced on `provider`'s `model` route.
    #[must_use]
    pub fn model(provider: &str, model: &str) -> Self {
        Self::with_extra(
            "model",
            &ModelSourceExtra {
                provider: provider.to_string(),
                model: model.to_string(),
                replay_state: None,
            },
        )
    }

    /// The result of the tool call `call_id`.
    #[must_use]
    pub fn tool(call_id: &str) -> Self {
        Self::with_extra(
            "tool",
            &ToolSourceExtra {
                call_id: call_id.to_string(),
            },
        )
    }

    /// Content a plugin contributed, in the given [`ContextForm`].
    #[must_use]
    pub fn plugin(plugin: &str, form: ContextForm) -> Self {
        Self::with_extra(
            "plugin",
            &PluginSourceExtra {
                plugin: plugin.to_string(),
                form: Some(form),
            },
        )
    }

    /// The flattened `extra` map of one of the typed views above.
    ///
    /// The views are plain structs that serialize to a JSON object, so the
    /// conversion cannot fail.
    fn with_extra<E: Serialize>(kind: &str, extra: &E) -> Self {
        let extra = match serde_json::to_value(extra)
            .expect("a source extra is plain data and always serializes")
        {
            Value::Object(map) => map,
            other => unreachable!("a source extra serializes to an object, got {other:?}"),
        };
        Self {
            kind: kind.to_string(),
            extra,
        }
    }
}

/// Producer kind: `user` | `plugin` | `model` | `tool` — the TS type is a
/// merge-extensible sum type, so keep this a free string and match on known
/// values; unknown kinds must not reject the log.
pub type SourceKind = String;

/// The kind of information in producer-supplied context (declared beside
/// provenance; semantic, never visual).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "lowercase")]
pub enum ContextForm {
    /// Standing instructions for the model.
    Instructions,
    /// An enumeration the model may pick from.
    Catalog,
    /// A point-in-time view assembled from named contributions.
    Snapshot {
        /// The named contributions, in assembly order.
        sections: Vec<ContextSnapshotSection>,
    },
    /// An account of something the harness did.
    Notice {
        /// One-line account of what happened (≤120 chars, ellipsized).
        summary: String,
    },
    /// Content relayed verbatim from elsewhere.
    Relay,
    /// Content recalled from earlier in the session.
    Recall,
}

/// One named contribution of a [`ContextForm::Snapshot`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSnapshotSection {
    /// Name of the contributing source.
    pub name: String,
    /// What it contributed.
    pub text: String,
}

/// Typed view over [`MessageSource::extra`] for `model` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSourceExtra {
    /// Provider route id.
    pub provider: String,
    /// Model id on that route.
    pub model: String,
    /// Adapter-private lossless-JSON replay state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_state: Option<Value>,
}

/// Typed view over [`MessageSource::extra`] for `tool` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSourceExtra {
    /// The call this message answers.
    pub call_id: String,
}

/// Typed view over [`MessageSource::extra`] for `plugin` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginSourceExtra {
    /// Name of the contributing plugin.
    pub plugin: String,
    /// What kind of content it contributed.
    ///
    /// Flattened: a [`ContextForm`] is internally tagged on `form`, and its
    /// own fields (a `notice`'s `summary`, a `snapshot`'s `sections`) sit
    /// beside `plugin` in the extras rather than nested under a `form` key.
    #[serde(default, flatten)]
    pub form: Option<ContextForm>,
}

/// Model-visible content block (discriminated by `type`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentBlock {
    /// Plain text visible to the end user.
    /// The text.
    Text {
        /// The text.
        text: String,
    },
    /// Reasoning / thinking content, distinct from visible text.
    Reasoning {
        /// The reasoning text.
        text: String,
    },
    /// Durable raster image reference (attachment service metadata only).
    Image {
        /// Metadata of the referenced attachment.
        attachment: ImageAttachmentRef,
    },
    /// A tool invocation requested by the model.
    ToolCall {
        /// Correlates the call with its result.
        id: String,
        /// Tool name the model asked for.
        name: String,
        /// Raw JSON string as produced by the model.
        arguments: String,
    },
    /// The result of a tool invocation, sent back to the model.
    ToolResult {
        #[serde(rename = "toolCallId")]
        /// The id of the tool call this answers.
        tool_call_id: String,
        /// What the tool returned, as model-facing blocks.
        content: Vec<ContentBlock>,
        #[serde(rename = "isError", default, skip_serializing_if = "Option::is_none")]
        /// `true` when the call failed.
        is_error: Option<bool>,
    },
}

/// Durable, serializable metadata for one immutable image object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageAttachmentRef {
    /// Opaque storage identifier; never a filesystem path or bearer URL.
    pub attachment_id: String,
    /// One of `image/png` | `image/jpeg` | `image/webp` | `image/gif`.
    pub media_type: String,
    /// Exact encoded byte length.
    pub bytes: u64,
    /// Intrinsic encoded width in pixels.
    pub width: u64,
    /// Intrinsic encoded height in pixels.
    pub height: u64,
    /// Optional display name, stripped of local path information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Raw streaming protocol emitted by adapters (the `assistant/chunk` payload).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StreamChunk {
    /// A new stream block opened.
    BlockStart {
        /// Index of the block within the stream.
        index: u64,
        /// One of `text` | `reasoning` | `image` | `tool-call` | `tool-result`.
        #[serde(rename = "blockType")]
        block_type: String,
    },
    /// More visible text arrived.
    TextDelta {
        /// Index of the block this extends.
        index: u64,
        /// The added text.
        text: String,
    },
    /// More reasoning text arrived.
    ReasoningDelta {
        /// Index of the block this extends.
        index: u64,
        /// The added text.
        text: String,
    },
    /// More of a tool call's argument JSON arrived.
    ToolCallDelta {
        /// Index of the block this extends.
        index: u64,
        /// Correlates the call with its result.
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        /// Tool name, once the adapter has resolved it.
        name: Option<String>,
        #[serde(rename = "argumentsDelta")]
        /// The added slice of the raw argument JSON.
        arguments_delta: String,
    },
    /// Carries the assembled block.
    BlockEnd {
        /// Index of the block that ended.
        index: u64,
        /// The block as assembled.
        block: ContentBlock,
    },
    /// The adapter reported token accounting.
    Usage {
        /// What it reported.
        usage: TokenUsage,
    },
    /// Terminal. `replayState` is adapter-private lossless-JSON state.
    Finish {
        /// Why the response stopped.
        reason: FinishReason,
        #[serde(
            rename = "replayState",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        /// Adapter-private lossless-JSON replay state.
        replay_state: Option<Value>,
    },
}

/// Why a model response stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FinishReason {
    /// The model ran out of things to say.
    Stop,
    /// The model asked for tools and is waiting on them.
    ToolCalls,
    /// The response hit its output-token ceiling.
    MaxTokens,
    /// A cancellation cut the response short.
    Aborted {
        /// What the adapter reported about the cancellation.
        failure: LlmFailure,
    },
    /// The call failed.
    Error {
        /// The provider or transport facts of the failure.
        failure: LlmFailure,
    },
}

/// Serializable provider or transport failure facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmFailure {
    /// Human-readable account of the failure.
    pub message: String,
    /// Stable provider-neutral machine-routing code.
    pub code: String,
    /// HTTP status returned by the provider, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u64>,
    /// Provider-requested delay in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_retry_after_ms: Option<f64>,
    /// Opaque provider-issued request identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Token accounting for one model call.
/// Counts are DISJOINT: `input_tokens` is uncached input only; billed input =
/// `input_tokens + cache_read_tokens + cache_write_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    /// Uncached input tokens.
    pub input_tokens: u64,
    /// Output tokens, reasoning included.
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Input tokens served from the provider's cache.
    pub cache_read_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Input tokens written into the provider's cache.
    pub cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The share of the output tokens that was reasoning.
    pub reasoning_tokens: Option<u64>,
}

// =============================================================================
// Plugin-merged event payloads
// =============================================================================

/// `permission/preset` — durable, log-only user intent (the selected preset).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionPresetData {
    /// The preset id.
    pub preset: String,
}

/// `sandbox/mode` — session sandbox-mode override; last event wins.
/// `mode` ∈ `read-only` | `workspace-write` | `danger-full-access`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxModeData {
    /// The sandbox mode to run under.
    pub mode: String,
    /// Present only when the override was seeded into a child at delegation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceMarker>,
}

/// `approval/policy` — session approval-policy override; last event wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalPolicyData {
    /// `ask` (default) or `never`.
    pub policy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Present only when the override was seeded into a child at delegation.
    pub source: Option<SourceMarker>,
}

/// `approval/asked` — an approval question was put to the answerer chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalAskedData {
    /// Pairs with the `approval/decided` that always follows.
    pub id: String,
    /// The tool the question is about.
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The call the question is about, when there is one.
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Why the approval was needed.
    pub reason: Option<String>,
}

/// `approval/decided` — outcome of a prior `approval/asked` (same `id`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalDecidedData {
    /// The `approval/asked` this decides.
    pub id: String,
    /// `allowed-once` | `rejected` | `cancelled` | `unavailable`.
    pub outcome: String,
}

/// `agent/inbox/spliced` — one normalized mutation of an agent's durable
/// pending-message lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InboxSplicedData {
    /// `next-turn` | `next-step`.
    pub target: String,
    /// Index the splice starts at.
    pub start: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// How many entries the splice removed.
    pub removed_count: Option<u64>,
    /// The entries the splice inserted.
    pub inserted: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// How the splice ended, when the backend reported it.
    pub outcome: Option<String>,
}

/// `session/title` — latest-wins session title snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTitleData {
    /// Normalized non-empty title text.
    pub title: String,
    /// Exact human `user/message` seqs used to derive this title; empty for an
    /// explicit user rename.
    pub message_seqs: Vec<u64>,
    /// Who supplied the title.
    pub source: TitleSource,
}

/// Whether the built-in fallback, a registered provider, or the user supplied
/// the title.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TitleSource {
    /// The built-in fallback named the session.
    Fallback,
    /// A title provider named the session.
    Provider {
        /// The title provider's id.
        provider: String,
        /// The route it answered on, when the provider reported one.
        model: Option<ModelProvenance>,
    },
    /// Explicit user rename: pins the title.
    User,
}

/// Provider/model provenance (route identity).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProvenance {
    /// Provider route id.
    pub provider: String,
    /// Model id on that route.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Context window of the route, in tokens, when it is known.
    pub context_window: Option<u64>,
}

/// `session/title-llm-request` — exact auxiliary title request, pre-dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TitleLlmRequestData {
    /// Registered title-provider identity.
    pub title_provider: String,
    /// Exact human `user/message` seqs the title was derived from; empty for
    /// an explicit user rename.
    pub message_seqs: Vec<u64>,
    /// Exact auxiliary LLM route.
    pub route: ModelProvenance,
    /// Exact auxiliary system prompt.
    pub system: String,
    /// Exact auxiliary message list.
    pub messages: Vec<Message>,
    /// Exact auxiliary output-token cap.
    pub max_tokens: u64,
}

/// `compaction/start` — marks the start of a compaction (log-only lock).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionStartData {
    /// Pairs this with its `compaction/end`.
    pub compaction_id: String,
    /// Human command that initiated a manual compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_command_id: Option<String>,
    /// Numbered owner turn; `null` for a standalone manual transaction
    /// between turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u64>,
}

/// `compaction/end` — marks the end of a compaction; `error` records an
/// unsuccessful attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEndData {
    /// The `compaction/start` this closes.
    pub compaction_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The slash command that asked for the compaction, when one did.
    pub source_command_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The turn the compaction ran in, when it ran inside one.
    pub turn: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// What went wrong, when the compaction failed.
    pub error: Option<String>,
}

/// `compaction/summary` — completed summary, its inputs, and model call facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryData {
    /// The compaction run that produced this summary.
    pub compaction_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The slash command that asked for the compaction, when one did.
    pub source_command_id: Option<String>,
    /// The summary content.
    pub summary: Vec<ContentBlock>,
    /// First/last surface-node seqs of the replaced range (a surface-POSITION
    /// span — after a prior replace, `start` can be GREATER than `end`).
    pub shadowed_range: RangeSpan,
    /// Seqs of all shadowed surface nodes, in surface order.
    pub shadowed_seqs: Vec<u64>,
    /// Token count of the events the summary shadows.
    pub shadowed_token_count: u64,
    /// The provider route that wrote the summary.
    pub provider: String,
    /// The model that wrote the summary.
    pub model: String,
    /// The generation cap the summarize call sent, when one applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Provider-reported token usage for the summarization request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Complete provider output; REQUIRED when `llm_stream_call` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Vec<ContentBlock>>,
    /// `true` when the summary identified exactly one call through the
    /// context's LLM seam.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_stream_call: Option<bool>,
}

/// `compaction/prune` — shadow price of one model-free prune replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionPruneData {
    /// Surface range the prune shadows.
    pub shadowed_range: RangeSpan,
    /// Seqs of the events it shadows.
    pub shadowed_seqs: Vec<u64>,
    /// Token count of the events it shadows.
    pub shadowed_token_count: u64,
}

/// An inclusive `start..=end` span of surface nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RangeSpan {
    /// First node, inclusive.
    pub start: u64,
    /// Last node, inclusive.
    pub end: u64,
}

/// `command/run` — a command started.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandRunData {
    /// Pairs this with its `command/done`.
    pub command_id: String,
    /// The command's name, without the leading slash.
    pub name: String,
    /// Absent when the command definition sets `recordInput: false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<String>,
    /// Currently only `{ kind: "user" }`.
    pub source: CommandSource,
}

/// `command/done` — the paired command settled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandDoneData {
    /// The `command/run` this closes.
    pub command_id: String,
    /// `success` | `error` (a thrown/aborted handler settles as `error`).
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// What the command reported, when it reported anything.
    pub text: Option<String>,
    /// A successful command may identify the authoritative domain event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_seq: Option<u64>,
}

/// Who issued a command line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CommandSource {
    /// The person at the keyboard invoked it.
    User,
}

/// `llm/retry` — one provider-routed retry scheduled after a failed attempt.
///
/// `max_retries` is present when `mode == "normal"` and absent for
/// `mode == "always"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmRetryData {
    /// Pairs this with its `llm/retry-started`.
    pub retry_id: String,
    /// The turn this belongs to.
    pub turn: u64,
    /// The step within the turn.
    pub step: u64,
    /// Provider route id.
    pub provider: String,
    /// `normal` | `always`.
    pub mode: String,
    /// Opaque policy key (serialized policy identity).
    pub policy_key: String,
    /// Which retry this is, counting from one.
    pub retry: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The retry ceiling of the policy, when it has one.
    pub max_retries: Option<u64>,
    /// How long the lane waits before it retries.
    pub delay_ms: f64,
    /// The failure that caused the retry.
    pub failure: LlmFailure,
}

/// `llm/retry-started` — the retry wait completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmRetryStartedData {
    /// The `llm/retry` this starts.
    pub retry_id: String,
    /// The turn this belongs to.
    pub turn: u64,
    /// The step within the turn.
    pub step: u64,
    /// Which retry this is, counting from one.
    pub retry: u64,
}

/// `agent-preset/selected` — the session's agent preset was chosen after
/// creation, while the session was still blank.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPresetSelectedData {
    /// The chosen preset id.
    pub agent_preset: String,
}

/// `web/deepseek-search-llm-request` — secret-free auxiliary DeepSeek search
/// request. `body` is the provider's exact wire body (kept opaque).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchLlmRequestData {
    /// The endpoint the search was sent to.
    pub endpoint: String,
    /// `anthropic-version` header value.
    pub api_version: String,
    /// Exact JSON body sent to the provider.
    pub body: Value,
}

/// Marks an override seeded into a child at delegation (`"delegation"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceMarker {
    /// The value was seeded into a child at delegation.
    Delegation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_parses() {
        let line = r#"{"type":"session","version":0,"id":"s1","createdAt":1787764689062,"cwd":"/x","delegationDepth":0,"agentPreset":"standard"}"#;
        let l: LogLine = serde_json::from_str(line).expect("a header line parses");
        match l {
            LogLine::Session(h) => {
                assert_eq!(h.version, 0);
                assert_eq!(h.delegation_depth, 0);
                assert_eq!(h.agent_preset.as_deref(), Some("standard"));
            }
            other => panic!("a `session` line decodes to a header, got {other:?}"),
        }
    }

    #[test]
    fn turn_end_reasons() {
        for (json, name) in [
            (r#"{"kind":"completed"}"#, "completed"),
            (
                r#"{"kind":"aborted","reason":{"kind":"user"}}"#,
                "aborted/user",
            ),
            (
                r#"{"kind":"aborted","reason":{"kind":"hook","reason":"x"}}"#,
                "aborted/hook",
            ),
            (
                r#"{"kind":"error","error":{"message":"m","code":"C"}}"#,
                "error",
            ),
            (r#"{"kind":"max-tokens"}"#, "max-tokens"),
            (r#"{"kind":"interrupted"}"#, "interrupted"),
            (r#"{"kind":"blocked"}"#, "blocked"),
        ] {
            let r: TurnEndReason =
                serde_json::from_str(json).unwrap_or_else(|e| panic!("{name} parses: {e}"));
            assert!(matches!(
                r,
                TurnEndReason::Completed
                    | TurnEndReason::Aborted { .. }
                    | TurnEndReason::Blocked
                    | TurnEndReason::Error { .. }
                    | TurnEndReason::MaxTokens
                    | TurnEndReason::Interrupted
            ));
        }
    }

    #[test]
    fn surface_op_roundtrip() {
        assert!(matches!(
            serde_json::from_str::<SurfaceOp>("\"append\"").expect("`append` parses"),
            SurfaceOp::Append
        ));
        let r: SurfaceOp = serde_json::from_str(r#"{"op":"replace","start":3,"end":7}"#)
            .expect("a replace op parses");
        assert!(matches!(r, SurfaceOp::Replace { start: 3, end: 7 }));
        assert_eq!(
            serde_json::to_string(&r).expect("a surface op serializes"),
            r#"{"op":"replace","start":3,"end":7}"#
        );
    }
}
