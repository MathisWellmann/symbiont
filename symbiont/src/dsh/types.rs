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
pub(super) enum LogLine {
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
    TurnStart(Event<TurnStartData>),
    #[serde(rename = "turn/end")]
    TurnEnd(Event<TurnEndData>),
    /// `step/end` has the identical payload shape.
    #[serde(rename = "step/start")]
    StepStart(Event<StepStartData>),
    #[serde(rename = "step/end")]
    StepEnd(Event<StepStartData>),
    /// The `data` payload is the whole user-role message.
    #[serde(rename = "user/message")]
    UserMessage(Event<Message>),
    #[serde(rename = "assistant/chunk")]
    AssistantChunk(Event<AssistantChunkData>),
    #[serde(rename = "assistant/message")]
    AssistantMessage(Event<AssistantMessageData>),
    #[serde(rename = "tool/call")]
    ToolCall(Event<ToolCallData>),
    #[serde(rename = "tool/result")]
    ToolResult(Event<ToolResultData>),
    #[serde(rename = "todo/write")]
    TodoWrite(Event<TodoWriteData>),
    #[serde(rename = "request/header")]
    RequestHeader(Event<RequestHeaderData>),
    #[serde(rename = "request/context")]
    RequestContext(Event<RequestContextData>),
    /// Empty payload (`data: {}`); marks the end of seed history.
    #[serde(rename = "session/end-seed")]
    EndSeed(Event<Value>),

    // ---- plugin-merged events (observed on disk, v0) ----
    #[serde(rename = "permission/preset")]
    PermissionPreset(Event<PermissionPresetData>),
    #[serde(rename = "sandbox/mode")]
    SandboxMode(Event<SandboxModeData>),
    #[serde(rename = "approval/policy")]
    ApprovalPolicy(Event<ApprovalPolicyData>),
    #[serde(rename = "approval/asked")]
    ApprovalAsked(Event<ApprovalAskedData>),
    #[serde(rename = "approval/decided")]
    ApprovalDecided(Event<ApprovalDecidedData>),
    #[serde(rename = "agent/inbox/spliced")]
    InboxSpliced(Event<InboxSplicedData>),
    #[serde(rename = "session/title")]
    SessionTitle(Event<SessionTitleData>),
    #[serde(rename = "session/title-llm-request")]
    TitleLlmRequest(Event<TitleLlmRequestData>),
    #[serde(rename = "compaction/start")]
    CompactionStart(Event<CompactionStartData>),
    #[serde(rename = "compaction/end")]
    CompactionEnd(Event<CompactionEndData>),
    #[serde(rename = "compaction/summary")]
    CompactionSummary(Event<CompactionSummaryData>),
    #[serde(rename = "compaction/prune")]
    CompactionPrune(Event<CompactionPruneData>),
    #[serde(rename = "command/run")]
    CommandRun(Event<CommandRunData>),
    #[serde(rename = "command/done")]
    CommandDone(Event<CommandDoneData>),
    #[serde(rename = "llm/retry")]
    LlmRetry(Event<LlmRetryData>),
    #[serde(rename = "llm/retry-started")]
    LlmRetryStarted(Event<LlmRetryStartedData>),
    #[serde(rename = "agent-preset/selected")]
    AgentPresetSelected(Event<AgentPresetSelectedData>),
    #[serde(rename = "web/deepseek-search-llm-request")]
    WebSearchLlmRequest(Event<WebSearchLlmRequestData>),
}

/// Immutable session header — the first JSONL line of every log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SessionHeaderLine {
    /// On-disk format version; `0` for all current logs.
    pub(super) version: u32,
    /// Session id (free-form branded string).
    pub(super) id: String,
    /// Unix epoch milliseconds.
    pub(super) created_at: u64,
    /// Absolute working directory the session was created in.
    #[serde(default)]
    pub(super) cwd: Option<String>,
    /// Session this one was forked from (seed lineage).
    #[serde(default)]
    pub(super) parent_session: Option<String>,
    /// How many leading events were inherited through a seed.
    #[serde(default)]
    pub(super) seed_length: Option<u64>,
    /// Only present for subagent children.
    #[serde(default)]
    pub(super) origin: Option<Origin>,
    /// Delegation depth: `0` for top-level sessions; REQUIRED on disk.
    pub(super) delegation_depth: u32,
    /// Agent preset id; durable because it decides tools + prompt on resume.
    #[serde(default)]
    pub(super) agent_preset: Option<String>,
}

/// Coarse origin marker for subagent sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Origin {
    Subagent,
}

// =============================================================================
// Event envelope
// =============================================================================

/// Shared envelope of every session event (all non-header, non-chunk-row lines).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Event<D> {
    /// Monotonic sequence number. Contiguous from 0 across the decoded log
    /// (packed chunk rows expand to `len` members), so for plain events
    /// `events[i].seq == i`.
    pub(super) seq: u64,
    /// Unix epoch milliseconds.
    pub(super) time: i64,
    /// Type-specific payload.
    pub(super) data: D,
    /// How this event entered the ordered surface. Present on surface events
    /// (`user/message`, `assistant/message`, `tool/result`); absent otherwise.
    #[serde(default)]
    pub(super) surface_op: Option<SurfaceOp>,
    /// Seqs of earlier events cited as sources (e.g. the chunk seqs that built
    /// an `assistant/message`). Surface events only; may be an empty array.
    #[serde(default)]
    pub(super) source_event_seqs: Option<Vec<u64>>,
    /// Written as `true` only on purely informational records a reader may
    /// safely skip when it does not recognize the event type. Absent means the
    /// event is REQUIRED — an unrecognized type without this marker must
    /// reject the log, not silently skip.
    #[serde(default)]
    pub(super) ignorable: Option<bool>,
}

/// How a surface event entered the ordered surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SurfaceOp {
    /// JSON: `"append"` — added to the tail.
    Append,
    /// JSON: `{"op":"replace","start":s,"end":e}` — replaces surface nodes
    /// `start..=end` (both inclusive) with this one. Used by compaction.
    Replace { start: u64, end: u64 },
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
pub(super) struct ChunkRow<D> {
    /// Seq of the first member.
    pub(super) seq0: u64,
    /// Time of the first member (epoch ms).
    pub(super) time0: i64,
    /// Run payload.
    pub(super) data: D,
}

/// Payload of `text-chunks` / `reasoning-chunks` rows: one entry per member —
/// never joined, token boundaries are data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TextRunData {
    pub(super) turn: u64,
    pub(super) step: u64,
    /// The stream block index every member shares.
    pub(super) index: u64,
    /// Epoch-ms gaps between consecutive members; `len == members - 1`.
    pub(super) dt: Vec<i64>,
    /// Per-member text deltas, in order.
    pub(super) texts: Vec<String>,
}

/// Payload of `tool-call-chunks` rows: the run-constant call identity plus
/// each member's raw arguments fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolCallRunData {
    pub(super) turn: u64,
    pub(super) step: u64,
    pub(super) index: u64,
    pub(super) dt: Vec<i64>,
    /// Provider-issued call id (constant across the run).
    pub(super) id: String,
    /// Present iff every member carried it, with one uniform value.
    #[serde(default)]
    pub(super) name: Option<String>,
    /// Per-member raw arguments fragments, in order.
    pub(super) args: Vec<String>,
}

impl ChunkRow<TextRunData> {
    /// Expand to the exact original `assistant/chunk` events.
    ///
    /// The payload shape is shared by `text-chunks` and `reasoning-chunks`
    /// rows; pass `true` for the latter to reconstruct `reasoning-delta`
    /// chunks. Validates member/gap counts and returns `Err` for corrupt rows.
    pub(super) fn expand(
        &self,
        reasoning: bool,
    ) -> Result<Vec<(u64, i64, StreamChunk)>, &'static str> {
        let members = self.data.texts.len();
        if members < 3 || self.data.dt.len() != members - 1 {
            return Err("invalid chunk row: dt length must be members - 1 (min 3 members)");
        }
        let mut out = Vec::with_capacity(members);
        let mut time = self.time0;
        for (k, text) in self.data.texts.iter().enumerate() {
            if k > 0 {
                time = time.wrapping_add(self.data.dt[k - 1]);
            }
            let chunk = if reasoning {
                StreamChunk::ReasoningDelta {
                    index: self.data.index,
                    text: text.clone(),
                }
            } else {
                StreamChunk::TextDelta {
                    index: self.data.index,
                    text: text.clone(),
                }
            };
            out.push((self.seq0 + k as u64, time, chunk));
        }
        Ok(out)
    }
}

impl ChunkRow<ToolCallRunData> {
    /// Expand to the exact original `assistant/chunk` events.
    /// Validates member/gap counts and returns `Err` for corrupt rows.
    pub(super) fn expand(&self) -> Result<Vec<(u64, i64, StreamChunk)>, &'static str> {
        let members = self.data.args.len();
        if members < 3 || self.data.dt.len() != members - 1 {
            return Err("invalid chunk row: dt length must be members - 1 (min 3 members)");
        }
        let mut out = Vec::with_capacity(members);
        let mut time = self.time0;
        for (k, fragment) in self.data.args.iter().enumerate() {
            if k > 0 {
                time = time.wrapping_add(self.data.dt[k - 1]);
            }
            out.push((
                self.seq0 + k as u64,
                time,
                StreamChunk::ToolCallDelta {
                    index: self.data.index,
                    id: self.data.id.clone(),
                    name: self.data.name.clone(),
                    arguments_delta: fragment.clone(),
                },
            ));
        }
        Ok(out)
    }
}

// =============================================================================
// Core event payloads (dsh-session `SessionEventMap`)
// =============================================================================

/// `turn/start`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TurnStartData {
    pub(super) turn: u64,
}

/// `turn/end`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TurnEndData {
    pub(super) turn: u64,
    pub(super) reason: TurnEndReason,
}

/// Why a turn ended (discriminated by `kind`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub(super) enum TurnEndReason {
    Completed,
    /// A cancellation request interrupted the live turn.
    Aborted {
        reason: TurnEndCancelCause,
    },
    Blocked,
    /// Structured LLM failure (verbatim provider facts, or flattened).
    Error {
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
pub(super) enum TurnEndCancelCause {
    User,
    Parent,
    Hook {
        reason: String,
    },
    Disposed,
    /// Imports whose original coarse record carried no cause.
    Legacy,
}

/// `step/start` and `step/end` share this payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct StepStartData {
    pub(super) turn: u64,
    pub(super) step: u64,
}

/// `assistant/chunk` — raw stream chunk, token-level replay fidelity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct AssistantChunkData {
    pub(super) turn: u64,
    pub(super) step: u64,
    pub(super) chunk: StreamChunk,
}

/// `assistant/message` — assembled assistant message for one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct AssistantMessageData {
    pub(super) turn: u64,
    pub(super) step: u64,
    pub(super) message: Message,
    /// Present when the adapter reported token accounting.
    #[serde(default)]
    pub(super) usage: Option<TokenUsage>,
}

/// `tool/call` — the model requested one tool invocation.
/// `arguments` is the raw JSON string exactly as the model produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolCallData {
    pub(super) turn: u64,
    pub(super) step: u64,
    /// Correlates the call with its `tool/result`.
    pub(super) call_id: String,
    pub(super) name: String,
    pub(super) arguments: String,
}

/// `tool/result` — a completed tool call's model-facing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolResultData {
    pub(super) turn: u64,
    pub(super) step: u64,
    /// Role `user`; content is exactly one `tool-result` block.
    pub(super) message: Message,
    /// Optional internal failure identity.
    #[serde(default)]
    pub(super) error: Option<ToolResultError>,
    /// Opaque tool-private presentation payload (JSON-serializable); e.g.
    /// `dsh-tool-fs` carries a result-time contextual diff here.
    #[serde(default)]
    pub(super) meta: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolResultError {
    pub(super) name: String,
    pub(super) code: String,
}

/// `todo/write` — whole-list snapshot; latest write wins on replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TodoWriteData {
    pub(super) todos: Vec<TodoItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TodoItem {
    pub(super) content: String,
    pub(super) status: TodoStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// `request/header` — full header for the next request (log-only; the latest
/// snapshot reconstructs the request header).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RequestHeaderData {
    pub(super) header: EpochHeader,
    pub(super) reason: RequestHeaderReason,
}

/// Why a `request/header` snapshot was appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum RequestHeaderReason {
    Initial,
    Resume,
    Change,
}

/// Logged request state outside derived history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct EpochHeader {
    pub(super) config: LlmCallConfig,
    /// Effective config fields materialized from the exact adapter.
    #[serde(default)]
    pub(super) adapter_defaults: Option<LlmCallConfigAdapterDefaults>,
    /// Rendered system prompt text; absent for a system-less request.
    #[serde(default)]
    pub(super) system: Option<String>,
    /// Assembled tool schemas; absent for a tool-less request.
    #[serde(default)]
    pub(super) tools: Option<Vec<ToolSchema>>,
}

/// Provider, model, reasoning effort, and sampling scalars of one
/// conversation's requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmCallConfig {
    pub(super) provider: String,
    pub(super) model: String,
    #[serde(default)]
    pub(super) reasoning_effort: Option<String>,
    #[serde(default)]
    pub(super) temperature: Option<f64>,
    #[serde(default)]
    pub(super) max_tokens: Option<u64>,
    #[serde(default)]
    pub(super) stop: Option<Vec<String>>,
}

/// Only ever `true` where present (a marker that the adapter supplied the
/// default).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmCallConfigAdapterDefaults {
    #[serde(default)]
    pub(super) reasoning_effort: Option<bool>,
    #[serde(default)]
    pub(super) max_tokens: Option<bool>,
}

/// JSON-schema description of a tool, as sent to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolSchema {
    pub(super) name: String,
    pub(super) description: String,
    /// JSON Schema object for the arguments (arbitrary JSON).
    pub(super) parameters: Value,
}

/// `request/context` — route metadata for the next request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RequestContextData {
    pub(super) provider: String,
    pub(super) model: String,
    #[serde(default)]
    pub(super) context_window: Option<u64>,
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
pub(super) struct Message {
    /// Stable identity (free-form branded string).
    pub(super) id: String,
    pub(super) role: Role,
    /// Exact model-facing blocks.
    pub(super) content: Vec<ContentBlock>,
    /// Producer-supplied provenance + plugin extras.
    pub(super) source: MessageSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Role {
    System,
    User,
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
pub(super) struct MessageSource {
    pub(super) kind: SourceKind,
    /// Kind-specific fields + forward-compatible plugin extras.
    #[serde(flatten)]
    pub(super) extra: Map<String, Value>,
}

/// Producer kind: `user` | `plugin` | `model` | `tool` — the TS type is a
/// merge-extensible sum type, so keep this a free string and match on known
/// values; unknown kinds must not reject the log.
pub(super) type SourceKind = String;

/// The kind of information in producer-supplied context (declared beside
/// provenance; semantic, never visual).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "lowercase")]
pub(super) enum ContextForm {
    Instructions,
    Catalog,
    Snapshot {
        /// The named contributions, in assembly order.
        sections: Vec<ContextSnapshotSection>,
    },
    Notice {
        /// One-line account of what happened (≤120 chars, ellipsized).
        summary: String,
    },
    Relay,
    Recall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ContextSnapshotSection {
    pub(super) name: String,
    pub(super) text: String,
}

/// Typed view over [`MessageSource.extra`] for `model` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelSourceExtra {
    pub(super) provider: String,
    pub(super) model: String,
    /// Adapter-private lossless-JSON replay state.
    #[serde(default)]
    pub(super) replay_state: Option<Value>,
}

/// Typed view over [`MessageSource.extra`] for `tool` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolSourceExtra {
    pub(super) call_id: String,
}

/// Typed view over [`MessageSource.extra`] for `plugin` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PluginSourceExtra {
    pub(super) plugin: String,
    #[serde(default)]
    pub(super) form: Option<ContextForm>,
}

/// Model-visible content block (discriminated by `type`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(super) enum ContentBlock {
    /// Plain text visible to the end user.
    Text { text: String },
    /// Reasoning / thinking content, distinct from visible text.
    Reasoning { text: String },
    /// Durable raster image reference (attachment service metadata only).
    Image { attachment: ImageAttachmentRef },
    /// A tool invocation requested by the model.
    ToolCall {
        id: String,
        name: String,
        /// Raw JSON string as produced by the model.
        arguments: String,
    },
    /// The result of a tool invocation, sent back to the model.
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        content: Vec<ContentBlock>,
        #[serde(default)]
        is_error: Option<bool>,
    },
}

/// Durable, serializable metadata for one immutable image object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ImageAttachmentRef {
    /// Opaque storage identifier; never a filesystem path or bearer URL.
    pub(super) attachment_id: String,
    /// One of `image/png` | `image/jpeg` | `image/webp` | `image/gif`.
    pub(super) media_type: String,
    /// Exact encoded byte length.
    pub(super) bytes: u64,
    /// Intrinsic encoded width in pixels.
    pub(super) width: u64,
    /// Intrinsic encoded height in pixels.
    pub(super) height: u64,
    /// Optional display name, stripped of local path information.
    #[serde(default)]
    pub(super) name: Option<String>,
}

/// Raw streaming protocol emitted by adapters (the `assistant/chunk` payload).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(super) enum StreamChunk {
    BlockStart {
        index: u64,
        /// One of `text` | `reasoning` | `image` | `tool-call` | `tool-result`.
        #[serde(rename = "blockType")]
        block_type: String,
    },
    TextDelta {
        index: u64,
        text: String,
    },
    ReasoningDelta {
        index: u64,
        text: String,
    },
    ToolCallDelta {
        index: u64,
        id: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(rename = "argumentsDelta")]
        arguments_delta: String,
    },
    /// Carries the assembled block.
    BlockEnd {
        index: u64,
        block: ContentBlock,
    },
    Usage {
        usage: TokenUsage,
    },
    /// Terminal. `replayState` is adapter-private lossless-JSON state.
    Finish {
        reason: FinishReason,
        #[serde(default)]
        replay_state: Option<Value>,
    },
}

/// Why a model response stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub(super) enum FinishReason {
    Stop,
    ToolCalls,
    MaxTokens,
    Aborted { failure: LlmFailure },
    Error { failure: LlmFailure },
}

/// Serializable provider or transport failure facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmFailure {
    pub(super) message: String,
    /// Stable provider-neutral machine-routing code.
    pub(super) code: String,
    /// HTTP status returned by the provider, when available.
    #[serde(default)]
    pub(super) status: Option<u64>,
    /// Provider-requested delay in milliseconds.
    #[serde(default)]
    pub(super) provider_retry_after_ms: Option<f64>,
    /// Opaque provider-issued request identifier.
    #[serde(default)]
    pub(super) request_id: Option<String>,
}

/// Token accounting for one model call.
/// Counts are DISJOINT: `input_tokens` is uncached input only; billed input =
/// `input_tokens + cache_read_tokens + cache_write_tokens`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TokenUsage {
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    #[serde(default)]
    pub(super) cache_read_tokens: Option<u64>,
    #[serde(default)]
    pub(super) cache_write_tokens: Option<u64>,
    #[serde(default)]
    pub(super) reasoning_tokens: Option<u64>,
}

// =============================================================================
// Plugin-merged event payloads
// =============================================================================

/// `permission/preset` — durable, log-only user intent (the selected preset).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PermissionPresetData {
    pub(super) preset: String,
}

/// `sandbox/mode` — session sandbox-mode override; last event wins.
/// `mode` ∈ `read-only` | `workspace-write` | `danger-full-access`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SandboxModeData {
    pub(super) mode: String,
    /// Present only when the override was seeded into a child at delegation.
    #[serde(default)]
    pub(super) source: Option<SourceMarker>,
}

/// `approval/policy` — session approval-policy override; last event wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ApprovalPolicyData {
    /// `ask` (default) or `never`.
    pub(super) policy: String,
    #[serde(default)]
    pub(super) source: Option<SourceMarker>,
}

/// `approval/asked` — an approval question was put to the answerer chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ApprovalAskedData {
    /// Pairs with the `approval/decided` that always follows.
    pub(super) id: String,
    pub(super) tool_name: String,
    #[serde(default)]
    pub(super) call_id: Option<String>,
    #[serde(default)]
    pub(super) reason: Option<String>,
}

/// `approval/decided` — outcome of a prior `approval/asked` (same `id`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ApprovalDecidedData {
    pub(super) id: String,
    /// `allowed-once` | `rejected` | `cancelled` | `unavailable`.
    pub(super) outcome: String,
}

/// `agent/inbox/spliced` — one normalized mutation of an agent's durable
/// pending-message lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct InboxSplicedData {
    /// `next-turn` | `next-step`.
    pub(super) target: String,
    pub(super) start: u64,
    #[serde(default)]
    pub(super) removed_count: Option<u64>,
    pub(super) inserted: Vec<Message>,
    #[serde(default)]
    pub(super) outcome: Option<String>,
}

/// `session/title` — latest-wins session title snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SessionTitleData {
    /// Normalized non-empty title text.
    pub(super) title: String,
    /// Exact human `user/message` seqs used to derive this title; empty for an
    /// explicit user rename.
    pub(super) message_seqs: Vec<u64>,
    pub(super) source: TitleSource,
}

/// Whether the built-in fallback, a registered provider, or the user supplied
/// the title.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub(super) enum TitleSource {
    Fallback,
    Provider {
        provider: String,
        model: Option<ModelProvenance>,
    },
    /// Explicit user rename: pins the title.
    User,
}

/// Provider/model provenance (route identity).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelProvenance {
    pub(super) provider: String,
    pub(super) model: String,
    #[serde(default)]
    pub(super) context_window: Option<u64>,
}

/// `session/title-llm-request` — exact auxiliary title request, pre-dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TitleLlmRequestData {
    /// Registered title-provider identity.
    pub(super) title_provider: String,
    pub(super) message_seqs: Vec<u64>,
    /// Exact auxiliary LLM route.
    pub(super) route: ModelProvenance,
    /// Exact auxiliary system prompt.
    pub(super) system: String,
    /// Exact auxiliary message list.
    pub(super) messages: Vec<Message>,
    /// Exact auxiliary output-token cap.
    pub(super) max_tokens: u64,
}

/// `compaction/start` — marks the start of a compaction (log-only lock).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionStartData {
    pub(super) compaction_id: String,
    /// Human command that initiated a manual compaction.
    #[serde(default)]
    pub(super) source_command_id: Option<String>,
    /// Numbered owner turn; `null` for a standalone manual transaction
    /// between turns.
    #[serde(default)]
    pub(super) turn: Option<u64>,
}

/// `compaction/end` — marks the end of a compaction; `error` records an
/// unsuccessful attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionEndData {
    pub(super) compaction_id: String,
    #[serde(default)]
    pub(super) source_command_id: Option<String>,
    #[serde(default)]
    pub(super) turn: Option<u64>,
    #[serde(default)]
    pub(super) error: Option<String>,
}

/// `compaction/summary` — completed summary, its inputs, and model call facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionSummaryData {
    pub(super) compaction_id: String,
    #[serde(default)]
    pub(super) source_command_id: Option<String>,
    /// The summary content.
    pub(super) summary: Vec<ContentBlock>,
    /// First/last surface-node seqs of the replaced range (a surface-POSITION
    /// span — after a prior replace, `start` can be GREATER than `end`).
    pub(super) shadowed_range: RangeSpan,
    /// Seqs of all shadowed surface nodes, in surface order.
    pub(super) shadowed_seqs: Vec<u64>,
    pub(super) shadowed_token_count: u64,
    /// The provider route that wrote the summary.
    pub(super) provider: String,
    /// The model that wrote the summary.
    pub(super) model: String,
    /// The generation cap the summarize call sent, when one applied.
    #[serde(default)]
    pub(super) max_tokens: Option<u64>,
    /// Provider-reported token usage for the summarization request.
    #[serde(default)]
    pub(super) usage: Option<TokenUsage>,
    /// Complete provider output; REQUIRED when `llm_stream_call` is set.
    #[serde(default)]
    pub(super) raw_output: Option<Vec<ContentBlock>>,
    /// `true` when the summary identified exactly one call through the
    /// context's LLM seam.
    #[serde(default)]
    pub(super) llm_stream_call: Option<bool>,
}

/// `compaction/prune` — shadow price of one model-free prune replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionPruneData {
    pub(super) shadowed_range: RangeSpan,
    pub(super) shadowed_seqs: Vec<u64>,
    pub(super) shadowed_token_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RangeSpan {
    pub(super) start: u64,
    pub(super) end: u64,
}

/// `command/run` — a command started.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CommandRunData {
    pub(super) command_id: String,
    pub(super) name: String,
    /// Absent when the command definition sets `recordInput: false`.
    #[serde(default)]
    pub(super) args: Option<String>,
    /// Currently only `{ kind: "user" }`.
    pub(super) source: CommandSource,
}

/// `command/done` — the paired command settled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CommandDoneData {
    pub(super) command_id: String,
    /// `success` | `error` (a thrown/aborted handler settles as `error`).
    pub(super) kind: String,
    #[serde(default)]
    pub(super) text: Option<String>,
    /// A successful command may identify the authoritative domain event.
    #[serde(default)]
    pub(super) source_event_seq: Option<u64>,
}

/// Who issued a command line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub(super) enum CommandSource {
    User,
}

/// `llm/retry` — one provider-routed retry scheduled after a failed attempt.
///
/// `max_retries` is present when `mode == "normal"` and absent for
/// `mode == "always"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmRetryData {
    pub(super) retry_id: String,
    pub(super) turn: u64,
    pub(super) step: u64,
    pub(super) provider: String,
    /// `normal` | `always`.
    pub(super) mode: String,
    /// Opaque policy key (serialized policy identity).
    pub(super) policy_key: String,
    pub(super) retry: u64,
    #[serde(default)]
    pub(super) max_retries: Option<u64>,
    pub(super) delay_ms: f64,
    pub(super) failure: LlmFailure,
}

/// `llm/retry-started` — the retry wait completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmRetryStartedData {
    pub(super) retry_id: String,
    pub(super) turn: u64,
    pub(super) step: u64,
    pub(super) retry: u64,
}

/// `agent-preset/selected` — the session's agent preset was chosen after
/// creation, while the session was still blank.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct AgentPresetSelectedData {
    pub(super) agent_preset: String,
}

/// `web/deepseek-search-llm-request` — secret-free auxiliary DeepSeek search
/// request. `body` is the provider's exact wire body (kept opaque).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct WebSearchLlmRequestData {
    pub(super) endpoint: String,
    /// `anthropic-version` header value.
    pub(super) api_version: String,
    /// Exact JSON body sent to the provider.
    pub(super) body: Value,
}

/// Marks an override seeded into a child at delegation (`"delegation"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum SourceMarker {
    Delegation,
}

// =============================================================================
// Reader helpers
// =============================================================================

/// Decode a whole `session.jsonl.zstd` file.
///
/// The zstd `Decoder` transparently consumes the file's independent frames
/// (one checksummed header frame + one frame per durable append batch) as a
/// single stream. Returns the header and the remaining records in log order.
pub(super) fn load_session_log(
    path: &std::path::Path,
) -> Result<(SessionHeaderLine, Vec<LogLine>), Box<dyn std::error::Error>> {
    use std::io::Read;
    let raw = std::fs::read(path)?;
    let mut jsonl = Vec::new();
    zstd::Decoder::new(&raw[..])?.read_to_end(&mut jsonl)?;
    let text = String::from_utf8(jsonl)?;

    let mut records: Vec<LogLine> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let rec = serde_json::from_str::<LogLine>(line).map_err(|e| format!("line {i}: {e}"))?;
        records.push(rec);
    }
    let (first, rest) = records
        .split_first()
        .ok_or_else(|| "empty log".to_string())?;
    let header = match first {
        LogLine::Session(h) => h.clone(),
        _ => return Err("first line is not a `session` header".into()),
    };
    Ok((header, rest.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_parses() {
        let line = r#"{"type":"session","version":0,"id":"s1","createdAt":1787764689062,"cwd":"/x","delegationDepth":0,"agentPreset":"standard"}"#;
        let l: LogLine = serde_json::from_str(line).unwrap();
        match l {
            LogLine::Session(h) => {
                assert_eq!(h.version, 0);
                assert_eq!(h.delegation_depth, 0);
                assert_eq!(h.agent_preset.as_deref(), Some("standard"));
            }
            _ => panic!(),
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
            let r: TurnEndReason = serde_json::from_str(json).unwrap();
            let _ = name;
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
            serde_json::from_str::<SurfaceOp>("\"append\"").unwrap(),
            SurfaceOp::Append
        ));
        let r: SurfaceOp = serde_json::from_str(r#"{"op":"replace","start":3,"end":7}"#).unwrap();
        assert!(matches!(r, SurfaceOp::Replace { start: 3, end: 7 }));
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"op":"replace","start":3,"end":7}"#
        );
    }

    #[test]
    fn chunk_row_expands() {
        let row: ChunkRow<TextRunData> = serde_json::from_str(
            r#"{"type":"text-chunks","seq0":100,"time0":1000,"data":{"turn":1,"step":1,"index":1,"dt":[2,3],"texts":["a","b","c"]}}"#,
        )
        .unwrap();
        let members = row.expand(false).unwrap();
        assert_eq!(members.len(), 3);
        assert_eq!(members[0].0, 100);
        assert_eq!(members[1].1, 1002);
        assert_eq!(members[2].1, 1005);
        assert!(matches!(
            members[2].2,
            StreamChunk::TextDelta { index: 1, .. }
        ));
    }
}
