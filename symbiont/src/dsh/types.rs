//! Rust/serde mirror of the DeepSeek Harness (DSH) session trajectory format.
//!
//! On disk: `~/.dsh/sessions/<project-key>/<session-dir>/session.jsonl.zstd`
//! where `<project-key>` is the cwd with `/` replaced by `-`.
//! The file is a concatenation of independent, checksummed Zstandard frames:
//! one frame holds the header line only, each later frame holds one append batch of JSONL records.
//! Each JSONL line decodes to a `LogLine`.

#![expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "Internal types only, TypedBuilder would be annoying here."
)]

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
/// Line 1 is always the [`SessionHeaderLine`] (`type: "session"`). Every
/// later line is a session event (envelope `seq`/`time`/`data`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(super) enum LogLine {
    /// Immutable session header (first line only).
    #[serde(rename = "session")]
    Session(SessionHeaderLine),

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
    #[serde(rename = "assistant/message")]
    /// The assembled assistant message of one step.
    AssistantMessage(Event<AssistantMessageData>),
    #[serde(rename = "tool/call")]
    /// The model asked for one tool invocation.
    ToolCall(Event<ToolCallData>),
    #[serde(rename = "tool/result")]
    /// One tool call answered.
    ToolResult(Event<ToolResultData>),
    #[serde(rename = "request/header")]
    /// The full header of the next model request.
    RequestHeader(Event<RequestHeaderData>),
    #[serde(rename = "request/context")]
    /// Route metadata of the next model request.
    RequestContext(Event<RequestContextData>),

    // ---- plugin-merged events (observed on disk, v0) ----
    #[serde(rename = "session/title")]
    /// A session title snapshot.
    SessionTitle(Event<SessionTitleData>),
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
    #[serde(rename = "llm/retry")]
    /// A model call failed and will be retried.
    LlmRetry(Event<LlmRetryData>),
    #[serde(rename = "llm/retry-started")]
    /// The retry announced by the matching `llm/retry` started.
    LlmRetryStarted(Event<LlmRetryStartedData>),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cwd: Option<String>,
    /// Session this one was forked from (seed lineage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) parent_session: Option<String>,
    /// How many leading events were inherited through a seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) seed_length: Option<u64>,
    /// Only present for subagent children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) origin: Option<Origin>,
    /// Delegation depth: `0` for top-level sessions; REQUIRED on disk.
    pub(super) delegation_depth: u32,
    /// Agent preset id; durable because it decides tools + prompt on resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) agent_preset: Option<String>,
}

/// Coarse origin marker for subagent sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Origin {
    /// The session is a child of a delegating parent.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) surface_op: Option<SurfaceOp>,
    /// Seqs of earlier events cited as sources (e.g. the chunk seqs that built
    /// an `assistant/message`). Surface events only; may be an empty array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) source_event_seqs: Option<Vec<u64>>,
    /// Written as `true` only on purely informational records a reader may
    /// safely skip when it does not recognize the event type. Absent means the
    /// event is REQUIRED — an unrecognized type without this marker must
    /// reject the log, not silently skip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) ignorable: Option<bool>,
}

impl<D> Event<D> {
    /// A log-only event: one that carries `data` but does not land on the
    /// ordered transcript.
    pub(super) fn new(seq: u64, time: i64, data: D) -> Self {
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
    pub(super) fn on_surface(mut self) -> Self {
        self.surface_op = Some(SurfaceOp::Append);
        self
    }
}

/// How a surface event entered the ordered surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SurfaceOp {
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
// Core event payloads (dsh-session `SessionEventMap`)
// =============================================================================

/// `turn/start`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TurnStartData {
    /// The turn that opened.
    pub(super) turn: u64,
}

/// `turn/end`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TurnEndData {
    /// The turn that closed.
    pub(super) turn: u64,
    /// Why it closed.
    pub(super) reason: TurnEndReason,
}

/// Why a turn ended (discriminated by `kind`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub(super) enum TurnEndReason {
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
pub(super) enum TurnEndCancelCause {
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
pub(super) struct StepStartData {
    /// The turn this step belongs to.
    pub(super) turn: u64,
    /// The step's index within the turn.
    pub(super) step: u64,
}

/// `assistant/message` — assembled assistant message for one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct AssistantMessageData {
    /// The turn this belongs to.
    pub(super) turn: u64,
    /// The step within the turn.
    pub(super) step: u64,
    /// The assembled assistant message.
    pub(super) message: Message,
    /// Present when the adapter reported token accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) usage: Option<TokenUsage>,
}

/// `tool/call` — the model requested one tool invocation.
/// `arguments` is the raw JSON string exactly as the model produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolCallData {
    /// The turn this belongs to.
    pub(super) turn: u64,
    /// The step within the turn.
    pub(super) step: u64,
    /// Correlates the call with its `tool/result`.
    pub(super) call_id: String,
    /// Tool name, as registered with the model.
    pub(super) name: String,
    /// The model's raw argument JSON, as a string.
    pub(super) arguments: String,
}

/// `tool/result` — a completed tool call's model-facing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolResultData {
    /// The turn this belongs to.
    pub(super) turn: u64,
    /// The step within the turn.
    pub(super) step: u64,
    /// Role `user`; content is exactly one `tool-result` block.
    pub(super) message: Message,
    /// Optional internal failure identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<ToolResultError>,
    /// Opaque tool-private presentation payload (JSON-serializable); e.g.
    /// `dsh-tool-fs` carries a result-time contextual diff here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) meta: Option<Value>,
}

/// The identity of a tool failure the harness recognized.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolResultError {
    /// Error name, for the reader.
    pub(super) name: String,
    /// Stable machine-routing code.
    pub(super) code: String,
}

/// `request/header` — full header for the next request (log-only; the latest
/// snapshot reconstructs the request header).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RequestHeaderData {
    /// The header the next request will carry.
    pub(super) header: EpochHeader,
    /// Why this snapshot was appended.
    pub(super) reason: RequestHeaderReason,
}

/// Why a `request/header` snapshot was appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum RequestHeaderReason {
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
pub(super) struct EpochHeader {
    /// Provider, model and sampling scalars.
    pub(super) config: LlmCallConfig,
    /// Effective config fields materialized from the exact adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) adapter_defaults: Option<LlmCallConfigAdapterDefaults>,
    /// Rendered system prompt text; absent for a system-less request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) system: Option<String>,
    /// Assembled tool schemas; absent for a tool-less request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tools: Option<Vec<ToolSchema>>,
}

/// Provider, model, reasoning effort, and sampling scalars of one
/// conversation's requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmCallConfig {
    /// Provider route id.
    pub(super) provider: String,
    /// Model id on that route.
    pub(super) model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Reasoning-effort level, when the route takes one.
    pub(super) reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Sampling temperature, when the route takes one.
    pub(super) temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Output-token ceiling, when one was set.
    pub(super) max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Stop sequences, when any were set.
    pub(super) stop: Option<Vec<String>>,
}

/// Only ever `true` where present (a marker that the adapter supplied the
/// default).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmCallConfigAdapterDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// `true` when the adapter, not the caller, set the reasoning effort.
    pub(super) reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// `true` when the adapter, not the caller, set the output ceiling.
    pub(super) max_tokens: Option<bool>,
}

/// JSON-schema description of a tool, as sent to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolSchema {
    /// Tool name the model calls.
    pub(super) name: String,
    /// What the tool does, in the model's words.
    pub(super) description: String,
    /// JSON Schema object for the arguments (arbitrary JSON).
    pub(super) parameters: Value,
}

/// `request/context` — route metadata for the next request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RequestContextData {
    /// Provider route id.
    pub(super) provider: String,
    /// Model id on that route.
    pub(super) model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Context window of the route, in tokens, when it is known.
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
    /// Who the message speaks as.
    pub(super) role: Role,
    /// Exact model-facing blocks.
    pub(super) content: Vec<ContentBlock>,
    /// Producer-supplied provenance + plugin extras.
    pub(super) source: MessageSource,
}

/// Who a message speaks as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Role {
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
pub(super) struct MessageSource {
    /// Who produced this message.
    pub(super) kind: SourceKind,
    /// Kind-specific fields + forward-compatible plugin extras.
    #[serde(flatten)]
    pub(super) extra: Map<String, Value>,
}

impl MessageSource {
    /// A message the human (or, here, the harness acting as one) typed.
    #[must_use]
    pub(super) fn user() -> Self {
        Self {
            kind: "user".to_string(),
            extra: Map::new(),
        }
    }

    /// A message the model produced on `provider`'s `model` route.
    #[must_use]
    pub(super) fn model(provider: &str, model: &str) -> Self {
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
    pub(super) fn tool(call_id: &str) -> Self {
        Self::with_extra(
            "tool",
            &ToolSourceExtra {
                call_id: call_id.to_string(),
            },
        )
    }

    /// Content a plugin contributed, in the given [`ContextForm`].
    #[must_use]
    pub(super) fn plugin(plugin: &str, form: ContextForm) -> Self {
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
pub(super) type SourceKind = String;

/// The kind of information in producer-supplied context (declared beside
/// provenance; semantic, never visual).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "lowercase")]
pub(super) enum ContextForm {
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
pub(super) struct ContextSnapshotSection {
    /// Name of the contributing source.
    pub(super) name: String,
    /// What it contributed.
    pub(super) text: String,
}

/// Typed view over [`MessageSource::extra`] for `model` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelSourceExtra {
    /// Provider route id.
    pub(super) provider: String,
    /// Model id on that route.
    pub(super) model: String,
    /// Adapter-private lossless-JSON replay state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) replay_state: Option<Value>,
}

/// Typed view over [`MessageSource::extra`] for `tool` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolSourceExtra {
    /// The call this message answers.
    pub(super) call_id: String,
}

/// Typed view over [`MessageSource::extra`] for `plugin` sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PluginSourceExtra {
    /// Name of the contributing plugin.
    pub(super) plugin: String,
    /// What kind of content it contributed.
    ///
    /// Flattened: a [`ContextForm`] is internally tagged on `form`, and its
    /// own fields (a `notice`'s `summary`, a `snapshot`'s `sections`) sit
    /// beside `plugin` in the extras rather than nested under a `form` key.
    #[serde(default, flatten)]
    pub(super) form: Option<ContextForm>,
}

/// Model-visible content block (discriminated by `type`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(super) enum ContentBlock {
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) name: Option<String>,
}

/// Serializable provider or transport failure facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmFailure {
    /// Human-readable account of the failure.
    pub(super) message: String,
    /// Stable provider-neutral machine-routing code.
    pub(super) code: String,
    /// HTTP status returned by the provider, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) status: Option<u64>,
    /// Provider-requested delay in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) provider_retry_after_ms: Option<f64>,
    /// Opaque provider-issued request identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) request_id: Option<String>,
}

/// Token accounting for one model call.
/// Counts are DISJOINT: `input_tokens` is uncached input only; billed input =
/// `input_tokens + cache_read_tokens + cache_write_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TokenUsage {
    /// Uncached input tokens.
    pub(super) input_tokens: u64,
    /// Output tokens, reasoning included.
    pub(super) output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Input tokens served from the provider's cache.
    pub(super) cache_read_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Input tokens written into the provider's cache.
    pub(super) cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The share of the output tokens that was reasoning.
    pub(super) reasoning_tokens: Option<u64>,
}

// =============================================================================
// Plugin-merged event payloads
// =============================================================================

/// `session/title` — latest-wins session title snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SessionTitleData {
    /// Normalized non-empty title text.
    pub(super) title: String,
    /// Exact human `user/message` seqs used to derive this title; empty for an
    /// explicit user rename.
    pub(super) message_seqs: Vec<u64>,
    /// Who supplied the title.
    pub(super) source: TitleSource,
}

/// Who supplied the title.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub(super) enum TitleSource {
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
pub(super) struct ModelProvenance {
    /// Provider route id.
    pub(super) provider: String,
    /// Model id on that route.
    pub(super) model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Context window of the route, in tokens, when it is known.
    pub(super) context_window: Option<u64>,
}

/// `compaction/start` — marks the start of a compaction (log-only lock).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionStartData {
    /// Pairs this with its `compaction/end`.
    pub(super) compaction_id: String,
    /// Human command that initiated a manual compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) source_command_id: Option<String>,
    /// Numbered owner turn; `null` for a standalone manual transaction
    /// between turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) turn: Option<u64>,
}

/// `compaction/end` — marks the end of a compaction; `error` records an
/// unsuccessful attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionEndData {
    /// The `compaction/start` this closes.
    pub(super) compaction_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The slash command that asked for the compaction, when one did.
    pub(super) source_command_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The turn the compaction ran in, when it ran inside one.
    pub(super) turn: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// What went wrong, when the compaction failed.
    pub(super) error: Option<String>,
}

/// `compaction/summary` — completed summary, its inputs, and model call facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionSummaryData {
    /// The compaction run that produced this summary.
    pub(super) compaction_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The slash command that asked for the compaction, when one did.
    pub(super) source_command_id: Option<String>,
    /// The summary content.
    pub(super) summary: Vec<ContentBlock>,
    /// First/last surface-node seqs of the replaced range (a surface-POSITION
    /// span — after a prior replace, `start` can be GREATER than `end`).
    pub(super) shadowed_range: RangeSpan,
    /// Seqs of all shadowed surface nodes, in surface order.
    pub(super) shadowed_seqs: Vec<u64>,
    /// Token count of the events the summary shadows.
    pub(super) shadowed_token_count: u64,
    /// The provider route that wrote the summary.
    pub(super) provider: String,
    /// The model that wrote the summary.
    pub(super) model: String,
    /// The generation cap the summarize call sent, when one applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) max_tokens: Option<u64>,
    /// Provider-reported token usage for the summarization request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) usage: Option<TokenUsage>,
    /// Complete provider output; REQUIRED when `llm_stream_call` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) raw_output: Option<Vec<ContentBlock>>,
    /// `true` when the summary identified exactly one call through the
    /// context's LLM seam.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) llm_stream_call: Option<bool>,
}

/// `compaction/prune` — shadow price of one model-free prune replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompactionPruneData {
    /// Surface range the prune shadows.
    pub(super) shadowed_range: RangeSpan,
    /// Seqs of the events it shadows.
    pub(super) shadowed_seqs: Vec<u64>,
    /// Token count of the events it shadows.
    pub(super) shadowed_token_count: u64,
}

/// An inclusive `start..=end` span of surface nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RangeSpan {
    /// First node, inclusive.
    pub(super) start: u64,
    /// Last node, inclusive.
    pub(super) end: u64,
}

/// `llm/retry` — one provider-routed retry scheduled after a failed attempt.
///
/// `max_retries` is present when `mode == "normal"` and absent for
/// `mode == "always"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmRetryData {
    /// Pairs this with its `llm/retry-started`.
    pub(super) retry_id: String,
    /// The turn this belongs to.
    pub(super) turn: u64,
    /// The step within the turn.
    pub(super) step: u64,
    /// Provider route id.
    pub(super) provider: String,
    /// `normal` | `always`.
    pub(super) mode: String,
    /// Opaque policy key (serialized policy identity).
    pub(super) policy_key: String,
    /// Which retry this is, counting from one.
    pub(super) retry: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The retry ceiling of the policy, when it has one.
    pub(super) max_retries: Option<u64>,
    /// How long the lane waits before it retries.
    pub(super) delay_ms: f64,
    /// The failure that caused the retry.
    pub(super) failure: LlmFailure,
}

/// `llm/retry-started` — the retry wait completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LlmRetryStartedData {
    /// The `llm/retry` this starts.
    pub(super) retry_id: String,
    /// The turn this belongs to.
    pub(super) turn: u64,
    /// The step within the turn.
    pub(super) step: u64,
    /// Which retry this is, counting from one.
    pub(super) retry: u64,
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
        let op: SurfaceOp = serde_json::from_str("\"append\"").expect("`append` parses");
        assert!(matches!(op, SurfaceOp::Append));
        assert_eq!(
            serde_json::to_string(&op).expect("a surface op serializes"),
            "\"append\""
        );
    }
}
