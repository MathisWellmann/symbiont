# Evolution Trace — Design Proposal

Goal: any caller of `Runtime::evolve` / `evolve_batch` / `evolve_batch_stream` should be able to
get the **full agent chat trajectory** of an evolution — every inference request and response,
every tool call and result, every retry (self-healing nudges, transient HTTP backoff, context
resets, repeat resets) with its diagnostics, per-turn token usage, and per-stage timings — as
structured data for later inspection, in the way other agent harnesses capture traces (OpenAI
Agents SDK request/response spans, Claude Code session JSONL).

The system prompt is deliberately **not** part of the trace. It is byte-identical across every
attempt of a lane and every lane of a batch, so per-trace storage is pure duplication of what is
routinely the largest single string in the process. Hosts reconstruct it by calling
`symbiont::system_prompt`, which this proposal promotes to public API (§4.5).

This is a design proposal; no code changes are implied yet.

---

## 1. What is already available (capture-point inventory)

| Item | Where it lives today |
|---|---|
| Per-attempt chat (prompt turn + assistant turns + tool calls/results) | `AgentRun.new_messages` (`= PromptResponse.messages`), already returned per run; `evolve_lane` extends its local `history: Vec<Message>` with it |
| Per-attempt aggregate token usage | `AgentRun.usage` (= `PromptResponse.usage`) |
| Per-HTTP-call details | `PromptResponse.completion_calls: Vec<CompletionCall>` — **dropped today** by the blanket impl in `evolution_agent.rs:67-71`. Far richer than a token count; see below |
| Tool calls + results | Inline in `new_messages` as `Assistant(ToolCall)` / `User(ToolResult)` content — derivable, no extra capture |
| Self-healing ladder decisions | Local to `evolve_lane` (error kind, nudge prompt, backoff, reset) — recordable in place |
| LLM and parse/validate timings | Measured by `evolve_no_backpressure` for `PIPELINE_STAGE_DURATION` (`runtime.rs:391`, `runtime.rs:407`) |
| Build-stage timings | Measured by `build_and_register` (`runtime.rs:458-534`), **not** by `evolve_lane`, and **absent on the dedup path** — see the row below |
| Source-dedup reuse | `build_and_register` returns early at `runtime.rs:467-473` when the candidate is byte-identical to a registered revision. No compile, no load, no timings — only `REVISION_DEDUP_HITS` |
| System prompt | Built by `system_prompt.rs:145` (`pub(crate)`), passed to `.preamble(…)` at `inference.rs:135`. **Not** a public rig field — see below |
| Per-HTTP request payload size | `MeteredHttpClient` (`REQUEST_BODY_BYTES`) |
| Fine-grained model-call spans | Rig's native OTel GenAI `tracing` spans (target `rig::agent_chat`, `gen_ai.*` fields) |

**`CompletionCall` is not a token counter.** Its actual shape
(`rig-agent/src/agent/prompt_request/mod.rs:331-381`) is:

| Field | Type | Note |
|---|---|---|
| `call_index` | `usize` | zero-based within the run |
| `usage` | `Usage` | not `Option`; all-zero is rig's documented "provider reported nothing" sentinel |
| `message_id` | `Option<String>` | provider-assigned assistant message id |
| `response_id` | `Option<String>` | provider-assigned response-scoped id |
| `provider_request_id` | `Option<String>` | transport request id (e.g. Anthropic `request-id`) |
| `finish_reason` | `Option<FinishReason>` | distinguishes a turn truncated at the output-token limit from a turn that had nothing to say |
| `raw` | `serde_json::Value` | **the provider's full response body, per call** |

`finish_reason` is the single most diagnostically valuable field here and costs nothing. `raw` is
the opposite: it would be the dominant memory term of the whole design while duplicating text the
trace already holds, so the blanket impl drops it (§6).

**The system prompt is not reachable from a rig `Agent`.** In rig 0.42,
`pub struct Agent { pub(crate) config: AgentConfig, pub(crate) tool_server_handle: ToolServerHandle }`
(`agent/completion.rs:573`); `preamble` is a `pub(crate)` field of `AgentConfig`
(`agent/completion.rs:593`) with no accessor — only builder setters (`builder.rs:141`,
`runner.rs:261`, `prompt_request/mod.rs:56`). The `pub preamble` at `hook.rs:840` belongs to
`RequestPatch` and is a per-turn override, not the agent's configured value. This is one of two
reasons the trace excludes it; the other is size (§2, §6).

**Key invariant.** The lane's `history` is the lane's complete ordered transcript. Rig's
`PromptResponse.messages` is the run's own `new_messages` — the prompt turn plus everything the
run produced — not input concatenated with new
(`rig-agent/src/agent/run/mod.rs:1092`), and the lane appends it wholesale
(`runtime.rs:388`). Attempt *k* therefore sends `system_prompt + history[..p_k]` and produces
`history[p_k..p_{k+1}]`, where `p_k` is the index of that attempt's user-prompt turn. Storing the
final `history` **once** plus per-attempt boundary indices captures the full trajectory in linear
memory — no O(attempts²) duplication of the (large, code-heavy) message content.

## 2. Data model

New module `symbiont/src/evolution_trace.rs`:

```rust
/// The full agent trajectory of one lane (or of a single-prompt `evolve`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvolutionTrace {
    /// Lane index; 0 for single-prompt `evolve`.
    pub lane: usize,
    /// The prompt the lane started with, before any nudge.
    pub base_prompt: String,
    /// The lane's complete ordered transcript: every user turn (base prompt,
    /// each corrective nudge), every assistant turn, every tool call and
    /// result. Owned once; attempts reference into it.
    pub history: Vec<Message>,
    /// One entry per lane iteration, in order. An iteration whose inference
    /// call failed outright is present too, with `run: None`.
    pub attempts: Vec<AttemptTrace>,
    pub outcome: TraceOutcome,
    /// Wall time of the whole lane.
    pub duration: Duration,
}

/// One iteration of the lane's self-healing ladder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttemptTrace {
    /// Position in the lane timeline. Dense: always `0..attempts.len()`.
    pub seq: usize,
    /// The lane's self-healing attempt counter at this iteration; matches
    /// `EvolveFailure::attempt` and the `attempt` metric label. **Not**
    /// unique across entries: a transient HTTP retry deliberately does not
    /// consume the budget (`runtime.rs:1105` decrements it), so consecutive
    /// entries can carry the same value. Index by `seq`, report `attempt`.
    pub attempt: usize,
    /// This iteration's user-prompt text (base prompt or nudge).
    pub prompt: String,
    /// The agent run, when there was one. `None` when `EvolutionAgent::run`
    /// itself returned `Err` — a transient HTTP error or a context-size
    /// error — in which case no messages, no usage and no completion calls
    /// exist for this iteration.
    pub run: Option<RunTrace>,
    /// How far the iteration got through the pipeline, and how long each
    /// stage took.
    pub stages: StageTimings,
    /// What the harness did in response to this iteration.
    pub ladder: LadderEvent,
    pub duration: Duration,
}

/// The parts of an iteration that only exist once the agent run succeeded.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunTrace {
    /// Range into `EvolutionTrace::history` of the messages this run
    /// produced: its own user-prompt turn plus all assistant turns and tool
    /// exchanges. The request this run *sent* was
    /// `history[..produced.start + 1]` plus the system prompt.
    pub produced: Range<usize>,
    /// Final assistant text.
    pub response: String,
    /// Aggregate token usage for this run (sum over its completion calls).
    pub usage: Usage,
    /// One entry per HTTP completion request, including rig-internal retries
    /// of an invalid tool call. Each entry's `raw` wire body is nulled by the
    /// blanket impl (§6); `usage`, `finish_reason` and the provider ids are
    /// kept.
    pub completion_calls: Vec<CompletionCall>,
}

/// Per-attempt mirror of the `PIPELINE_STAGE_DURATION` histogram. A field is
/// `None` when the iteration failed before reaching that stage, which is what
/// makes "attempt 3 spent 90s compiling, then failed" recoverable from the
/// trace.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StageTimings {
    pub llm: Option<Duration>,
    pub parse_validate: Option<Duration>,
    pub build: Option<BuildRecord>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BuildRecord {
    /// The candidate was compiled and loaded.
    Built {
        slot_wait: Duration,
        compile: Duration,
        load: Duration,
    },
    /// The candidate was byte-identical to an already-registered revision, so
    /// no build was spent (`runtime.rs:467-473`). There is no compile and no
    /// load duration to report — only the build-slot wait that preceded the
    /// check.
    Deduped {
        slot_wait: Duration,
        revision: Revision,
    },
}

/// One step of the harness's reaction ladder. Every attempt has exactly one.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LadderEvent {
    /// This attempt registered the revision and ended the lane.
    Registered { revision: Revision },
    /// The failure was fed back as a corrective nudge. `kind` uses the same
    /// labels as `EvolveFailure::kind`: `no_rust_code`, `parse`, `max_turns`,
    /// `signature`, `unsafe`, `forbidden`, `compile`.
    SelfHeal { kind: &'static str, diagnostics: String },
    /// Transient inference error; the same prompt is retried after `backoff`
    /// and the attempt budget is not consumed.
    TransientRetry { backoff: Duration, cause: String },
    /// Context-size error; `messages_dropped` chat messages were discarded and
    /// the lane restarted from the base prompt.
    ContextReset { messages_dropped: usize, brief: String },
    /// Verbatim-repeat detection; the lane's context was reset likewise.
    RepeatReset { messages_dropped: usize, brief: String },
    /// This attempt's failure ended the lane; `reason` is the final error.
    Terminal { reason: String },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceOutcome {
    Registered { revision: Revision },
    Failed { reason: String },
}
```

Notes:

- **`LadderEvent` is non-optional and includes `Registered`.** Making it the timeline's spine
  means every attempt has exactly one outcome, and the trace reads as the recovery path directly:
  `SelfHeal → SelfHeal → Terminal` vs. `TransientRetry → SelfHeal → Registered`.
- **`run: Option<RunTrace>` is load-bearing.** A lane iteration can fail before any run exists:
  a transient HTTP error (`runtime.rs:1085-1108`) or a context-size error
  (`runtime.rs:1040-1079`) both come from `agent.run` returning `Err`. Hoisting `produced`,
  `response`, `usage` and `completion_calls` into `RunTrace` is what makes those iterations
  representable at all.
- **`seq` and `attempt` are different numbers.** See the field docs; conflating them yields
  duplicate indices on every transient retry.
- **No system prompt field.** The preamble embeds the full generated host-API documentation
  (`doc_string::write_prelude_doc_string`) and is identical for every lane in a batch. Excluding
  it removes the design's largest constant-per-lane term, and it is not lost: hosts call
  `symbiont::system_prompt(opt_crate_name)` (§4.5) once and store it beside the traces.
- `TraceOutcome::Failed` carries a `String`, not `Error`, so the trace stays serializable; the
  actual error lives in `EvolveError` (§3).
- **Serde needs no adapters.** `Message` (`rig-core/src/completion/message.rs:18`), `Usage`
  (`rig-core/src/completion/request.rs:535`) and `CompletionCall`
  (`rig-agent/src/agent/prompt_request/mod.rs:330`) all derive `Serialize + Deserialize`;
  `std::time::Duration` has serde impls in `serde_core` (ser `:685`, de `:2129`); `Revision`
  derives both once serde becomes mandatory (§5). No newtype wrapper is required.

## 3. Delivery API: attach the trace to the per-lane result (recommended)

```rust
// evolve_info.rs — add one field
pub struct EvolveInfo {
    revision: Revision,
    usage: Usage,
    trace: EvolutionTrace,   // new
}

// new evolve_error.rs
/// Failure of an evolve call, with the full trajectory of the lane that failed.
pub struct EvolveError {
    #[source]
    error: Error,            // the final error out of the ladder
    trace: EvolutionTrace,
}
impl std::error::Error for EvolveError { ... }
impl std::fmt::Display for EvolveError { ... }
impl From<Error> for EvolveError { ... }
impl From<EvolveError> for Error { ... }   // lossy: drops the trace
```

and the three public methods become

```rust
pub fn evolve<AgentT>(&self, agent: &AgentT, base_prompt: &str)
    -> impl Future<Output = Result<EvolveInfo, EvolveError>> + Send;

pub fn evolve_batch<'a, AgentT, S>(&'a self, agent: &'a AgentT, prompts: &'a [S])
    -> impl Future<Output = Vec<Result<EvolveInfo, EvolveError>>> + Send + 'a;

pub fn evolve_batch_stream<'a, AgentT, S>(&'a self, agent: &'a AgentT, prompts: &'a [S])
    -> impl Stream<Item = (usize, Result<EvolveInfo, EvolveError>)> + Send + 'a;
```

**Why not a `Runtime`-level trace buffer + `take_evolution_traces()`:**

- `Runtime` is a process-wide `OnceLock` singleton (`runtime.rs:118`) and `evolve_batch_stream`
  (`runtime.rs:885`) explicitly supports overlapping rounds that share one inference budget. A
  shared "recent traces" buffer would conflate lanes across overlapping calls — the same
  attribution hazard the `EvolveFailure` docs already flag ("group by lane").
- Traces are large (transcript + nudges). Buffering them on the singleton until drained means
  unbounded retention under a slow drainer, unlike the small failure records.
- Per-lane attachment is race-free by construction: each lane builds its own `EvolutionTrace` in
  local state and hands it up with its result. **No new `Runtime` state at all** — the singleton
  stays a singleton.

### Breaking-change cost (crate is 0.x)

- **`EvolveInfo` loses `Copy`, `PartialEq` and `Eq`.** It is currently
  `#[derive(Debug, Clone, Copy, PartialEq, Eq, Getters, CopyGetters)]` (`evolve_info.rs:18`); a
  `Vec`-bearing trace field removes all three. `symbiont/tests/evolve_info.rs` depends on them.
  The constructor is `pub(crate)`, so external construction cannot break — only the derives and
  the new getter.
- **`From<EvolveError> for Error` silently discards the trace**, and that is what every `?` in
  host code does — including `examples/evolving-trader/src/main.rs:705`. This must be documented
  on the impl, not left implicit. Hosts that want the trace on the error path must match instead
  of propagating.
- **Accessor is an explicit `error()` getter, not `Deref<Target = Error>`** (decided). `Deref` to
  an error type makes `err.to_string()` and `?` resolution ambiguous at a glance, and combining
  it with `#[source]` on the same value double-reports that error in a printed chain. The
  convenience it buys — existing `match` arms continuing to compile — is worth less than the
  ambiguity, given the call sites must be touched for the `Result` type change regardless.
  Call sites that inspect the error write `match err.error() { … }`.
- **Migration surface**: 17 call sites across `examples/*/src/main.rs` and `symbiont/tests/*`.
- The workspace lints `missing_docs = "warn"`, so every new public item needs a doc comment.

*Fallback if non-breaking is a hard requirement:* keep the existing signatures and add
`evolve_traced` / `evolve_batch_traced` variants. Cost: 3× API surface and two parallel
implementations of `evolve_lane`. Not recommended.

## 4. Capture architecture

### 4.1 `EvolutionAgent` and `AgentRun` (`evolution_agent.rs`)

```rust
pub struct AgentRun {
    pub output: String,
    pub new_messages: Vec<Message>,
    pub usage: Usage,
    pub completion_calls: Vec<CompletionCall>,  // new: rig already computes this;
}                                               // the blanket impl currently drops it
```

- The `EvolutionAgent` trait is **unchanged**. An earlier draft of this document proposed a
  provided `system_prompt()` method whose blanket impl read `self.preamble`; that field is
  `pub(crate)` with no accessor (§1), so the method could not have been implemented, and the
  trace no longer needs it.
- The blanket impl for rig `Agent` passes `response.completion_calls` through, nulling each
  entry's `raw` field first (§6): the wire response body is redundant with `history` and
  `RunTrace::response`, while `usage`, `finish_reason` and the provider ids are kept.
- Adding a field to `AgentRun` breaks custom implementors that construct it by value — a minor
  semver bump; hosts in practice use the blanket impl.

### 4.2 Lane-side bookkeeping (`runtime.rs`, `evolve_lane`)

The ladder already has every decision as a local value; recording is additive:

```text
let mut trace = EvolutionTrace::new(lane, base_prompt.clone());
let mut history: Vec<Message> = Vec::new();
loop {
    let produced_start = history.len();
    let t_attempt = Instant::now();
    let mut stages = StageTimings::default();
    match self.evolve_no_backpressure(agent, &prompt, &mut history, &mut usage, &mut stages).await {
        Ok((revision, run)) => {
            trace.push(AttemptTrace {
                seq: trace.attempts.len(),
                attempt: attempts,
                prompt: prompt.clone(),
                run: Some(RunTrace {
                    produced: produced_start..history.len(),
                    response: run.output,
                    usage: run.usage,
                    completion_calls: run.completion_calls,
                }),
                stages,
                ladder: LadderEvent::Registered { revision },
                duration: t_attempt.elapsed(),
            });
        }
        Err(e) => {
            // Two shapes. If the agent run itself failed (transient HTTP,
            // context overflow) there is no `RunTrace`: push with `run: None`.
            // Otherwise the run succeeded and a later stage rejected it, so
            // `run` is `Some` and `stages` records how far it got.
            // Each ladder arm then supplies its own `LadderEvent`, reusing
            // the exact strings it already builds for metrics, the
            // `EvolveFailure` record, and the next prompt.
        }
    }
}
// at the end: trace.history = history (moved, not cloned); trace.outcome = …;
// on success: EvolveInfo { revision, usage, trace }
// on failure: EvolveError { error, trace }
```

Details:

- `history` is *moved* into the trace at the end (the lane's existing local becomes the trace's
  field) — zero extra copies of transcript content.
- The per-attempt `prompt` string is a clone of a value the ladder already owns (nudges are built
  as `String` anyway).
- **`StageTimings` needs plumbing.** `evolve_no_backpressure` already measures the LLM and
  parse/validate stages (`runtime.rs:391`, `runtime.rs:407`) but only feeds them to histograms,
  and `build_and_register` measures the build stage (`runtime.rs:458-534`) yet returns just
  `Revision`. Both must surface their measurements — an out-param threaded down, or a richer
  return type. `build_and_register`'s dedup early return (`runtime.rs:467-473`) reports
  `BuildRecord::Deduped`.
- **`evolve_no_backpressure` must return the `AgentRun`, not just the revision.** It currently
  consumes `run.output` and drops the rest into `history`; the trace needs `usage` and
  `completion_calls` per attempt, and the lane needs them even on the rejecting paths.
- The existing `EvolveFailure` push into the drain buffer stays untouched (its docs promise the
  drain; the trace *subsumes* it — document that the trace is the superset; deprecating
  `EvolveFailure` can wait).

### 4.3 Tool calls

No new capture needed: they are already inline in `new_messages` as `Assistant(ToolCall)` /
`User(ToolResult)`. `EvolutionTrace::render()` (§5) walks them for the human-readable view.

**Known v1 gap:** rig's *invalid tool-call* handling (`PromptHook::on_invalid_tool_call`) leaves
only the corrective user message in the transcript, not a labelled event with rig's action
(fail / retry / repair / skip).

*Phase 2:* attach a recording hook in the blanket impl via `PromptRequest::add_hook`
(`prompt_request/mod.rs:288` — note the name; there is no `with_hook`). The user's own hooks do
not need to be composed manually: `Agent.hooks` is `pub(crate)` (`agent/completion.rs:616`) and
therefore unreadable, but rig already stacks the agent's default hooks with request-level hooks
internally, so a recorder added on the `PromptRequest` runs alongside the user's, with the
user's `Terminate`/`Skip` actions winning. The same hook's `on_completion_call` provides exact
per-request message snapshots if we later want wire-level fidelity.

### 4.4 Raw payloads

**Response payloads reach us but are dropped.** `CompletionCall.raw` carries the provider's full
response body for every call (§1); the blanket impl nulls it before the trace ever sees it. The
parsed assistant text is retained — in `history` and in `RunTrace::response` — which is what the
trace's goal actually requires; the wire framing around it is not. Rationale and the rejected
opt-in are in §6.

**Request payloads are not.** `MeteredHttpClient` sees them (`REQUEST_BODY_BYTES`) but only
measures their size. Capturing the bytes needs a task-local sink plumbed through the gate scope;
propose as an opt-in extension rather than default. This also answers the `TODO.md` item "Expose
the rendered payload itself to the host."

**Rig's per-completion spans** are deliberately out of scope: they already flow through `tracing`
(target `rig::agent_chat`, OTel GenAI fields). Hosts running an OTel subscriber get fine-grained
model-call spans; our trace adds lane attribution, the self-healing ladder, the build stage, and
the revision outcome. A span/event bridge from the trace is a cheap follow-up.

### 4.5 Publishing `system_prompt`

Since the trace omits the preamble, hosts need a supported way to obtain it. `system_prompt.rs:145`
is `pub(crate) async fn system_prompt(opt_crate_name: Option<&str>) -> Result<String>` inside a
private module (`lib.rs:32`). Promotion means: make the function `pub`, re-export it from
`lib.rs`, and document it (the workspace lints `missing_docs`).

It is the single place symbiont's preamble is produced — `inference.rs:130` calls it and feeds
the result to `.preamble(…)` at `inference.rs:135` — so a host calling it reproduces exactly what
the agent received, **provided** it passes the same `opt_crate_name` and did not override the
preamble on its own builder. Both caveats belong in the function's doc comment. Note also that
the function is not free: with `Some(crate_name)` it shells out to rustdoc via
`write_prelude_doc_string`. Hosts should call it once and cache, which is the same thing they
would have done with a per-trace copy, minus the duplication.

## 5. Export

`serde` becomes a **mandatory** dependency of `symbiont`. It is already a non-optional transitive
dependency through `rig-core`, `serde_json` and `rustdoc-types`, so making it direct costs
nothing at build time while removing a feature gate that would otherwise make the entire export
surface conditional. The optional `serde` feature and its eight `#[cfg_attr(feature = "serde", …)]`
sites go away (§7).

Every trace type derives `Serialize` **and** `Deserialize`, so persisted traces can be read back
for offline analysis:

```rust
impl EvolutionTrace {
    /// Human-readable transcript: per-attempt blocks
    /// (prompt → response → tool calls → ladder event → usage → stage timings).
    pub fn render(&self) -> String;

    /// One JSON object per attempt (JSONL). The full transcript is included
    /// once, on the first line; attempt lines reference it by index range.
    /// Append-friendly across evolve rounds — the same shape as Claude Code
    /// session JSONL.
    pub fn write_jsonl<W: std::io::Write>(&self, w: W) -> Result<(), std::io::Error>;

    /// Single pretty-printed JSON document for the whole trace.
    pub fn to_json_pretty(&self) -> String;
}
```

Caller UX — the system prompt is written once for the whole batch, not once per lane:

```rust
let system_prompt = symbiont::system_prompt(Some(env!("CARGO_PKG_NAME"))).await?;
std::fs::write("traces/system-prompt.md", &system_prompt)?;

let results = runtime.evolve_batch(&agent, &prompts).await;
for (i, res) in results.into_iter().enumerate() {
    let trace = match res {
        Ok(info) => info.trace(),
        Err(err) => err.trace(),
    };
    std::fs::write(format!("traces/lane-{i}.json"), trace.to_json_pretty())?;
}
```

JSONL is the recommended default for hosts that want one appendable file per lane across batches;
pretty JSON for one-off inspection.

## 6. Memory / cost

- **Transcript**: owned **once** per lane — it is the lane's existing `history` local, taken
  instead of dropped.
- **Annotations**: O(attempts) small strings (nudges, diagnostics — data the ladder already
  builds) plus O(attempts) `StageTimings`, which are a handful of `Option<Duration>`.
- **System prompt**: not stored. Excluding it is what removes the design's dominant
  constant-per-lane term; with many lanes and a large documented API surface, a per-lane copy of
  the preamble would have outweighed everything else on this list.
- **`completion_calls`**: O(HTTP calls). Rig computes them regardless; we stop discarding them.
  The metadata fields are negligible.
- **`CompletionCall.raw` is dropped** (decided). It would otherwise be the dominant term: one
  full provider response body per HTTP call, per attempt, per lane, held until the caller drops
  the result. Its payload is also redundant — the assistant text it wraps is already in `history`
  as a `Message::Assistant` *and* in `RunTrace::response`, so retaining it would store the
  largest string in the trace three times, and evolve responses are whole Rust source files by
  construction.

  The blanket impl sets `raw = serde_json::Value::Null` on every `CompletionCall` before storing
  it in `AgentRun` — one `map` at the single construction point (`evolution_agent.rs:67-71`).
  `usage`, `finish_reason` and the three provider ids are always kept: they are small, they are
  not recoverable from the transcript, and `finish_reason` is what distinguishes a turn truncated
  at the output-token limit from a turn that had nothing to say.

  What this forfeits is provider-specific data rig does not normalize (reasoning traces, logprobs,
  cache breakdowns), which lives only in `raw`. That is an escape hatch for wire-level debugging,
  not something the trace's stated goal needs, and it is recoverable by other means — see the
  note below.

  Because `raw` is `#[serde(skip_serializing_if = "Value::is_null")]`
  (`rig-agent/src/agent/prompt_request/mod.rs:375`), nulling it also removes the key from the
  serialized trace entirely rather than emitting `"raw": null`.

- **No knob on the blanket impl.** An opt-in was considered and rejected: the blanket impl is
  `impl EvolutionAgent for Agent` and has no access to `Runtime` or to any symbiont config, so a
  toggle would have to become trait state, a `Runtime` field threaded into a call the runtime
  does not make, or a cargo feature — each of which buys back the API surface this design is
  trying not to add. A host that genuinely needs wire fidelity writes its own `EvolutionAgent`
  impl (the trait is public and the blanket impl is ~20 lines) or attaches a rig `PromptHook`
  (§4.3), both of which see the unmodified `PromptResponse`.

- **Default-on is the right call** for the trace as a whole: nothing new is computed, no feature
  flag, no parallel API. The only cost is keeping the transcript alive until the call returns.
  Hosts that don't care drop the `EvolveInfo` / `EvolveError` values as they do today.

## 7. Affected files (implementation phase)

| File | Change |
|---|---|
| `symbiont/src/evolution_trace.rs` (new) | `EvolutionTrace`, `AttemptTrace`, `RunTrace`, `StageTimings`, `BuildRecord`, `LadderEvent`, `TraceOutcome` + `render` / `write_jsonl` / `to_json_pretty` |
| `symbiont/src/evolution_agent.rs` | `AgentRun.completion_calls`; blanket impl passes them through with each entry's `raw` nulled. Trait itself unchanged |
| `symbiont/src/runtime.rs` | `evolve_no_backpressure` returns the `AgentRun` and reports `StageTimings`; `build_and_register` reports `BuildRecord` (incl. `Deduped`); `evolve_lane` accumulates the trace; `evolve` / `evolve_batch` / `evolve_batch_stream` return `Result<EvolveInfo, EvolveError>` |
| `symbiont/src/evolve_error.rs` (new) | `EvolveError { error, trace }` with `std::error::Error`, `Display`, `error()` / `trace()` getters, and `From` both ways (documenting the lossy direction) |
| `symbiont/src/evolve_info.rs` | add `trace` field + getter; drop `Copy`, `PartialEq`, `Eq` |
| `symbiont/src/system_prompt.rs` | `system_prompt` becomes `pub` and documented (§4.5) |
| `symbiont/src/lib.rs` | re-export `system_prompt`, the trace types, `EvolveError`, and `CompletionCall` (from `rig_agent::agent`) |
| `Cargo.toml` (workspace) | add `serde = { version = "1", features = ["derive"] }` to `[workspace.dependencies]` |
| `symbiont/Cargo.toml` | `serde` moves from optional to required; delete the `serde` entry from `[features]` |
| `symbiont/src/{evolve_info,evolve_failure,profile,thinking_level,dylib_config,revision,dylib_dependency}.rs` | eight `#[cfg_attr(feature = "serde", derive(…))]` → plain `#[derive(…)]` (`evolve_info.rs:19`, `evolve_failure.rs:30`, `profile.rs:7`, `thinking_level.rs:16`, `dylib_config.rs:23`, `revision.rs:36`, `dylib_dependency.rs:17`, `dylib_dependency.rs:51`) |
| `examples/*/src/main.rs`, `symbiont/tests/*` | migrate the 17 call sites to `Result<EvolveInfo, EvolveError>`; `tests/evolve_info.rs` loses the `Copy` / `Eq` assumptions |
| `TODO.md` | update "expose the rendered payload" / "capture the inference cost" items (per-call usage is now exposed; request-payload capture deferred to the opt-in extension) |

Tests:

- Unit, with a hand-written fake `EvolutionAgent` returning canned runs/errors: `history`
  concatenation invariants, `produced` ranges, `usage` totals, ladder sequences.
- A transient-retry iteration: asserts `run: None`, a `TransientRetry` ladder event, and that
  `attempt` repeats across the retry while `seq` stays dense.
- A dedup case: asserts `BuildRecord::Deduped` with no compile/load durations.
- A partially-failed attempt: asserts `stages.llm.is_some()`, `stages.build.is_none()` when the
  candidate was rejected at parse/validate.
- Serde round-trip (`Serialize` → `Deserialize`) of a populated trace.
- Integration against the existing fake OpenAI endpoint: trace count matches lane count; failed
  lanes carry a `Terminal` ladder event; success lanes carry `Registered` outcomes.

## 8. Open questions

**Resolved since the first draft:**

- *Serde gating* — `serde` becomes a mandatory dependency; the feature is removed (§5).
- *System-prompt storage* — excluded from the trace, derived upstream via the newly public
  `symbiont::system_prompt` (§4.5). This also disposes of the `Arc<str>` interning follow-up.
- *`EvolveError` accessor* — an explicit `error()` getter; no `Deref` (§3).
- *`CompletionCall.raw` retention* — dropped unconditionally by the blanket impl; the parsed text
  is retained in `history` and `RunTrace::response`, and no opt-in is added (§6).

**Open:**

1. **`StageTimings` on the dedup path** — `BuildRecord::Deduped` reports `slot_wait` only. Is the
   slot wait worth reporting on its own, or should the dedup case carry no timing at all?
2. **Breaking vs. additive** — is changing `evolve`'s error type to `EvolveError` acceptable at
   0.x (recommended), or are `_traced` variants required?
3. **`take_evolve_failures`** — the trace now contains everything `EvolveFailure` records. Keep
   both (recommended, for stability) or mark the drain deprecated now?
4. **Raw request-payload capture** — opt-in `MeteredHttpClient` task-local sink in v1, or defer
   to v2?
