# Evolution Trace — Design Proposal

Goal: any caller of `Runtime::evolve` / `evolve_batch` / `evolve_batch_stream` should be able to
get the **full agent chat trajectory** of an evolution — system prompt, every inference request
and response, every tool call and result, every retry (self-healing nudges, transient HTTP
backoff, context resets, repeat resets) with its diagnostics, per-turn token usage, and timings —
as structured data for later inspection, in the way other agent harnesses capture traces
(OpenAI Agents SDK request/response spans, Claude Code session JSONL).

This is a design proposal; no code changes are implied yet.

---

## 1. What is already available (capture-point inventory)

| Item | Where it lives today |
|---|---|
| System prompt | `Agent.preamble: Option<String>` (public rig field, sent on every request); built by `system_prompt.rs`, logged only via `info!` |
| Per-attempt chat (prompt turn + assistant turns + tool calls/results) | `AgentRun.new_messages` (`= PromptResponse.messages`), already returned per run; `evolve_lane` extends its local `history: Vec<Message>` with it |
| Per-attempt aggregate token usage | `AgentRun.usage` (= `PromptResponse.usage`) |
| Per-HTTP-call token usage | `PromptResponse.completion_calls: Vec<CompletionCall { call_index, usage: Option<Usage> }>` — **dropped today** by the blanket impl in `evolution_agent.rs` |
| Tool calls + results | Inline in `new_messages` as `Assistant(ToolCall)` / `User(ToolResult)` content — derivable, no extra capture |
| Self-healing ladder decisions | Local to `evolve_lane` (error kind, nudge prompt, backoff, reset) — recordable in place |
| Build-stage timings | Already measured by `evolve_lane` for the `EVOLVE_*` metrics (build-slot wait, compile, load) |
| Per-HTTP payload size | `MeteredHttpClient` (`REQUEST_BODY_BYTES`) |
| Fine-grained model-call spans | Rig's native OTel GenAI `tracing` spans (target `rig::agent_chat`, `gen_ai.*` fields) |

**Key invariant.** The lane's `history` is the lane's complete ordered transcript. Attempt *k*
sends `system_prompt + history[..p_k]` and produces `history[p_k..p_{k+1}]`, where `p_k` is the
index of that attempt's user-prompt turn. Storing the final `history` **once** plus per-attempt
boundary indices therefore captures the full trajectory in linear memory — no O(attempts²)
duplication of the (large, code-heavy) message content.

## 2. Data model

New module `symbiont/src/evolution_trace.rs`:

```rust
/// The full agent trajectory of one lane (or of a single-prompt `evolve`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvolutionTrace {
    /// Lane index; 0 for single-prompt `evolve`.
    pub lane: usize,
    /// The system prompt sent with every request (the agent's preamble), if any.
    pub system_prompt: Option<String>,
    /// The prompt the lane started with, before any nudge.
    pub base_prompt: String,
    /// The lane's complete ordered transcript: every user turn (base prompt,
    /// each corrective nudge), every assistant turn, every tool call and
    /// result. Owned once; attempts reference into it.
    pub history: Vec<Message>,
    /// One entry per `EvolutionAgent::run` invocation, in order.
    pub attempts: Vec<AttemptTrace>,
    /// Timings of the build/registration stage; set on success.
    pub build: Option<BuildRecord>,
    pub outcome: TraceOutcome,
}

/// One agent run inside the lane's self-healing ladder.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AttemptTrace {
    /// 1-based; matches `EvolveFailure::attempt` and the `attempt` metric label.
    pub index: usize,
    /// Range into `history` of the messages this attempt produced: its own
    /// user-prompt turn plus all assistant turns and tool exchanges. The
    /// request this attempt *sent* was `history[..produced.start + 1]` plus
    /// the system prompt.
    pub produced: Range<usize>,
    /// Convenience copy of this attempt's user-prompt text (base prompt or nudge).
    pub prompt: String,
    /// Final assistant text; `None` when the run failed before producing text.
    pub response: Option<String>,
    /// Aggregate token usage for this attempt (sum over its completion calls).
    pub usage: Usage,
    /// One entry per HTTP completion request, including rig-internal retries
    /// of an invalid tool call.
    pub completion_calls: Vec<CompletionCall>,
    /// What the harness did in response to this attempt's failure; `None` on
    /// the attempt that registered the revision.
    pub ladder: Option<LadderEvent>,
    pub duration: Duration,
}

/// One step of the harness's reaction ladder.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LadderEvent {
    /// The failure was fed back as a corrective nudge. `kind` uses the same
    /// labels as `EvolveFailure::kind`: `no_rust_code`, `parse`, `max_turns`,
    /// `signature`, `unsafe`, `forbidden`, `compile`.
    SelfHeal { kind: &'static str, diagnostics: String },
    /// Transient inference error; the same prompt is retried after `backoff`.
    TransientRetry { backoff: Duration, cause: String },
    /// Context-size error; `messages_dropped` chat messages were discarded and
    /// the lane continued with `brief`.
    ContextReset { messages_dropped: usize, brief: String },
    /// Verbatim-repeat detection; the lane's context was reset likewise.
    RepeatReset { messages_dropped: usize, brief: String },
    /// This attempt's failure ended the lane; `reason` is the final error.
    Terminal { reason: String },
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceOutcome {
    Registered { revision: Revision },
    Failed { reason: String },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BuildRecord {
    pub build_slot_wait: Duration,
    pub compile: Duration,
    pub load: Duration,
}
```

Notes:

- `Message`, `Usage`, `CompletionCall` are rig types that already derive `Serialize`; no new
  dependency. (If `Duration`'s serde shape ever proves awkward, wrap it in a small newtype.)
- `TraceOutcome::Failed` carries a `String`, not `Error`, so the trace stays serializable; the
  actual error lives in `EvolveError` (§3).
- `ladder` makes the trace a record of the *recovery path*: `SelfHeal → SelfHeal → Terminal`
  vs. `TransientRetry → SelfHeal → Registered`.

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
impl Deref<Target = Error> for EvolveError { ... }
impl From<Error> for EvolveError { ... }
impl From<EvolveError> for Error { ... }
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

- `Runtime` is a process-wide `OnceLock` singleton and `evolve_batch_stream` explicitly supports
  overlapping rounds that share one inference budget. A shared "recent traces" buffer would
  conflate lanes across overlapping calls — the same attribution hazard the `EvolveFailure`
  docs already flag ("group by lane").
- Traces are large (transcript + nudges). Buffering them on the singleton until drained means
  unbounded retention under a slow drainer, unlike the small failure records.
- Per-lane attachment is race-free by construction: each lane builds its own
  `EvolutionTrace` in local state and hands it up with its result. **No new `Runtime` state at
  all** — the singleton stays a singleton.

Breaking-change cost (crate is 0.x): `EvolveError: Deref<Target = Error>` + `From` in both
directions keep most match arms compiling; `EvolveInfo` gains one field (its constructor is
`pub(crate)`, so external construction cannot break — only new getters).

*Fallback if non-breaking is a hard requirement:* keep the existing signatures and add
`evolve_traced` / `evolve_batch_traced` variants. Cost: 3× API surface and two parallel
implementations of `evolve_lane`. Not recommended.

## 4. Capture architecture

### 4.1 `EvolutionAgent` and `AgentRun` (`evolution_agent.rs`)

```rust
pub trait EvolutionAgent {
    fn run(&self, prompt: &str, history: Vec<Message>)
        -> impl Future<Output = Result<AgentRun, PromptError>> + Send;

    /// The system prompt / preamble this agent sends on every request.
    /// Used for trace capture; `None` when the agent has no preamble.
    fn system_prompt(&self) -> Option<String> { None }   // new, with default
}

pub struct AgentRun {
    pub output: String,
    pub new_messages: Vec<Message>,
    pub usage: Usage,
    pub completion_calls: Vec<CompletionCall>,  // new: rig already computes this;
}                                               // the blanket impl currently drops it
```

- `system_prompt()` is a *provided* method → existing user impls of the public trait keep
  compiling (additive trait extension).
- The blanket impl for rig `Agent` returns `self.preamble.clone()`.
- Adding a field to `AgentRun` breaks custom implementors that construct it by value — a minor
  semver bump; hosts in practice use the blanket impl.

### 4.2 Lane-side bookkeeping (`runtime.rs`, `evolve_lane`)

The ladder already has every decision as a local value; recording is additive:

```text
let mut trace = EvolutionTrace::new(lane, agent.system_prompt(), base_prompt.clone());
let mut history: Vec<Message> = Vec::new();
loop {
    let produced_start = history.len();
    let run = agent.run(&prompt, history.clone()).await;  // existing call; clone is already done today
    let run = match run { Ok(run) => run, Err(e) => /* ladder arm records below */ };
    let produced_end = produced_start + run.new_messages.len();
    // history.extend(run.new_messages)  (existing)
    trace.attempts.push(AttemptTrace {
        index,
        produced: produced_start..produced_end,
        prompt: prompt.clone(),
        response: Some(run.output.clone()),
        usage: run.usage.clone(),
        completion_calls: run.completion_calls,
        ladder: None,
        duration,
    });
    // each ladder arm then sets trace.attempts.last_mut().ladder = Some(LadderEvent::…);
    // nudge / backoff / reset / terminal decisions reuse the exact strings the
    // ladder already builds for metrics, EvolveFailure records, and the prompt.
}
// at the end: trace.history = history (moved, not cloned); trace.outcome = …;
// on success: EvolveInfo { revision, usage, trace }
// on failure: EvolveError { error, trace }
```

Details:

- `history` is *moved* into the trace at the end (the lane's existing local becomes the
  trace's field) — zero extra copies of transcript content.
- The per-attempt `prompt` string is a clone of a value the ladder already owns (nudges are
  built as `String` anyway).
- The existing `EvolveFailure` push into the drain buffer stays untouched (its docs promise the
  drain; the trace *subsumes* it — document that the trace is the superset; deprecating
  `EvolveFailure` can wait).

### 4.3 Tool calls

No new capture needed: they are already inline in `new_messages` as
`Assistant(ToolCall)` / `User(ToolResult)`. `EvolutionTrace::render()` (§5) walks them for the
human-readable view.

**Known v1 gap:** rig's *invalid tool-call* handling (`PromptHook::on_invalid_tool_call`) leaves
only the corrective user message in the transcript, not a labelled event with rig's action
(fail / retry / repair / skip).

*Phase 2:* attach a recording `PromptHook` in the blanket impl via `PromptRequest::with_hook`.
Rig's `Agent.hook` is a public field, so a composite hook can run the user's hook and the
recorder in sequence (the user's `Terminate`/`Skip` actions win). The same hook's
`on_completion_call` provides exact per-request message snapshots if we later want wire-level
fidelity.

### 4.4 Deliberately not captured in v1

- **Raw wire payloads** (exact JSON body of each HTTP request): `MeteredHttpClient` already sees
  them (`REQUEST_BODY_BYTES`). Capturing the bytes per request needs a task-local sink plumbed
  through the gate scope — propose as an opt-in extension rather than default. This also answers
  the `TODO.md` item "Expose the rendered payload itself to the host."
- **Rig's per-completion spans**: they already flow through `tracing` (target
  `rig::agent_chat`, OTel GenAI fields). Hosts running an OTel subscriber get fine-grained
  model-call spans; our trace adds lane attribution, the self-healing ladder, the build stage,
  and the revision outcome. A span/event bridge from the trace is a cheap follow-up.

## 5. Export

Everything is serde-serializable, so:

```rust
impl EvolutionTrace {
    /// Human-readable transcript: system prompt, then per-attempt blocks
    /// (prompt → response → tool calls → ladder event → usage → duration).
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

Caller UX:

```rust
let results = runtime.evolve_batch(&agent, &prompts).await;
for (i, res) in results.into_iter().enumerate() {
    let trace = match res {
        Ok(info) => info.trace(),
        Err(err) => err.trace(),
    };
    std::fs::write(format!("traces/lane-{i}.json"), trace.to_json_pretty())?;
}
```

JSONL is the recommended default for hosts that want one appendable file per lane across
batches; pretty JSON for one-off inspection.

## 6. Memory / cost

- Transcript: owned **once** per lane — it is the lane's existing `history` local, taken instead
  of dropped.
- Annotations: O(attempts) small strings (nudges, diagnostics — data the ladder already builds).
- `completion_calls`: O(HTTP calls); rig computes them regardless; we stop discarding them.
- System prompt: duplicated per lane (identical across lanes of a batch). With many lanes and a
  large documented API surface this is the main new constant; acceptable, and interning to
  `Arc<str>` is a one-line follow-up if it ever matters.
- **Default-on** is the right call: nothing new is computed, no feature flag, no parallel API.
  The only cost is keeping the transcript alive until the call returns. Hosts that don't care
  drop the `EvolveInfo` / `EvolveError` values as they do today. (The alternatives — a feature
  flag or `_traced` variant APIs — add surface for no benefit.)

## 7. Affected files (implementation phase)

| File | Change |
|---|---|
| `symbiont/src/evolution_trace.rs` (new) | `EvolutionTrace`, `AttemptTrace`, `LadderEvent`, `TraceOutcome`, `BuildRecord` + `render` / `write_jsonl` / `to_json_pretty` |
| `symbiont/src/evolution_agent.rs` | `AgentRun.completion_calls`; `EvolutionAgent::system_prompt()` default; blanket impl passes `completion_calls` through and reads `agent.preamble` |
| `symbiont/src/runtime.rs` | `evolve_lane` accumulates the trace; `evolve` / `evolve_batch` / `evolve_batch_stream` return `Result<EvolveInfo, EvolveError>` |
| `symbiont/src/evolve_error.rs` (new) | `EvolveError { error, trace }` with `Deref<Target = Error>` and `From` both ways |
| `symbiont/src/evolve_info.rs` | add `trace` field + getter |
| `symbiont/src/lib.rs` | re-export `EvolutionTrace`, `AttemptTrace`, `LadderEvent`, `TraceOutcome`, `BuildRecord`, `EvolveError`, and `CompletionCall` (from `rig_core::agent`) |
| `TODO.md` | update "expose the rendered payload" / "capture the inference cost" items (per-call usage is now exposed; payload capture deferred to the opt-in extension) |

Tests:

- Unit, with a hand-written fake `EvolutionAgent` returning canned runs/errors: assert
  `history` concatenation invariants, `produced` ranges, `usage` totals, ladder sequences.
- Serde round-trip of a populated trace.
- Integration against the existing fake OpenAI endpoint: trace count matches lane count; failed
  lanes carry a `Terminal` ladder event; success lanes carry `Registered` outcomes.

## 8. Open questions

1. **Breaking vs. additive** — is changing `evolve`'s error type to `EvolveError` acceptable at
   0.x (recommended, with `Deref`), or are `_traced` variants required?
2. **`system_prompt` storage** — store the full preamble in every lane's trace (simple,
   recommended), or intern it to `Arc<str>` from the start?
3. **`take_evolve_failures`** — the trace now contains everything `EvolveFailure` records.
   Keep both (recommended, for stability) or mark the drain deprecated now?
4. **Raw payload capture** — opt-in `MeteredHttpClient` task-local sink in v1, or defer to v2?
