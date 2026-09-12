// SPDX-License-Identifier: MPL-2.0
//! The runtime module contains the primary `Runtime`,
//! managing the lifecycle of the temporary dylib crate: creation, compilation,
//! loading, and hot-reloading.

#[cfg(miri)]
use std::time::Instant;
use std::{
    collections::HashMap,
    fmt::Write,
    path::{
        Path,
        PathBuf,
    },
    sync::{
        Arc,
        OnceLock,
        RwLock,
        atomic::{
            AtomicPtr,
            AtomicU64,
            Ordering,
        },
    },
    time::Duration,
};

use futures_util::stream::{
    self,
    Stream,
    StreamExt,
};
use libloading::Library;
use metrics::{
    counter,
    gauge,
    histogram,
};
#[cfg(not(miri))]
use minstant::Instant;
use owo_colors::OwoColorize;
use rig_core::{
    completion::Usage,
    message::Message,
};
use tracing::{
    debug,
    info,
    warn,
};

use crate::{
    AgentRun,
    AppliedFix,
    BuildRecord,
    Diagnostic,
    DocIndex,
    DylibConfig,
    EXPECT_WRITE,
    EvolutionAgent,
    EvolutionTrace,
    EvolvableDecl,
    EvolveError,
    EvolveInfo,
    FullSource,
    LadderEvent,
    Lane,
    PartialRun,
    Profile,
    RunError,
    RunTrace,
    StageTimings,
    TraceOutcome,
    api_hints::{
        api_hint_names,
        render_api_hints,
    },
    compiler::compile_dylib,
    diagnostics::{
        apply_machine_applicable,
        render_fixes_for_prompt,
    },
    edit::{
        self,
        EditBase,
    },
    error::{
        Error,
        Result,
    },
    inference::{
        InferenceGate,
        Priority,
        is_context_size_error,
        is_transient_http_error,
    },
    layout::{
        assemble_lib_rs,
        harness_glue,
        initial_candidate,
    },
    observability::{
        BUILD_SLOT_WAIT,
        COMPILE_AUTOFIXES,
        DYLIB_SIZE_BYTES,
        DYLIB_SOURCE_BYTES,
        EVOLVE_ATTEMPTS,
        EVOLVE_BATCH_DURATION,
        EVOLVE_BATCH_LANES,
        EVOLVE_BATCH_SIZE,
        EVOLVE_CONTEXT_RESETS,
        EVOLVE_DURATION,
        EVOLVE_EDITS,
        EVOLVE_FAILURES,
        EVOLVE_REPEAT_RESETS,
        EVOLVE_TOOLS_WITHDRAWN,
        INFERENCE_ERRORS,
        LLM_RETRY_BACKOFF,
        LLM_RUN_INPUT_TOKENS,
        LLM_RUN_MESSAGES,
        LLM_RUN_OUTPUT_TOKENS,
        LLM_RUNS,
        LLM_TOKENS,
        LLM_TRANSIENT_RETRIES,
        PIPELINE_STAGE_DURATION,
        REVISION_ACTIVATIONS,
        REVISION_ACTIVE,
        REVISION_DEDUP_HITS,
        failure_kind_of,
        inference_error_reason,
        stage,
    },
    parser::{
        Candidate,
        Fence,
        parse_candidate,
        parse_rust_code,
    },
    revision::{
        Revision,
        RevisionEntry,
        RevisionFn,
    },
    tools::context::ToolContext,
    utils::{
        find_so,
        scaffold_dylib_crate,
        versioned_so_path,
    },
    validation::{
        check_implementation_bodies,
        default_body_tokens,
        validate_generated_ast,
    },
};

/// Singleton runtime instance.
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Whether a successfully registered revision should also become the active
/// one. Batch lanes register without publishing, so the host can evaluate all
/// candidates before choosing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(missing_docs, reason = "Self explanatory")]
pub enum Publish {
    Yes,
    No,
}

/// What one iteration of the ladder sends to the agent, beside the
/// transcript.
struct AttemptRequest<'a> {
    /// The user prompt of this iteration: the base prompt, or the corrective
    /// nudge built from the previous failure.
    prompt: &'a str,
    /// Index into the lane's transcript of the first message the agent still
    /// sees. A context or repeat reset moves it forward.
    history_base: usize,
    /// The lane's state for the revision tools, and the previous candidate
    /// a response may edit, with its compiler errors.
    tools_ctx: &'a ToolContext,
    /// Whether the agent may call tools in this iteration. Withdrawn for the
    /// rest of the lane once a run exhausted its turn budget without
    /// producing code - see [`Runtime::evolve_lane`].
    tools: ToolAccess,
}

/// Whether an iteration of the ladder lets the agent call tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolAccess {
    Allowed,
    Withdrawn,
}

/// The prompt of a transient retry whose failed run had already exchanged
/// tool calls: those exchanges stay in the history, and the model is asked to
/// go on from them rather than to start over.
const TRANSIENT_CONTINUE_NUDGE: &str = "nudge: The connection to the model failed after your last \
    tool call; its result is above. Continue from there. If you have what you need, respond with \
    the complete Rust code block now.";

/// Count `usage` into the token counters.
fn record_token_usage(usage: &Usage) {
    counter!(LLM_TOKENS, "kind" => "input").increment(usage.input_tokens);
    counter!(LLM_TOKENS, "kind" => "output").increment(usage.output_tokens);
    if usage.cached_input_tokens > 0 {
        counter!(LLM_TOKENS, "kind" => "cached_input").increment(usage.cached_input_tokens);
    }
}

/// What [`Runtime::compile_with_autofix`] ended with. Either way the fixes
/// that were applied to get there come along, for the build record.
enum Compiled {
    /// The dylib was built from this candidate, which is now the artifact at
    /// the unversioned `so_path`.
    Fresh(String, Vec<AppliedFix>),
    /// The autofixed candidate is the source of this revision already;
    /// nothing was built.
    Registered(Revision, Vec<AppliedFix>),
}

/// What a response answered with; see [`Runtime::answer_of`].
enum Answer {
    /// A registered revision the agent chose.
    Chosen(Revision),
    /// The source of a validated candidate, for the build.
    Candidate(String),
}

/// Cached pointer to the dylib's `__symbiont_take_panic` function.
/// Updated on each reload alongside the evolvable function pointers.
pub(crate) static TAKE_PANIC_PTR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Manages the lifecycle of the temporary dylib crate: creation, compilation,
/// loading, and hot-reloading.
///
/// Function dispatch is lock-free: each evolvable function reads its cached
/// pointer via a single `AtomicPtr::load`.
///
/// Every successfully loaded dylib is retained in a keep-all revision
/// registry — see [`Revision`]. Earlier evolutions therefore stay loaded and
/// callable for the lifetime of the process, without ever parsing or
/// compiling them again.
///
/// # Contract
///
/// **All evolvable function calls must have returned before [`Runtime::evolve`]
/// is called.** This is the natural shape of the feedback loop — run functions,
/// collect results, evolve, repeat. The contract is enforced with an assertion
/// in debug builds and is zero-cost in release. Retained revisions are never
/// unmapped, so a violating in-flight call executes stale but still-mapped
/// code; the contract remains so a swap cannot tear a multi-function revision
/// apart mid-use.
pub struct Runtime {
    /// Path to the temporary dylib crate directory.
    crate_dir: PathBuf,
    /// Path to the unversioned `.so` / `.dylib` / `.dll` produced by cargo,
    /// used as the copy source for the per-revision versioned files.
    so_path: PathBuf,
    /// Function signatures for validation of LLM-generated code.
    fn_sigs: Vec<String>,
    /// Normalized default body per declared function name, the echo
    /// reference for the stub/echo check in the evolve loop.
    default_bodies: HashMap<String, String>,
    /// Path prefixes denied in LLM-generated code, from
    /// [`DylibConfig::denied_paths`].
    denied_paths: Vec<String>,
    /// Every successfully loaded dylib revision, retained for the lifetime of
    /// the process (keep-all). The index into this vec is the revision id.
    /// Entries are reference-counted so [`crate::RevisionFn`] handles can pin
    /// them. The lock is never taken on the hot path.
    revisions: RwLock<Vec<Arc<RevisionEntry>>>,
    /// Id of the revision currently published to the dispatch pointers.
    active: AtomicU64,
    /// Declarations (kept for fn_ptr updates on reload).
    decls: &'static [EvolvableDecl],
    /// Compilation profile (`debug` or `release`).
    profile: Profile,
    /// Rust source snippets that are part of the dylib's `lib.rs` on every
    /// (re)compilation. This includes inline items declared inside
    /// `evolvable! { ... }` and configured imports such as
    /// `use host::prelude::*;`.
    prelude: Vec<String>,
    /// Everything `lib.rs` holds after the candidate: the prelude, the panic
    /// protocol and the export wrappers. See [`crate::layout`]. Rendered once
    /// here, appended to every candidate.
    glue: String,
    /// Serializes the compile-and-register critical section.
    ///
    /// Everything guarded by it is process-wide shared state: the generated
    /// `crate_dir/src/lib.rs`, the unversioned `so_path` cargo writes, and the
    /// dense revision id (which is the registry length, so it can only be
    /// chosen by whoever is about to push). Cargo additionally takes an
    /// exclusive lock on its own build directory, so concurrent builds in one
    /// crate dir would serialize regardless — this just makes the boundary
    /// explicit and keeps the id assignment correct.
    ///
    /// A `tokio` mutex rather than a `std` one: the guard is held across the
    /// `cargo build` await, and holding a `std` guard there would make
    /// `evolve`'s future `!Send`.
    build_slot: tokio::sync::Mutex<()>,
    /// Caps how many inference requests are resident at the endpoint at once,
    /// process-wide and across overlapping calls.
    ///
    /// Deliberately *not* a cap on lanes: a lane holds a slot only while it is
    /// actually talking to the model, and none of it while it parses,
    /// waits for [`Runtime::build_slot`], compiles or loads. That is what lets
    /// [`Runtime::evolve_batch`] run every lane at once without overrunning
    /// the endpoint, and what lets the lanes that are compiling be covered by
    /// lanes that are generating. See [`InferenceGate`].
    inference_gate: InferenceGate,
    /// The name of the host crate the dylib depends on as `host`, if the
    /// [`DylibConfig`] has one. Its [`DocIndex`] documents the API that
    /// compile errors about an invented method or field are attached with.
    host_crate: Option<String>,
    /// The index of `host_crate`, built on the first compile failure that
    /// needs it. A build runs `cargo rustdoc`, which is slow, so a runtime
    /// whose candidates compile never pays for it; a host that already built
    /// the index for its system prompt shares it through the process-wide
    /// cache of [`DocIndex::host`].
    doc_index: tokio::sync::OnceCell<Option<Arc<DocIndex>>>,
}

impl Runtime {
    /// Maximum number of attempts [`Runtime::evolve`] will make before giving
    /// up and returning [`Error::MaxRetriesExceeded`]. Prevents a misbehaving
    /// agent from hanging the runtime indefinitely.
    pub const MAX_EVOLVE_ATTEMPTS: usize = 10;

    /// Maximum number of retries for transient HTTP errors (429, 5xx, 529)
    /// and connection-level failures, including requests that hit
    /// [`INFERENCE_REQUEST_TIMEOUT`](crate::INFERENCE_REQUEST_TIMEOUT).
    ///
    /// These are retried with exponential backoff and do not count against
    /// [`Self::MAX_EVOLVE_ATTEMPTS`]. The retries are also bounded in wall
    /// time by [`Self::MAX_TRANSIENT_WALL_CLOCK`].
    pub const MAX_TRANSIENT_RETRIES: usize = 6;

    /// The wall-clock budget of one lane for transient failures: the summed
    /// duration of the attempts that ended in a transient HTTP error, backoff
    /// included. Once it is spent, the next transient error is terminal even
    /// if [`Self::MAX_TRANSIENT_RETRIES`] has retries left.
    ///
    /// A count alone does not bound time. A request that times out costs
    /// [`INFERENCE_REQUEST_TIMEOUT`](crate::INFERENCE_REQUEST_TIMEOUT), and a
    /// run whose twentieth tool turn times out costs the nineteen before it
    /// as well; six of those on an overloaded endpoint kept a lane busy for
    /// eight hours doing nothing. Two long timeouts is where a lane stops
    /// waiting for the endpoint to recover.
    pub const MAX_TRANSIENT_WALL_CLOCK: Duration = Duration::from_secs(30 * 60);

    /// Maximum number of times a lane may discard its accumulated chat
    /// history and restart from the base prompt after a context-overflow
    /// error. Unlike transient retries, each restart counts against
    /// [`Self::MAX_EVOLVE_ATTEMPTS`]: resending is a consumed attempt, not a
    /// free one, so a lane that keeps overflowing cannot retry without limit.
    pub const MAX_CONTEXT_RESETS: usize = 3;

    /// Maximum number of candidates one lane may send to the compiler
    /// through the revision tools (`build_revision`, `edit_revision`; see
    /// [`crate::tools`]). Once spent, the tools refuse to build and tell the
    /// agent to submit what it has. The budget is separate from
    /// [`Self::MAX_EVOLVE_ATTEMPTS`], which bounds response rounds; a lane
    /// with the tools can therefore spend at most the sum of both in builds.
    /// Candidates that fail to parse or validate spend nothing, and neither
    /// does code the compiler already rejected in the lane or code that is
    /// byte-identical to a registered revision: the budget counts compiles.
    pub const MAX_TOOL_BUILDS: usize = 10;

    /// Initialize the symbiont runtime.
    ///
    /// Creates a temporary dylib crate from the declarations generated by `evolvable!`,
    /// compiles it, and loads the resulting shared library.
    ///
    /// Use [`Profile::Release`] when benchmarking evolved functions — the
    /// optimizer can make orders-of-magnitude difference for compute-heavy code.
    /// [`Profile::Debug`] compiles faster and is fine for correctness-only workloads.
    ///
    /// # Arguments:
    /// - `decls` should be the generated `SYMBIONT_DECLS` constant from the `evolvable` macro.
    /// - `generated_prelude` should be the generated `SYMBIONT_PRELUDE` constant from the macro.
    /// - `dylib_config` defines the compilation profile, dylib dependencies, and configured imports.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    pub async fn new(
        decls: &'static [EvolvableDecl],
        generated_prelude: &'static [&'static str],
        dylib_config: impl Into<DylibConfig>,
    ) -> Result<&'static Runtime> {
        let config = dylib_config.into();
        if decls.is_empty() {
            return Err(Error::NoEvolvableFunctions);
        }

        let fn_sigs = Vec::from_iter(decls.iter().map(|d| d.signature.to_string()));

        // The echo reference of the stub/echo check: every default body as
        // a normalized token string, keyed by function name.
        let default_bodies = decls
            .iter()
            .map(|d| {
                let item: syn::ItemFn = syn::parse_str(d.full_source)
                    .expect("full_source is generated by evolvable! and must parse");
                (d.name.to_string(), default_body_tokens(&item))
            })
            .collect();

        let crate_dir = scaffold_dylib_crate(decls, &config)?;

        let mut prelude = Vec::with_capacity(4);
        prelude.extend(
            generated_prelude
                .iter()
                .copied()
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
        );
        prelude.extend(config.prelude().iter().cloned());

        // The initial revision: every declared function at its default body.
        let glue = harness_glue(decls, &prelude);
        let candidate = initial_candidate(decls);
        let initial_source_bytes = candidate.len();

        // Compile
        compile_dylib(
            &crate_dir,
            config.profile(),
            &candidate,
            &assemble_lib_rs(&candidate, &glue),
        )
        .await?;

        // Copy the build output to the revision-0 path: later `cargo build`
        // runs replace the unversioned artifact, while the versioned copy
        // stays stable for the lifetime of the registry.
        let so_path = find_so(&crate_dir, config.profile())?;
        let v0_path = versioned_so_path(&crate_dir, Revision::INITIAL.as_u64());
        std::fs::copy(&so_path, &v0_path).map_err(|e| {
            Error::DylibLoad(format!(
                "Failed to copy compiled dylib {} to revision-0 path {}: {e}",
                so_path.display(),
                v0_path.display()
            ))
        })?;
        let lib = unsafe {
            Library::new(&v0_path).map_err(|e| {
                Error::DylibLoad(format!("Failed to load {}: {e}", v0_path.display()))
            })?
        };

        // Resolve and cache the function pointers of the initial revision
        // (dispatch is lock-free after this point) and register it.
        let initial = unsafe { RevisionEntry::resolve(lib, decls, candidate)? };
        initial.publish(decls);

        let runtime = Runtime {
            crate_dir,
            so_path,
            fn_sigs,
            default_bodies,
            denied_paths: config.denied_paths().clone(),
            revisions: RwLock::new(vec![Arc::new(initial)]),
            active: AtomicU64::new(Revision::INITIAL.as_u64()),
            decls,
            profile: config.profile(),
            prelude,
            glue,
            build_slot: tokio::sync::Mutex::new(()),
            // Unlimited until a caller asks for a limit, so a host that only
            // ever calls `evolve` is unaffected.
            inference_gate: InferenceGate::unlimited(),
            host_crate: config.host_crate().map(str::to_string),
            doc_index: tokio::sync::OnceCell::new(),
        };

        RUNTIME
            .set(runtime)
            .map_err(|_| Error::AlreadyInitialized)?;

        histogram!(DYLIB_SOURCE_BYTES).record(initial_source_bytes as f64);
        if let Ok(meta) = std::fs::metadata(&v0_path) {
            histogram!(DYLIB_SIZE_BYTES).record(meta.len() as f64);
        }
        gauge!(REVISION_ACTIVE).set(Revision::INITIAL.as_u64() as f64);
        gauge!(crate::observability::REVISIONS_LOADED).set(1.0);

        Ok(RUNTIME.get().expect("just set"))
    }

    /// Generate an LLM response, then parse, validate, compile and register it.
    /// Validation errors are not caught here and fed back to the LLM — that is
    /// [`Runtime::evolve_lane`]'s job — which keeps the prompting behaviour
    /// customizable.
    ///
    /// On success, returns the [`Revision`] the new implementation was
    /// registered under. The dispatch pointers are left alone: publishing is
    /// the caller's decision, because batch lanes register without activating.
    ///
    /// `history` is the whole transcript of the lane. Only
    /// `history[request.history_base..]` goes to the agent. A context or
    /// repeat reset advances `history_base` instead of truncating. The
    /// request therefore gets smaller, but the transcript keeps everything
    /// the lane exchanged.
    ///
    /// `run_out` and `stages` hold the trace records. This method writes them
    /// as the attempt progresses instead of returning them, so the caller
    /// keeps what it recorded before a failure. `run_out` stays `None` only
    /// when the agent run itself failed. That is how the caller separates an
    /// attempt that never got to the model from one the pipeline rejected.
    ///
    /// A run that fails inside the tool-calling loop (turn budget exhausted,
    /// cancelled, unknown tool call) has still produced messages before the
    /// abort. Rig ships that partial transcript out in the error, and the
    /// failure path appends it to `history`: the retry then sees the tool
    /// exchanges the aborted run already made instead of replaying the
    /// identical request that just exhausted its budget.
    ///
    /// `request.tools_ctx` holds the previous candidate with its compiler
    /// errors, if there is one. A response may then edit it instead of
    /// repeating it (see [`crate::edit`]); the edited text is the candidate.
    /// The context is installed on the run's task, so the revision tools the
    /// agent calls during the run find their lane; the builds they made are
    /// drained into `stages` after the run, whether it succeeded or not.
    async fn evolve_no_backpressure<AgentT>(
        &self,
        agent: &AgentT,
        request: AttemptRequest<'_>,
        history: &mut Vec<Message>,
        run_out: &mut Option<AgentRun>,
        stages: &mut StageTimings,
    ) -> Result<Revision>
    where
        AgentT: EvolutionAgent,
    {
        let AttemptRequest {
            prompt,
            history_base,
            tools_ctx,
            tools,
        } = request;
        debug!("prompt: {}", prompt.green());
        let t0 = Instant::now();
        let visible = history.get(history_base..).unwrap_or_default().to_vec();
        let visible_len = visible.len();
        debug!("chat history: {visible:?}");

        // The agent implementation drives any tool-calling turns to
        // completion internally and returns only the final text.
        let run = tools_ctx
            .scope(async {
                match tools {
                    ToolAccess::Allowed => agent.run(prompt, visible).await,
                    ToolAccess::Withdrawn => agent.run_without_tools(prompt, visible).await,
                }
            })
            .await;
        // The candidates the tools built during the run belong to this
        // attempt, whichever way the run ended.
        stages.set_tool_builds(tools_ctx.take_builds());
        let run = match run {
            Ok(run) => run,
            Err(RunError { error, partial }) => {
                let err = Error::from(error);
                counter!(LLM_RUNS, "outcome" => "error").increment(1);
                counter!(
                    INFERENCE_ERRORS,
                    "reason" => inference_error_reason(&err),
                )
                .increment(1);
                stages.set_llm(Some(t0.elapsed()));
                // A run that died mid-loop still produced messages and paid
                // for tokens. Keep both: the messages go into the history so
                // the next request extends the conversation that broke off
                // instead of repeating it from the start, and the usage goes
                // into the counters and the attempt's trace so a lane that
                // spends an hour on tool turns and then times out does not
                // account as an hour of nothing. The agent's own record is
                // preferred; the transcript some rig errors carry (input
                // history included) is the fallback.
                let recovered = partial.map(|partial| *partial).or_else(|| {
                    err.aborted_run_messages(visible_len)
                        .filter(|messages| !messages.is_empty())
                        .map(|new_messages| PartialRun {
                            new_messages,
                            ..PartialRun::default()
                        })
                });
                if let Some(partial) = recovered {
                    debug!(
                        "Recovered {} messages and {} answered request(s) from the failed run",
                        partial.new_messages.len(),
                        partial.completion_calls.len()
                    );
                    record_token_usage(&partial.usage);
                    history.extend(partial.new_messages.iter().cloned());
                    *run_out = Some(AgentRun {
                        output: String::new(),
                        new_messages: partial.new_messages,
                        usage: partial.usage,
                        completion_calls: partial.completion_calls,
                    });
                }
                return Err(err);
            }
        };
        counter!(LLM_RUNS, "outcome" => "ok").increment(1);
        record_token_usage(&run.usage);
        histogram!(LLM_RUN_INPUT_TOKENS).record(run.usage.input_tokens as f64);
        histogram!(LLM_RUN_OUTPUT_TOKENS).record(run.usage.output_tokens as f64);
        histogram!(LLM_RUN_MESSAGES).record(run.new_messages.len() as f64);
        debug!("llm_response: {}", run.output.blue());
        info!("token usage for this run: {:?}", run.usage);

        // `new_messages` contains the prompt, assistant turns and any
        // tool exchanges of this run, so extending is sufficient.
        history.extend(run.new_messages.iter().cloned());
        let llm_response = run.output.clone();
        let llm_time = t0.elapsed().as_millis();
        stages.set_llm(Some(t0.elapsed()));
        *run_out = Some(run);
        histogram!(
            PIPELINE_STAGE_DURATION,
            "stage" => stage::LLM
        )
        .record(t0.elapsed().as_secs_f64());

        let candidate = match self.answer_of(&llm_response, tools_ctx, stages)? {
            Answer::Chosen(revision) => {
                info!("Agent chose revision {revision}. LLM generation: {llm_time}ms.");
                return Ok(revision);
            }
            Answer::Candidate(candidate) => candidate,
        };

        // Compile, load and retain the new revision. Whether it also becomes
        // the active one is up to the caller.
        let revision = self
            .build_and_register(candidate, stages.build_mut())
            .await?;

        info!("Built revision {revision}. LLM generation: {llm_time}ms.");

        Ok(revision)
    }

    /// What a response answers with, in order of precedence:
    ///
    /// 1. The revision the agent chose with `submit_revision` during the
    ///    run. It is built and registered already. The choice wins over a
    ///    code block in the same reply: the agent named what it wants, and
    ///    the block is most likely a quote of it.
    /// 2. The code block, parsed and validated (see
    ///    [`Runtime::parse_and_validate`]), as before the tools existed.
    /// 3. Without a code block, a `revision: N` line naming a revision the
    ///    tools built: the text form of the choice, for a run whose tools
    ///    were withdrawn.
    ///
    /// A reply with none of these fails with [`Error::UnsubmittedRevisions`]
    /// when the tools built revisions in this lane, so the nudge asks for
    /// the choice, and with [`Error::NoRustCode`] otherwise.
    fn answer_of(
        &self,
        response: &str,
        tools_ctx: &ToolContext,
        stages: &mut StageTimings,
    ) -> Result<Answer> {
        if let Some(revision) = tools_ctx.take_submitted() {
            return Ok(Answer::Chosen(revision));
        }
        let edit_base = tools_ctx.edit_base();
        match self.parse_and_validate(stages, |stages| {
            self.candidate_of(response, edit_base.as_ref(), stages)
        }) {
            Ok(candidate) => Ok(Answer::Candidate(candidate)),
            Err(Error::NoRustCode) => {
                let built = tools_ctx.built();
                if built.is_empty() {
                    return Err(Error::NoRustCode);
                }
                match crate::parser::submission_line(response) {
                    Some(revision) if built.contains(&revision) => Ok(Answer::Chosen(revision)),
                    _ => Err(Error::UnsubmittedRevisions { built }),
                }
            }
            Err(e) => Err(e),
        }
    }

    /// The parse and validate stage: `parse` yields the candidate, which
    /// must then match the declared signatures and implement something. On
    /// success the result is the candidate's source, ready for the build.
    ///
    /// The candidate that goes on to the build is the text as the agent
    /// wrote it, never a re-rendering of the AST: the compiler's line numbers
    /// then point into text the agent has seen. The `syn` AST never leaves
    /// this method: `syn` trees are `!Send`, and holding one across the
    /// compile `await` would make the calling future `!Send`.
    ///
    /// The stage's duration is recorded in `stages` whether it passes or not:
    /// a rejected candidate must still report the time its parse and
    /// validation took.
    pub(crate) fn parse_and_validate(
        &self,
        stages: &mut StageTimings,
        parse: impl FnOnce(&mut StageTimings) -> Result<Candidate>,
    ) -> Result<String> {
        let t1 = Instant::now();
        let result = parse(stages).and_then(|candidate| {
            // Validate signatures match declarations
            validate_generated_ast(candidate.ast(), &self.fn_sigs, &self.denied_paths)?;
            // Reject stub bodies outright, and candidates that implement
            // nothing: an echo of every declared default body. A candidate
            // that genuinely evolves one function while leaving others at
            // their defaults is a partial evolution and passes.
            check_implementation_bodies(candidate.ast(), &self.default_bodies)?;
            Ok(candidate.into_source())
        });
        stages.set_parse_validate(Some(t1.elapsed()));
        if result.is_ok() {
            histogram!(
                PIPELINE_STAGE_DURATION,
                "stage" => stage::PARSE_VALIDATE
            )
            .record(t1.elapsed().as_secs_f64());
        }
        result
    }

    /// The candidate a response describes: the whole code block, or the
    /// `edit_base` with the response's edits applied. An edit is recorded in
    /// `stages` for the trace.
    fn candidate_of(
        &self,
        response: &str,
        edit_base: Option<&EditBase>,
        stages: &mut StageTimings,
    ) -> Result<Candidate> {
        let Some(base) = edit_base else {
            return parse_rust_code(response);
        };
        self.edited_candidate(base, &crate::parser::fences(response), stages, || {
            parse_rust_code(response)
        })
    }

    /// The candidate `fences` describe against `base`: the base with the
    /// edits applied, or, when the fences carry no edits, whatever `whole`
    /// parses as the complete candidate. An edit is recorded in `stages` for
    /// the trace; an edit that does not apply is [`Error::EditFailed`] with
    /// the base unchanged.
    pub(crate) fn edited_candidate(
        &self,
        base: &EditBase,
        fences: &[Fence],
        stages: &mut StageTimings,
        whole: impl FnOnce() -> Result<Candidate>,
    ) -> Result<Candidate> {
        let declared: Vec<&str> = self.decls.iter().map(|decl| decl.name).collect();
        match edit::resolve(base, fences, &declared) {
            Ok(edit::Resolved::Edited { source, edits }) => {
                counter!(EVOLVE_EDITS).increment(edits.total() as u64);
                info!(
                    "Applied {} edit(s) to the previous candidate ({} anchor(s), {} hunk(s), {} item(s)).",
                    edits.total(),
                    edits.anchors,
                    edits.hunks,
                    edits.items
                );
                stages.set_edits(Some(edits));
                parse_candidate(source)
            }
            Ok(edit::Resolved::Whole) => whole(),
            Err(error) => Err(Error::EditFailed {
                code: base.source().to_string(),
                err: error.to_string(),
            }),
        }
    }

    /// Write the corrective nudge for `e` to `out`: the text of
    /// [`Error::nudge`], followed by the definitions of the host types a
    /// compile error shows the agent misusing (see [`Runtime::api_hints`]).
    /// Returns the names of those types, for the trace.
    ///
    /// The same text goes to the agent whether the failure came from a
    /// response or from a tool call, so a repair reads the same either way.
    ///
    /// Hands `e` back when it is not a failure the agent can repair
    /// (provider, IO, dylib load).
    pub(crate) async fn render_nudge(
        &self,
        e: Error,
        out: &mut String,
    ) -> std::result::Result<Vec<String>, Error> {
        // The host types an invented API was called on. Read before the
        // error is consumed by the nudge; rendered after it, so the
        // definitions follow the errors they explain.
        let (api_hints, hinted_types) = match &e {
            Error::CompilationFailed { diagnostics, .. } => self.api_hints(diagnostics).await,
            _ => (String::new(), Vec::new()),
        };
        e.nudge(out)?;
        out.push_str(&api_hints);
        Ok(hinted_types)
    }

    /// The definitions of the host types that `diagnostics` show the agent
    /// misusing, rendered for the nudge. Empty without a host crate, without
    /// such an error, or if the host's documentation cannot be built.
    async fn api_hints(&self, diagnostics: &[Diagnostic]) -> (String, Vec<String>) {
        let mut out = String::new();
        if api_hint_names(diagnostics).is_empty() {
            return (out, Vec::new());
        }
        let Some(host_crate) = &self.host_crate else {
            return (out, Vec::new());
        };
        let index = self
            .doc_index
            .get_or_init(async || match DocIndex::host(host_crate).await {
                Ok(index) => Some(index),
                Err(err) => {
                    warn!("Cannot build the host API index for compile-error hints: {err}");
                    None
                }
            })
            .await;
        let attached = match index {
            Some(index) => render_api_hints(index, diagnostics, &mut out),
            None => Vec::new(),
        };
        (out, attached)
    }

    /// Compile `candidate`, load the resulting dylib, and retain it in the
    /// registry under a fresh revision id.
    ///
    /// Does **not** touch the dispatch pointers: the returned revision is
    /// registered and callable through [`crate::RevisionFn`] handles, but the
    /// active revision is unchanged. Use [`Runtime::publish_revision`] to make
    /// it the one `evolvable!` call sites dispatch to.
    ///
    /// The whole body runs under [`Runtime::build_slot`], which is what makes
    /// concurrent lanes safe: the shared crate dir, the shared `so_path`, and
    /// the id assignment are all inside one critical section.
    ///
    /// A candidate that is byte-identical to an already-registered revision
    /// reuses it instead of being built again — see
    /// [`Runtime::registered_with_source`].
    ///
    /// `record` receives the result of the build stage for the trace. This
    /// method writes it before every early return. A candidate that the
    /// compiler rejects therefore still reports the time its compile took.
    pub(crate) async fn build_and_register(
        &self,
        candidate: String,
        record: &mut Option<BuildRecord>,
    ) -> Result<Revision> {
        let t_wait = Instant::now();
        let _build_permit = self.build_slot.lock().await;
        let waited = t_wait.elapsed();
        histogram!(BUILD_SLOT_WAIT).record(waited.as_secs_f64());

        // Identical source compiles to an identical dylib, so there is nothing
        // to gain from building it twice. The check runs inside the build
        // permit, which is what makes it airtight for a batch: two lanes that
        // generated the same code cannot both miss and then both build.
        if let Some(existing) = self.registered_with_source(&candidate)? {
            counter!(REVISION_DEDUP_HITS).increment(1);
            info!(
                "Candidate is byte-identical to revision {existing}; reusing it instead of spending a build."
            );
            *record = Some(BuildRecord::Deduped {
                slot_wait: waited,
                revision: existing,
                autofixes: Vec::new(),
            });
            return Ok(existing);
        }

        debug!("candidate: {candidate}");

        let t_compile = Instant::now();
        // A compile failure is the common self-healing case, and its duration
        // is the most useful number of the whole attempt. Record it before the
        // error propagates.
        let mut applied = Vec::new();
        let (candidate, autofixes) = match self
            .compile_with_autofix(candidate, &mut applied)
            .await
            .inspect_err(|_| {
                *record = Some(BuildRecord::Built {
                    slot_wait: waited,
                    compile: t_compile.elapsed(),
                    load: Duration::ZERO,
                    autofixes: std::mem::take(&mut applied),
                });
            })? {
            Compiled::Fresh(candidate, autofixes) => (candidate, autofixes),
            Compiled::Registered(existing, autofixes) => {
                counter!(REVISION_DEDUP_HITS).increment(1);
                info!("Autofixed candidate is byte-identical to revision {existing}; reusing it.");
                *record = Some(BuildRecord::Deduped {
                    slot_wait: waited,
                    revision: existing,
                    autofixes,
                });
                return Ok(existing);
            }
        };
        let source_bytes = candidate.len();
        let compile_time = t_compile.elapsed();
        histogram!(
            PIPELINE_STAGE_DURATION,
            "stage" => stage::COMPILE
        )
        .record(compile_time.as_secs_f64());

        // Copy the build output to the next revision's own path (which also
        // defeats dlopen path caching) and load it. The id is the registry
        // length, read while holding the build permit so no other lane can
        // claim the same one — and therefore not the same versioned path.
        let t_load = Instant::now();
        let id = {
            let revisions = self.revisions.read().map_err(|_| Error::MutexPoison)?;
            u64::try_from(revisions.len()).expect("registry length fits in u64")
        };
        let versioned_so = versioned_so_path(&self.crate_dir, id);
        std::fs::copy(&self.so_path, &versioned_so)?;
        let dylib_size = std::fs::metadata(&versioned_so).ok().map(|meta| meta.len());
        let new_lib = unsafe {
            Library::new(&versioned_so).map_err(|e| {
                Error::DylibLoad(format!("Failed to load {}: {e}", versioned_so.display()))
            })?
        };

        // Resolve the new revision's symbols and retain it in the registry.
        // Every earlier library stays loaded (keep-all), so earlier revisions
        // remain callable for the lifetime of the process.
        let entry = unsafe { RevisionEntry::resolve(new_lib, self.decls, candidate)? };
        {
            let mut revisions = self.revisions.write().map_err(|_| Error::MutexPoison)?;
            debug_assert_eq!(
                u64::try_from(revisions.len()).expect("registry length fits in u64"),
                id,
                "the registry grew while the build permit was held"
            );
            revisions.push(Arc::new(entry));
            metrics::gauge!(crate::observability::REVISIONS_LOADED)
                .set(u64::try_from(revisions.len()).expect("registry length fits in u64") as f64);
        }

        histogram!(DYLIB_SOURCE_BYTES).record(source_bytes as f64);
        if let Some(bytes) = dylib_size {
            histogram!(DYLIB_SIZE_BYTES).record(bytes as f64);
        }
        histogram!(
            PIPELINE_STAGE_DURATION,
            "stage" => stage::LOAD
        )
        .record(t_load.elapsed().as_secs_f64());
        *record = Some(BuildRecord::Built {
            slot_wait: waited,
            compile: compile_time,
            load: t_load.elapsed(),
            autofixes,
        });

        info!(
            "Registered revision {id}. Timings: build slot wait: {}ms, compilation: {}ms, load: {}ms.",
            waited.as_millis(),
            compile_time.as_millis(),
            t_load.elapsed().as_millis(),
        );

        Ok(Revision::new(id))
    }

    /// Compile `candidate`; when it fails only in ways rustc knows how to
    /// fix, apply those fixes and compile once more.
    ///
    /// Returns the candidate that compiled, which is the input or the
    /// patched text. A patched candidate is what gets registered: the source
    /// a revision reports must be the source its dylib was built from. When
    /// the patched text is a registered revision already (two lanes that made
    /// the same slip converge on the same fix), that revision is returned
    /// instead and nothing is built.
    ///
    /// The second build is the last: its diagnostics describe the patched
    /// text and are reported as they are, together with the fixes that were
    /// applied, so the model's picture of the code matches what the compiler
    /// saw. Applying fixes a second time would need the diagnostics to be
    /// relocated first and buys little; a candidate that needs two rounds of
    /// mechanical fixes has bigger problems the model should look at.
    ///
    /// `applied` receives the fixes as soon as they are applied, so a build
    /// record written on the error path still says what was tried.
    async fn compile_with_autofix(
        &self,
        candidate: String,
        applied: &mut Vec<AppliedFix>,
    ) -> Result<Compiled> {
        let first = match compile_dylib(
            &self.crate_dir,
            self.profile,
            &candidate,
            &assemble_lib_rs(&candidate, &self.glue),
        )
        .await
        {
            Ok(()) => return Ok(Compiled::Fresh(candidate, Vec::new())),
            Err(err) => err,
        };
        let Error::CompilationFailed {
            diagnostics: first_diagnostics,
            ..
        } = &first
        else {
            return Err(first);
        };
        let Some((patched, fixes)) = apply_machine_applicable(&candidate, first_diagnostics) else {
            return Err(first);
        };
        counter!(COMPILE_AUTOFIXES).increment(fixes.len() as u64);
        info!(
            "Applied {} machine-applicable compiler suggestion(s) to the candidate; compiling again.",
            fixes.len()
        );
        debug!("autofixed candidate: {patched}");
        applied.clone_from(&fixes);
        if let Some(existing) = self.registered_with_source(&patched)? {
            return Ok(Compiled::Registered(existing, fixes));
        }
        match compile_dylib(
            &self.crate_dir,
            self.profile,
            &patched,
            &assemble_lib_rs(&patched, &self.glue),
        )
        .await
        {
            Ok(()) => Ok(Compiled::Fresh(patched, fixes)),
            Err(Error::CompilationFailed {
                code,
                mut err,
                diagnostics,
            }) => {
                let mut report = String::new();
                render_fixes_for_prompt(&fixes, &mut report);
                report.push_str(
                    "The code still failed to compile with these fixes applied. The errors below \
                     are located in the code *after* these fixes, so their line numbers may differ \
                     from your code block where a fix added or removed a line:\n",
                );
                err.insert_str(0, &report);
                Err(Error::CompilationFailed {
                    code,
                    err,
                    diagnostics,
                })
            }
            Err(other) => Err(other),
        }
    }

    /// The revision whose source is exactly `source`, if one is registered.
    ///
    /// A linear scan with a full string comparison rather than a hash index:
    /// the registry holds tens to hundreds of entries of a few KB each, so a
    /// miss costs microseconds against a build that costs seconds, and there
    /// is no collision case to get wrong.
    fn registered_with_source(&self, source: &str) -> Result<Option<Revision>> {
        let revisions = self.revisions.read().map_err(|_| Error::MutexPoison)?;
        Ok(revisions
            .iter()
            .position(|entry| entry.source() == source)
            .map(|idx| Revision::new(u64::try_from(idx).expect("registry index fits in u64"))))
    }

    /// Point every `evolvable!` dispatch wrapper at `revision` and record it as
    /// active. `source` is the `source` label of
    /// [`crate::observability::REVISION_ACTIVATIONS`].
    ///
    /// This is the only operation that mutates the swappable dispatch
    /// pointers, so it is where the feedback-loop contract is enforced.
    fn publish_revision(&self, revision: Revision, source: &'static str) -> Result<()> {
        Self::assert_no_calls_in_flight();

        {
            let revisions = self.revisions.read().map_err(|_| Error::MutexPoison)?;
            let entry = usize::try_from(revision.as_u64())
                .ok()
                .and_then(|idx| revisions.get(idx))
                .ok_or_else(|| Error::UnknownRevision {
                    requested: revision,
                    latest: Revision::new(
                        u64::try_from(revisions.len()).expect("registry length fits in u64") - 1,
                    ),
                })?;
            entry.publish(self.decls);
        }
        self.active.store(revision.as_u64(), Ordering::Release);

        gauge!(REVISION_ACTIVE).set(revision.as_u64() as f64);
        counter!(
            REVISION_ACTIVATIONS,
            "source" => source
        )
        .increment(1);
        Ok(())
    }

    /// Assert (debug builds only) that no evolvable function calls are in
    /// flight. Retained revisions are never unmapped, so a violation is no
    /// longer a use-after-unload — but a swap concurrent with running calls
    /// could still publish a torn set of pointers from two different
    /// revisions, so the feedback-loop contract remains.
    ///
    /// Only publishing can tear the pointers, which is why registration
    /// ([`Runtime::build_and_register`]) does not check this — a revision that
    /// is merely retained is invisible to running calls.
    fn assert_no_calls_in_flight() {
        #[cfg(debug_assertions)]
        {
            use crate::debug_call_counter::IN_FLIGHT_CALLS;

            let in_flight = IN_FLIGHT_CALLS.load(Ordering::Acquire);
            assert!(
                in_flight == 0,
                "the active revision was swapped while {in_flight} evolvable function(s) are still executing. \
                 All callers must return before evolve() or activate_revision() — this is the feedback loop contract."
            );
        }
    }

    /// Exponential backoff (capped at 30s) for transient retry attempt `n`.
    fn transient_backoff(n: usize) -> Duration {
        let secs = 1u64 << n.min(5);
        Duration::from_secs(secs.min(30))
    }

    /// Prompt the LLM, validate the response, compile, and hot-swap.
    ///
    /// On success, returns an [`EvolveInfo`] carrying the [`Revision`] the
    /// new implementation was registered under, plus the token usage of the
    /// LLM runs that produced it. The revision stays loaded for the lifetime
    /// of the process, so it can be pointed at again later.
    ///
    /// If the agent produced source byte-identical to an already-registered
    /// revision, that revision is returned and activated instead of being
    /// compiled again — so a returned id is not necessarily a *new* id. Watch
    /// [`crate::observability::REVISION_DEDUP_HITS`] if you need to tell the
    /// two apart.
    ///
    /// If constrained generation fails (parse error, signature mismatch, or
    /// compilation failure), the next turn contains only the latest correction;
    /// prior context remains available in chat history. The LLM retries until it
    /// produces valid code, up to [`Self::MAX_EVOLVE_ATTEMPTS`] attempts. After
    /// that, [`Error::MaxRetriesExceeded`] is returned so a
    /// misbehaving agent cannot hang the runtime indefinitely.
    ///
    /// The chat history is scoped to this call: it starts empty, accumulates
    /// the retry turns, and is discarded on return. Nothing carries over
    /// between `evolve` calls, so request sizes stay bounded in long-lived
    /// processes; callers that want cross-call continuity must render it
    /// into `base_prompt` themselves.
    ///
    /// Transient HTTP errors from the LLM provider (HTTP 429, 5xx, 529
    /// "overloaded") and connection-level failures (timeouts, resets, DNS)
    /// are retried separately with exponential backoff up to
    /// [`Self::MAX_TRANSIENT_RETRIES`] times, and do not count against the
    /// self-healing attempt budget.
    ///
    /// A request that exceeds the model's context window cannot succeed by
    /// resending, so the chat history is discarded and the next request
    /// restarts from `base_prompt`. Such restarts are capped at
    /// [`Self::MAX_CONTEXT_RESETS`] and each one consumes an attempt. If a
    /// request overflows with an already-empty history, `base_prompt` itself
    /// is too large and the error is returned unwrapped.
    ///
    /// If the agent answers a correction with the exact same rejected code
    /// as the previous attempt (weak models echo their own broken answer
    /// out of the chat history), the history is discarded and the next
    /// request restarts from `base_prompt` with an explicit do-not-repeat
    /// instruction. Such attempts still count against the retry budget.
    ///
    /// Every attempt, rejected or registered, is in the [`EvolutionTrace`]
    /// of the result: [`EvolveInfo::trace`] on success, [`EvolveError::trace`]
    /// on failure. A rejected attempt records the candidate the pipeline
    /// turned down and the diagnostics it fed back, so a host can persist
    /// the compiler output of failed attempts for offline analysis.
    ///
    /// # Contract
    ///
    /// All evolvable function calls must have returned before this is called.
    /// This is the natural shape of the feedback loop: run functions, collect
    /// results, evolve, repeat.
    #[expect(
        clippy::manual_async_fn,
        reason = "Ensure the future is `Send` such that it works better with tokios multi-thread runtime"
    )]
    pub fn evolve<AgentT>(
        &self,
        agent: &AgentT,
        base_prompt: &str,
    ) -> impl Future<Output = std::result::Result<EvolveInfo, EvolveError>> + Send
    where
        AgentT: EvolutionAgent + Sync,
    {
        async move {
            // Checked up front as well as in `publish_revision`, so a contract
            // violation surfaces before minutes of inference rather than after.
            Self::assert_no_calls_in_flight();

            self.evolve_lane(agent, base_prompt, Lane::from(0), Publish::Yes)
                .await
        }
    }

    /// Evolve one candidate implementation per prompt, concurrently.
    ///
    /// Each prompt gets its own lane: its own chat history, its own
    /// self-healing retry budget, and its own [`Revision`] on success. Lanes
    /// are independent, so eight slightly different prompts can converge on
    /// eight entirely different implementations.
    /// The returned vector is positionally aligned with `prompts`, and a lane that
    /// exhausts its budget yields `Err` without affecting its siblings.
    ///
    /// Lanes that converge on byte-identical source share one revision rather
    /// than compiling it repeatedly, so the returned ids are not guaranteed to
    /// be distinct. Repeated ids are a useful signal in their own right: the
    /// prompt variants are not diversifying the output. Deduplicate before
    /// evaluating if your fitness function is expensive, and watch
    /// [`crate::observability::REVISION_DEDUP_HITS`] to quantify the collapse.
    ///
    /// Every lane runs concurrently. That is not the same as every lane being
    /// *sent* concurrently: [`Runtime::set_max_in_flight`] caps how many
    /// inference requests reach the endpoint at a time, which is the quantity a
    /// server's batch width and a provider's rate limit are expressed in.
    ///
    /// Results arrive as one `Vec` when the slowest lane is done. To act on
    /// each candidate as it lands — and overlap the next round's generation
    /// with this round's evaluation — use [`Runtime::evolve_batch_stream`].
    ///
    /// # The active revision is not changed
    ///
    /// Unlike [`Runtime::evolve`], no lane publishes. Every successful lane is
    /// compiled, loaded and retained, but the `evolvable!` call sites keep
    /// dispatching to whatever was active before. Evaluate the candidates
    /// through the `<name>_fn` accessors — which return [`crate::RevisionFn`]
    /// handles that pin their own revision — and then commit to a winner with
    /// [`Runtime::activate_revision`]:
    ///
    /// ```rust,ignore
    /// runtime.set_max_in_flight(16);
    /// let results = runtime.evolve_batch(&agent, &prompts).await;
    /// let best = results
    ///     .iter()
    ///     .filter_map(|r| r.as_ref().ok())
    ///     .max_by_key(|info| score(solve_fn(info.revision).expect("just registered").get()))
    ///     .expect("at least one lane succeeded")
    ///     .revision;
    /// runtime.activate_revision(best)?;
    /// ```
    ///
    /// # Failures
    ///
    /// Each lane's rejected attempts are in its own [`EvolutionTrace`], on the
    /// `Ok` and on the `Err` side alike. The trace sits at the lane's index,
    /// so what each prompt variant struggled with needs no attribution step.
    #[expect(
        clippy::manual_async_fn,
        reason = "Ensure the future is `Send` such that it works better with tokios multi-thread runtime"
    )]
    pub fn evolve_batch<'a, AgentT, S>(
        &'a self,
        agent: &'a AgentT,
        prompts: &'a [S],
    ) -> impl Future<Output = Vec<std::result::Result<EvolveInfo, EvolveError>>> + Send + 'a
    where
        AgentT: EvolutionAgent + Sync,
        S: AsRef<str> + Sync,
    {
        async move {
            if prompts.is_empty() {
                return Vec::new();
            }

            info!(
                "Evolving a batch of {} prompts, at most {} inference requests in flight.",
                prompts.len(),
                self.inference_gate.capacity(),
            );
            let t_batch = Instant::now();
            histogram!(EVOLVE_BATCH_SIZE).record(prompts.len() as f64);

            // Completion order in, input order out: the reordering buffer is
            // the whole difference between this and
            // [`Runtime::evolve_batch_stream`].
            let mut slots = Vec::<Option<std::result::Result<EvolveInfo, EvolveError>>>::from_iter(
                prompts.iter().map(|_| None),
            );
            {
                let mut lanes = std::pin::pin!(self.evolve_batch_stream(agent, prompts));
                while let Some((lane, result)) = lanes.next().await {
                    slots[lane] = Some(result);
                }
            }
            let results = Vec::<std::result::Result<EvolveInfo, EvolveError>>::from_iter(
                slots
                    .into_iter()
                    .map(|slot| slot.expect("every lane yields exactly one result")),
            );

            let elapsed = t_batch.elapsed();
            histogram!(EVOLVE_BATCH_DURATION).record(elapsed.as_secs_f64());
            info!(
                "Batch of {} lanes finished in {}ms ({} succeeded).",
                results.len(),
                elapsed.as_millis(),
                results.iter().filter(|r| r.is_ok()).count(),
            );

            results
        }
    }

    /// [`Runtime::evolve_batch`] yielding each lane the moment it finishes,
    /// tagged with its index into `prompts`, instead of collecting the whole
    /// batch first.
    ///
    /// Use this when the caller has something to do with a winner before the
    /// batch is over — evaluate it, score it, and submit the next round's
    /// prompts. A `Vec` return is a barrier by construction: the batch ends
    /// when its slowest lane ends, so with a collected batch the endpoint
    /// spends the tail of every round running one repair loop at concurrency
    /// one. Overlapping rounds is the only thing that fills that tail, and it
    /// requires results to be observable early.
    ///
    /// Because the limit lives on the runtime rather than on the call
    /// ([`Runtime::set_max_in_flight`]), overlapping rounds share one budget:
    /// submitting round `n + 1` while round `n` still has stragglers does not
    /// overrun the endpoint, it just stops the endpoint from going idle.
    ///
    /// ```rust,ignore
    /// runtime.set_max_in_flight(16);
    /// let mut lanes = std::pin::pin!(runtime.evolve_batch_stream(&agent, &prompts));
    /// while let Some((lane, result)) = lanes.next().await {
    ///     if let Ok(info) = result {
    ///         // Scored while the remaining lanes are still generating.
    ///         record(lane, score(solve_fn(info.revision)?.get()));
    ///     }
    /// }
    /// ```
    ///
    /// Every lane carries its own [`EvolutionTrace`], so overlapping rounds
    /// share no state that one round could clobber for another.
    pub fn evolve_batch_stream<'a, AgentT, S>(
        &'a self,
        agent: &'a AgentT,
        prompts: &'a [S],
    ) -> impl Stream<Item = (usize, std::result::Result<EvolveInfo, EvolveError>)> + Send + 'a
    where
        AgentT: EvolutionAgent + Sync,
        S: AsRef<str> + Sync,
    {
        // Constructing a lane future does no work, so building them all up
        // front is free — and it pins the lifetimes.
        let lanes = Vec::from_iter(prompts.iter().enumerate().map(|(lane, prompt)| {
            let evolve =
                self.evolve_lane(agent, prompt.as_ref(), Lane::from(lane as u32), Publish::No);
            async move {
                let result = evolve.await;
                let outcome = if result.is_ok() { "ok" } else { "error" };
                counter!(EVOLVE_BATCH_LANES, "outcome" => outcome).increment(1);
                (lane, result)
            }
        }));

        // Every lane at once. A lane that is not generating holds nothing the
        // endpoint can see, so oversubscribing them relative to
        // [`Runtime::max_in_flight`] is exactly what keeps the endpoint full;
        // the gate does the shaping.
        let admitted = lanes.len().max(1);
        stream::iter(lanes).buffer_unordered(admitted)
    }

    /// Cap max number of inference requests this process sends to the inference endpoint at once.
    /// Applies to everything the runtime sends from now on,
    /// including calls already in flight.
    ///
    /// Lowering it never cancels a request already sent; the surplus drains.
    /// Values below 1 are treated as 1.
    ///
    /// See [`Runtime::evolve_batch`] for what the limit does and does not bound.
    pub fn set_max_in_flight(&self, max_in_flight: u16) {
        self.inference_gate.set_capacity(max_in_flight);
    }

    /// The current inference concurrency limit, as set by
    /// [`Runtime::set_max_in_flight`].
    ///
    /// [`u16::MAX`] until one of them is called: an unconfigured runtime
    /// admits whatever it is asked to send.
    #[must_use]
    pub fn max_in_flight(&self) -> u16 {
        self.inference_gate.capacity()
    }

    /// One independent evolution: the self-healing retry ladder around
    /// [`Runtime::evolve_no_backpressure`] for a single prompt.
    ///
    /// This is the body shared by [`Runtime::evolve`] (one lane, publishing)
    /// and [`Runtime::evolve_batch`] (`n` concurrent lanes, not publishing).
    ///
    /// `lane` only labels the [`EvolutionTrace`] this lane produces.
    ///
    /// On success, the returned [`EvolveInfo`] carries the lane's total
    /// token usage across all of its attempts — a rejected attempt's tokens
    /// are counted too.
    #[expect(
        clippy::manual_async_fn,
        reason = "Ensure the future is `Send` such that it works better with tokios multi-thread runtime"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "The retry policy is one sequential decision ladder; splitting it would obscure the order of the recovery rules"
    )]
    pub fn evolve_lane<AgentT>(
        &self,
        agent: &AgentT,
        base_prompt: &str,
        lane: Lane,
        publish: Publish,
    ) -> impl Future<Output = std::result::Result<EvolveInfo, EvolveError>> + Send
    where
        AgentT: EvolutionAgent + Sync,
    {
        async move {
            let t_start = Instant::now();
            let mut prompt = base_prompt.to_string();
            let mut history: Vec<Message> = Vec::with_capacity(32);
            // A context or repeat reset retires everything before this index.
            // Those messages stay in the transcript of the trace, but leave
            // the request. A reset advances this index instead of truncating,
            // so the trace keeps what the lane exchanged.
            let mut history_base: usize = 0;
            let mut attempts: usize = 0;
            let mut context_resets: usize = 0;
            let mut transient_attempts: usize = 0;
            // Wall time the lane has lost to transient failures so far, the
            // second bound on them beside the count.
            let mut transient_elapsed = Duration::ZERO;
            // Candidate of the agent's most recent answer (`None` for an
            // answer without code), used to detect an agent that echoes the
            // same broken code back verbatim.
            let mut last_failed_code: Option<String> = None;
            // The lane's state for the revision tools, and the candidate the
            // next response may edit instead of retyping: the most recent
            // one that parsed and validated, with the compiler errors the
            // agent was shown about it. No base on the first attempt and
            // after a reset, when the agent no longer sees the code the
            // base would refer to.
            let tools_ctx = ToolContext::new(Self::MAX_TOOL_BUILDS);
            // Tools are withdrawn for the rest of the lane the first time a
            // run spends its whole turn budget on them without answering.
            // Nudging such a run to answer while the tools stay on does not
            // work: v0.28 traces show a lane spend ten attempts of fifty
            // turns each on the same failing documentation lookup. Without
            // tools the model has one move left, and the definitions it did
            // fetch are still in its history.
            let mut tools = ToolAccess::Allowed;
            let mut trace = EvolutionTrace::new(
                agent.provider().to_string(),
                agent.model().to_string(),
                lane,
                agent.system_prompt(),
                base_prompt.to_string(),
            );

            // Finish the lane. Move the transcript into the trace and set the
            // outcome. Every exit path calls this.
            macro_rules! finish {
                ($outcome:expr) => {{
                    trace.set_history(std::mem::take(&mut history));
                    trace.set_outcome($outcome);
                    trace.set_duration(t_start.elapsed());
                    trace
                }};
            }

            loop {
                attempts += 1;
                let t_attempt = Instant::now();
                let produced_start = history.len();
                let mut run_out: Option<AgentRun> = None;
                let mut stages = StageTimings::default();
                let attempt_prompt = prompt.clone();

                // Build the `RunTrace` of this attempt from what the pipeline
                // got far enough to produce.
                macro_rules! run_trace {
                    () => {
                        run_out.take().map(|run| {
                            RunTrace::builder()
                                .produced(produced_start..history.len())
                                .response(run.output)
                                .usage(run.usage)
                                .completion_calls(run.completion_calls)
                                .build()
                        })
                    };
                }
                // The gate is entered inside `evolve_no_backpressure` and
                // left again before the compile stage, so a lane queued on
                // `build_slot` is not also occupying a slot at the endpoint.
                //
                // Priority rises with the attempt number. A lane deep in its
                // repair ladder is the one that decides when the batch ends,
                // and its request is a prefix-extension of the request before
                // it — so serving it ahead of freshly admitted lanes is both
                // the shortest path to finishing and the cheapest prefill in
                // the batch.
                match self
                    .inference_gate
                    .scope(
                        Priority::attempt(attempts),
                        self.evolve_no_backpressure(
                            agent,
                            AttemptRequest {
                                prompt: &prompt,
                                history_base,
                                tools_ctx: &tools_ctx,
                                tools,
                            },
                            &mut history,
                            &mut run_out,
                            &mut stages,
                        ),
                    )
                    .await
                {
                    Ok(revision) => {
                        // The registered source is the candidate the build
                        // accepted, autofixes included.
                        let candidate = self.revision_code(revision);
                        if publish == Publish::Yes {
                            // The revision built and registered. Only the step
                            // that makes it active can still fail. The trace is
                            // complete at this point and worth keeping.
                            if let Err(e) = self.publish_revision(revision, "evolve") {
                                let reason = e.to_string();
                                trace.push_attempt(
                                    attempts,
                                    attempt_prompt,
                                    run_trace!(),
                                    stages,
                                    candidate,
                                    LadderEvent::Terminal {
                                        reason: reason.clone(),
                                    },
                                    t_attempt.elapsed(),
                                );
                                return Err(EvolveError::new(
                                    e,
                                    finish!(TraceOutcome::Failed { reason }),
                                ));
                            }
                            info!("Hot-reloaded evolvable dylib (revision {revision}).");
                        }
                        histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                        histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                        trace.push_attempt(
                            attempts,
                            attempt_prompt,
                            run_trace!(),
                            stages,
                            candidate,
                            LadderEvent::Registered { revision },
                            t_attempt.elapsed(),
                        );
                        return Ok(EvolveInfo::new(
                            revision,
                            finish!(TraceOutcome::Registered { revision }),
                        ));
                    }
                    Err(e) => {
                        counter!(
                            EVOLVE_FAILURES,
                            "kind" => failure_kind_of(&e)
                        )
                        .increment(1);
                        // The text the pipeline rejected, for the trace. Every
                        // exit of this arm records exactly one attempt, so the
                        // owned copy moves into whichever `push_attempt` runs.
                        let candidate = e.candidate().map(str::to_owned);
                        // A verbatim repeat of the previously rejected code.
                        // The reference is the last answer the agent gave,
                        // with code or without: the same code after a prose
                        // answer is not an echo. A failed edit is not an
                        // answer of its own (it left the previous candidate
                        // standing), and a transport failure is not one
                        // either; neither moves the reference.
                        let answered = candidate.is_some()
                            || matches!(e, Error::NoRustCode | Error::UnsubmittedRevisions { .. })
                            || e.exhausted_tool_turns();
                        let repeated = candidate.as_deref().is_some_and(|code| !code.is_empty())
                            && last_failed_code == candidate;
                        if answered {
                            last_failed_code.clone_from(&candidate);
                        }
                        // What the next response may edit. A candidate the
                        // compiler rejected is a valid edit base: it parsed,
                        // it validated, and the agent is about to see its
                        // errors by number. Any other failure keeps the base
                        // as it was: a response whose edits did not apply
                        // left the base untouched, and a response without
                        // valid code gave the agent nothing new to refer to.
                        if let Error::CompilationFailed {
                            code, diagnostics, ..
                        } = &e
                        {
                            tools_ctx.set_edit_base(Some(EditBase::new(
                                code.clone(),
                                diagnostics.clone(),
                            )));
                        }
                        // A request that exceeds the model's context window can
                        // never succeed by resending: shrink it instead.
                        // Discard the accumulated retry history and restart
                        // from the base prompt. If even a fresh request
                        // overflows (empty history), the base prompt itself is
                        // too large and only the caller can slim it down.
                        if is_context_size_error(&e) {
                            if history.len() == history_base {
                                warn!(
                                    "Request exceeds the model's context window even without \
                                     chat history; the base prompt is too large: {e}"
                                );
                                histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                                histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                                let reason = e.to_string();
                                trace.push_attempt(
                                    attempts,
                                    attempt_prompt,
                                    run_trace!(),
                                    stages,
                                    candidate,
                                    LadderEvent::Terminal {
                                        reason: reason.clone(),
                                    },
                                    t_attempt.elapsed(),
                                );
                                return Err(EvolveError::new(
                                    e,
                                    finish!(TraceOutcome::Failed { reason }),
                                ));
                            }
                            if context_resets >= Self::MAX_CONTEXT_RESETS {
                                warn!(
                                    "Context-overflow restart budget exhausted \
                                     ({context_resets}/{}); giving up. Last error: {e}",
                                    Self::MAX_CONTEXT_RESETS
                                );
                                histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                                histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                                let reason = e.to_string();
                                trace.push_attempt(
                                    attempts,
                                    attempt_prompt,
                                    run_trace!(),
                                    stages,
                                    candidate,
                                    LadderEvent::Terminal {
                                        reason: reason.clone(),
                                    },
                                    t_attempt.elapsed(),
                                );
                                return Err(EvolveError::new(
                                    Error::MaxRetriesExceeded {
                                        attempts,
                                        last_error: Box::new(e),
                                    },
                                    finish!(TraceOutcome::Failed { reason }),
                                ));
                            }
                            context_resets += 1;
                            let dropped = history.len() - history_base;
                            warn!(
                                "Request exceeded the model's context window (restart \
                                 {context_resets}/{}); discarding {dropped} history messages and \
                                 restarting from the base prompt",
                                Self::MAX_CONTEXT_RESETS,
                            );
                            counter!(EVOLVE_CONTEXT_RESETS).increment(1);
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                candidate,
                                LadderEvent::ContextReset {
                                    messages_dropped: dropped,
                                    brief: e.to_string(),
                                },
                                t_attempt.elapsed(),
                            );
                            history_base = history.len();
                            prompt.clear();
                            prompt.push_str(base_prompt);
                            // The agent no longer sees the code an edit
                            // would refer to.
                            tools_ctx.set_edit_base(None);
                            // Withdrawing tools relied on the definitions the
                            // agent fetched staying in its history; the reset
                            // discards them, so it must be able to fetch
                            // them again.
                            tools = ToolAccess::Allowed;
                            // The restart consumes this attempt: unlike
                            // transient retries, an overflowing request is
                            // not the LLM's fault but it must not be free.
                            continue;
                        }

                        // Transient HTTP errors (rate limits, overload, gateway
                        // failures) are not the LLM's fault: retry with
                        // exponential backoff and don't count against the
                        // self-healing attempt budget.
                        if is_transient_http_error(&e) {
                            transient_elapsed += t_attempt.elapsed();
                            let out_of_time = transient_elapsed >= Self::MAX_TRANSIENT_WALL_CLOCK;
                            if transient_attempts >= Self::MAX_TRANSIENT_RETRIES || out_of_time {
                                warn!(
                                    "Transient HTTP error retry budget exhausted ({transient_attempts}/{} retries, \
                                     {:.0?} of {:?} wall clock); giving up. Last error: {e}",
                                    Self::MAX_TRANSIENT_RETRIES,
                                    transient_elapsed,
                                    Self::MAX_TRANSIENT_WALL_CLOCK,
                                );
                                histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                                histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                                let reason = e.to_string();
                                trace.push_attempt(
                                    attempts,
                                    attempt_prompt,
                                    run_trace!(),
                                    stages,
                                    candidate,
                                    LadderEvent::Terminal {
                                        reason: reason.clone(),
                                    },
                                    t_attempt.elapsed(),
                                );
                                return Err(EvolveError::new(
                                    e,
                                    finish!(TraceOutcome::Failed { reason }),
                                ));
                            }
                            let backoff = Self::transient_backoff(transient_attempts);
                            transient_attempts += 1;
                            transient_elapsed += backoff;
                            counter!(LLM_TRANSIENT_RETRIES).increment(1);
                            histogram!(LLM_RETRY_BACKOFF).record(backoff.as_secs_f64());
                            warn!(
                                "Transient HTTP error from LLM provider (retry {transient_attempts}/{} in {:?}): {e}",
                                Self::MAX_TRANSIENT_RETRIES,
                                backoff,
                            );
                            // A run that got some tool turns answered before
                            // the endpoint failed left them in the history
                            // (`evolve_no_backpressure`). Resending the same
                            // prompt after them would open the task a second
                            // time; ask the model to carry on instead. A run
                            // that never got an answer left nothing, and the
                            // prompt goes out again as it was.
                            let progressed = run_out
                                .as_ref()
                                .is_some_and(|run| !run.completion_calls.is_empty());
                            if progressed {
                                prompt.clear();
                                prompt.push_str(TRANSIENT_CONTINUE_NUDGE);
                            }
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                candidate,
                                LadderEvent::TransientRetry {
                                    backoff,
                                    cause: e.to_string(),
                                },
                                t_attempt.elapsed(),
                            );
                            // Don't count this against the self-healing budget.
                            attempts -= 1;
                            tokio::time::sleep(backoff).await;
                            continue;
                        }

                        if attempts >= Self::MAX_EVOLVE_ATTEMPTS {
                            warn!(
                                "Evolution failed after {attempts} attempts; giving up. Last error: {e}"
                            );
                            histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                            histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                            let reason = e.to_string();
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                candidate,
                                LadderEvent::Terminal {
                                    reason: reason.clone(),
                                },
                                t_attempt.elapsed(),
                            );
                            return Err(EvolveError::new(
                                Error::MaxRetriesExceeded {
                                    attempts,
                                    last_error: Box::new(e),
                                },
                                finish!(TraceOutcome::Failed { reason }),
                            ));
                        }

                        info!(
                            "Function evolution error (attempt {attempts}/{}): {e}.\nSelf-healing from error...",
                            Self::MAX_EVOLVE_ATTEMPTS
                        );

                        prompt.clear();

                        // A verbatim repeat of already-rejected code means the
                        // correction nudge is not working: the agent is echoing
                        // its own broken answer from the chat history (weak
                        // models do this persistently). Quoting the same code
                        // back a third time only reinforces the echo, so
                        // discard the history and restart from the base prompt
                        // with an explicit do-not-repeat instruction that does
                        // NOT quote the rejected code.
                        if repeated {
                            counter!(EVOLVE_REPEAT_RESETS).increment(1);
                            let dropped = history.len() - history_base;
                            warn!(
                                "Agent repeated the same rejected code verbatim; discarding \
                                 {dropped} history messages and restarting from the base prompt",
                            );
                            history_base = history.len();
                            tools_ctx.set_edit_base(None);
                            write!(
                                prompt,
                                "{base_prompt}\n\nYour previous attempt was rejected: {}\n\
                                 You already answered with that exact code before and it was \
                                 rejected with the same error, so do NOT repeat it. Respond \
                                 with a different, valid implementation.",
                                e
                            )
                            .expect(EXPECT_WRITE);
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                candidate,
                                LadderEvent::RepeatReset {
                                    messages_dropped: dropped,
                                    brief: e.to_string(),
                                },
                                t_attempt.elapsed(),
                            );
                            continue;
                        }

                        // The nudge that the ladder builds below is the same
                        // text as the diagnostics that go to the agent. Record
                        // the ladder event after the match writes that nudge.
                        let kind = failure_kind_of(&e).to_string();

                        if e.exhausted_tool_turns() && tools == ToolAccess::Allowed {
                            warn!(
                                "Agent spent its tool-call turn budget without producing code; \
                                 withdrawing tools for the rest of the lane"
                            );
                            counter!(EVOLVE_TOOLS_WITHDRAWN).increment(1);
                            tools = ToolAccess::Withdrawn;
                        }

                        // Add a nudge prompt.
                        let hinted_types = match self.render_nudge(e, &mut prompt).await {
                            Ok(hinted_types) => hinted_types,
                            Err(e) => {
                                warn!("Unhandled error: {e}");
                                let reason = e.to_string();
                                trace.push_attempt(
                                    attempts,
                                    attempt_prompt,
                                    run_trace!(),
                                    stages,
                                    candidate,
                                    LadderEvent::Terminal {
                                        reason: reason.clone(),
                                    },
                                    t_attempt.elapsed(),
                                );
                                return Err(EvolveError::new(
                                    e,
                                    finish!(TraceOutcome::Failed { reason }),
                                ));
                            }
                        };
                        // Without tools the agent cannot call `submit_revision`
                        // for the revisions it built with them. Name the text
                        // form of the choice, so those builds are not lost.
                        if tools == ToolAccess::Withdrawn {
                            let built = tools_ctx.built();
                            if !built.is_empty() {
                                write!(
                                    prompt,
                                    " You built revisions {} with the tools before they were \
                                     withdrawn. To activate one of them, reply with the single \
                                     line `revision: N` instead of code.",
                                    crate::tools::revision_list(&built)
                                )
                                .expect(EXPECT_WRITE);
                            }
                        }

                        trace.push_attempt(
                            attempts,
                            attempt_prompt,
                            run_trace!(),
                            stages,
                            candidate,
                            LadderEvent::SelfHeal {
                                kind,
                                diagnostics: prompt.clone(),
                                api_hints: hinted_types,
                            },
                            t_attempt.elapsed(),
                        );
                    }
                }
            }
        }
    }

    /// Retrieve and clear the last panic message from the **active**
    /// revision's dylib.
    ///
    /// Returns `Some(message)` if the most recent evolvable function call
    /// panicked, `None` otherwise. The stored message is cleared on read.
    ///
    /// Call this after each evolvable function invocation to detect panics
    /// that were caught inside the dylib. Note that calls through a
    /// [`crate::RevisionFn`] handle store their panics in *that* revision's
    /// buffer — read those with [`crate::RevisionFn::take_panic`].
    pub fn take_panic(&self) -> Option<String> {
        let ptr = TAKE_PANIC_PTR.load(Ordering::Acquire);
        // SAFETY: TAKE_PANIC_PTR is only ever set from `__symbiont_take_panic`
        // symbols resolved out of libraries the registry keeps loaded.
        unsafe { crate::revision::read_panic_buffer(ptr.cast_const()) }
    }

    /// Path to the temporary crate directory.
    pub fn crate_dir(&self) -> &Path {
        &self.crate_dir
    }

    /// Get the function signature strings for all evolvable functions.
    pub fn fn_sigs(&self) -> &[String] {
        &self.fn_sigs
    }

    /// Get the prelude source injected into the generated dylib.
    ///
    /// This includes inline items from `evolvable!` and configured imports such
    /// as `use host::prelude::*;`.
    pub fn fn_prelude(&self) -> Vec<FullSource<'_>> {
        Vec::from_iter(self.prelude.iter().map(|v| FullSource(v)))
    }

    /// Get the full function signatures, including doc comments and default function body.
    ///
    /// Returns each source wrapped in [`FullSource`], which preserves real line
    /// breaks when pretty-printed (`{:#?}`) so logs stay readable.
    ///
    /// The other relevant imports/items that a function may require can be found in `fn_preludes`.
    pub fn fn_full_sources(&self) -> Vec<FullSource<'static>> {
        Vec::from_iter(self.decls.iter().map(|d| FullSource(d.full_source)))
    }

    /// Get the current LLM-generated code, byte for byte as the agent wrote
    /// it (no prelude, no panic protocol, no export wrappers). Suitable for
    /// feeding back into the LLM prompt or displaying to the user.
    ///
    /// This is the source of the revision the dispatch pointers currently
    /// point at: the latest successful evolution.
    pub fn current_code(&self) -> String {
        self.revision_code(self.active_revision())
            .expect("the active revision is always registered")
    }

    /// The revision whose code the `evolvable!` dispatch wrappers currently
    /// execute.
    pub fn active_revision(&self) -> Revision {
        Revision::new(self.active.load(Ordering::Acquire))
    }

    /// Number of registered revisions: the initial build plus one per
    /// successful evolution. Valid revision ids are `0..revision_count()`.
    pub fn revision_count(&self) -> u64 {
        let revisions = self
            .revisions
            .read()
            .expect("revisions RwLock is not poisoned");
        u64::try_from(revisions.len()).expect("registry length fits in u64")
    }

    /// The generated source of `revision` as the agent wrote it (no prelude,
    /// no panic protocol, no export wrappers), or `None` if no such revision
    /// was registered.
    pub fn revision_code(&self, revision: Revision) -> Option<String> {
        let idx = usize::try_from(revision.as_u64()).ok()?;
        let revisions = self
            .revisions
            .read()
            .expect("revisions RwLock is not poisoned");
        revisions.get(idx).map(|entry| entry.source().to_owned())
    }

    /// Re-activate a previously registered revision.
    ///
    /// Republishes the revision's function pointers, which were resolved once
    /// when its dylib was first loaded: afterwards all `evolvable!` call
    /// sites dispatch to `revision`'s code and [`Runtime::current_code`]
    /// returns its source. No parsing or compilation is involved — the dylib
    /// has stayed loaded since it was hot-swapped, so activation costs a
    /// handful of atomic stores instead of an evolution round.
    ///
    /// Use it to roll back to the best revision a search discovered, to
    /// implement undo, or to re-deploy a known-good implementation for a
    /// final evaluation.
    ///
    /// Returns [`Error::UnknownRevision`] if `revision` was never registered;
    /// the active revision is left unchanged in that case.
    ///
    /// # Contract
    ///
    /// Same as [`Runtime::evolve`]: all evolvable function calls must have
    /// returned before this is called. Enforced with an assertion in debug
    /// builds, zero-cost in release.
    pub fn activate_revision(&self, revision: Revision) -> Result<()> {
        self.publish_revision(revision, "manual")?;
        info!("Activated revision {revision}.");
        Ok(())
    }

    /// Return the function signature and body for a single function base on its `fn_name`
    pub fn current_function(&self, fn_name: &str) -> Option<syn::ItemFn> {
        let code = self.current_code();
        let file: syn::File = syn::parse_str(&code).ok()?; // Its always valid code.
        file.items.into_iter().find_map(|item| match item {
            syn::Item::Fn(f) if f.sig.ident == fn_name => Some(f),
            _ => None,
        })
    }
}

/// Internal lookup behind the `<name>_fn` accessors generated by `evolvable!`:
/// resolve an untyped [`RevisionFn`] for the declaration whose dispatch static
/// is `fn_ptr_static` (identified by pointer identity, no strings involved).
///
/// Returns `None` if the runtime is not initialized, the declaration is not
/// registered, or `revision` does not exist. The generated accessor casts the
/// result to the concrete `fn` type it was expanded with.
///
/// Not part of the public API — used by `evolvable!` expansion.
#[doc(hidden)]
pub fn revision_fn_lookup(
    fn_ptr_static: &'static AtomicPtr<()>,
    revision: Revision,
) -> Option<RevisionFn<*const ()>> {
    let runtime = RUNTIME.get()?;
    let idx = runtime
        .decls
        .iter()
        .position(|decl| std::ptr::eq(decl.fn_ptr, fn_ptr_static))?;
    let revisions = runtime
        .revisions
        .read()
        .expect("revisions RwLock is not poisoned");
    let entry = revisions.get(usize::try_from(revision.as_u64()).ok()?)?;
    Some(RevisionFn::new_untyped(
        revision,
        entry.fn_ptr_at(idx),
        Arc::clone(entry),
    ))
}
