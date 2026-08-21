// SPDX-License-Identifier: MPL-2.0
//! The runtime module contains the primary `Runtime`,
//! managing the lifecycle of the temporary dylib crate: creation, compilation,
//! loading, and hot-reloading.

#[cfg(miri)]
use std::time::Instant;
use std::{
    collections::hash_map::DefaultHasher,
    fmt::Write,
    hash::{
        Hash,
        Hasher,
    },
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
use prettyplease::unparse;
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
    BuildRecord,
    DylibConfig,
    EvolutionAgent,
    EvolutionTrace,
    EvolvableDecl,
    EvolveError,
    EvolveFailure,
    EvolveInfo,
    FullSource,
    LadderEvent,
    Profile,
    RunTrace,
    StageTimings,
    TraceOutcome,
    compiler::compile_dylib,
    error::{
        Error,
        Result,
    },
    inference_gate::{
        InferenceGate,
        Priority,
    },
    observability::{
        BUILD_SLOT_WAIT,
        DYLIB_SIZE_BYTES,
        DYLIB_SOURCE_BYTES,
        EVOLVE_ATTEMPTS,
        EVOLVE_BATCH_DURATION,
        EVOLVE_BATCH_LANES,
        EVOLVE_BATCH_SIZE,
        EVOLVE_CONTEXT_RESETS,
        EVOLVE_DURATION,
        EVOLVE_FAILURES,
        EVOLVE_REPEAT_RESETS,
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
        stage,
    },
    parser::parse_rust_code,
    revision::{
        Revision,
        RevisionEntry,
        RevisionFn,
    },
    utils::{
        find_so,
        generate_cargo_toml,
        generate_lib_rs,
        is_context_size_error,
        is_transient_http_error,
        versioned_so_path,
    },
    validation::validate_generated_ast,
};

/// Singleton runtime instance.
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Whether a successfully registered revision should also become the active
/// one. Batch lanes register without publishing, so the host can evaluate all
/// candidates before choosing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Publish {
    Yes,
    No,
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
    /// Rust source snippets prepended to the dylib's `lib.rs` on every
    /// (re)compilation. This includes inline items declared inside
    /// `evolvable! { ... }` and configured imports such as
    /// `use host::prelude::*;`.
    prelude: Vec<String>,
    /// Failed attempts of the most recent [`Runtime::evolve`] call that fed
    /// backpressure to the agent; drained by
    /// [`Runtime::take_evolve_failures`].
    evolve_failures: RwLock<Vec<EvolveFailure>>,
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
    /// [`Self::MAX_EVOLVE_ATTEMPTS`].
    pub const MAX_TRANSIENT_RETRIES: usize = 6;

    /// Maximum number of times a lane may discard its accumulated chat
    /// history and restart from the base prompt after a context-overflow
    /// error. Unlike transient retries, each restart counts against
    /// [`Self::MAX_EVOLVE_ATTEMPTS`]: resending is a consumed attempt, not a
    /// free one, so a lane that keeps overflowing cannot retry without limit.
    pub const MAX_CONTEXT_RESETS: usize = 3;

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

        // Create a stable temp directory based on function names
        let mut hasher = DefaultHasher::new();
        for d in decls {
            d.name.hash(&mut hasher);
        }
        let hash = hasher.finish();
        let crate_dir = std::env::temp_dir().join(format!("symbiont-evolvable-{hash:x}"));
        std::fs::create_dir_all(crate_dir.join("src"))?;

        // Write Cargo.toml
        let cargo_toml = generate_cargo_toml(config.dependencies(), config.patches());
        std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;

        let mut prelude = Vec::with_capacity(4);
        prelude.extend(
            generated_prelude
                .iter()
                .copied()
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
        );
        prelude.extend(config.prelude().iter().cloned());

        // Write src/lib.rs from all default_source entries
        let lib_rs = generate_lib_rs(decls, &prelude);
        let initial_source_bytes = lib_rs.len();

        // Compile
        compile_dylib(&crate_dir, config.profile(), &lib_rs).await?;

        // Copy the build output to the revision-0 path: later `cargo build`
        // runs replace the unversioned artifact, while the versioned copy
        // stays stable for the lifetime of the registry.
        let so_path = find_so(&crate_dir, config.profile())?;
        let v0_path = versioned_so_path(&crate_dir, Revision::INITIAL.as_u64());
        std::fs::copy(&so_path, &v0_path)?;
        let lib = unsafe {
            Library::new(&v0_path).map_err(|e| {
                Error::DylibLoad(format!("Failed to load {}: {e}", v0_path.display()))
            })?
        };

        // Resolve and cache the function pointers of the initial revision
        // (dispatch is lock-free after this point) and register it.
        let initial = unsafe { RevisionEntry::resolve(lib, decls, lib_rs)? };
        initial.publish(decls);

        let runtime = Runtime {
            crate_dir,
            so_path,
            fn_sigs,
            denied_paths: config.denied_paths().clone(),
            revisions: RwLock::new(vec![Arc::new(initial)]),
            active: AtomicU64::new(Revision::INITIAL.as_u64()),
            decls,
            profile: config.profile(),
            prelude,
            evolve_failures: RwLock::new(Vec::new()),
            build_slot: tokio::sync::Mutex::new(()),
            // Unlimited until a caller asks for a limit, so a host that only
            // ever calls `evolve` is unaffected.
            inference_gate: InferenceGate::unlimited(),
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
    /// The run's token usage is added to `usage` either way, so a lane's
    /// total includes attempts that were later rejected.
    ///
    /// `history` is the whole transcript of the lane. Only
    /// `history[history_base..]` goes to the agent. A context or repeat reset
    /// advances `history_base` instead of truncating. The request therefore
    /// gets smaller, but the transcript keeps everything the lane exchanged.
    ///
    /// `run_out` and `stages` hold the trace records. This method writes them
    /// as the attempt progresses instead of returning them, so the caller
    /// keeps what it recorded before a failure. `run_out` stays `None` only
    /// when the agent run itself failed. That is how the caller separates an
    /// attempt that never got to the model from one the pipeline rejected.
    #[expect(
        clippy::too_many_arguments,
        reason = "One attempt's inputs, its two accumulators and its two trace out-params; bundling them would only move the same fields behind a single-use struct"
    )]
    async fn evolve_no_backpressure<AgentT>(
        &self,
        agent: &AgentT,
        prompt: &str,
        history: &mut Vec<Message>,
        history_base: usize,
        usage: &mut Usage,
        run_out: &mut Option<AgentRun>,
        stages: &mut StageTimings,
    ) -> Result<Revision>
    where
        AgentT: EvolutionAgent,
    {
        info!("prompt: {}", prompt.green());
        let t0 = Instant::now();
        let visible = history.get(history_base..).unwrap_or_default().to_vec();
        debug!("chat history: {visible:?}");

        // The agent implementation drives any tool-calling turns to
        // completion internally and returns only the final text.
        let run = match agent.run(prompt, visible).await {
            Ok(run) => run,
            Err(e) => {
                counter!(LLM_RUNS, "outcome" => "error").increment(1);
                stages.llm = Some(t0.elapsed());
                return Err(e.into());
            }
        };
        // Counted even if a later step rejects the result: a rejected
        // attempt is still a paid-for LLM run.
        *usage += run.usage;
        counter!(LLM_RUNS, "outcome" => "ok").increment(1);
        counter!(LLM_TOKENS, "kind" => "input").increment(run.usage.input_tokens);
        counter!(LLM_TOKENS, "kind" => "output").increment(run.usage.output_tokens);
        if run.usage.cached_input_tokens > 0 {
            counter!(LLM_TOKENS, "kind" => "cached_input").increment(run.usage.cached_input_tokens);
        }
        histogram!(LLM_RUN_INPUT_TOKENS).record(run.usage.input_tokens as f64);
        histogram!(LLM_RUN_OUTPUT_TOKENS).record(run.usage.output_tokens as f64);
        histogram!(LLM_RUN_MESSAGES).record(run.new_messages.len() as f64);
        info!("llm_response: {}", run.output.blue());
        info!("token usage for this run: {:?}", run.usage);

        // `new_messages` contains the prompt, assistant turns and any
        // tool exchanges of this run, so extending is sufficient.
        history.extend(run.new_messages.iter().cloned());
        let llm_response = run.output.clone();
        let llm_time = t0.elapsed().as_millis();
        stages.llm = Some(t0.elapsed());
        *run_out = Some(run);
        histogram!(
            PIPELINE_STAGE_DURATION,
            "stage" => stage::LLM
        )
        .record(t0.elapsed().as_secs_f64());

        // Parse Rust from markdown fences, validate signatures, and render
        // the candidate source. Scoped so the `syn` AST is dropped before the
        // compile `await` below: `syn` trees are `!Send`, and holding one
        // across an await would make this future `!Send`.
        let clean_ast_str = {
            let t1 = Instant::now();
            // Recorded before `?` propagates. A rejected candidate must still
            // report the time its parse and validation took.
            let mut ast = parse_rust_code(&llm_response).inspect_err(|_| {
                stages.parse_validate = Some(t1.elapsed());
            })?;

            // Validate signatures match declarations
            validate_generated_ast(&mut ast, &self.fn_sigs, &self.denied_paths).inspect_err(
                |_| {
                    stages.parse_validate = Some(t1.elapsed());
                },
            )?;
            stages.parse_validate = Some(t1.elapsed());
            histogram!(
                PIPELINE_STAGE_DURATION,
                "stage" => stage::PARSE_VALIDATE
            )
            .record(t1.elapsed().as_secs_f64());

            // Re-inject the prelude (inline helper items and configured imports)
            // so the dylib still sees the same API surface used at initialization.
            // The LLM is asked to emit only the function bodies, so we control
            // the prelude here rather than relying on the model to repeat it.
            if !self.prelude.is_empty() {
                let mut combined: Vec<syn::Item> = Vec::new();
                for part in &self.prelude {
                    if part.is_empty() {
                        continue;
                    }
                    let prelude_file: syn::File = syn::parse_str(part)
                        .expect("prelude was successfully parsed at init; should still be valid");
                    combined.extend(prelude_file.items);
                }
                combined.append(&mut ast.items);
                ast.items = combined;
            }
            unparse(&ast)
        };

        // Compile, load and retain the new revision. Whether it also becomes
        // the active one is up to the caller.
        let revision = self
            .build_and_register(clean_ast_str, &mut stages.build)
            .await?;

        info!("Built revision {revision}. LLM generation: {llm_time}ms.");

        Ok(revision)
    }

    /// Compile `clean_ast_str`, load the resulting dylib, and retain it in the
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
    async fn build_and_register(
        &self,
        clean_ast_str: String,
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
        if let Some(existing) = self.registered_with_source(&clean_ast_str)? {
            counter!(REVISION_DEDUP_HITS).increment(1);
            info!(
                "Candidate is byte-identical to revision {existing}; reusing it instead of spending a build."
            );
            *record = Some(BuildRecord::Deduped {
                slot_wait: waited,
                revision: existing,
            });
            return Ok(existing);
        }

        let source_bytes = clean_ast_str.len();
        debug!("clean_ast_str: {clean_ast_str}");

        let t_compile = Instant::now();
        // A compile failure is the common self-healing case, and its duration
        // is the most useful number of the whole attempt. Record it before the
        // error propagates.
        compile_dylib(&self.crate_dir, self.profile, &clean_ast_str)
            .await
            .inspect_err(|_| {
                *record = Some(BuildRecord::Built {
                    slot_wait: waited,
                    compile: t_compile.elapsed(),
                    load: Duration::ZERO,
                });
            })?;
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
        let entry = unsafe { RevisionEntry::resolve(new_lib, self.decls, clean_ast_str)? };
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
        });

        info!(
            "Registered revision {id}. Timings: build slot wait: {}ms, compilation: {}ms, load: {}ms.",
            waited.as_millis(),
            compile_time.as_millis(),
            t_load.elapsed().as_millis(),
        );

        Ok(Revision::new(id))
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
    /// Every failure that feeds backpressure to the agent is recorded and
    /// can be drained afterwards with [`Runtime::take_evolve_failures`],
    /// e.g. to persist the compiler diagnostics of failed attempts for
    /// offline analysis.
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

            self.evolve_failures
                .write()
                .map_err(|_| Error::MutexPoison)?
                .clear();

            self.evolve_lane(agent, base_prompt, 0, Publish::Yes).await
        }
    }

    /// Evolve one candidate implementation per prompt, concurrently.
    ///
    /// Each prompt gets its own lane: its own chat history, its own
    /// self-healing retry budget, and its own [`Revision`] on success. Lanes
    /// are independent, so eight slightly different prompts can converge on
    /// eight entirely different implementations — which is the point. The
    /// returned vector is positionally aligned with `prompts`, and a lane that
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
    /// # Why this is faster than a loop
    ///
    /// Against an OpenAI-compatible endpoint there is no batch API to call:
    /// batching *is* issuing the requests concurrently and letting the server's
    /// continuous batcher merge them into one forward pass. Two effects
    /// compound:
    ///
    /// - Decode is memory-bandwidth-bound. At batch 1 the model weights are
    ///   read once per token; at batch `n` they are read once for `n` tokens.
    /// - All lanes share the symbiont system preamble — which embeds the
    ///   rustdoc-derived API surface and is by far the largest part of the
    ///   request — so with prefix caching enabled every lane after the first
    ///   skips that prefill.
    ///
    /// Keep the varying part of each prompt at the *end*. Prefix reuse stops at
    /// the first differing token, so a per-lane preamble throws the second
    /// effect away.
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
    /// # The limit caps requests, not lanes
    ///
    /// Every lane starts immediately, however large the batch and however small
    /// [`Runtime::max_in_flight`]. A lane holds a slot only while it is talking
    /// to the model, and gives it back before it parses, queues for the build
    /// slot, compiles and loads — so lanes doing local work are covered by other
    /// lanes generating in their place. The limit belongs to the endpoint, not
    /// to one call: it lives on the runtime and is shared by everything the
    /// process sends.
    ///
    /// # Contract
    ///
    /// Because nothing is published, this method is **exempt** from the
    /// feedback-loop contract that [`Runtime::evolve`] imposes: a
    /// retained-but-inactive revision is invisible to running calls, so a batch
    /// may generate while evolvable functions from an earlier round are still
    /// executing. That is what makes it possible to overlap evaluation of round
    /// `n` with generation of round `n + 1`.
    ///
    /// # Failures
    ///
    /// The failure buffer is cleared once for the whole batch, then filled by
    /// all lanes in completion order. Group the drained records by
    /// [`EvolveFailure::lane`] to see what each prompt variant struggled with.
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
            match self.evolve_failures.write() {
                Ok(mut failures) => failures.clear(),
                // Every lane would fail on the same poisoned lock; report it
                // once per lane rather than pretending the batch ran.
                Err(_) => {
                    return prompts
                        .iter()
                        .map(|_| Err(EvolveError::from(Error::MutexPoison)))
                        .collect();
                }
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
            let mut slots: Vec<Option<std::result::Result<EvolveInfo, EvolveError>>> =
                Vec::from_iter(prompts.iter().map(|_| None));
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
    /// # Failures
    ///
    /// Unlike [`Runtime::evolve_batch`] this does **not** clear the
    /// failure buffer: a round that cleared it would discard the records of an
    /// overlapping round that is still running, which is the case this method
    /// exists for. Drain it yourself with
    /// [`Runtime::take_evolve_failures`] — grouping by [`EvolveFailure::lane`]
    /// is only unambiguous within one round, so drain between rounds if you
    /// need to attribute them.
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
            let evolve = self.evolve_lane(agent, prompt.as_ref(), lane, Publish::No);
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
    /// It does not clear the failure buffer — the caller owns that, since a
    /// batch clears once for the whole round rather than once per lane.
    ///
    /// `lane` only labels the [`EvolveFailure`] records this lane produces.
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
    fn evolve_lane<AgentT>(
        &self,
        agent: &AgentT,
        base_prompt: &str,
        lane: usize,
        publish: Publish,
    ) -> impl Future<Output = std::result::Result<EvolveInfo, EvolveError>> + Send
    where
        AgentT: EvolutionAgent + Sync,
    {
        async move {
            let t_start = Instant::now();
            let mut prompt = base_prompt.to_string();
            // Scoped to this call; see the doc comment above.
            let mut history: Vec<Message> = Vec::new();
            // A context or repeat reset retires everything before this index.
            // Those messages stay in the transcript of the trace, but leave
            // the request. A reset advances this index instead of truncating,
            // so the trace keeps what the lane exchanged.
            let mut history_base: usize = 0;
            let mut attempts: usize = 0;
            let mut context_resets: usize = 0;
            let mut usage = Usage::new();
            let mut transient_attempts: usize = 0;
            // Code of the most recent rejected attempt, used to detect an
            // agent that echoes the same broken code back verbatim.
            let mut last_failed_code: Option<String> = None;
            let mut trace = EvolutionTrace::new(lane, base_prompt.to_string());

            // Finish the lane. Move the transcript into the trace and set the
            // outcome. Every exit path calls this.
            macro_rules! finish {
                ($outcome:expr) => {{
                    trace.history = std::mem::take(&mut history);
                    trace.outcome = $outcome;
                    trace.duration = t_start.elapsed();
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
                        run_out.take().map(|run| RunTrace {
                            produced: produced_start..history.len(),
                            response: run.output,
                            usage: run.usage,
                            completion_calls: run.completion_calls,
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
                            &prompt,
                            &mut history,
                            history_base,
                            &mut usage,
                            &mut run_out,
                            &mut stages,
                        ),
                    )
                    .await
                {
                    Ok(revision) => {
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
                            LadderEvent::Registered { revision },
                            t_attempt.elapsed(),
                        );
                        return Ok(EvolveInfo::new(
                            revision,
                            usage,
                            finish!(TraceOutcome::Registered { revision }),
                        ));
                    }
                    Err(e) => {
                        counter!(
                            EVOLVE_FAILURES,
                            "kind" => failure_kind_of(&e)
                        )
                        .increment(1);
                        // Record every failure that will feed backpressure to
                        // the agent (including the one that exhausts the
                        // retry budget) so hosts can drain and persist them
                        // via `take_evolve_failures` for offline analysis.
                        // Along the way, detect a verbatim repeat of the
                        // previously rejected code.
                        let mut repeated = false;
                        if let Some(failure) =
                            EvolveFailure::from_error(&e, attempts).map(|f| f.with_lane(lane))
                        {
                            let code = failure.generated_code();
                            repeated = !code.is_empty()
                                && last_failed_code.as_deref() == Some(code.as_str());
                            last_failed_code = Some(code.clone());
                            match self.evolve_failures.write() {
                                Ok(mut failures) => failures.push(failure),
                                Err(_) => {
                                    let reason = MutexPoison.to_string();
                                    trace.push_attempt(
                                        attempts,
                                        attempt_prompt,
                                        run_trace!(),
                                        stages,
                                        LadderEvent::Terminal {
                                            reason: reason.clone(),
                                        },
                                        t_attempt.elapsed(),
                                    );
                                    return Err(EvolveError::new(
                                        MutexPoison,
                                        finish!(TraceOutcome::Failed { reason }),
                                    ));
                                }
                            }
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
                                    LadderEvent::Terminal {
                                        reason: reason.clone(),
                                    },
                                    t_attempt.elapsed(),
                                );
                                return Err(EvolveError::new(
                                    MaxRetriesExceeded {
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
                            let brief = first_line_of(&e);
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                LadderEvent::ContextReset {
                                    messages_dropped: dropped,
                                    brief,
                                },
                                t_attempt.elapsed(),
                            );
                            history_base = history.len();
                            prompt.clear();
                            prompt.push_str(base_prompt);
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
                            if transient_attempts >= Self::MAX_TRANSIENT_RETRIES {
                                warn!(
                                    "Transient HTTP error retry budget exhausted ({transient_attempts}/{}); giving up. Last error: {e}",
                                    Self::MAX_TRANSIENT_RETRIES
                                );
                                histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                                histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                                let reason = e.to_string();
                                trace.push_attempt(
                                    attempts,
                                    attempt_prompt,
                                    run_trace!(),
                                    stages,
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
                            counter!(LLM_TRANSIENT_RETRIES).increment(1);
                            histogram!(LLM_RETRY_BACKOFF).record(backoff.as_secs_f64());
                            warn!(
                                "Transient HTTP error from LLM provider (retry {transient_attempts}/{} in {:?}): {e}",
                                Self::MAX_TRANSIENT_RETRIES,
                                backoff,
                            );
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
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
                                LadderEvent::Terminal {
                                    reason: reason.clone(),
                                },
                                t_attempt.elapsed(),
                            );
                            return Err(EvolveError::new(
                                MaxRetriesExceeded {
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
                            // Only the first line of the error: the full
                            // diagnostics quote the rejected code, which is
                            // exactly the echo source being removed here.
                            let brief = first_line_of(&e);
                            write!(
                                prompt,
                                "{base_prompt}\n\nYour previous attempt was rejected: {brief}\n\
                                 You already answered with that exact code before and it was \
                                 rejected with the same error, so do NOT repeat it. Respond \
                                 with a different, valid implementation."
                            )
                            .expect("Can write to prompt");
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                LadderEvent::RepeatReset {
                                    messages_dropped: dropped,
                                    brief,
                                },
                                t_attempt.elapsed(),
                            );
                            continue;
                        }

                        // The nudge that the ladder builds below is the same
                        // text as the diagnostics that go to the agent. Record
                        // the ladder event after the match writes that nudge.
                        let kind = failure_kind_of(&e).to_string();

                        use Error::*;
                        match e {
                        NoRustCode => prompt.push_str(
                            "Your response did not contain a rust code block. Please try again and make sure its wrapped like this: ```CODE```",
                        ),
                        CouldNotParseRust { code, err } => write!(prompt,
                            "Your generated code ```{code}``` is not valid Rust. Parse error: ```{err}```. Fix the syntax error and respond with the full corrected code.",
                        ).expect("Can write to prompt"),
                        RigPrompt(rig_agent::completion::PromptError::MaxTurnsError { .. }) => prompt.push_str(
                            "You exhausted the tool-call turn budget before producing code. Respond with the final Rust code block now.",
                        ),
                        WriteLib(_) => todo!(),
                        SignatureMismatch {
                            code,
                            expected,
                            got,
                        } => write!(prompt,
                            "Signature mismatch in {got}. Expected `{expected}`. Fix ONLY this function's signature (argument types and return type must match exactly; argument names may differ). Full code: ```{code}```",
                        ).expect("Can write to prompt"),
                        UnsafeCode { code, construct } => write!(prompt,
                            "Your generated code contains {construct}, but unsafe code is forbidden in evolvable code. \
                            Rewrite it in safe Rust only: no `unsafe` blocks, `unsafe fn`, `unsafe impl`, `unsafe trait`, \
                            `extern` blocks, unsafe attributes, or `unsafe` tokens inside macros. \
                            Keep the logic and the function signatures unchanged. Full code: ```{code}```",
                        ).expect("Can write to prompt"),
                        ForbiddenConstruct { code, construct, reason } => write!(prompt,
                            "Your generated code contains {construct}, which is forbidden in evolvable code: {reason}. \
                            Rewrite the code without it, keeping the logic and the function signatures unchanged. Full code: ```{code}```",
                        ).expect("Can write to prompt"),
                        CompilationFailed{code, err} => write!(prompt,
                            "Your generated code ```{code}``` failed to compile. Compiler output:\n```\n{err}\n```\n\
                            Fix the compilation errors while preserving the existing logic and behaviour. \
                            Change only the expressions the compiler diagnostics point at (match the `src/lib.rs:<line>:<col>` markers); \
                            do not rewrite, restructure, rename, reformat or otherwise alter the rest of the code. \
                            A trait error (E0277) means you used an operator or conversion the type does not implement: \
                            consult the documented `impl ... for ...` blocks for that type and use only listed impls, \
                            adjusting the operand types instead of forcing an unsupported operation.",
                        ).expect("Can write to prompt"),
                        e => {
                            warn!("Unhandled error: {e}");
                            let reason = e.to_string();
                            trace.push_attempt(
                                attempts,
                                attempt_prompt,
                                run_trace!(),
                                stages,
                                LadderEvent::Terminal { reason: reason.clone() },
                                t_attempt.elapsed(),
                            );
                            return Err(EvolveError::new(e, finish!(TraceOutcome::Failed { reason })))
                        },
                    }

                        trace.push_attempt(
                            attempts,
                            attempt_prompt,
                            run_trace!(),
                            stages,
                            LadderEvent::SelfHeal {
                                kind,
                                diagnostics: prompt.clone(),
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

    /// Drain the failed attempts recorded during the most recent
    /// [`Runtime::evolve`] or [`Runtime::evolve_batch`] call.
    ///
    /// Each entry is one failure that fed backpressure to the agent inside
    /// the self-healing loop: missing code blocks, parse errors, exhausted
    /// tool-call turn budgets, signature mismatches, and compilation
    /// failures (with the full rustc diagnostics). Transient provider errors
    /// and context-window resets are not recorded.
    ///
    /// The buffer is cleared at the start of every `evolve` call and by this
    /// method, so drain it right after `evolve` returns — including on
    /// `Err`, where the recorded failures explain what exhausted the retry
    /// budget. Persist them (e.g. to a database) to analyze common failure
    /// patterns of the generation agent offline.
    ///
    /// A batch clears the buffer once for the whole round, not once per lane,
    /// then fills it from all lanes in completion order. Group by
    /// [`EvolveFailure::lane`] to attribute records back to their prompt.
    pub fn take_evolve_failures(&self) -> Vec<EvolveFailure> {
        self.evolve_failures
            .write()
            .map(|mut failures| std::mem::take(&mut *failures))
            .unwrap_or_default()
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

    /// Get the current, clean LLM-generated code (without panic-catching wrappers or preamble).
    /// Suitable for feeding back into the LLM prompt or displaying to the user.
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

    /// The clean generated source of `revision` (without panic-catching
    /// wrappers or preamble), or `None` if no such revision was registered.
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

/// The first line of the display form of an error.
///
/// A reset nudge quotes only this line. The full diagnostics repeat the
/// rejected code, which is the same text that the reset removes.
fn first_line_of(e: &Error) -> String {
    let text = e.to_string();
    text.lines().next().unwrap_or_default().to_string()
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
