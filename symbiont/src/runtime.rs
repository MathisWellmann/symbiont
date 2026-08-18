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
    DylibConfig,
    EvolutionAgent,
    EvolvableDecl,
    EvolveFailure,
    EvolveInfo,
    FullSource,
    Profile,
    compiler::compile_dylib,
    error::{
        Error,
        Result,
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
}

impl Runtime {
    /// Maximum number of attempts [`Runtime::evolve`] will make before giving
    /// up and returning [`Error::MaxRetriesExceeded`]. Prevents a misbehaving
    /// agent from hanging the runtime indefinitely.
    pub const MAX_EVOLVE_ATTEMPTS: usize = 10;

    /// Maximum number of retries for transient HTTP errors (429, 5xx, 529).
    ///
    /// These are retried with exponential backoff and do not count against
    /// [`Self::MAX_EVOLVE_ATTEMPTS`].
    pub const MAX_TRANSIENT_RETRIES: usize = 6;

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
    /// - `config` defines the compilation profile, dylib dependencies, and configured imports.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    pub async fn new(
        decls: &'static [EvolvableDecl],
        generated_prelude: &'static [&'static str],
        config: impl Into<DylibConfig>,
    ) -> Result<&'static Runtime> {
        let config = config.into();
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
    async fn evolve_no_backpressure<AgentT>(
        &self,
        agent: &AgentT,
        prompt: &str,
        history: &mut Vec<Message>,
        usage: &mut Usage,
    ) -> Result<Revision>
    where
        AgentT: EvolutionAgent,
    {
        info!("prompt: {}", prompt.green());
        let t0 = Instant::now();
        debug!("chat history: {history:?}");

        // The agent implementation drives any tool-calling turns to
        // completion internally and returns only the final text.
        let run = match agent.run(prompt, history.clone()).await {
            Ok(run) => run,
            Err(e) => {
                counter!(LLM_RUNS, "outcome" => "error").increment(1);
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
        history.extend(run.new_messages);
        let llm_response = run.output;
        let llm_time = t0.elapsed().as_millis();
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
            let mut ast = parse_rust_code(&llm_response)?;

            // Validate signatures match declarations
            validate_generated_ast(&mut ast, &self.fn_sigs, &self.denied_paths)?;
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
        let revision = self.build_and_register(clean_ast_str).await?;

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
    async fn build_and_register(&self, clean_ast_str: String) -> Result<Revision> {
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
            return Ok(existing);
        }

        let source_bytes = clean_ast_str.len();
        debug!("clean_ast_str: {clean_ast_str}");

        let t_compile = Instant::now();
        compile_dylib(&self.crate_dir, self.profile, &clean_ast_str).await?;
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
    /// "overloaded") are retried separately with exponential backoff up to
    /// [`Self::MAX_TRANSIENT_RETRIES`] times, and do not count against the
    /// self-healing attempt budget.
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
    ) -> impl Future<Output = Result<EvolveInfo>> + Send
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
    /// Every lane defaults to running concurrently; use
    /// [`Runtime::evolve_batch_limited`] to cap how many are in flight at once.
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
    /// Confirming the second effect is provider-dependent.
    /// [`crate::observability::LLM_TOKENS`]`{kind="cached_input"}` reports it
    /// only for backends that fill in the OpenAI-compatible
    /// `usage.prompt_tokens_details.cached_tokens` field. vLLM does not, even
    /// when its prefix cache is demonstrably working — there the figure lives
    /// on its own `/metrics` endpoint as `vllm:prefix_cache_hits_total`.
    ///
    /// Keep the varying part of each prompt at the *end*. Prefix reuse stops at
    /// the first differing token, so a per-lane preamble throws the second
    /// effect away.
    ///
    /// Note that a server must be configured to batch: vLLM and SGLang do it by
    /// default, but `llama-server` runs one sequence at a time unless started
    /// with `--parallel n`.
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
    pub fn evolve_batch<'a, AgentT, S>(
        &'a self,
        agent: &'a AgentT,
        prompts: &'a [S],
    ) -> impl Future<Output = Vec<Result<EvolveInfo>>> + Send + 'a
    where
        AgentT: EvolutionAgent + Sync,
        S: AsRef<str> + Sync,
    {
        self.evolve_batch_limited(agent, prompts, prompts.len())
    }

    /// [`Runtime::evolve_batch`] with a ceiling on how many lanes are in flight
    /// at once. The remaining lanes start as slots free up; results stay
    /// positionally aligned with `prompts` either way.
    ///
    /// Cap this below `prompts.len()` when the endpoint is rate limited — eight
    /// concurrent requests into a per-minute quota produce eight independent
    /// 429 backoffs — or when the server's own batch width is smaller than the
    /// batch, in which case the excess only queues server-side where you cannot
    /// see it. For a local server, matching its `--max-num-seqs` is a good
    /// default.
    ///
    /// `max_in_flight` is clamped to `1..=prompts.len()`.
    #[expect(
        clippy::manual_async_fn,
        reason = "Ensure the future is `Send` such that it works better with tokios multi-thread runtime"
    )]
    pub fn evolve_batch_limited<'a, AgentT, S>(
        &'a self,
        agent: &'a AgentT,
        prompts: &'a [S],
        max_in_flight: usize,
    ) -> impl Future<Output = Vec<Result<EvolveInfo>>> + Send + 'a
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
                Err(_) => return prompts.iter().map(|_| Err(Error::MutexPoison)).collect(),
            }

            let in_flight = max_in_flight.clamp(1, prompts.len());
            info!(
                "Evolving a batch of {} prompts, at most {in_flight} in flight.",
                prompts.len()
            );
            let t_batch = Instant::now();
            histogram!(EVOLVE_BATCH_SIZE).record(prompts.len() as f64);

            // Constructing a lane future does no work, so building them all up
            // front is free — and it pins the lifetimes, which a lazy
            // `.map()` closure returning `impl Future` cannot do.
            let lanes =
                Vec::from_iter(prompts.iter().enumerate().map(|(lane, prompt)| {
                    self.evolve_lane(agent, prompt.as_ref(), lane, Publish::No)
                }));

            // `buffered` polls up to `in_flight` lanes concurrently and yields
            // results in input order, so the ordering guarantee costs nothing.
            let results: Vec<Result<EvolveInfo>> = stream::iter(lanes)
                .buffered(in_flight)
                .collect::<Vec<_>>()
                .await;

            for result in &results {
                let outcome = if result.is_ok() { "ok" } else { "error" };
                counter!(EVOLVE_BATCH_LANES, "outcome" => outcome).increment(1);
            }
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
    ) -> impl Future<Output = Result<EvolveInfo>> + Send
    where
        AgentT: EvolutionAgent + Sync,
    {
        async move {
            let t_start = Instant::now();
            let mut prompt = base_prompt.to_string();
            // Scoped to this call; see the doc comment above.
            let mut history: Vec<Message> = Vec::new();
            let mut attempts: usize = 0;
            let mut usage = Usage::new();
            let mut transient_attempts: usize = 0;
            // Code of the most recent rejected attempt, used to detect an
            // agent that echoes the same broken code back verbatim.
            let mut last_failed_code: Option<String> = None;

            loop {
                attempts += 1;
                match self
                    .evolve_no_backpressure(agent, &prompt, &mut history, &mut usage)
                    .await
                {
                    Ok(revision) => {
                        if publish == Publish::Yes {
                            self.publish_revision(revision, "evolve")?;
                            info!("Hot-reloaded evolvable dylib (revision {revision}).");
                        }
                        histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                        histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                        return Ok(EvolveInfo::new(revision, usage));
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
                            self.evolve_failures
                                .write()
                                .map_err(|_| MutexPoison)?
                                .push(failure);
                        }
                        // A request that exceeds the model's context window can
                        // never succeed by resending: shrink it instead.
                        // Discard the accumulated retry history and restart
                        // from the base prompt. If even a fresh request
                        // overflows (empty history), the base prompt itself is
                        // too large and only the caller can slim it down.
                        if is_context_size_error(&e) {
                            if history.is_empty() {
                                warn!(
                                    "Request exceeds the model's context window even without \
                                     chat history; the base prompt is too large: {e}"
                                );
                                histogram!(EVOLVE_ATTEMPTS).record(attempts as f64);
                                histogram!(EVOLVE_DURATION).record(t_start.elapsed().as_secs_f64());
                                return Err(e);
                            }
                            warn!(
                                "Request exceeded the model's context window; discarding {} \
                                 history messages and restarting from the base prompt",
                                history.len()
                            );
                            counter!(EVOLVE_CONTEXT_RESETS).increment(1);
                            history.clear();
                            prompt.clear();
                            prompt.push_str(base_prompt);
                            // Not the LLM's fault: don't count against the
                            // self-healing budget.
                            attempts -= 1;
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
                                return Err(e);
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
                            return Err(MaxRetriesExceeded {
                                attempts,
                                last_error: Box::new(e),
                            });
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
                            warn!(
                                "Agent repeated the same rejected code verbatim; discarding {} \
                                 history messages and restarting from the base prompt",
                                history.len()
                            );
                            history.clear();
                            // Only the first line of the error: the full
                            // diagnostics quote the rejected code, which is
                            // exactly the echo source being removed here.
                            let brief = e.to_string();
                            let brief = brief.lines().next().unwrap_or_default().to_string();
                            write!(
                                prompt,
                                "{base_prompt}\n\nYour previous attempt was rejected: {brief}\n\
                                 You already answered with that exact code before and it was \
                                 rejected with the same error, so do NOT repeat it. Respond \
                                 with a different, valid implementation."
                            )
                            .expect("Can write to prompt");
                            continue;
                        }

                        use Error::*;
                        match e {
                        NoRustCode => prompt.push_str(
                            "Your response did not contain a rust code block. Please try again and make sure its wrapped like this: ```CODE```",
                        ),
                        CouldNotParseRust { code, err } => write!(prompt,
                            "Your generated code ```{code}``` is not valid Rust. Parse error: ```{err}```. Fix the syntax error and respond with the full corrected code.",
                        ).expect("Can write to prompt"),
                        RigPrompt(rig_core::completion::PromptError::MaxTurnsError { .. }) => prompt.push_str(
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
                            return Err(e)
                        },
                    }
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
