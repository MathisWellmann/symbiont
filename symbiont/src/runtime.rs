// SPDX-License-Identifier: MPL-2.0
//! The runtime module contains the primary `Runtime`,
//! managing the lifecycle of the temporary dylib crate: creation, compilation,
//! loading, and hot-reloading.

use std::{
    collections::hash_map::DefaultHasher,
    ffi::CString,
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
        Mutex,
        OnceLock,
        atomic::{
            AtomicPtr,
            AtomicU64,
            Ordering,
        },
    },
};

use libloading::{
    Library,
    Symbol,
};
use minstant::Instant;
use owo_colors::OwoColorize;
use prettyplease::unparse;
use rig::completion::Prompt;
use tracing::{
    debug,
    info,
    warn,
};

use crate::{
    EvolvableDecl,
    FullSource,
    compiler::{
        Profile,
        compile_dylib,
    },
    error::{
        Error,
        Result,
    },
    parser::parse_rust_code,
    utils::{
        dylib_extension,
        find_so,
        generate_cargo_toml,
        generate_lib_rs,
    },
    validation::validate_generated_ast,
};

/// Singleton runtime instance.
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Cached pointer to the dylib's `__symbiont_take_panic` function.
/// Updated on each reload alongside the evolvable function pointers.
static TAKE_PANIC_PTR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

// ---------- debug-mode call counter ----------
//
// Tracks in-flight evolvable function calls so that `evolve()` can assert
// none are running. Compiled away entirely in release builds — zero overhead.

#[cfg(debug_assertions)]
static IN_FLIGHT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// RAII guard that decrements the in-flight call counter on drop.
/// Only exists in debug builds.
#[cfg(debug_assertions)]
pub struct CallGuard;

#[cfg(debug_assertions)]
impl Drop for CallGuard {
    fn drop(&mut self) {
        IN_FLIGHT_CALLS.fetch_sub(1, Ordering::Release);
    }
}

/// Increment the in-flight call counter and return a guard that decrements
/// it on drop (including during unwind). Debug builds only.
#[cfg(debug_assertions)]
pub fn enter_call() -> CallGuard {
    IN_FLIGHT_CALLS.fetch_add(1, Ordering::Acquire);
    CallGuard
}

// -------------------------------------------------

/// Manages the lifecycle of the temporary dylib crate: creation, compilation,
/// loading, and hot-reloading.
///
/// Function dispatch is lock-free: each evolvable function reads its cached
/// pointer via a single `AtomicPtr::load`.
///
/// # Contract
///
/// **All evolvable function calls must have returned before [`Runtime::evolve`]
/// is called.** This is the natural shape of the feedback loop — run functions,
/// collect results, evolve, repeat. The contract is enforced with an assertion
/// in debug builds and is zero-cost in release.
pub struct Runtime {
    /// Path to the temporary dylib crate directory.
    crate_dir: PathBuf,
    /// Path to the compiled `.so` / `.dylib` / `.dll` file.
    so_path: PathBuf,
    /// Monotonically increasing version counter for versioned `.so` paths
    /// to defeat `dlopen` caching.
    version: AtomicU64,
    /// Function signatures for validation of LLM-generated code.
    fn_sigs: Vec<String>,
    /// The currently loaded library.
    /// Safe to replace because the caller guarantees no in-flight calls
    /// during evolution. The Mutex is only taken during reload, never on
    /// the hot path.
    library: Mutex<Option<Library>>,
    /// Declarations (kept for fn_ptr updates on reload).
    decls: &'static [EvolvableDecl],
    /// Compilation profile (`debug` or `release`).
    profile: Profile,
    /// The currently active AST of the agent code, in String form, to make it `Send`
    current_clean_ast: Mutex<String>,
}

/// Look up all declared symbols in `lib` and store their addresses
/// in the corresponding `AtomicPtr` fields of each declaration.
///
/// # Safety
///
/// The caller must ensure `lib` is a valid loaded library containing
/// symbols with signatures matching the declarations.
unsafe fn update_fn_ptrs(lib: &Library, decls: &[EvolvableDecl]) -> Result<()> {
    for decl in decls {
        let c_name =
            CString::new(decl.name).expect("function name must not contain interior NUL bytes");
        let sym: Symbol<*const ()> = unsafe {
            lib.get(c_name.as_bytes_with_nul())
                .map_err(|e| Error::DylibLoad(format!("symbol '{}' not found: {e}", decl.name)))?
        };
        decl.fn_ptr.store(*sym as *mut (), Ordering::Release);
    }

    // Resolve the panic-retrieval symbol.
    let panic_sym: Symbol<*const ()> = unsafe {
        lib.get(b"__symbiont_take_panic\0").map_err(|e| {
            Error::DylibLoad(format!("symbol '__symbiont_take_panic' not found: {e}"))
        })?
    };
    TAKE_PANIC_PTR.store(*panic_sym as *mut (), Ordering::Release);

    Ok(())
}

impl Runtime {
    /// Maximum number of attempts [`Runtime::evolve`] will make before giving
    /// up and returning [`Error::MaxRetriesExceeded`]. Prevents a misbehaving
    /// agent from hanging the runtime indefinitely.
    pub const MAX_EVOLVE_ATTEMPTS: usize = 10;

    /// Initialize the symbiont runtime.
    ///
    /// Creates a temporary dylib crate from the declarations generated by `evolvable!`,
    /// compiles it, and loads the resulting shared library.
    ///
    /// Use [`Profile::Release`] when benchmarking evolved functions — the
    /// optimizer can make orders-of-magnitude difference for compute-heavy code.
    /// [`Profile::Debug`] compiles faster and is fine for correctness-only workloads.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    pub async fn init(
        decls: &'static [EvolvableDecl],
        profile: Profile,
    ) -> Result<&'static Runtime> {
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
        let cargo_toml = generate_cargo_toml();
        std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml)?;

        // Write src/lib.rs from all default_source entries
        let lib_rs = generate_lib_rs(decls);
        let mut ast = syn::parse_str(&lib_rs)?;

        // Compile
        compile_dylib(&crate_dir, profile, &mut ast, &lib_rs).await?;

        // Find and load the .so
        let so_path = find_so(&crate_dir, profile)?;
        let lib = unsafe {
            Library::new(&so_path).map_err(|e| {
                Error::DylibLoad(format!("Failed to load {}: {e}", so_path.display()))
            })?
        };

        // Cache function pointers (lock-free after this point)
        unsafe { update_fn_ptrs(&lib, decls)? };

        let runtime = Runtime {
            crate_dir,
            so_path,
            version: AtomicU64::new(1),
            fn_sigs,
            library: Mutex::new(Some(lib)),
            decls,
            profile,
            current_clean_ast: Mutex::new(lib_rs),
        };

        RUNTIME
            .set(runtime)
            .map_err(|_| Error::AlreadyInitialized)?;
        Ok(RUNTIME.get().expect("just set"))
    }

    /// Generate LLM response, then parse, validate, compile, and hot-swap.
    /// It does not catch validation errors and feed it back to the LLM, allowing the user to customize prompting behaviour.
    ///
    /// # Contract
    ///
    /// All evolvable function calls must have returned before this is called.
    /// In debug builds this is enforced with an assertion; in release it is
    /// the caller's responsibility.
    async fn evolve_no_backpressure<AgentT>(&self, agent: &AgentT, prompt: &str) -> Result<()>
    where
        AgentT: Prompt,
    {
        #[cfg(debug_assertions)]
        {
            let in_flight = IN_FLIGHT_CALLS.load(Ordering::Acquire);
            assert!(
                in_flight == 0,
                "evolve() called while {in_flight} evolvable function(s) are still executing. \
                 All callers must return before evolving — this is the feedback loop contract."
            );
        }

        info!("prompt: {}", prompt.green());
        let t0 = Instant::now();
        let llm_response = agent.prompt(prompt).await?;
        let llm_time = t0.elapsed().as_millis();
        info!("llm_response: {}", llm_response.blue());

        // Parse Rust from markdown fences
        let mut ast = parse_rust_code(&llm_response).map_err(|_| Error::CouldNotParseRust)?;

        // Validate signatures match declarations
        validate_generated_ast(&mut ast, &self.fn_sigs)?;

        // Recompile
        let t0 = Instant::now();
        let clean_ast_str = unparse(&ast);
        debug!("clean_ast_str: {clean_ast_str}");
        compile_dylib(&self.crate_dir, self.profile, &mut ast, &clean_ast_str).await?;
        {
            *self
                .current_clean_ast
                .lock()
                .expect("Can lock the clean ast mutex") = clean_ast_str;
        }
        let compile_time = t0.elapsed().as_millis();

        // Copy .so to versioned path to defeat dlopen caching
        let version = self.version.fetch_add(1, Ordering::SeqCst);
        let versioned_so = self.crate_dir.join(format!(
            "libsymbiont_evolvable_v{version}{}",
            dylib_extension()
        ));
        std::fs::copy(&self.so_path, &versioned_so)?;

        // Load new library
        let new_lib = unsafe {
            Library::new(&versioned_so).map_err(|e| {
                Error::DylibLoad(format!("Failed to load {}: {e}", versioned_so.display()))
            })?
        };

        // Update cached function pointers (atomic stores with Release ordering).
        unsafe { update_fn_ptrs(&new_lib, self.decls)? };

        // Replace the library. Safe to drop the old one because the caller
        // guarantees no evolvable functions are executing (feedback loop contract).
        *self.library.lock().expect("library Mutex poisoned") = Some(new_lib);

        info!(
            "Hot-reloaded evolvable dylib (version {version}). Timings: LLM generation: {llm_time}ms, compilation: {compile_time}ms.",
        );

        Ok(())
    }

    /// Prompt the LLM, validate the response, compile, and hot-swap.
    ///
    /// If the constrained generation fails (parse error, signature mismatch,
    /// compilation failure), the error is appended to the prompt and the LLM
    /// retries until it produces valid code, up to [`Self::MAX_EVOLVE_ATTEMPTS`]
    /// attempts. After that, [`Error::MaxRetriesExceeded`] is returned so a
    /// misbehaving agent cannot hang the runtime indefinitely.
    ///
    /// # Contract
    ///
    /// All evolvable function calls must have returned before this is called.
    /// This is the natural shape of the feedback loop: run functions, collect
    /// results, evolve, repeat.
    pub async fn evolve<AgentT>(&self, agent: &AgentT, base_prompt: &str) -> Result<()>
    where
        AgentT: Prompt,
    {
        let mut prompt = base_prompt.to_string();
        let mut attempts: usize = 0;

        loop {
            attempts += 1;
            match self.evolve_no_backpressure(agent, &prompt).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempts >= Self::MAX_EVOLVE_ATTEMPTS {
                        warn!(
                            "Evolution failed after {attempts} attempts; giving up. Last error: {e}"
                        );
                        return Err(MaxRetriesExceeded {
                            attempts,
                            last_error: Box::new(e),
                        });
                    }

                    info!(
                        "Function evolution error (attempt {attempts}/{}): {e}.\nSelf-healing from error...",
                        Self::MAX_EVOLVE_ATTEMPTS
                    );

                    prompt = base_prompt.to_string();

                    use Error::*;
                    match e {
                        NoRustCode => prompt.push_str(
                            "Your response did not contain a rust code block. Please try again and make sure its wrapped like this: ```CODE```",
                        ),
                        CouldNotParseRust => prompt.push_str(
                            "Your response did not contain valid Rust code. Please try again",
                        ),
                        WriteLib(_) => todo!(),
                        SignatureMismatch {
                            code,
                            expected,
                        } => write!(prompt,
                            " Generated function signature miss-match. Expected ```{expected}```, Got Code ```{code}```",
                        ).expect("Can write to prompt"),
                        CompilationFailed{code, err} => write!(prompt,
                            " Your generated code ```{}``` failed to compile. Compiler output:\n```\n{}\n```\nPlease fix the compilation errors.", code.blue(), err.red()
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

    /// Retrieve and clear the last panic message from the loaded dylib.
    ///
    /// Returns `Some(message)` if the most recent evolvable function call
    /// panicked, `None` otherwise. The stored message is cleared on read.
    ///
    /// Call this after each evolvable function invocation to detect panics
    /// that were caught inside the dylib.
    pub fn take_panic(&self) -> Option<String> {
        let ptr = TAKE_PANIC_PTR.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        // Signature: unsafe fn __symbiont_take_panic(buf: *mut u8, buf_len: usize) -> usize
        let take_panic: unsafe fn(*mut u8, usize) -> usize = unsafe { std::mem::transmute(ptr) };
        let mut buf = [0u8; 512];
        let len = unsafe { take_panic(buf.as_mut_ptr(), buf.len()) };
        if len == 0 {
            None
        } else {
            Some(String::from_utf8_lossy(&buf[..len]).into_owned())
        }
    }

    /// Path to the temporary crate directory.
    pub fn crate_dir(&self) -> &Path {
        &self.crate_dir
    }

    /// Get the function signature strings for all evolvable functions.
    pub fn fn_sigs(&self) -> &[String] {
        &self.fn_sigs
    }

    /// Get the full function signatures, including doc comments and default function body.
    ///
    /// Returns each source wrapped in [`FullSource`], which preserves real line
    /// breaks when pretty-printed (`{:#?}`) so logs stay readable.
    pub fn fn_full_sources(&self) -> Vec<FullSource<'static>> {
        Vec::from_iter(self.decls.iter().map(|d| FullSource(d.full_source)))
    }

    /// Get the current, ,clean LLM-generated code (without panic-catching wrappers or preamble).
    /// Suitable for feeding back into the LLM prompt or displaying to the user.
    pub fn current_code(&self) -> String {
        self.current_clean_ast
            .lock()
            .expect("Can lock the mutex to get clean AST")
            .clone()
    }
}
