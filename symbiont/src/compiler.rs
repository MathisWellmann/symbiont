// SPDX-License-Identifier: MPL-2.0
use std::path::Path;
#[cfg(miri)]
use std::time::Instant;

#[cfg(not(miri))]
use minstant::Instant;
use prettyplease::unparse;
use tokio::process::Command;
use tracing::info;

use crate::{
    error::{
        Error,
        Result,
    },
    profile::Profile,
    unwind::{
        PANIC_PREAMBLE,
        wrap_bodies_in_catch_unwind,
    },
};

/// Compile a dylib crate at the given directory.
///
/// Runs `cargo build --manifest-path <crate_dir>/Cargo.toml`,
/// adding `--release` when the profile is [`Profile::Release`].
///
/// Cargo is spawned through [`tokio::process`] and awaited, so the multi-second
/// build occupies no runtime worker: the calling task yields and the workers
/// stay free to drive other tasks. That matters for
/// [`crate::Runtime::evolve_batch`], where sibling lanes are streaming tokens
/// from the inference server while this one builds — a blocking
/// [`std::process::Command`] here would park a worker, and with enough lanes
/// starve the I/O driver that those inference responses depend on.
///
/// The generated crate allows all warnings: the code is machine-generated
/// and its only reader is the compiler-feedback loop on failed builds, where
/// warnings would drown out the errors the evolution agent has to fix.
pub(crate) async fn compile_dylib(
    crate_dir: &Path,
    profile: Profile,
    clean_ast_str: &str,
) -> Result<()> {
    let t0 = Instant::now();

    // Scoped so the `syn` AST is dropped before the `await` below: `syn`
    // trees are `!Send` (a `proc_macro2` token may wrap a `proc_macro` one),
    // and holding one across the await would make every caller's future
    // `!Send` too.
    let formatted = {
        let mut clean_ast: syn::File = syn::parse_str(clean_ast_str)?;
        // Wrap function bodies in catch_unwind so panics stay inside the dylib.
        wrap_bodies_in_catch_unwind(&mut clean_ast);

        // Final lib.rs: warning suppression + preamble + wrapped code.
        format!(
            "#![allow(warnings)]\n{PANIC_PREAMBLE}\n{}",
            unparse(&clean_ast)
        )
    };
    std::fs::write(crate_dir.join("src").join("lib.rs"), formatted).map_err(|e| {
        Error::DylibLoad(format!(
            "Failed to write {}: {e}",
            crate_dir.join("src").join("lib.rs").display()
        ))
    })?;
    info!("Created temp dylib crate at {}", crate_dir.display());

    let manifest_path = crate_dir.join("Cargo.toml");
    info!(
        "Compiling evolvable dylib ({profile}) at {}...",
        manifest_path.display()
    );
    let manifest_str = manifest_path.to_string_lossy();
    // `cargo rustc` instead of `cargo build`: the extra flags after `--` apply
    // to the dylib itself, not to its dependencies.
    let mut args = vec!["rustc", "--manifest-path", &manifest_str];
    if profile == Profile::Release {
        args.push("--release");
    }
    if cfg!(target_os = "linux") {
        // Bind the dylib's calls to its own exported functions at link time.
        //
        // An exported (`#[unsafe(no_mangle)]`) symbol in an ELF dylib has
        // default visibility and is therefore preemptible: the compiler emits
        // the dylib's *own* calls to it through the GOT, and the loader
        // resolves those against the global scope — the host executable and
        // everything loaded with it, libc included — before the dylib itself.
        // A generated function whose name collides with a libc symbol (e.g.
        // `qsort`) then hijacks the call with mismatched arguments and
        // segfaults the host. `-Bsymbolic-functions` pre-binds intra-dylib
        // calls to the local definitions and removes that whole failure mode.
        // Only Rust symbols undefined in the artifact (libc, a shared std)
        // still resolve dynamically.
        args.extend_from_slice(&["--", "-Clink-arg=-Wl,-Bsymbolic-functions"]);
    }

    let output = Command::new("cargo")
        .args(&args)
        // This nested build has its own artifact lookup rooted at `crate_dir`.
        // Inherited `CARGO_TARGET_DIR` (commonly set by CI) would redirect
        // cargo elsewhere and make `find_so` report a missing dylib.
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .await
        .map_err(|e| Error::CompilationFailed {
            code: clean_ast_str.to_string(),
            err: format!("Failed to spawn cargo: {e}"),
        })?;

    if output.status.success() {
        info!(
            "Evolvable dylib compiled successfully in {}ms",
            t0.elapsed().as_millis()
        );
        Ok(())
    } else {
        let err = String::from_utf8_lossy(&output.stderr).to_string();
        Err(Error::CompilationFailed {
            code: clean_ast_str.to_string(),
            err,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_display() {
        assert_eq!(&Profile::Release.to_string(), "release");
        assert_eq!(&Profile::Debug.to_string(), "debug");
    }
}
