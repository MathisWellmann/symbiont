// SPDX-License-Identifier: MPL-2.0
use std::path::Path;
#[cfg(miri)]
use std::time::Instant;

#[cfg(not(miri))]
use minstant::Instant;
use tokio::process::Command;
use tracing::info;

use crate::{
    diagnostics::{
        Diagnostic,
        parse_cargo_json,
        render_for_prompt,
    },
    error::{
        Error,
        Result,
    },
    profile::Profile,
};

/// Compile the dylib crate at `crate_dir` from `lib_rs`.
///
/// `lib_rs` is the complete `src/lib.rs`, assembled by
/// [`crate::layout::assemble_lib_rs`]: the candidate first, the harness glue
/// after it. `candidate` is the agent's part of it, quoted in the error.
///
/// Runs `cargo rustc --manifest-path <crate_dir>/Cargo.toml`, adding
/// `--release` when the profile is [`Profile::Release`].
///
/// Cargo is spawned through [`tokio::process`] and awaited, so the multi-second
/// build occupies no runtime worker: the calling task yields and the workers
/// stay free to drive other tasks. That matters for
/// [`crate::Runtime::evolve_batch`], where sibling lanes are streaming tokens
/// from the inference server while this one builds — a blocking
/// [`std::process::Command`] here would park a worker, and with enough lanes
/// starve the I/O driver that those inference responses depend on.
///
/// Lints are capped at `allow` for the generated crate: the code is
/// machine-generated and its only reader is the compiler-feedback loop on
/// failed builds, where warnings would drown out the errors the evolution
/// agent has to fix. `--cap-lints` is passed to rustc directly so it also
/// overrides a `RUSTFLAGS="-D warnings"` inherited from the environment
/// (CI sets exactly that), which a crate-level `#![allow(warnings)]` would
/// not: `-D` on the command line outranks an attribute in the source.
///
/// Cargo emits `--message-format=json`, so a failed build returns
/// [`Error::CompilationFailed`] with one [`Diagnostic`] per error, located
/// in `candidate`, next to the text rendered for the model. Should the JSON
/// hold no error (cargo itself failed, say on a broken manifest), the error
/// text is cargo's stderr instead.
pub(crate) async fn compile_dylib(
    crate_dir: &Path,
    profile: Profile,
    candidate: &str,
    lib_rs: &str,
) -> Result<()> {
    let t0 = Instant::now();

    std::fs::write(crate_dir.join("src").join("lib.rs"), lib_rs).map_err(|e| {
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
    let mut args = vec![
        "rustc",
        "--manifest-path",
        &manifest_str,
        "--message-format=json",
    ];
    if profile == Profile::Release {
        args.push("--release");
    }
    args.extend_from_slice(&["--", "--cap-lints", "allow"]);
    if cfg!(target_os = "linux") {
        // Bind the dylib's calls to its own exported functions at link time.
        //
        // An exported symbol in an ELF dylib has default visibility and is
        // therefore preemptible: the compiler emits the dylib's *own* calls
        // to it through the GOT, and the loader resolves those against the
        // global scope — the host executable and everything loaded with it,
        // libc included — before the dylib itself. A generated function whose
        // name collides with a libc symbol (e.g. `qsort`) then hijacks the
        // call with mismatched arguments and segfaults the host.
        // `-Bsymbolic-functions` pre-binds intra-dylib calls to the local
        // definitions and removes that whole failure mode. Only Rust symbols
        // undefined in the artifact (libc, a shared std) still resolve
        // dynamically.
        args.push("-Clink-arg=-Wl,-Bsymbolic-functions");
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
            code: candidate.to_string(),
            err: format!("Failed to spawn cargo: {e}"),
            diagnostics: Vec::new(),
        })?;

    if output.status.success() {
        info!(
            "Evolvable dylib compiled successfully in {}ms",
            t0.elapsed().as_millis()
        );
        Ok(())
    } else {
        let diagnostics = parse_cargo_json(&String::from_utf8_lossy(&output.stdout), candidate);
        Err(compilation_failed(
            candidate,
            diagnostics,
            &String::from_utf8_lossy(&output.stderr),
        ))
    }
}

/// The error of a failed build: the located diagnostics and the text the
/// model reads. Without any located error, cargo's stderr is the text.
fn compilation_failed(candidate: &str, diagnostics: Vec<Diagnostic>, stderr: &str) -> Error {
    let err = if diagnostics.is_empty() {
        stderr.to_string()
    } else {
        let mut err = String::new();
        render_for_prompt(&diagnostics, &mut err);
        err
    };
    Error::CompilationFailed {
        code: candidate.to_string(),
        err,
        diagnostics,
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
