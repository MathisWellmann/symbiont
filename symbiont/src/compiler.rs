// SPDX-License-Identifier: MPL-2.0
use std::path::Path;

use minstant::Instant;
use prettyplease::unparse;
use tokio::process::Command;
use tracing::{
    debug,
    info,
};

use crate::{
    error::{
        Error,
        Result,
    },
    unwind::{
        PANIC_PREAMBLE,
        wrap_bodies_in_catch_unwind,
    },
};

/// Compilation profile for the evolvable dylib.
///
/// Controls whether `cargo build` is invoked with or without `--release`.
/// Use [`Profile::Release`] when benchmarking evolved functions — the
/// optimizer can make orders-of-magnitude difference for compute-heavy code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// `cargo build` (unoptimized, fast compilation).
    #[default]
    Debug,
    /// `cargo build --release` (optimized, slower compilation).
    Release,
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Profile::Debug => f.write_str("debug"),
            Profile::Release => f.write_str("release"),
        }
    }
}

/// Compile a dylib crate at the given directory.
///
/// Runs `cargo build --manifest-path <crate_dir>/Cargo.toml`,
/// adding `--release` when the profile is [`Profile::Release`].
/// Blocks (async) until compilation finishes.
pub(crate) async fn compile_dylib(
    crate_dir: &Path,
    profile: Profile,
    clean_ast: &mut syn::File,
) -> Result<()> {
    let t0 = Instant::now();

    let clean_code = unparse(clean_ast);
    debug!("clean_code: {clean_code}");
    let clean_path = crate_dir.join("src").join("clean.rs");
    std::fs::write(&clean_path, &clean_code)
        .map_err(|e| Error::WriteLib(format!("Failed to write clean.rs: {e}")))?;

    // Wrap function bodies in catch_unwind so panics stay inside the dylib.
    wrap_bodies_in_catch_unwind(clean_ast);

    // Write final lib.rs (preamble + wrapped code) for compilation.
    let formatted = format!("{PANIC_PREAMBLE}\n{}", unparse(clean_ast));
    std::fs::write(crate_dir.join("src").join("lib.rs"), formatted)?;
    info!("Created temp dylib crate at {}", crate_dir.display());

    let manifest_path = crate_dir.join("Cargo.toml");
    info!(
        "Compiling evolvable dylib ({profile}) at {}...",
        manifest_path.display()
    );
    let manifest_str = manifest_path.to_string_lossy();
    let mut args = vec!["build", "--manifest-path", &manifest_str];
    if profile == Profile::Release {
        args.push("--release");
    }

    let output = Command::new("cargo")
        .args(&args)
        .output()
        .await
        .map_err(|e| Error::CompilationFailed {
            code: clean_code.clone(),
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
            code: clean_code.clone(),
            err,
        })
    }
}
