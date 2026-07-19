// SPDX-License-Identifier: MPL-2.0
use std::{
    path::Path,
    process::Command,
};

use minstant::Instant;
use prettyplease::unparse;
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
/// Blocks (async) until compilation finishes.
///
/// The generated crate allows all warnings: the code is machine-generated
/// and its only reader is the compiler-feedback loop on failed builds, where
/// warnings would drown out the errors the evolution agent has to fix.
pub(crate) fn compile_dylib(crate_dir: &Path, profile: Profile, clean_ast_str: &str) -> Result<()> {
    let t0 = Instant::now();

    let mut clean_ast: syn::File = syn::parse_str(clean_ast_str)?;
    // Wrap function bodies in catch_unwind so panics stay inside the dylib.
    wrap_bodies_in_catch_unwind(&mut clean_ast);

    // Write final lib.rs (warning suppression + preamble + wrapped code) for
    // compilation.
    let formatted = format!(
        "#![allow(warnings)]\n{PANIC_PREAMBLE}\n{}",
        unparse(&clean_ast)
    );
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
        // This nested build has its own artifact lookup rooted at `crate_dir`.
        // Inherited `CARGO_TARGET_DIR` (commonly set by CI) would redirect
        // cargo elsewhere and make `find_so` report a missing dylib.
        .env_remove("CARGO_TARGET_DIR")
        .output()
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
