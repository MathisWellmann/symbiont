use std::path::Path;

use minstant::Instant;
use tokio::process::Command;
use tracing::info;

use crate::error::{
    Error,
    Result,
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
pub(crate) async fn compile_dylib(crate_dir: &Path, profile: Profile) -> Result<()> {
    let t0 = Instant::now();

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
        .map_err(|e| Error::CompilationFailed(format!("Failed to spawn cargo: {e}")))?;

    if output.status.success() {
        info!(
            "Evolvable dylib compiled successfully in {}ms",
            t0.elapsed().as_millis()
        );
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(Error::CompilationFailed(stderr))
    }
}
