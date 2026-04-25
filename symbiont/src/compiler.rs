use std::path::Path;

use minstant::Instant;
use tokio::process::Command;
use tracing::info;

use crate::error::{
    Error,
    Result,
};

/// Compile a dylib crate at the given directory.
///
/// Runs `cargo build --manifest-path <crate_dir>/Cargo.toml`.
/// Blocks (async) until compilation finishes.
pub(crate) async fn compile_dylib(crate_dir: &Path) -> Result<()> {
    let t0 = Instant::now();

    let manifest_path = crate_dir.join("Cargo.toml");

    info!(
        "Compiling evolvable dylib at {}...",
        manifest_path.display()
    );
    // TODO: optional optimization level here.
    let output = Command::new("cargo")
        .args(["build", "--manifest-path", &manifest_path.to_string_lossy()])
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
