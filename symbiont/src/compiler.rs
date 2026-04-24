use tokio::process::Command;
use tracing::info;

use crate::error::{
    Error,
    Result,
};

/// Compile `symbiont-lib` by invoking `cargo build -p symbiont-lib`.
/// Blocks (async) until compilation finishes.
/// Returns `Ok(())` on success, or `Err(CompilationFailed(stderr))` on failure.
pub(crate) async fn compile_lib() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    info!("Compiling symbiont-lib...");
    let output = Command::new("cargo")
        // TODO: might want to set the optimization profile.
        .args(["build", "-p", "symbiont-lib"])
        .current_dir(format!("{manifest_dir}/.."))
        .output()
        .await
        .map_err(|e| Error::CompilationFailed(format!("Failed to spawn cargo: {e}")))?;

    if output.status.success() {
        info!("symbiont-lib compiled successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(Error::CompilationFailed(stderr))
    }
}
