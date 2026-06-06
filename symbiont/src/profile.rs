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
        use Profile::*;
        match self {
            Debug => f.write_str("debug"),
            Release => f.write_str("release"),
        }
    }
}
