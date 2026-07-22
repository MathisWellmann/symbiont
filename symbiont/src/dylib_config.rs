use std::path::PathBuf;

use getset::{
    CopyGetters,
    Getters,
};

use crate::dylib_dependency::{
    DylibDependency,
    DylibPatch,
};

/// Configuration for the temporary dylib crate compiled by [`crate::Runtime`].
///
/// The generated dylib is a normal Cargo crate. When an evolvable signature uses
/// host-defined or dependency-defined types, configure the dylib with matching
/// dependencies and imports instead of copying type source into the dylib.
///
/// [`DylibConfig::host_package`] is the ergonomic path for examples and
/// single-package applications: it depends on the current package's library
/// target as the crate alias `host` and prepends `use host::prelude::*;`.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct DylibConfig {
    /// Compilation profile (`debug` or `release`).
    #[getset(get_copy = "pub")]
    profile: crate::Profile,

    /// Rust source snippets prepended to every generated dylib source file.
    /// Typically contains imports such as `use host::prelude::*;`.
    #[getset(get = "pub")]
    prelude: Vec<String>,

    /// Dependencies added to the generated dylib's `Cargo.toml`.
    #[getset(get = "pub")]
    dependencies: Vec<DylibDependency>,

    /// `[patch]` sections added to the generated dylib's `Cargo.toml`.
    #[getset(get = "pub")]
    patches: Vec<DylibPatch>,

    /// Path prefixes (e.g. `std::fs`) that LLM-generated code must not
    /// reference. Enforced on the parsed AST before compilation; violations
    /// are fed back to the agent as backpressure.
    ///
    /// Defaults to [`DylibConfig::default_denied_paths`]. Hosts widen or
    /// narrow the capability surface with [`DylibConfig::with_denied_path`]
    /// and [`DylibConfig::with_allowed_path`].
    #[getset(get = "pub")]
    denied_paths: Vec<String>,
}

impl DylibConfig {
    /// The path prefixes denied in LLM-generated code by default:
    /// process control and spawning (`std::process` — `exit`/`abort` kill
    /// the host instantly, bypassing panic capture), threads
    /// (`std::thread` — spawned threads escape the feedback-loop contract),
    /// filesystem and network I/O (`std::fs`, `std::net`), host environment
    /// (`std::env`), OS extension traits (`std::os`), and blocking stdin
    /// (`std::io::stdin`).
    ///
    /// This bounds what evolvable code can *name*; it is a guidance
    /// mechanism for the evolution loop, not a security sandbox.
    #[must_use]
    pub fn default_denied_paths() -> Vec<String> {
        [
            "std::process",
            "std::thread",
            "std::fs",
            "std::net",
            "std::env",
            "std::os",
            "std::io::stdin",
        ]
        .map(String::from)
        .to_vec()
    }

    /// Create a config for a Cargo package's library target.
    ///
    /// The dylib gets a path dependency on `package_dir`, renamed to the crate
    /// alias `host`, and imports `host::prelude::*` by default.
    ///
    /// This assumes the package has a `lib` target. Binary-only packages should
    /// move shared boundary types and methods into `src/lib.rs`.
    #[must_use]
    pub fn host_package(
        profile: crate::Profile,
        package_name: impl Into<String>,
        package_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            profile,
            prelude: vec!["use host::prelude::*;".to_string()],
            dependencies: vec![DylibDependency::path_renamed(
                "host",
                package_name,
                package_dir,
            )],
            patches: Vec::new(),
            denied_paths: Self::default_denied_paths(),
        }
    }

    /// Create a config with no dylib dependencies and the supplied profile.
    #[must_use]
    pub fn standalone(profile: crate::Profile) -> Self {
        Self {
            profile,
            prelude: Vec::new(),
            dependencies: Vec::new(),
            patches: Vec::new(),
            denied_paths: Self::default_denied_paths(),
        }
    }

    /// Add a Rust source snippet to the generated dylib prelude.
    #[must_use]
    pub fn with_prelude(mut self, prelude: impl Into<String>) -> Self {
        self.prelude.push(prelude.into());
        self
    }

    /// Add a dependency to the generated dylib crate.
    #[must_use]
    pub fn with_dependency(mut self, dependency: DylibDependency) -> Self {
        self.dependencies.push(dependency);
        self
    }

    /// Add a `[patch]` section to the generated dylib crate.
    ///
    /// The generated crate is its own Cargo build root; `[patch]` sections of
    /// the host workspace do not reach it. Mirror any host-workspace patch
    /// that affects types crossing the dylib boundary.
    #[must_use]
    pub fn with_patch(mut self, patch: DylibPatch) -> Self {
        self.patches.push(patch);
        self
    }

    /// Deny an additional path prefix in LLM-generated code, e.g.
    /// `host::dangerous` or `std::collections::BTreeMap`.
    #[must_use]
    pub fn with_denied_path(mut self, path: impl Into<String>) -> Self {
        self.denied_paths.push(path.into());
        self
    }

    /// Allow a path prefix that is denied by default, e.g. `std::fs` for a
    /// host whose evolvable functions legitimately operate on files.
    ///
    /// Removes every denied entry equal to `path` or nested inside it.
    #[must_use]
    pub fn with_allowed_path(mut self, path: &str) -> Self {
        let nested = format!("{path}::");
        self.denied_paths
            .retain(|denied| denied != path && !denied.starts_with(&nested));
        self
    }
}

impl From<crate::Profile> for DylibConfig {
    fn from(profile: crate::Profile) -> Self {
        Self::standalone(profile)
    }
}
