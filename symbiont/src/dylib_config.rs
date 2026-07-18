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
}

impl DylibConfig {
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
}

impl From<crate::Profile> for DylibConfig {
    fn from(profile: crate::Profile) -> Self {
        Self::standalone(profile)
    }
}
