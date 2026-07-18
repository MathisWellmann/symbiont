use std::path::PathBuf;

use getset::{
    CopyGetters,
    Getters,
};

/// A `[patch.<source>]` entry for the generated dylib crate's `Cargo.toml`.
///
/// The generated dylib crate is its own Cargo build root, so `[patch]`
/// sections of the host workspace do **not** apply to it. When the host
/// workspace patches a dependency (e.g. to a local fork), mirror that patch
/// into the dylib crate with [`crate::DylibConfig::with_patch`] — otherwise
/// the dylib compiles the unpatched upstream source and the two sides of the
/// `dlopen` boundary disagree.
#[derive(Debug, Clone, PartialEq, Eq, Getters)]
pub struct DylibPatch {
    /// The patched source: `"crates-io"` for the default registry, or a git
    /// URL such as `"https://github.com/foo/bar"`.
    #[getset(get = "pub")]
    source: String,

    /// The dependency entry that replaces the patched package.
    #[getset(get = "pub")]
    dependency: DylibDependency,
}

impl DylibPatch {
    /// Patch a crates.io package.
    #[must_use]
    pub fn crates_io(dependency: DylibDependency) -> Self {
        Self {
            source: "crates-io".to_string(),
            dependency,
        }
    }

    /// Patch a git-sourced package identified by its repository URL.
    #[must_use]
    pub fn git(url: impl Into<String>, dependency: DylibDependency) -> Self {
        Self {
            source: url.into(),
            dependency,
        }
    }
}

/// A dependency entry for the generated dylib crate's `Cargo.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct DylibDependency {
    /// Dependency key in the generated `Cargo.toml`, e.g. `host`.
    #[getset(get = "pub")]
    name: String,

    /// Optional package rename, e.g. `{ package = "my-app", path = "..." }`.
    #[getset(get = "pub")]
    package: Option<String>,

    /// Local path dependency.
    #[getset(get = "pub")]
    path: Option<PathBuf>,

    /// Registry version requirement.
    #[getset(get = "pub")]
    version: Option<String>,

    /// Enabled features.
    #[getset(get = "pub")]
    features: Vec<String>,

    /// Whether default features are enabled.
    #[getset(get_copy = "pub")]
    default_features: bool,
}

impl DylibDependency {
    /// Create a path dependency.
    #[must_use]
    pub fn with_path(name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            package: None,
            path: Some(path.into()),
            version: None,
            features: Vec::new(),
            default_features: true,
        }
    }

    /// Create a path dependency with a crate rename.
    #[must_use]
    pub fn path_renamed(
        name: impl Into<String>,
        package: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            package: Some(package.into()),
            path: Some(path.into()),
            version: None,
            features: Vec::new(),
            default_features: true,
        }
    }

    /// Create a registry dependency.
    #[must_use]
    pub fn with_version(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            package: None,
            path: None,
            version: Some(version.into()),
            features: Vec::new(),
            default_features: true,
        }
    }

    /// Set the package name used when it differs from the dependency key.
    #[must_use]
    pub fn with_package(mut self, package: impl Into<String>) -> Self {
        self.package = Some(package.into());
        self
    }

    /// Set features for this dependency.
    #[must_use]
    pub fn with_features(mut self, features: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.features = features.into_iter().map(Into::into).collect();
        self
    }

    /// Enable or disable default features for this dependency.
    #[must_use]
    pub fn with_default_features(mut self, enabled: bool) -> Self {
        self.default_features = enabled;
        self
    }
}
