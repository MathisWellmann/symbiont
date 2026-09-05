// SPDX-License-Identifier: MPL-2.0
use std::{
    collections::hash_map::DefaultHasher,
    fmt::Write,
    hash::{
        Hash,
        Hasher,
    },
    path::{
        Path,
        PathBuf,
    },
};

use tracing::info;

use crate::{
    DylibConfig,
    DylibDependency,
    DylibPatch,
    Error,
    EvolvableDecl,
    Profile,
    Result,
};

/// Create the dylib crate for `decls` under the temp dir and write its
/// manifest and lockfile. Returns the crate directory.
///
/// The directory is named after a hash of the function names, so the same
/// set of evolvables reuses one crate (and its `target/`) across runs.
/// The manifest and lockfile are rewritten on every start, so a reused
/// directory never keeps a stale manifest or lock from an earlier host.
pub(crate) fn scaffold_dylib_crate(
    decls: &[EvolvableDecl],
    config: &DylibConfig,
) -> Result<PathBuf> {
    let mut hasher = DefaultHasher::new();
    for d in decls {
        d.name.hash(&mut hasher);
    }
    let hash = hasher.finish();
    let crate_dir = std::env::temp_dir().join(format!("symbiont-evolvable-{hash:x}"));
    std::fs::create_dir_all(crate_dir.join("src")).map_err(|e| {
        Error::DylibLoad(format!(
            "Failed to create dylib crate directory {}: {e}",
            crate_dir.display()
        ))
    })?;

    let cargo_toml = generate_cargo_toml(config.dependencies(), config.patches());
    std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml).map_err(|e| {
        Error::DylibLoad(format!(
            "Failed to write {}: {e}",
            crate_dir.join("Cargo.toml").display()
        ))
    })?;

    if let Some(lockfile) = host_lockfile(config.dependencies()) {
        info!("Seeding dylib crate lockfile from {}", lockfile.display());
        std::fs::copy(&lockfile, crate_dir.join("Cargo.lock")).map_err(|e| {
            Error::DylibLoad(format!(
                "Failed to copy {} into the dylib crate: {e}",
                lockfile.display()
            ))
        })?;
    }
    Ok(crate_dir)
}
pub(crate) fn generate_cargo_toml(
    dependencies: &[DylibDependency],
    patches: &[DylibPatch],
) -> String {
    let mut toml = r#"[package]
name = "symbiont-evolvable"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["dylib"]

# Ensure panics unwind rather than abort so that `symbiont::catch_panic`
# can intercept them across the dylib boundary.
[profile.dev]
panic = "unwind"

[profile.release]
panic = "unwind"

[dependencies]
"#
    .to_string();

    for dependency in dependencies {
        write_dependency(&mut toml, dependency);
    }

    for patch in patches {
        // `crates-io` is a bare key; git URLs must be quoted.
        let source = patch.source();
        if source == "crates-io" {
            let _ = write!(toml, "\n[patch.crates-io]\n");
        } else {
            let _ = write!(toml, "\n[patch.{source:?}]\n");
        }
        write_dependency(&mut toml, patch.dependency());
    }

    toml
}

fn write_dependency(toml: &mut String, dependency: &DylibDependency) {
    toml.push_str(dependency.name());
    toml.push_str(" = ");

    let simple_version = dependency.package().is_none()
        && dependency.path().is_none()
        && dependency.git().is_none()
        && dependency.features().is_empty()
        && dependency.default_features();
    if simple_version && let Some(version) = &dependency.version() {
        let _ = writeln!(toml, "{version:?}");
        return;
    }

    toml.push_str("{ ");
    let mut needs_comma = false;
    let mut push_field = |toml: &mut String, name: &str, value: &str| {
        if needs_comma {
            toml.push_str(", ");
        }
        let _ = write!(toml, "{name} = {value:?}");
        needs_comma = true;
    };

    if let Some(package) = dependency.package() {
        push_field(toml, "package", package);
    }
    if let Some(path) = dependency.path() {
        push_field(toml, "path", &path.display().to_string());
    }
    if let Some(git) = dependency.git() {
        push_field(toml, "git", git);
    }
    if let Some(rev) = dependency.rev() {
        push_field(toml, "rev", rev);
    }
    if let Some(version) = dependency.version() {
        push_field(toml, "version", version);
    }
    if !dependency.default_features() {
        if needs_comma {
            toml.push_str(", ");
        }
        toml.push_str("default-features = false");
        needs_comma = true;
    }
    if !dependency.features().is_empty() {
        if needs_comma {
            toml.push_str(", ");
        }
        toml.push_str("features = [");
        for (idx, feature) in dependency.features().iter().enumerate() {
            if idx > 0 {
                toml.push_str(", ");
            }
            let _ = write!(toml, "{feature:?}");
        }
        toml.push(']');
    }

    toml.push_str(" }\n");
}

/// The `Cargo.lock` that governs the dylib's path dependencies, if any.
///
/// A path dependency (the host package) is built inside the dylib crate,
/// but the dylib crate is its own package: without a lockfile, cargo
/// resolves the host's dependencies from scratch, and a `git` dependency
/// then lands on the remote's current HEAD rather than the revision the
/// host was written and tested against. Copying the host's lockfile into
/// the dylib crate keeps every shared dependency at the host's pinned
/// version; cargo adds the dylib package itself and prunes entries nothing
/// depends on.
///
/// The lockfile is found by walking up from the path dependency's directory,
/// which handles both a standalone package and a workspace member (whose
/// lockfile sits at the workspace root). The first path dependency that
/// leads to one wins.
pub(crate) fn host_lockfile(dependencies: &[DylibDependency]) -> Option<PathBuf> {
    dependencies
        .iter()
        .filter_map(|dependency| dependency.path().as_deref())
        .find_map(|path| {
            path.ancestors()
                .map(|dir| dir.join("Cargo.lock"))
                .find(|lock| lock.is_file())
        })
}

pub(crate) fn dylib_extension() -> &'static str {
    if cfg!(target_os = "macos") {
        ".dylib"
    } else if cfg!(target_os = "windows") {
        ".dll"
    } else {
        ".so"
    }
}

/// Path of the retained copy of a revision's compiled shared library.
/// One file per revision id, which also defeats `dlopen` path caching.
pub(crate) fn versioned_so_path(crate_dir: &Path, revision_id: u64) -> PathBuf {
    crate_dir.join(format!(
        "libsymbiont_evolvable_v{revision_id}{}",
        dylib_extension()
    ))
}

/// Find the compiled shared library in the temp crate's target directory.
pub(crate) fn find_so(crate_dir: &Path, profile: Profile) -> Result<PathBuf> {
    let subdir = match profile {
        Profile::Debug => "debug",
        Profile::Release => "release",
    };
    let target_dir = crate_dir.join("target").join(subdir);

    let prefix = if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    };
    let name = format!("{prefix}symbiont_evolvable{ext}", ext = dylib_extension());
    let so_path = target_dir.join(&name);

    if so_path.exists() {
        Ok(so_path)
    } else {
        Err(Error::DylibLoad(format!(
            "Compiled dylib not found at {}",
            so_path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_lockfile_is_found_at_the_workspace_root() {
        let root =
            std::env::temp_dir().join(format!("symbiont_host_lockfile_{}", std::process::id()));
        let member = root.join("examples").join("member");
        std::fs::create_dir_all(&member).expect("can create fixture dirs");
        std::fs::write(root.join("Cargo.lock"), "").expect("can write lockfile");

        let deps = [
            DylibDependency::with_version("serde", "1"),
            DylibDependency::path_renamed("host", "member", &member),
        ];
        assert_eq!(host_lockfile(&deps), Some(root.join("Cargo.lock")));
        assert_eq!(
            host_lockfile(&deps[..1]),
            None,
            "a registry dependency has no lockfile to inherit"
        );

        std::fs::remove_dir_all(&root).expect("can remove fixture");
    }

    #[test]
    fn cargo_toml_renders_pinned_git_dependency() {
        let toml = generate_cargo_toml(
            &[DylibDependency::with_git(
                "const-decimal",
                "https://github.com/MathisWellmann/const-decimal",
                "ec766197d62bc89f15b64971fc22c5e8721e90d5",
            )],
            &[],
        );

        assert!(toml.contains(
            "const-decimal = { git = \"https://github.com/MathisWellmann/const-decimal\", rev = \"ec766197d62bc89f15b64971fc22c5e8721e90d5\" }"
        ));
    }

    #[test]
    fn cargo_toml_renders_patch_sections() {
        let deps = [DylibDependency::path_renamed("host", "my-app", "/tmp/app")];
        let patches = [
            DylibPatch::git(
                "https://github.com/foo/bar",
                DylibDependency::with_path("bar", "/tmp/bar"),
            ),
            DylibPatch::crates_io(DylibDependency::with_path("baz", "/tmp/baz")),
        ];
        let toml = generate_cargo_toml(&deps, &patches);
        assert!(
            toml.contains("[patch.\"https://github.com/foo/bar\"]\nbar = { path = \"/tmp/bar\" }"),
            "got: {toml}"
        );
        assert!(
            toml.contains("[patch.crates-io]\nbaz = { path = \"/tmp/baz\" }"),
            "got: {toml}"
        );
    }
}
