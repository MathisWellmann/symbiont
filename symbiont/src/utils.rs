// SPDX-License-Identifier: MPL-2.0
use std::{
    fmt::Write,
    path::{
        Path,
        PathBuf,
    },
};

use crate::{
    DylibDependency,
    DylibPatch,
    Error,
    Profile,
    Result,
};

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
