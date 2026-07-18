use std::{
    fmt::Write,
    path::{
        Path,
        PathBuf,
    },
};

// SPDX-License-Identifier: MPL-2.0
use syn::ItemFn;

use crate::{
    DylibDependency,
    DylibPatch,
    Error,
    EvolvableDecl,
    Profile,
    Result,
};

/// If `true`, the function has visibiliity `pub`
#[inline(always)]
pub(crate) fn is_pub(item_fn: &ItemFn) -> bool {
    matches!(item_fn.vis, syn::Visibility::Public(_))
}

/// If `true`, the function is annotated with `#[unsafe(no_mangle)]`
#[inline]
pub(crate) fn is_no_mangle(item_fn: &ItemFn) -> bool {
    item_fn.attrs.iter().any(|attr| {
        // Only match exactly #[unsafe(no_mangle)]
        attr.path().is_ident("unsafe")
            && matches!(&attr.meta, syn::Meta::List(list) if list.tokens.to_string() == "no_mangle")
    })
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

pub(crate) fn generate_lib_rs(decls: &[EvolvableDecl], prelude: &[String]) -> String {
    let mut src = String::with_capacity(1_000);
    for part in prelude {
        if part.is_empty() {
            continue;
        }
        src.push_str(part);
        if !part.ends_with('\n') {
            src.push('\n');
        }
        src.push('\n');
    }
    for d in decls {
        src.push_str(d.full_source);
        src.push_str("\n\n");
    }
    src
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

/// Return `true` for [`Error`] values that represent transient failures of
/// the LLM provider (rate-limits, server overload, gateway errors) and are
/// safe to retry without modifying the prompt.
pub(crate) fn is_transient_http_error(err: &Error) -> bool {
    let http_err = match err {
        Error::RigPrompt(rig_core::completion::PromptError::CompletionError(
            rig_core::completion::CompletionError::HttpError(http_err),
        )) => http_err,
        Error::RigHttp(http_err) => http_err,
        _ => return false,
    };

    use rig_core::http_client::Error::*;
    let status = match http_err {
        InvalidStatusCode(s) => s,
        InvalidStatusCodeWithMessage(s, _) => s,
        // Connection-level errors (timeouts, resets, DNS, etc.) are also
        // transient by nature.
        Instance(_) => return true,
        _ => return false,
    };

    let code = status.as_u16();
    // 408 Request Timeout, 425 Too Early, 429 Too Many Requests,
    // 5xx Server errors (incl. 529 Site Overloaded used by Anthropic).
    matches!(code, 408 | 425 | 429 | 500..=599)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_unsafe_no_mangle_detected() {
        let code: ItemFn = syn::parse_quote! {
            #[unsafe(no_mangle)]
            pub fn step(counter: &mut usize) {}
        };
        assert!(is_no_mangle(&code));
    }

    #[test]
    fn test_plain_no_mangle_rejected() {
        let code: ItemFn = syn::parse_quote! {
            #[no_mangle]
            pub fn step(counter: &mut usize) {}
        };
        assert!(!is_no_mangle(&code));
    }

    #[test]
    fn test_no_attribute_returns_false() {
        let code: ItemFn = syn::parse_quote! {
            pub fn step(counter: &mut usize) {}
        };
        assert!(!is_no_mangle(&code));
    }

    #[test]
    fn test_other_attribute_returns_false() {
        let code: ItemFn = syn::parse_quote! {
            #[allow(dead_code)]
            pub fn step(counter: &mut usize) {}
        };
        assert!(!is_no_mangle(&code));
    }
}
