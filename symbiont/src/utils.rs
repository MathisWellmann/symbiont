use std::path::{
    Path,
    PathBuf,
};

// SPDX-License-Identifier: MPL-2.0
use syn::ItemFn;

use crate::{
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

pub(crate) fn generate_cargo_toml() -> String {
    r#"[package]
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
    .to_string()
}

pub(crate) fn generate_lib_rs(decls: &[EvolvableDecl]) -> String {
    let mut src = String::with_capacity(1_000);
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
        Error::RigPrompt(rig::completion::PromptError::CompletionError(
            rig::completion::CompletionError::HttpError(http_err),
        )) => http_err,
        Error::RigHttp(http_err) => http_err,
        _ => return false,
    };

    use rig::http_client::Error::*;
    let status = match http_err {
        InvalidStatusCode(s) => *s,
        InvalidStatusCodeWithMessage(s, _) => *s,
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
