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

/// Return `true` for [`Error`] values indicating the request exceeded the
/// model's context window.
///
/// Such a request can never succeed by resending, but it is recoverable by
/// shrinking the conversation: [`crate::Runtime::evolve`] responds by
/// discarding the accumulated retry history and restarting from the base
/// prompt.
pub(crate) fn is_context_size_error(err: &Error) -> bool {
    let Some(http_err) = http_error_of(err) else {
        return false;
    };

    let rig_core::http_client::Error::InvalidStatusCodeWithMessage(status, msg) = http_err else {
        return false;
    };

    let markers: &[&str] = match status.as_u16() {
        400 => &CONTEXT_SIZE_MARKERS,
        500 => &LLAMA_CPP_500_MARKERS,
        _ => return false,
    };
    let msg = msg.to_ascii_lowercase();
    markers.iter().any(|marker| msg.contains(marker))
}

/// Lower-case substrings that identify a context-window overflow in the body
/// of a `400 Bad Request` from an inference endpoint.
///
/// Every provider phrases the same condition differently, and none of them
/// expose a machine-readable code that all of them agree on, so matching the
/// body is the only portable option. (llama.cpp's `500` overflow variants
/// live in [`LLAMA_CPP_500_MARKERS`].)
const CONTEXT_SIZE_MARKERS: [&str; 5] = [
    // llama.cpp: `{"error":{"code":400,"message":"request (258963 tokens)
    // exceeds the available context size (256000 tokens), try increasing
    // it","type":"exceed_context_size_error",...}}`.
    "exceed_context_size",
    // OpenAI and other OpenAI-compatible servers:
    // `{"error":{"code":"context_length_exceeded",...}}`.
    "context_length_exceeded",
    // Anthropic: "prompt is too long: 215591 tokens > 204798 maximum".
    "prompt is too long",
    // vLLM, request validation: "This model's maximum context length is
    // 65536 tokens. However, you requested 70000 tokens (69000 in the
    // messages, 1000 in the completion). Please reduce the length of the
    // messages or completion."
    "maximum context length",
    // vLLM, engine-side validation: "The decoder prompt (length 70000) is
    // longer than the maximum model length of 65536."
    "longer than the maximum model length",
];

/// Lower-case substrings that identify a context-window overflow reported by
/// llama.cpp as a `500 Internal Server Error`.
const LLAMA_CPP_500_MARKERS: [&str; 3] = [
    // The KV cache cannot fit even a single-token decode batch:
    // `err = "Context size has been exceeded."` in `update_slots()`. It is
    // sent to *every* processing slot and then re-broadcast by
    // `abort_all_slots("decode() failed: " + err)`, hence the substring
    // match: both bodies must be recognized.
    "context size has been exceeded",
    // The prompt exceeds the physical batch size.
    "increase the physical batch size",
    // Generation reached `n_ctx` while `--no-context-shift` is in effect:
    // "context shift is disabled".
    "context shift is disabled",
];

/// The provider HTTP error inside `err`, if that is what it wraps.
fn http_error_of(err: &Error) -> Option<&rig_core::http_client::Error> {
    match err {
        Error::RigPrompt(rig_agent::completion::PromptError::CompletionError(
            rig_core::completion::CompletionError::HttpError(http_err),
        )) => Some(http_err),
        Error::RigHttp(http_err) => Some(http_err),
        _ => None,
    }
}

/// Return `true` for [`Error`] values that represent transient failures of
/// the LLM provider (rate-limits, server overload, gateway errors) and are
/// safe to retry without modifying the prompt.
pub(crate) fn is_transient_http_error(err: &Error) -> bool {
    let Some(http_err) = http_error_of(err) else {
        return false;
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

    /// The [`Error`] a provider surfaces for `status` with `body`.
    fn http_status(status: http::StatusCode, body: &str) -> Error {
        Error::RigHttp(rig_core::http_client::Error::InvalidStatusCodeWithMessage(
            status,
            body.to_string(),
        ))
    }

    /// The [`Error`] a provider surfaces for a `400 Bad Request` with `body`.
    fn http_400(body: &str) -> Error {
        http_status(http::StatusCode::BAD_REQUEST, body)
    }

    #[test]
    fn context_size_error_detects_every_provider_wording() {
        for body in [
            // llama.cpp
            r#"{"error":{"code":400,"message":"request (258963 tokens) exceeds the available context size (256000 tokens), try increasing it","type":"exceed_context_size_error"}}"#,
            // OpenAI-compatible
            r#"{"error":{"message":"too long","code":"context_length_exceeded"}}"#,
            // Anthropic
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 215591 tokens > 204798 maximum"}}"#,
            // vLLM, request validation
            r#"{"object":"error","message":"This model's maximum context length is 65536 tokens. However, you requested 70000 tokens (69000 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}"#,
            // vLLM, engine-side validation
            r#"{"object":"error","message":"The decoder prompt (length 70000) is longer than the maximum model length of 65536. Make sure that `max_model_len` is no smaller than the number of text tokens.","type":"BadRequestError","param":null,"code":400}"#,
        ] {
            assert!(
                is_context_size_error(&http_400(body)),
                "overflow not detected: {body}"
            );
        }
    }

    #[test]
    fn context_size_error_matching_ignores_case() {
        assert!(is_context_size_error(&http_400(
            "This Model's Maximum Context Length Is 65536 Tokens."
        )));
    }

    #[test]
    fn context_size_error_seen_through_the_rig_prompt_wrapper() {
        // The shape the runtime actually receives: rig wraps the provider
        // error in `PromptError::CompletionError`.
        let err = Error::RigPrompt(rig_agent::completion::PromptError::CompletionError(
            rig_core::completion::CompletionError::HttpError(
                rig_core::http_client::Error::InvalidStatusCodeWithMessage(
                    http::StatusCode::BAD_REQUEST,
                    "This model's maximum context length is 65536 tokens.".to_string(),
                ),
            ),
        ));
        assert!(is_context_size_error(&err));
    }

    #[test]
    fn other_bad_requests_are_not_context_size_errors() {
        for body in [
            // vLLM rejects this before it ever looks at the context length.
            r#"{"object":"error","message":"max_tokens must be at least 1, got -45.","type":"BadRequestError","code":400}"#,
            r#"{"error":{"message":"model `qwen3.6` not found","code":"model_not_found"}}"#,
            "",
        ] {
            assert!(
                !is_context_size_error(&http_400(body)),
                "false positive: {body}"
            );
        }
    }

    #[test]
    fn llama_cpp_500_context_overflow_is_detected() {
        for body in [
            // KV cache full, single-token batch.
            r#"{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}"#,
            // The same condition after `abort_all_slots`.
            r#"{"error":{"code":500,"message":"decode() failed: Context size has been exceeded.","type":"server_error"}}"#,
            // Prompt larger than the physical batch.
            r#"{"error":{"code":500,"message":"input (70000 tokens) is too large to process. increase the physical batch size (current batch size: 2048)","type":"server_error"}}"#,
            // Generation hit `n_ctx` with context shifting turned off.
            r#"{"error":{"code":500,"message":"context shift is disabled","type":"server_error"}}"#,
        ] {
            let err = http_status(http::StatusCode::INTERNAL_SERVER_ERROR, body);
            assert!(is_context_size_error(&err), "overflow not detected: {body}");
            // Every 5xx is transient too, so the two predicates overlap here:
            // `Runtime::evolve` must keep testing this one first, otherwise
            // the request is resent unchanged until the retry budget is gone.
            assert!(is_transient_http_error(&err), "not transient: {body}");
        }
    }

    #[test]
    fn other_providers_context_wording_in_a_5xx_is_ignored() {
        // Only llama.cpp reports an overflow as a 5xx. A 5xx carrying another
        // provider's overflow wording is a server failure, not an oversized
        // prompt: it stays retryable without shrinking the conversation.
        let err = http_status(
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "This model's maximum context length is 65536 tokens",
        );
        assert!(!is_context_size_error(&err));
        assert!(is_transient_http_error(&err));
    }

    #[test]
    fn non_http_errors_are_not_context_size_errors() {
        assert!(!is_context_size_error(&Error::NoRustCode));
    }
}
