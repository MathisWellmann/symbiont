use crate::Error;

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

/// Return `true` for [`Error`] values indicating the request exceeded the
/// model's context window.
///
/// Such a request can never succeed by resending, but it is recoverable by
/// shrinking the conversation: [`crate::Runtime::evolve`] responds by
/// discarding the accumulated retry history and restarting from the base
/// prompt.
pub(crate) fn is_context_size_error(err: &Error) -> bool {
    let (Some(status), Some(body)) = provider_response_of(err) else {
        return false;
    };

    let markers: &[&str] = match status {
        400 => &CONTEXT_SIZE_MARKERS,
        500 => &LLAMA_CPP_500_MARKERS,
        _ => return false,
    };
    let body = body.to_ascii_lowercase();
    markers.iter().any(|marker| body.contains(marker))
}

/// The status and the raw body of the provider response inside `err`, when
/// `err` preserves one.
///
/// rig keeps a failed provider response in one of three shapes, and which one
/// a call gets depends on the provider and on the transport, not on the
/// failure: a plain `HttpError`, an `HttpError` that also kept the response
/// headers, or a `ProviderResponse` for the providers with a request-id
/// contract. All three mean the same thing here, so classification reads them
/// through the accessors of rig instead of matching one variant.
///
/// The status can be a `2xx`: some providers send an error envelope with a
/// success status. A caller must not read failure out of the status alone.
fn provider_response_of(err: &Error) -> (Option<u16>, Option<&str>) {
    match err {
        Error::RigPrompt(rig_agent::completion::PromptError::CompletionError(e)) => (
            e.provider_response_status().map(|status| status.as_u16()),
            e.provider_response_body(),
        ),
        Error::RigHttp(e) => {
            use rig_core::http_client::Error::*;
            match e {
                InvalidStatusCode(status) => (Some(status.as_u16()), None),
                InvalidStatusCodeWithMessage(status, body) => {
                    (Some(status.as_u16()), Some(body.as_str()))
                }
                InvalidStatusCodeWithDetails { status, body, .. } => {
                    (Some(status.as_u16()), Some(body.as_str()))
                }
                _ => (None, None),
            }
        }
        _ => (None, None),
    }
}

/// The status of the provider response inside `err`, when `err` preserves
/// one. A `Some` result means the endpoint answered and the answer is the
/// failure.
pub(crate) fn provider_status_of(err: &Error) -> Option<u16> {
    provider_response_of(err).0
}

/// Return `true` for [`Error`] values that represent transient failures of
/// the LLM provider (rate-limits, server overload, gateway errors) and are
/// safe to retry without modifying the prompt.
pub(crate) fn is_transient_http_error(err: &Error) -> bool {
    // Connection-level errors (timeouts, resets, DNS) carry no status and are
    // transient by nature.
    if is_connection_error(err) {
        return true;
    }
    let Some(status) = provider_status_of(err) else {
        return false;
    };

    // 408 Request Timeout, 425 Too Early, 429 Too Many Requests,
    // 5xx Server errors (incl. 529 Site Overloaded used by Anthropic).
    matches!(status, 408 | 425 | 429 | 500..=599)
}

/// Return `true` when the request never got an answer from the endpoint.
fn is_connection_error(err: &Error) -> bool {
    let http_err = match err {
        Error::RigPrompt(rig_agent::completion::PromptError::CompletionError(
            rig_core::completion::CompletionError::HttpError(e),
        )) => e,
        Error::RigHttp(e) => e,
        _ => return false,
    };
    matches!(http_err, rig_core::http_client::Error::Instance(_))
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// The three shapes rig preserves a failed provider response in. Which
    /// one a call gets depends on the provider and on the transport: a
    /// provider with a request-id contract reports `ProviderResponse`, and a
    /// transport that kept the response headers reports the details variant.
    fn response_shapes(status: http::StatusCode, body: &str) -> [Error; 3] {
        use rig_core::{
            completion::CompletionError,
            http_client::Error as HttpError,
        };
        [
            http_status(status, body),
            Error::RigHttp(HttpError::InvalidStatusCodeWithDetails {
                status,
                body: body.to_string(),
                headers: Box::new(http::HeaderMap::new()),
            }),
            Error::RigPrompt(rig_agent::completion::PromptError::CompletionError(
                CompletionError::from_http_response_with_request_id(status, body, None),
            )),
        ]
    }

    #[test]
    fn every_response_shape_is_classified_alike() {
        for err in response_shapes(
            http::StatusCode::BAD_REQUEST,
            "This model's maximum context length is 65536 tokens.",
        ) {
            assert!(is_context_size_error(&err), "overflow not detected: {err}");
        }
        for err in response_shapes(http::StatusCode::TOO_MANY_REQUESTS, "rate limited") {
            assert!(is_transient_http_error(&err), "not transient: {err}");
            assert!(!is_context_size_error(&err));
        }
        for err in response_shapes(http::StatusCode::UNAUTHORIZED, "invalid api key") {
            assert!(!is_transient_http_error(&err), "wrongly transient: {err}");
            assert!(!is_context_size_error(&err));
        }
    }

    #[test]
    fn connection_errors_are_transient() {
        let err = Error::RigHttp(rig_core::http_client::Error::Instance(Box::new(
            std::io::Error::from(std::io::ErrorKind::ConnectionReset),
        )));
        assert!(is_transient_http_error(&err));
        assert!(!is_context_size_error(&err));
    }
}
