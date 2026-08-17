// SPDX-License-Identifier: MPL-2.0
//! [`MeteredHttpClient`]: the HTTP backend decorator that measures what is
//! actually sent to the inference endpoint.
//!
//! Token usage is only known after the fact, is reported by the provider, and
//! [`crate::AgentRun::usage`] aggregates it over every turn of an agentic run
//! — none of which answers "how big was the request we just sent?". This
//! decorator answers it at the only place where the payload is complete:
//! immediately before it goes on the wire.

use bytes::Bytes;
use metrics::histogram;
use rig_core::{
    http_client::{
        HttpClientExt,
        LazyBody,
        MultipartForm,
        Request,
        ReqwestClient,
        Response,
        Result,
        StreamingResponse,
    },
    wasm_compat::WasmCompatSend,
};

use crate::observability::REQUEST_BODY_BYTES;

/// An [`HttpClientExt`] decorator recording the serialized size of every
/// outbound request body as [`REQUEST_BODY_BYTES`].
///
/// This is the HTTP backend of the agents returned by
/// [`crate::agent_builder`] and [`crate::init_agent`], so hosts using those
/// get the metric for free. Hosts that assemble their own rig client can opt
/// in by passing it to rig's builder:
///
/// ```no_run
/// use rig_core::{
///     client::CompletionClient,
///     providers::openrouter,
///     http_client::ReqwestClient,
/// };
/// use symbiont::MeteredHttpClient;
///
/// # fn example() -> symbiont::Result<()> {
/// let client = openrouter::Client::builder()
///     .api_key("")
///     .base_url("http://127.0.0.1:8000/v1")
///     .http_client(MeteredHttpClient::new(ReqwestClient::default()))
///     .build()?;
/// let agent = client.agent("qwen3.6").build();
/// # Ok(())
/// # }
/// ```
///
/// # What is measured
///
/// The number of bytes of the serialized request body: the whole chat
/// completion payload — system preamble, accumulated chat history, tool
/// definitions and the new turn — plus its JSON framing. One recording per
/// HTTP request, so every turn of a tool-calling loop and every retry is
/// counted separately, unlike the per-run token metrics.
///
/// Bytes are not tokens; expect roughly 3-4 bytes per token for English text
/// and Rust source. Dividing this metric by
/// [`crate::observability::LLM_RUN_INPUT_TOKENS`] over the same window yields
/// the actual ratio for the deployed tokenizer, which turns this into a
/// context-window budget in the units the server enforces.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeteredHttpClient<H = ReqwestClient> {
    inner: H,
}

impl<H> MeteredHttpClient<H> {
    /// Wrap `inner`, measuring every request body that passes through it.
    pub const fn new(inner: H) -> Self {
        Self { inner }
    }

    /// The wrapped HTTP backend.
    pub const fn inner(&self) -> &H {
        &self.inner
    }
}

impl<H> HttpClientExt for MeteredHttpClient<H>
where
    H: HttpClientExt,
{
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.inner.send(measured(req))
    }

    /// Multipart bodies (audio, transcription) are not part of the prompt
    /// path and are forwarded unmeasured: sizing them would require draining
    /// the form.
    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        self.inner.send_streaming(measured(req))
    }
}

/// Record the serialized size of the body of `req` and return it with the body
/// converted to [`Bytes`] — the conversion the HTTP backend performs anyway.
///
/// Bodyless requests (the provider's credential check, for instance) are not
/// recorded: a zero-byte sample would drag the payload distribution down.
fn measured<T>(req: Request<T>) -> Request<Bytes>
where
    T: Into<Bytes>,
{
    let (parts, body) = req.into_parts();
    let body: Bytes = body.into();
    if !body.is_empty() {
        histogram!(REQUEST_BODY_BYTES).record(body.len() as f64);
    }
    Request::from_parts(parts, body)
}

#[cfg(test)]
mod tests {
    use metrics_util::debugging::{
        DebugValue,
        DebuggingRecorder,
    };
    use rig_core::http_client::NoBody;

    use super::*;

    /// An [`HttpClientExt`] that answers every request with `200 OK` and an
    /// empty body, and remembers the body size it was handed.
    #[derive(Debug, Clone, Copy, Default)]
    struct SpyClient;

    impl HttpClientExt for SpyClient {
        fn send<T, U>(
            &self,
            req: Request<T>,
        ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            T: Into<Bytes>,
            U: From<Bytes> + WasmCompatSend,
        {
            let body: Bytes = req.into_body().into();
            async move {
                let echoed: LazyBody<U> = Box::pin(async move { Ok(U::from(body)) });
                Ok(Response::builder()
                    .status(200)
                    .body(echoed)
                    .expect("Can build response"))
            }
        }

        #[expect(
            clippy::manual_async_fn,
            reason = "The signature must mirror the trait declaration"
        )]
        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            async { unimplemented!("not exercised by these tests") }
        }

        #[expect(
            clippy::manual_async_fn,
            reason = "The signature must mirror the trait declaration"
        )]
        fn send_streaming<T>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
        where
            T: Into<Bytes> + WasmCompatSend,
        {
            async { unimplemented!("not exercised by these tests") }
        }
    }

    /// The values [`REQUEST_BODY_BYTES`] holds in `snapshotter`.
    fn recorded_sizes(snapshotter: &metrics_util::debugging::Snapshotter) -> Vec<f64> {
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter(|(key, _, _, _)| key.key().name() == REQUEST_BODY_BYTES)
            .flat_map(|(_, _, _, value)| match value {
                DebugValue::Histogram(values) => {
                    values.into_iter().map(f64::from).collect::<Vec<_>>()
                }
                _ => panic!("{REQUEST_BODY_BYTES} must be a histogram"),
            })
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg_attr(
        miri,
        ignore = "crossbeam-epoch (via metrics-util) violates Stacked Borrows; known third-party false positive"
    )]
    async fn every_request_body_is_measured_once() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        // Unlike `with_local_recorder`, this survives the `.await` below.
        let _guard = metrics::set_default_local_recorder(&recorder);

        let client = MeteredHttpClient::new(SpyClient);
        let bodies = [
            r#"{"model":"qwen3.6","messages":[{"role":"user","content":"evolve"}]}"#,
            r#"{"model":"qwen3.6","messages":[]}"#,
        ];
        for body in bodies {
            let req = Request::post("http://localhost:8000/v1/chat/completions")
                .body(Bytes::from_static(body.as_bytes()))
                .expect("Can build request");
            let response = client
                .send::<_, Bytes>(req)
                .await
                .expect("SpyClient answers");
            assert_eq!(
                response.into_body().await.expect("Can read body").len(),
                body.len(),
                "the wrapper must forward the body unchanged"
            );
        }

        let expected: Vec<f64> = bodies.iter().map(|body| body.len() as f64).collect();
        assert_eq!(
            recorded_sizes(&snapshotter),
            expected,
            "one sample per request, sized in bytes of the serialized payload"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg_attr(
        miri,
        ignore = "crossbeam-epoch (via metrics-util) violates Stacked Borrows; known third-party false positive"
    )]
    async fn bodyless_requests_are_not_measured() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let client = MeteredHttpClient::new(SpyClient);
        let req = Request::get("http://localhost:8000/v1/key")
            .body(NoBody)
            .expect("Can build request");
        client
            .send::<_, Bytes>(req)
            .await
            .expect("SpyClient answers");

        assert!(
            recorded_sizes(&snapshotter).is_empty(),
            "a credential check carries no prompt"
        );
    }
}
