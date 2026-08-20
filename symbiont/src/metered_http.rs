// SPDX-License-Identifier: MPL-2.0
//! [`MeteredHttpClient`]: the HTTP backend decorator that measures — and
//! admits — what is actually sent to the inference endpoint.
//!
//! Token usage is only known after the fact, is reported by the provider, and
//! [`crate::AgentRun::usage`] aggregates it over every turn of an agentic run
//! — none of which answers "how big was the request we just sent?". This
//! decorator answers it at the only place where the payload is complete:
//! immediately before it goes on the wire.
//!
//! That same place is the only one that knows a request *exists*, which is why
//! the concurrency limit is enforced here too. See [`crate::InferenceGate`].

use std::{
    pin::Pin,
    task::{
        Context,
        Poll,
    },
};

use bytes::Bytes;
use futures_util::Stream;
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
        sse::BoxedStream,
    },
    wasm_compat::WasmCompatSend,
};

use crate::{
    inference_gate::{
        GatePermit,
        GateScope,
    },
    observability::REQUEST_BODY_BYTES,
};

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
///     providers::openrouter,
///     http_client::ReqwestClient,
/// };
/// use rig_agent::client::AgentClientExt;
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
///
/// # What is admitted
///
/// If the calling task runs inside an inference-gate scope — which is what
/// [`crate::Runtime`] puts every evolution attempt in — the request first waits
/// for one of the gate's slots and holds it until the response body has been
/// read. Outside such a scope nothing is gated and this is a pure measurement
/// decorator.
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
        // Read the ambient scope *here*, synchronously on the caller's task:
        // the future this returns is `'static` and may be polled anywhere,
        // where the task-local would no longer be visible.
        let scope = GateScope::current();
        // Constructing the inner future performs no I/O — the backend only
        // assembles the request — so building it before acquiring is safe.
        let send = self.inner.send(measured(req));
        async move {
            let permit = match scope {
                Some(scope) => Some(scope.acquire().await),
                None => None,
            };
            // On error the permit drops here, releasing the slot.
            let (parts, body) = send.await?.into_parts();
            // The completion is streamed into the body, so the request still
            // occupies the endpoint after the headers arrive. Move the permit
            // into the body future rather than releasing it early.
            let body: LazyBody<U> = Box::pin(async move {
                let body = body.await;
                drop(permit);
                body
            });
            Ok(Response::from_parts(parts, body))
        }
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
        let scope = GateScope::current();
        let send = self.inner.send_streaming(measured(req));
        async move {
            let permit = match scope {
                Some(scope) => Some(scope.acquire().await),
                None => None,
            };
            let (parts, body) = send.await?.into_parts();
            // Held until the event stream ends or its consumer drops it,
            // which is when this request stops occupying the endpoint.
            let body: BoxedStream = Box::pin(GatedStream { body, permit });
            Ok(Response::from_parts(parts, body))
        }
    }
}

/// A response stream that keeps its [`GatePermit`] alive for as long as the
/// response is still arriving.
struct GatedStream<S: ?Sized> {
    body: Pin<Box<S>>,
    /// Dropped with the stream, releasing the slot.
    permit: Option<GatePermit>,
}

impl<S> Stream for GatedStream<S>
where
    S: Stream + ?Sized,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Safe without `unsafe`: the only pinned field is already behind its
        // own `Box`, and `permit` is `Unpin`.
        let this = self.as_mut().get_mut();
        let polled = this.body.as_mut().poll_next(cx);
        if matches!(polled, Poll::Ready(None)) {
            // Release at the end of the stream rather than waiting for the
            // consumer to get around to dropping it.
            this.permit = None;
        }
        polled
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
    use crate::inference_gate::{
        InferenceGate,
        Priority,
    };

    /// An [`HttpClientExt`] that answers every request with `200 OK` and an
    /// empty body, and remembers the body size it was handed.
    #[derive(Debug, Clone, Copy, Default)]
    struct MockClient;

    impl HttpClientExt for MockClient {
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

        let client = MeteredHttpClient::new(MockClient);
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

    /// Poll `fut` exactly once. `futures_util`'s `poll!` needs its
    /// `async-await` feature, which this crate does not enable.
    fn poll_once<F: Future>(fut: Pin<&mut F>) -> Poll<F::Output> {
        fut.poll(&mut Context::from_waker(std::task::Waker::noop()))
    }

    fn chat_request() -> Request<Bytes> {
        Request::post("http://localhost:8000/v1/chat/completions")
            .body(Bytes::from_static(br#"{"model":"qwen3.6"}"#))
            .expect("Can build request")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_gated_request_holds_its_slot_until_the_body_is_read() {
        let gate = InferenceGate::new(1);
        let client = MeteredHttpClient::new(MockClient);

        gate.scope(Priority::FIRST_ATTEMPT, async {
            let response = client
                .send::<_, Bytes>(chat_request())
                .await
                .expect("SpyClient answers");
            assert_eq!(
                gate.in_flight(),
                1,
                "the completion arrives in the body, so headers do not end the request"
            );

            // A second request must wait: the first still occupies the
            // endpoint even though its future has already resolved.
            let mut second = Box::pin(client.send::<_, Bytes>(chat_request()));
            assert!(poll_once(second.as_mut()).is_pending());
            assert_eq!(gate.queued(), 1);

            response.into_body().await.expect("Can read body");
            second
                .await
                .expect("SpyClient answers")
                .into_body()
                .await
                .expect("Can read body");
        })
        .await;

        assert_eq!(gate.in_flight(), 0, "every slot is returned");
        assert_eq!(gate.queued(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_dropped_response_releases_the_slot() {
        let gate = InferenceGate::new(1);
        let client = MeteredHttpClient::new(MockClient);

        gate.scope(Priority::FIRST_ATTEMPT, async {
            let response = client
                .send::<_, Bytes>(chat_request())
                .await
                .expect("SpyClient answers");
            assert_eq!(gate.in_flight(), 1);
            // A caller that never reads the body must not leak the slot.
            drop(response);
            assert_eq!(gate.in_flight(), 0);
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn requests_outside_a_scope_are_not_gated() {
        let gate = InferenceGate::new(1);
        let client = MeteredHttpClient::new(MockClient);

        // No `gate.scope`, so the decorator is pure measurement and the two
        // responses can be held open simultaneously.
        let first = client
            .send::<_, Bytes>(chat_request())
            .await
            .expect("SpyClient answers");
        let second = client
            .send::<_, Bytes>(chat_request())
            .await
            .expect("SpyClient answers");
        assert_eq!(gate.in_flight(), 0, "an unscoped request takes no slot");
        drop((first, second));
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

        let client = MeteredHttpClient::new(MockClient);
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
