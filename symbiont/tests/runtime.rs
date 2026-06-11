#![expect(
    unused_crate_dependencies,
    missing_docs,
    reason = "Integration tests don't use them all"
)]

use rig_core::{
    client::CompletionClient,
    completion::{
        Chat,
        Completion,
        CompletionRequestBuilder,
    },
    providers::openrouter::{
        Client,
        CompletionModel,
    },
};
use symbiont::{
    FullSource,
    Profile,
    Runtime,
};

#[tokio::test]
#[tracing_test::traced_test]
async fn runtime() {
    symbiont::evolvable! {
        /// Should increment the counter by a value in the range 5..20
        fn step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");
    assert_eq!(&rt.fn_sigs(), &["fn step(counter: &mut usize)".to_string()]);
    assert_eq!(
        &rt.fn_full_sources(),
        &[FullSource(
            "/// Should increment the counter by a value in the range 5..20\n#[unsafe(no_mangle)]\npub fn step(counter: &mut usize) {\n    *counter += 1;\n}\n"
        )]
    );
    assert_eq!(
        rt.fn_prelude(),
        Vec::new(),
        "No prelude items in this function."
    );
    assert_eq!(
        &rt.current_code(),
        "/// Should increment the counter by a value in the range 5..20\n#[unsafe(no_mangle)]\npub fn step(counter: &mut usize) {\n    *counter += 1;\n}\n\n\n"
    );
    let mut counter = 0;
    step(&mut counter);
    assert_eq!(counter, 1);

    let agent = MockAgent;
    let prompt = format!("Implement this function in rust: ```{}```", rt.fn_sigs()[0]);
    rt.evolve(&agent, &prompt).await.expect("Can evolve");
    assert_eq!(&rt.fn_sigs(), &["fn step(counter: &mut usize)".to_string()]);
    assert_eq!(
        &rt.fn_full_sources(),
        &[FullSource(
            "/// Should increment the counter by a value in the range 5..20\n#[unsafe(no_mangle)]\npub fn step(counter: &mut usize) {\n    *counter += 1;\n}\n"
        )]
    );
    assert_eq!(
        &rt.current_code(),
        "#[unsafe(no_mangle)]\npub fn step(counter: &mut usize) {\n    *counter += 5;\n}\n",
        "Code has evolved"
    );
    assert_eq!(
        rt.fn_prelude(),
        Vec::new(),
        "No prelude items in this function."
    );
    step(&mut counter);
    assert_eq!(counter, 6);
}

struct MockAgent;

/// The canned LLM reply containing the evolved function.
const MOCK_LLM_REPLY: &str = "```
            pub fn step(counter: &mut usize) { *counter += 5; }
            ```";

/// Spawn a one-shot HTTP server which answers any request with a canned
/// `OpenRouter` completion response containing [`MOCK_LLM_REPLY`].
///
/// Returns the base URL of the server.
fn spawn_mock_completion_server() -> String {
    use std::io::{
        Read,
        Write,
    };

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("Can bind a local TCP port");
    let addr = listener.local_addr().expect("Listener has a local address");

    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("Can accept a connection");

        // Read the full request (headers + body) before responding.
        let mut request = Vec::new();
        let mut buf = [0_u8; 4096];
        loop {
            let n = stream.read(&mut buf).expect("Can read from the connection");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if let Some(end_of_headers) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end_of_headers]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if request.len() >= end_of_headers + 4 + content_length {
                    break;
                }
            }
        }

        let body = serde_json::json!({
            "id": "mock-completion-id",
            "object": "chat.completion",
            "created": 0,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "native_finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": MOCK_LLM_REPLY,
                },
                "finish_reason": "stop",
            }],
            "system_fingerprint": null,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(response.as_bytes())
            .expect("Can write the response");
        stream.flush().expect("Can flush the response");
    });

    format!("http://{addr}")
}

impl Completion<CompletionModel> for MockAgent {
    fn completion<I, T>(
        &self,
        prompt: impl Into<rig_core::message::Message> + rig_core::wasm_compat::WasmCompatSend,
        chat_history: I,
    ) -> impl Future<
        Output = Result<
            CompletionRequestBuilder<CompletionModel>,
            rig_core::completion::CompletionError,
        >,
    > + rig_core::wasm_compat::WasmCompatSend
    where
        I: IntoIterator<Item = T> + rig_core::wasm_compat::WasmCompatSend,
        T: Into<rig_core::message::Message>,
    {
        let prompt = prompt.into();
        let history = Vec::from_iter(chat_history.into_iter().map(Into::into));
        async move {
            let base_url = spawn_mock_completion_server();
            let client: Client = Client::builder()
                .api_key("mock-api-key")
                .base_url(&base_url)
                .build()
                .expect("Can build the mock OpenRouter client");
            let model = client.completion_model("mock-model");
            Ok(CompletionRequestBuilder::new(model, prompt).messages(history))
        }
    }
}

impl Chat for MockAgent {
    fn chat(
        &self,
        _prompt: impl Into<rig_core::message::Message> + rig_core::wasm_compat::WasmCompatSend,
        _chat_history: &mut Vec<rig_core::message::Message>,
    ) -> impl Future<Output = Result<String, rig_core::completion::PromptError>>
    + rig_core::wasm_compat::WasmCompatSend {
        async { Ok(MOCK_LLM_REPLY.to_string()) }
    }
}
