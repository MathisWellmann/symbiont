#![expect(
    unused_crate_dependencies,
    missing_docs,
    reason = "Integration tests don't use them all"
)]

use rig_core::{
    completion::{
        Chat,
        Completion,
        CompletionRequestBuilder,
    },
    providers::openrouter::{
        self,
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
        todo!()
    }
}

impl Chat for MockAgent {
    fn chat(
        &self,
        _prompt: impl Into<rig_core::message::Message> + rig_core::wasm_compat::WasmCompatSend,
        _chat_history: &mut Vec<rig_core::message::Message>,
    ) -> impl Future<Output = Result<String, rig_core::completion::PromptError>>
    + rig_core::wasm_compat::WasmCompatSend {
        async {
            Ok("```
            pub fn step(counter: &mut usize) { *counter += 5; }
            ```"
            .to_string())
        }
    }
}
