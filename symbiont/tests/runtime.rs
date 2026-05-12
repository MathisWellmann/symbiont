#![expect(
    unused_crate_dependencies,
    missing_docs,
    reason = "Integration tests don't use them all"
)]

use rig::completion::Prompt;
use symbiont::{
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
    let rt = Runtime::init(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");
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
    assert_eq!(
        &rt.current_code(),
        "#[unsafe(no_mangle)]\npub fn step(counter: &mut usize) {\n    *counter += 5;\n}\n",
        "Code has evolved"
    );
    step(&mut counter);
    assert_eq!(counter, 6);
}

struct MockAgent;

impl Prompt for MockAgent {
    fn prompt(
        &self,
        _prompt: impl Into<rig::message::Message> + rig::wasm_compat::WasmCompatSend,
    ) -> impl IntoFuture<
        Output = Result<String, rig::completion::PromptError>,
        IntoFuture: rig::wasm_compat::WasmCompatSend,
    > {
        async {
            Ok("```
            pub fn step(counter: &mut usize) { *counter += 5; }
            ```"
            .to_string())
        }
    }
}
