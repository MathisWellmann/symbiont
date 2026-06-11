#![expect(
    unused_crate_dependencies,
    missing_docs,
    reason = "Integration tests don't use them all"
)]

use rig_core::{
    completion::{
        PromptError,
        Usage,
    },
    message::Message,
};
use symbiont::{
    AgentRun,
    EvolutionAgent,
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

impl EvolutionAgent for MockAgent {
    async fn run(&self, prompt: &str, _history: Vec<Message>) -> Result<AgentRun, PromptError> {
        Ok(AgentRun {
            output: MOCK_LLM_REPLY.to_string(),
            new_messages: vec![Message::user(prompt), Message::assistant(MOCK_LLM_REPLY)],
            usage: Usage::new(),
        })
    }
}
