// SPDX-License-Identifier: MPL-2.0
//! Compiler suggestions rustc marks `MachineApplicable` are applied by the
//! harness before the model is asked. A candidate whose only errors are of
//! that kind compiles without a second inference round, and the registered
//! source is the patched text; a candidate with other errors left gets those
//! errors back together with the list of fixes already applied.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use common::{
    ScriptedAgent,
    Turn,
};
use symbiont::{
    BuildRecord,
    Profile,
    Runtime,
};

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn machine_applicable_suggestions_are_applied_before_asking_the_model() {
    symbiont::evolvable! {
        fn autofix_total(values: &[u64]) -> u64 {
            values.len() as u64
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // Two slips rustc fixes with `MachineApplicable` suggestions: an integer
    // literal where a float is expected (`2` -> `2.0`) and a missing
    // dereference (`r` -> `*r`).
    let slips = "fn autofix_total(values: &[u64]) -> u64 {\n\
                 \x20   let scale: f64 = 2;\n\
                 \x20   let first: u64 = values.first().copied().unwrap_or(0);\n\
                 \x20   let r: &u64 = &first;\n\
                 \x20   let plain: u64 = r;\n\
                 \x20   values.iter().sum::<u64>() * scale as u64 + plain\n\
                 }";
    let agent = ScriptedAgent::new([Turn::reply(&format!("```rust\n{slips}\n```"))]);

    let trace = rt
        .evolve(&agent, "Sum the values. Code only.")
        .await
        .expect("the harness applies the compiler's fixes itself")
        .into_trace();
    assert_eq!(
        agent.calls(),
        1,
        "no second inference round for a mechanical fix"
    );
    // The build record of the one attempt names both fixes, in source order.
    match trace.attempts()[0].stages().build() {
        Some(BuildRecord::Built { autofixes, .. }) => {
            assert_eq!(autofixes.len(), 2, "{autofixes:#?}");
            assert_eq!((autofixes[0].line, autofixes[1].line), (2, 5));
            assert_eq!(autofixes[0].code.as_deref(), Some("E0308"));
            assert_eq!(autofixes[0].after.trim(), "let scale: f64 = 2.0;");
        }
        other => panic!("expected a built record with autofixes, got {other:?}"),
    }

    let code = rt.current_code();
    assert_eq!(
        code,
        slips.replace("= 2;", "= 2.0;").replace("= r;", "= *r;"),
        "the registered source is the patched candidate, otherwise untouched"
    );
    assert_eq!(autofix_total(&[3, 3, 4]), (3 + 3 + 4) * 2 + 3);
    assert_eq!(rt.take_panic(), None);
    assert!(
        rt.take_evolve_failures().is_empty(),
        "an autofixed build is not a failure the model has to see"
    );

    // A candidate with a mechanical slip *and* a real error: the fix is
    // applied, the real error is reported, and the model is told about the
    // fix so its picture of the code matches what was compiled.
    let mixed = "fn autofix_total(values: &[u64]) -> u64 {\n\
                 \x20   let scale: f64 = 2;\n\
                 \x20   let wrong: u64 = \"definitely not a u64\";\n\
                 \x20   values.iter().sum::<u64>() * scale as u64 + wrong\n\
                 }";
    let agent = ScriptedAgent::new([
        Turn::reply(&format!("```rust\n{mixed}\n```")),
        Turn::reply(
            "```rust\nfn autofix_total(values: &[u64]) -> u64 { values.iter().sum() }\n```",
        ),
    ]);
    rt.evolve(&agent, "Sum the values. Code only.")
        .await
        .expect("the model fixes the real error on the retry");
    assert_eq!(agent.calls(), 2);

    let retry_prompt = agent.prompt(1);
    assert!(
        retry_prompt.contains("already applied"),
        "the nudge must list the fixes the harness applied, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains(
            "- line 2 (use a float literal for E0308):\n    - `let scale: f64 = 2;`\n    + `let scale: f64 = 2.0;`"
        ),
        "the applied fix must be spelled out, got: {retry_prompt}"
    );
    assert!(
        retry_prompt.contains("definitely not a u64"),
        "the remaining error must be reported, got: {retry_prompt}"
    );
    let errors = retry_prompt
        .split("still failed to compile")
        .nth(1)
        .expect("the nudge separates the fixes from the remaining errors");
    assert!(
        errors.contains("[E1]") && !errors.contains("[E2]"),
        "exactly the one remaining error is reported, got: {errors}"
    );
    assert!(
        !errors.contains("f64"),
        "the fixed error must not be reported again, got: {errors}"
    );
    // The failure record holds the patched candidate: that is the text the
    // recorded diagnostics are located in, so spans index it directly.
    let failures = rt.take_evolve_failures();
    assert_eq!(failures.len(), 1);
    assert!(
        failures[0]
            .generated_code()
            .contains("let scale: f64 = 2.0;"),
        "the failure record holds the text the diagnostics refer to: {}",
        failures[0].generated_code()
    );
    assert_eq!(autofix_total(&[3, 3, 4]), 10);
}
