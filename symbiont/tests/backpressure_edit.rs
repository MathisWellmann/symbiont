// SPDX-License-Identifier: MPL-2.0
//! A repair round may edit the previous candidate instead of retyping it:
//! by `E<n>` anchor into the reported errors, by token-matched
//! `SEARCH`/`REPLACE`, or by sending only the item that changes. An edit
//! that does not apply is reported and the base stays as it was.
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
    Profile,
    Runtime,
};

const PROMPT: &str = "Sum the values plus an offset. Code only.";

/// Two functions, two errors: `E1` on line 3 (a `&str` where a `u64` is
/// due, primary span `"one"`), `E2` on line 8 (`f64` times integer, primary
/// span the `*` operator).
const BROKEN: &str = "fn edit_total(values: &[u64]) -> u64 {\n\
                      \x20   let base = offset();\n\
                      \x20   let bonus: u64 = \"one\";\n\
                      \x20   values.iter().sum::<u64>() + base + bonus\n\
                      }\n\
                      \n\
                      fn offset() -> u64 {\n\
                      \x20   (1.5 * 2) as u64\n\
                      }";

fn broken_reply() -> Turn {
    Turn::reply(&format!("```rust\n{BROKEN}\n```"))
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn repair_rounds_edit_the_previous_candidate() {
    symbiont::evolvable! {
        fn edit_total(values: &[u64]) -> u64 {
            values.len() as u64
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    anchors(rt).await;
    assert_eq!(edit_total(&[1, 2]), 1 + 2 + 2 + 1);

    search_replace(rt).await;
    assert_eq!(edit_total(&[1, 2]), 1 + 2 + 3 + 10);

    item_edit(rt).await;
    assert_eq!(edit_total(&[1, 2]), 1 + 2 + 7 + 100);

    failed_edit(rt).await;
    assert_eq!(edit_total(&[1, 2]), 1 + 2 + 2 + 1);
}

/// Both errors fixed by number, nothing retyped. `E2` underlines only the
/// `*`, and the nudge says so; replacing that one token with `as u64 *`
/// turns `1.5 * 2` into `1.5 as u64 * 2`.
async fn anchors(rt: &Runtime) {
    let agent = ScriptedAgent::new([
        broken_reply(),
        Turn::reply("```rust-edit\nE1 => 1\nE2 => as u64 *\n```"),
    ]);
    rt.evolve(&agent, PROMPT)
        .await
        .expect("anchored edits repair the candidate");
    assert_eq!(agent.calls(), 2);
    let nudge = agent.prompt(1);
    assert!(
        nudge.contains("[E1] replaces `\"one\"` on line 3")
            && nudge.contains("[E2] replaces `*` on line 8"),
        "each error names the text its anchor replaces, got: {nudge}"
    );
    assert_eq!(
        rt.current_code(),
        BROKEN
            .replace("\"one\"", "1")
            .replace("(1.5 * 2)", "(1.5 as u64 * 2)"),
        "the registered source is the base with exactly the two spans replaced"
    );
}

/// Search/replace, matched on tokens. The agent's hunks space the
/// expressions differently from the base and spread one over two lines;
/// each still matches exactly one place.
async fn search_replace(rt: &Runtime) {
    let agent = ScriptedAgent::new([
        broken_reply(),
        Turn::reply(
            "```rust-edit\n\
             <<<<<<< SEARCH\n\
             let bonus:u64=\n\
             \"one\";\n\
             =======\n\
             let bonus: u64 = 10;\n\
             >>>>>>> REPLACE\n\
             <<<<<<< SEARCH\n\
             (1.5*2) as u64\n\
             =======\n\
             3\n\
             >>>>>>> REPLACE\n\
             ```",
        ),
    ]);
    rt.evolve(&agent, PROMPT)
        .await
        .expect("search/replace edits repair the candidate");
    assert_eq!(agent.calls(), 2);
    let code = rt.current_code();
    assert!(
        code.contains("    let bonus: u64 = 10;\n")
            && code.contains("fn offset() -> u64 {\n    3\n}"),
        "the base's text around each match is kept: {code}"
    );
}

/// An item edit: only `offset` is sent; `edit_total` (still broken in the
/// base) is fixed by an anchor in the same response.
async fn item_edit(rt: &Runtime) {
    let agent = ScriptedAgent::new([
        broken_reply(),
        Turn::reply(
            "```rust-edit\nE1 => 100\n```\n\
             And the helper:\n\
             ```rust\nfn offset() -> u64 {\n    7\n}\n```",
        ),
    ]);
    rt.evolve(&agent, PROMPT)
        .await
        .expect("an item edit plus an anchor repair the candidate");
    assert_eq!(agent.calls(), 2);
    let code = rt.current_code();
    assert!(
        code.starts_with(
            "fn edit_total(values: &[u64]) -> u64 {\n    let base = offset();\n    let bonus: u64 = 100;\n"
        ),
        "the declared function is the base's with the anchor applied: {code}"
    );
    assert!(
        code.ends_with("fn offset() -> u64 {\n    7\n}"),
        "the helper is the response's item: {code}"
    );
}

/// An edit that does not apply. The base is unchanged, the agent is told
/// why, and the next edit works against the same base.
async fn failed_edit(rt: &Runtime) {
    let agent = ScriptedAgent::new([
        broken_reply(),
        Turn::reply("```rust-edit\nE3 => 1\n```"),
        Turn::reply("```rust-edit\nE1 => 1\nE2 => as u64 *\n```"),
    ]);
    rt.evolve(&agent, PROMPT)
        .await
        .expect("the third response repairs the candidate");
    assert_eq!(agent.calls(), 3);
    let nudge = agent.prompt(2);
    assert!(
        nudge.contains("could not be applied") && nudge.contains("E3 does not exist"),
        "the failed edit is explained, got: {nudge}"
    );
    assert!(
        nudge.contains("previous code is unchanged"),
        "the agent is told the base still stands, got: {nudge}"
    );
    let failures = rt.take_evolve_failures();
    let kinds: Vec<&str> = failures.iter().map(|f| f.kind()).collect();
    assert!(
        kinds.contains(&"edit"),
        "the failed edit is recorded as its own failure kind, got: {kinds:?}"
    );
}
