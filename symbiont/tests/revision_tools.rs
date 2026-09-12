// SPDX-License-Identifier: MPL-2.0
//! The revision tools: the pipeline as tool calls, inside one agent run.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton,
//! so every scenario runs in sequence in the one test below.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use std::sync::{
    Arc,
    Mutex,
};

use common::{
    ScriptedAgent,
    Turn,
};
use rig_core::tool::PortableTool;
use symbiont::{
    BuildRecord,
    BuildRevisionArgs,
    BuildRevisionTool,
    EditRevisionArgs,
    EditRevisionTool,
    Profile,
    Revision,
    RevisionToolError,
    Runtime,
    SubmitRevisionArgs,
    SubmitRevisionTool,
    ToolBuildOutcome,
};

/// What the scripted model read back from its tool calls, in call order.
type Verdicts = Arc<Mutex<Vec<String>>>;

async fn build(rt: &'static Runtime, code: &str) -> Result<String, RevisionToolError> {
    PortableTool::call(&BuildRevisionTool::new(rt), BuildRevisionArgs::new(code)).await
}

async fn edit(rt: &'static Runtime, args: EditRevisionArgs) -> Result<String, RevisionToolError> {
    PortableTool::call(&EditRevisionTool::new(rt), args).await
}

async fn submit(revision: u64) -> Result<String, RevisionToolError> {
    PortableTool::call(
        &SubmitRevisionTool,
        SubmitRevisionArgs::new(Revision::new(revision)),
    )
    .await
}

/// The model's first run: it builds through the tool, reads each verdict,
/// and answers with the code that built.
async fn build_reject_repeat_register(rt: &'static Runtime, seen: Verdicts) -> String {
    let record = |verdict: String| seen.lock().expect("not poisoned").push(verdict);
    // An invented helper: a compile error rustc has no fix for.
    let broken = "fn tool_step(x: f64) -> f64 { triple(x) }";
    record(build(rt, broken).await.expect("a rejection is an answer"));
    // The same code again is answered from memory.
    record(build(rt, broken).await.expect("a repeat is an answer"));
    // A fenced candidate is unwrapped; this one builds.
    record(
        build(rt, "```rust\nfn tool_step(x: f64) -> f64 { x * 3.0 }\n```")
            .await
            .expect("a registration is an answer"),
    );
    // A parse failure costs no build.
    record(
        build(rt, "fn tool_step(x: f64) -> f64 { x * ")
            .await
            .expect("a parse rejection is an answer"),
    );
    // Code that is already a registered revision costs no build either.
    record(
        build(rt, "fn tool_step(x: f64) -> f64 { x * 3.0 }")
            .await
            .expect("a deduplication is an answer"),
    );
    // The model answers with the code it built: the response path finds it
    // registered already.
    "Built and checked.\n```rust\nfn tool_step(x: f64) -> f64 { x * 3.0 }\n```".to_string()
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
#[expect(
    clippy::too_many_lines,
    reason = "The Runtime singleton allows one runtime per process, so every scenario lives in one sequential test"
)]
async fn revision_tools_drive_the_pipeline_from_inside_a_run() {
    symbiont::evolvable! {
        fn tool_step(x: f64) -> f64 {
            x
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init");

    // -- Outside an evolution the tools have no lane and refuse -------------

    let err = build(rt, "fn tool_step(x: f64) -> f64 { x * 2.0 }")
        .await
        .expect_err("no lane is attached outside `evolve`");
    assert_eq!(err, RevisionToolError::OutsideEvolve);
    assert_eq!(rt.revision_count(), 1, "nothing was built");

    // -- `build_revision`: reject, remember, register, then answer ----------

    let verdicts: Verdicts = Arc::default();
    let seen = Arc::clone(&verdicts);
    let agent = ScriptedAgent::new([Turn::with_tools(move || {
        build_reject_repeat_register(rt, seen)
    })]);
    let info = rt
        .evolve(&agent, "Triple the input.")
        .await
        .expect("the run ends in a registered revision");
    assert_eq!(info.revision(), Revision::new(1));
    assert_eq!(rt.active_revision(), Revision::new(1));
    assert_eq!(tool_step(2.0), 6.0, "the tool-built revision is active");
    assert_eq!(rt.revision_count(), 2, "one revision, built once");

    let verdicts = verdicts.lock().expect("not poisoned").clone();
    assert_eq!(verdicts.len(), 5);
    assert!(
        verdicts[0].contains("failed to compile") && verdicts[0].contains("[E1]"),
        "the compile verdict is the response nudge: {}",
        verdicts[0]
    );
    assert!(
        verdicts[1].starts_with("You already sent this exact code"),
        "{}",
        verdicts[1]
    );
    assert!(
        verdicts[1].ends_with(&verdicts[0]),
        "the repeat quotes the earlier verdict"
    );
    assert!(
        verdicts[2].starts_with("Registered revision 1.\n"),
        "{}",
        verdicts[2]
    );
    assert!(
        verdicts[2].contains("Revisions built in this lane: 1. Build budget: 2 of 10 used."),
        "the repeat and the parse failure spent no build: {}",
        verdicts[2]
    );
    assert!(
        verdicts[3].contains("not valid Rust"),
        "the parse verdict is the response nudge: {}",
        verdicts[3]
    );
    assert!(
        verdicts[4].starts_with("Your code is byte-identical to revision 1")
            && verdicts[4].contains("Build budget: 2 of 10 used."),
        "a deduplicated candidate is refunded its build: {}",
        verdicts[4]
    );

    // The trace records every tool build under the attempt whose run made
    // it, and the response's own build as a deduplication.
    let trace = info.trace();
    assert_eq!(trace.attempts().len(), 1);
    let stages = trace.attempts()[0].stages();
    let outcomes: Vec<&ToolBuildOutcome> = stages
        .tool_builds()
        .iter()
        .map(|build| build.outcome())
        .collect();
    assert_eq!(outcomes.len(), 5);
    assert!(
        matches!(outcomes[0], ToolBuildOutcome::Rejected { kind, .. } if kind == "compile"),
        "{outcomes:?}"
    );
    assert_eq!(outcomes[1], &ToolBuildOutcome::Repeated);
    assert_eq!(
        outcomes[2],
        &ToolBuildOutcome::Registered {
            revision: Revision::new(1)
        }
    );
    assert!(
        matches!(outcomes[3], ToolBuildOutcome::Rejected { kind, .. } if kind == "parse"),
        "{outcomes:?}"
    );
    assert_eq!(
        outcomes[4],
        &ToolBuildOutcome::Registered {
            revision: Revision::new(1)
        }
    );
    assert!(
        stages
            .tool_builds()
            .iter()
            .all(|build| build.tool() == "build_revision"),
        "{stages:?}"
    );
    assert!(
        matches!(
            stages.build(),
            Some(BuildRecord::Deduped { revision, .. }) if *revision == Revision::new(1)
        ),
        "the response's code was the tool-built revision: {:?}",
        stages.build()
    );

    // -- `submit_revision`: two candidates, a choice, no code in the reply --

    let verdicts: Verdicts = Arc::default();
    let seen = Arc::clone(&verdicts);
    let agent = ScriptedAgent::new([Turn::with_tools(move || build_two_and_choose(rt, seen))]);
    let info = rt
        .evolve(&agent, "Try two variants and keep the better one.")
        .await
        .expect("the choice is the answer");
    assert_eq!(
        info.revision(),
        Revision::new(3),
        "the chosen one, not the last built"
    );
    assert_eq!(tool_step(2.0), 12.0, "the chosen revision is active");
    assert_eq!(rt.revision_count(), 4);
    let verdicts = verdicts.lock().expect("not poisoned").clone();
    assert!(
        matches!(&verdicts[..], [_, _, refused, chosen]
            if refused.contains("not built in this lane") && chosen.starts_with("Revision 3 is chosen.")),
        "{verdicts:#?}"
    );
    let stages = info.trace().attempts()[0].stages();
    assert_eq!(stages.tool_builds().len(), 2);
    assert!(
        stages.build().is_none(),
        "a chosen revision spends no build in the response path: {:?}",
        stages.build()
    );

    // -- A run that builds but does not choose is nudged for the choice ----

    let agent = ScriptedAgent::new([
        Turn::with_tools(move || build_and_forget(rt)),
        // The nudge names the text form; the model uses it.
        Turn::reply("Keeping the faster one.\nrevision: 4"),
    ]);
    let info = rt
        .evolve(&agent, "Halve the input.")
        .await
        .expect("the text form of the choice is accepted");
    assert_eq!(info.revision(), Revision::new(4));
    assert_eq!(agent.calls(), 2);
    let nudge = agent.prompt(1);
    assert!(
        nudge.contains("You built revisions 4 with the tools but did not choose one"),
        "{nudge}"
    );
    assert!(nudge.contains("`revision: N`"), "{nudge}");
    assert!(
        matches!(
            info.trace().attempts()[0].ladder(),
            symbiont::LadderEvent::SelfHeal { kind, .. } if kind == "unsubmitted"
        ),
        "{:?}",
        info.trace().attempts()[0].ladder()
    );

    // -- `edit_revision`: anchors on the last verdict, hunks on a revision --

    let verdicts: Verdicts = Arc::default();
    let seen = Arc::clone(&verdicts);
    let agent = ScriptedAgent::new([Turn::with_tools(move || edit_repair_and_vary(rt, seen))]);
    let info = rt
        .evolve(&agent, "Scale the input.")
        .await
        .expect("the edited candidate is chosen");
    assert_eq!(info.revision(), Revision::new(6));
    assert_eq!(tool_step(1.0), 8.0);
    let verdicts = verdicts.lock().expect("not poisoned").clone();
    assert_eq!(verdicts.len(), 7, "{verdicts:#?}");
    assert!(verdicts[0].contains("nothing to edit"), "{}", verdicts[0]);
    assert!(
        verdicts[1].contains("[E1] replaces `scale`"),
        "the verdict names the anchor: {}",
        verdicts[1]
    );
    assert!(
        verdicts[2].starts_with("Registered revision 5."),
        "the anchor repaired the last candidate: {}",
        verdicts[2]
    );
    assert!(
        verdicts[3].starts_with("Registered revision 6."),
        "the hunk edited a registered revision: {}",
        verdicts[3]
    );
    assert!(
        verdicts[4].contains("could not be applied"),
        "a search without a match is the edit nudge: {}",
        verdicts[4]
    );
    assert!(
        verdicts[5].contains("revision 99 is not registered"),
        "{}",
        verdicts[5]
    );
    assert_eq!(
        rt.revision_code(Revision::new(6)).as_deref(),
        Some("fn tool_step(x: f64) -> f64 { x * 8.0 }")
    );
    let stages = info.trace().attempts()[0].stages();
    let tools: Vec<&str> = stages
        .tool_builds()
        .iter()
        .map(|build| build.tool().as_str())
        .collect();
    assert_eq!(
        tools,
        [
            "build_revision",
            "edit_revision",
            "edit_revision",
            "edit_revision"
        ],
        "a refused call is no build"
    );
    assert_eq!(
        stages.tool_builds()[1]
            .stages()
            .edits()
            .map(|edits| edits.anchors),
        Some(1)
    );
    assert_eq!(
        stages.tool_builds()[2]
            .stages()
            .edits()
            .map(|edits| edits.hunks),
        Some(1)
    );
}

/// A broken candidate, repaired by an anchor on its verdict, then varied
/// by a hunk against the registered revision; two refusals on the way.
async fn edit_repair_and_vary(rt: &'static Runtime, seen: Verdicts) -> String {
    let record = |verdict: String| seen.lock().expect("not poisoned").push(verdict);
    // A fresh lane has nothing to edit yet.
    record(
        edit(rt, EditRevisionArgs::new("E1 => 7.0"))
            .await
            .expect_err("no base yet")
            .to_string(),
    );
    // An undefined name: the compiler underlines exactly `scale`.
    record(
        build(rt, "fn tool_step(x: f64) -> f64 { x * scale }")
            .await
            .expect("rejected"),
    );
    record(
        edit(rt, EditRevisionArgs::new("E1 => 7.0"))
            .await
            .expect("registered"),
    );
    record(
        edit(
            rt,
            EditRevisionArgs::against(
                Revision::new(5),
                "<<<<<<< SEARCH\n7.0\n=======\n8.0\n>>>>>>> REPLACE",
            ),
        )
        .await
        .expect("registered"),
    );
    record(
        edit(
            rt,
            EditRevisionArgs::new("<<<<<<< SEARCH\n9.0\n=======\n10.0\n>>>>>>> REPLACE"),
        )
        .await
        .expect("a failed edit is an answer"),
    );
    record(
        edit(
            rt,
            EditRevisionArgs::against(Revision::new(99), "E1 => 1.0"),
        )
        .await
        .expect_err("not registered")
        .to_string(),
    );
    record(submit(6).await.expect("built here"));
    "Revision 6 scales by eight.".to_string()
}

/// Two candidates through the tool; the model then tries to choose a
/// revision it did not build, and chooses one it did.
async fn build_two_and_choose(rt: &'static Runtime, seen: Verdicts) -> String {
    let record = |verdict: String| seen.lock().expect("not poisoned").push(verdict);
    record(
        build(rt, "fn tool_step(x: f64) -> f64 { x * 5.0 }")
            .await
            .expect("registered"),
    );
    record(
        build(rt, "fn tool_step(x: f64) -> f64 { x * 6.0 }")
            .await
            .expect("registered"),
    );
    record(
        submit(1)
            .await
            .expect_err("revision 1 belongs to the earlier lane")
            .to_string(),
    );
    record(submit(3).await.expect("built here"));
    "Revision 3 multiplies by six, which the task asked for.".to_string()
}

/// One candidate through the tool, then a reply that neither chooses it
/// nor carries code.
async fn build_and_forget(rt: &'static Runtime) -> String {
    build(rt, "fn tool_step(x: f64) -> f64 { x / 2.0 }")
        .await
        .expect("registered");
    "I built a candidate that halves the input.".to_string()
}
