// SPDX-License-Identifier: MPL-2.0
//! Batch integration test for [`symbiont::Runtime::evolve_batch_stream`]: lanes
//! are yielded as they finish rather than at a barrier, each tagged with its
//! index into the prompts, and a straggler delays only itself.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

mod common;

use common::{
    ANY_PROMPT,
    RoutedAgent,
};
use futures_util::StreamExt;
use symbiont::{
    Profile,
    Runtime,
};

/// Source for a lane that returns `value`.
fn implementation(value: usize) -> String {
    format!("```rust\npub fn stream_step(counter: &mut usize) {{ *counter += {value}; }}\n```")
}

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
#[tracing_test::traced_test]
async fn lanes_are_yielded_as_they_finish() {
    symbiont::evolvable! {
        fn stream_step(counter: &mut usize) {
            *counter += 1;
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");

    // The limit is a property of the endpoint, so it is set on the runtime
    // rather than on the call — which is what lets rounds overlap.
    assert_eq!(
        rt.max_in_flight(),
        u16::MAX,
        "an unconfigured runtime admits whatever it is asked to send"
    );
    rt.set_max_in_flight(2);
    assert_eq!(rt.max_in_flight(), 2);

    let prompts = [
        "Implement the function. Code only. Hint: strategy alpha",
        "Implement the function. Code only. Hint: strategy beta",
        "Implement the function. Code only. Hint: strategy gamma",
    ];
    // Lane 1 needs a repair round: its first answer does not parse as Rust,
    // and the correction prompt drops the lane's hint, so the retry is caught
    // by the trailing catch-all route.
    let agent = RoutedAgent::new([
        ("strategy alpha", vec![implementation(10)]),
        (
            "strategy beta",
            vec!["not rust, not even a code fence".to_string()],
        ),
        ("strategy gamma", vec![implementation(30)]),
        (ANY_PROMPT, vec![implementation(20)]),
    ]);

    let mut order = Vec::new();
    let mut values = vec![None; prompts.len()];
    {
        let mut lanes = std::pin::pin!(rt.evolve_batch_stream(&agent, &prompts));
        while let Some((lane, result)) = lanes.next().await {
            order.push(lane);
            values[lane] =
                Some(result.expect("every lane is eventually given a compiling implementation"));
        }
    }

    assert_eq!(order.len(), prompts.len(), "each lane yields exactly once");
    let mut seen = order.clone();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2], "every lane is accounted for, once");

    // The tag is what makes completion order usable: it is the only way back
    // to the prompt that produced the result.
    for (lane, expected) in [(0_usize, 10_usize), (1, 20), (2, 30)] {
        let info = values[lane]
            .as_ref()
            .expect("every lane yielded a result above");
        let handle = stream_step_fn(info.revision()).expect("registered revisions are retained");
        let mut counter = 0;
        (handle.get())(&mut counter);
        assert_eq!(
            counter, expected,
            "lane {lane} must be tagged with the result of its own prompt"
        );
    }

    // No assertion on `order` itself: completion order is the point of this
    // API and it is deliberately not deterministic. The repair lane does not
    // reliably come last either — its second attempt outranks the untouched
    // lanes at the gate, which is exactly what the priority is for.
    assert_eq!(agent.calls(), 4, "three lanes, one of them retried once");

    // The stream deliberately leaves the failure buffer alone: clearing it
    // would discard the records of an overlapping round.
    let failures = rt.take_evolve_failures();
    assert_eq!(failures.len(), 1, "the rejected answer of lane 1");
    assert_eq!(failures[0].lane(), 1);
}
