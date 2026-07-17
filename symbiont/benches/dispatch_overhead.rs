// SPDX-License-Identifier: MPL-2.0
//! Benchmark comparing direct function calls, `evolvable!` dispatch
//! (atomic pointer load + indirect call), and calls through a hoisted
//! [`symbiont::RevisionFn`] handle.

#![expect(unused_crate_dependencies, reason = "benches don't need everything.")]

use std::hint::black_box;

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

// Direct (baseline) — no indirection, just a regular function call.
#[inline(never)]
fn step_direct(counter: &mut usize) {
    *counter += 1;
}

// Evolvable — goes through RwLock read lock + libloading symbol lookup.
symbiont::evolvable! {
    fn step_evolvable(counter: &mut usize) {
        *counter += 1;
    }
}

fn bench_dispatch_overhead(c: &mut Criterion) {
    // Initialize the symbiont runtime. Compile the dylib in release so the
    // callee is as optimized as the direct baseline and the measured
    // difference is pure dispatch overhead, not debug codegen.
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        symbiont::Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Release)
            .await
            .expect("runtime init")
    });

    let mut group = c.benchmark_group("step_dispatch");

    group.bench_function("direct", |b| {
        let mut counter = 0usize;
        b.iter(|| {
            step_direct(black_box(&mut counter));
        });
    });

    group.bench_function("evolvable", |b| {
        let mut counter = 0usize;
        b.iter(|| {
            step_evolvable(black_box(&mut counter));
        });
    });

    group.bench_function("revision_handle", |b| {
        // Fetch once (registry read + Arc clone), hoist the bare fn pointer,
        // then every call is a plain indirect call — no atomic load.
        let handle =
            step_evolvable_fn(symbiont::Revision::INITIAL).expect("initial revision is retained");
        let f = handle.get();
        let mut counter = 0usize;
        b.iter(|| {
            f(black_box(&mut counter));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_dispatch_overhead);
criterion_main!(benches);
