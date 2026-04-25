//! Benchmark comparing direct function calls vs `evolvable!` dispatch
//! to measure the overhead of the RwLock + dlsym wrapper.

#![expect(
    unused_crate_dependencies,
    missing_docs,
    reason = "benches don't need everything."
)]

use criterion::{
    Criterion,
    black_box,
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
    // Initialize the symbiont runtime (compiles the temp dylib).
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        symbiont::Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Debug)
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

    group.finish();
}

criterion_group!(benches, bench_dispatch_overhead);
criterion_main!(benches);
