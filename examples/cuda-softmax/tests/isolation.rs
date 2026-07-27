// SPDX-License-Identifier: MPL-2.0
//! End-to-end test of the process isolation that makes the search survivable.
//!
//! Drives the real child protocol against the example's own binary: a kernel
//! that writes out of bounds takes a child process down with it, and the
//! parent goes on to measure the next candidate as if nothing happened. That
//! is the property the whole `Isolated` design exists for, and it cannot be
//! observed from inside a single process — once a sticky CUDA error lands,
//! everything in that process stays broken.

#![allow(
    unused_crate_dependencies,
    reason = "This test only exercises the isolation protocol of the library target."
)]

use cuda_softmax_example::{
    Isolated,
    KernelPlan,
    NAIVE_KERNEL,
    ROWS,
};

/// The example's binary, which dispatches to `evaluator_main` when staged as
/// a child. `current_exe` would point at this test harness instead.
const EVALUATOR: &str = env!("CARGO_BIN_EXE_cuda-softmax-example");

fn naive_plan() -> KernelPlan {
    KernelPlan {
        source: NAIVE_KERNEL.to_string(),
        grid: (u32::try_from(ROWS.div_ceil(256)).expect("fits"), 1, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    }
}

fn out_of_bounds_plan() -> KernelPlan {
    KernelPlan {
        source: r#"
extern "C" __global__ void softmax(const float* input, float* output, int rows, int cols) {
    output[(size_t)(blockIdx.x + 1) * 1000000000ull] = 1.0f;
}
"#
        .to_string(),
        ..naive_plan()
    }
}

#[test]
fn a_faulting_kernel_costs_a_child_process_and_nothing_else() {
    let isolated = Isolated::with_executable(EVALUATOR).expect("scratch dir");

    // Establish that this machine can run the benchmark at all; without a GPU
    // the child reports a device error and there is nothing to test.
    match isolated.evaluate(&naive_plan()) {
        Ok(measurement) => assert!(measurement.gb_per_s > 0.0),
        Err(failure) => {
            eprintln!("skipping: {failure}");
            return;
        }
    }

    let failure = isolated
        .evaluate(&out_of_bounds_plan())
        .expect_err("an out-of-bounds write must not be reported as a measurement");
    assert!(
        failure.poisons_context(),
        "a dead child is a launch fault: {failure}"
    );

    // The point: the parent is untouched and the search continues.
    let after = isolated
        .evaluate(&naive_plan())
        .expect("the parent can still evaluate candidates after a fault");
    assert!(after.gb_per_s > 0.0);
}
