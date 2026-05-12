// SPDX-License-Identifier: MPL-2.0
//! Test-driven function evolution: the LLM must satisfy a test suite.
//!
//! Declares an evolvable `fizzbuzz` function that classifies numbers by
//! divisibility rules. The harness runs the function against a battery of
//! test cases after each evolution, feeds pass/fail results back into the
//! prompt, and lets the constrained-generation loop converge on a correct
//! implementation.
//!
//! This showcases symbiont's core strength: the LLM receives **compiler
//! errors** (via backpressure) *and* **logical errors** (via the test
//! report) as feedback, iterating until the code is both valid Rust and
//! semantically correct.

use symbiont::Runtime;
use tracing::{
    info,
    warn,
};

// Encoding:
//   0 → the number itself (not divisible by 3 or 5)
//   1 → Fizz  (divisible by 3 only)
//   2 → Buzz  (divisible by 5 only)
//   3 → FizzBuzz (divisible by both 3 and 5)
//
// The default body is intentionally wrong — always returns 0.
// The LLM must evolve it to pass the test suite.
symbiont::evolvable! {
    fn fizzbuzz(n: usize) -> usize {
        let _ = n;
        0
    }
}

/// The ground-truth oracle for fizzbuzz classification.
fn expected(n: usize) -> usize {
    match (n.is_multiple_of(3), n.is_multiple_of(5)) {
        (true, true) => 3,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 0,
    }
}

/// Human-readable label for an encoded fizzbuzz result.
fn label(code: usize) -> &'static str {
    match code {
        0 => "Number",
        1 => "Fizz",
        2 => "Buzz",
        3 => "FizzBuzz",
        _ => "???",
    }
}

/// Run the test suite and return (passed, total, report_string).
fn run_tests() -> (usize, usize, String) {
    let test_inputs = Vec::<usize>::from_iter(1..=30);
    let total = test_inputs.len();
    let mut passed = 0;
    let mut failures = Vec::new();

    for &n in &test_inputs {
        let got = fizzbuzz(n);
        let want = expected(n);
        if got == want {
            passed += 1;
        } else {
            failures.push(format!(
                "  n={n:>2}: got {got} ({}) expected {want} ({})",
                label(got),
                label(want),
            ));
        }
    }

    let mut report = format!("Test results: {passed}/{total} passed.\n");
    if failures.is_empty() {
        report.push_str("All tests passed!");
    } else {
        report.push_str("Failures:\n");
        for f in &failures {
            report.push_str(f);
            report.push('\n');
        }
    }

    (passed, total, report)
}

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    let runtime = Runtime::init(SYMBIONT_DECLS, SYMBIONT_PRELUDE, symbiont::Profile::Debug).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;

    // -- Round 0: run the default (wrong) implementation ----------------
    println!("\n=== Round 0: default implementation ===");
    let (mut passed, mut total, mut report) = run_tests();
    println!("{report}");

    if passed == total {
        println!("Default implementation already correct — nothing to evolve.");
        return Ok(());
    }

    // -- Evolution loop --------------------------------------------------
    let max_rounds = 5;
    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM ===");

        // Build a prompt that includes the function signature, the encoding
        // contract, and the concrete test failures from the previous run.
        let prompt = format!(
            "Implement this function:\n\
             ```\n{sig}\n```\n\n\
             Encoding contract:\n\
             - Return 0 if `n` is not divisible by 3 or 5\n\
             - Return 1 if `n` is divisible by 3 only  (Fizz)\n\
             - Return 2 if `n` is divisible by 5 only  (Buzz)\n\
             - Return 3 if `n` is divisible by both 3 and 5 (FizzBuzz)\n\n\
             Previous {report}\n\n\
             Fix the failures. Code only.",
            sig = fn_sigs[0],
        );

        info!("Evolution prompt:\n{prompt}");

        runtime
            .evolve(&agent, &prompt)
            .await
            .expect("evolution should succeed");

        // Re-run tests with the newly hot-swapped implementation.
        (passed, total, report) = run_tests();
        println!("{report}");

        if passed == total {
            println!("All {total} tests passed after {round} evolution round(s)!");
            return Ok(());
        }

        warn!("{passed}/{total} correct after round {round} — retrying.");
    }

    println!("\nDid not converge after {max_rounds} rounds.");
    Ok(())
}
