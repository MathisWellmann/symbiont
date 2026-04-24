// Test that the trivial counter function is correctly parsed and implemented by the Agent harness.
#[test]
fn test_counter_body() {
    use crate::function_parser::parse_functions;

    // 1. Verify the function signature is correctly parsed
    let fn_sigs = parse_functions().expect("failed to parse functions");
    assert_eq!(
        fn_sigs.len(),
        1,
        "Expected exactly 1 public no_mangle function, found {}",
        fn_sigs.len()
    );
    assert_eq!(fn_sigs[0], "fn step(counter: &mut usize)");

    // TODO: here, the Symbiont Agent Harness should let the LLM generate the body and verify its correctness.

    // 2. Verify the function body works: counter should increment
    let mut counter: usize = 0;
    for i in 0..1000 {
        assert_eq!(counter, i, "step() should increment counter by 1");
        symbiont_lib::step(&mut counter);
    }
}
