- Repair rounds: an end-to-end test of the compile-error API hints (`api_hints.rs`) against a
  host crate with a `prelude` (`struct-support-example`, an invented `step_by` on `GameState`).
  It cannot be a workspace test: the nested dylib build resolves the workspace's git
  dependencies afresh and needs the network, so it belongs with the CI smoke tests.
- Repair rounds, phase 2: expose the edit verbs as a rig `PortableTool` in the tool `DocMode`s
  that only *stages* edits into the per-request `ToolContext` and returns the instant
  parse/validate result; the ladder applies staged edits after `run()` and compiles as today.
  Never put `cargo` inside an inference-gate scope. Measure the response-format version first.
- Show multi function evolution with example.
- Show example of using external dependency in generated dylibs, if configured.
- Proper eval pipeline to compare model performance across tasks. My own benchmark suite so to say, aka `symbiont-eval`
- Run Harness for my symbolic regression evaluation comparison, to see if it beats SOTA for ~150 optimization targets.
- Capture the inference cost in the responses, if available. Per-request token usage is now
  exposed: `AgentRun::completion_calls` carries one `CompletionCall` per HTTP request, and
  `EvolutionTrace` records them per attempt. A monetary cost still needs provider pricing,
  which rig does not report.
- Prefix-cache visibility is provider-dependent: `LLM_TOKENS{kind="cached_input"}` comes from
  `usage.prompt_tokens_details.cached_tokens`, which vLLM never populates (verified: identical
  923-token prompt sent twice, field stays `null`) even though its prefix cache is working.
  The throughput bench works around this by scraping `vllm:prefix_cache_hits_total` from
  `/metrics`. Consider an optional hook so hosts can feed a server-side cache metric back in.
- Split the generated crate directory per lane if `evolve_batch` ever becomes build-bound.
  Cargo locks its build directory, so real build parallelism needs a target dir per lane, which
  means paying dependency compilation per lane. Only worth it once
  `symbiont_build_slot_wait_seconds` says so — see [CAVEATS.md](CAVEATS.md).
- Track the context length of the prompt (system + user) and make it available to query.
  `symbiont_llm_request_body_bytes` now measures the serialized payload of every outbound
  request, which sizes the prompt against the endpoint's context window, but in bytes and
  only as a metric. A token count is now partly answered: each `CompletionCall` carries the
  input-token count the provider reported for that request. The rendered *request* payload is
  still unexposed — capturing it needs a task-local sink in `MeteredHttpClient`, plumbed
  through the gate scope. (Response bodies reach us via `CompletionCall::raw` but are dropped
  on purpose: they duplicate the transcript. See `docs/evolution-trace-design.md` §6.)
- Provide a way to call `info`, `debug` and `trace` like logging functions in the code and have them feed into the context in a smart way.
  Maybe its possible to re-use `tracing` here, depending on if its safe to do across dylib boundaries.
  It would need to be its own buffer though.
- Natively support storing the correctly generated rust code in a DB.
  Maybe rig has some native DB support?
- Support passing in images if the LLM supports multi-modality.
  Giving Agents image context might help improve the reasoning in certain problem cases.
- example of evolving a CUDA kernel.
- example for an interactive background daemon that creates dynamic wallpapers based on user prompts / revision.
  * Similar to fractal studio
  * Could be CPU native or CUDA accelerated
