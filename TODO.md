- Show multi function evolution with example.
- Show example of using external dependency in generated dylibs, if configured.
- Proper eval pipeline to compare model performance across tasks. My own benchmark suite so to say, aka `symbiont-eval`
- Run Harness for my symbolic regression evaluation comparison, to see if it beats SOTA for ~150 optimization targets.
- Capture the inference cost in the responses, if available.
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
