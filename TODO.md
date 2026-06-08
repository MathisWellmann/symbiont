- Show multi-threading support with example.
- Show multi function evolution with example.
- Show example of using external dependency in generated dylibs, if configured.
- Support disallowing methods, crates or unsafe, enforced by the harness.
- Proper eval pipeline to compare model performance across tasks. My own benchmark suite so to say, aka `symbiont-eval`
- Compare examples with SOTA equivalent search strategies, see if it beats any already.
- Run Harness for my symbolic regression evaluation comparison, to see if it beats SOTA for ~150 optimization targets.
- Bidirectionality, like evolving a fractal rendering function using `evolvable` and a UI in the main harness binary shows the results.
- Capture the number of evolution failures by category, e.g how many compile errors, how many parse errors, how many HTTP errors, etc.
- Capture the inference cost in the responses, if available.
- Track the context length of the prompt (system + user) and make it available to query.
- Cap the runtime of agent code to a user-specified maximum to prevent infine loops in agent code.
  This is not really possible, unless the function signature has cooperative cancellation code passed in like an `stop: AtomicBool` and the agent must ensure to check it in each loop round.
- Provide a way to call `info`, `debug` and `trace` like logging functions in the code and have them feed into the context in a smart way.
  Maybe its possible to re-use `tracing` here, depending on if its safe to do across dylib boundaries.
  It would need to be its own buffer though.
- Natively support storing the correctly generated rust code in a DB.
  Maybe rig has some native DB support?
- Support passing in images if the LLM supports multi-modality.
  Giving Agents image context might help improve the reasoning in certain problem cases.
