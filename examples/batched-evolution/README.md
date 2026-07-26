# Batched Evolution — Population Search in One Round

This example runs **eight prompt variants concurrently** through
`Runtime::evolve_batch`, benchmarks the eight resulting implementations against
each other, and only then activates the winner.

The task is counting primes below `n`. It is chosen because *strategy*
dominates it: trial division, a sieve of Eratosthenes, a bit-packed sieve and
wheel factorization are all a few dozen lines and all obviously correct to a
model, yet they span three orders of magnitude in runtime. Eight prompts that
differ only in a trailing strategy hint therefore produce eight genuinely
different programs — which is what makes comparing them worthwhile.

The default implementation is deliberately the worst reasonable one: trial
division by *every* smaller integer.

## What it demonstrates

**A batch is a population, not a faster loop.** `evolve_batch` registers every
candidate but activates none of them — eight implementations with no fitness
signal between them is not a hill to climb. The harness measures each candidate
through a `RevisionFn` handle, which pins its own revision and bypasses the
dispatch pointers entirely, so all eight are timed against the same baseline
with nothing hot-swapped in between. Only then does `activate_revision` commit
to one.

**Prompt layout decides the cost.** Everything before the strategy hint is
byte-identical across lanes. Prefix reuse on the server stops at the first
differing token, so the hint goes last — a per-lane preamble would throw the
shared prefill away.

**Failures are per lane.** Each lane owns its retry budget and chat history, so
one lane answering with prose ten times does not cost its siblings anything.
The report groups `EvolveFailure` records by `lane()`, which is how you tell an
unproductive prompt variant from a broken batch.

## Running

```bash
# Requires API_KEY, BASE_URL, and MODEL env vars (or a local llama-cpp server).
# Note that llama-server needs --parallel n to actually batch the lanes;
# devenv.nix sets it to 8.
cargo run -p batched-evolution-example
```

Set `STRICT=1` to require that at least one lane produces a correct
implementation. By default a round where every lane fails is reported rather
than fatal — whether a small model invents a sieve is a property of the model,
not of the harness, and everything the example exercises has already run by
that point.

## Output

```
| Lane | Revision | Correct | Median      | Speedup | Strategy
|------|----------|---------|-------------|---------|---------
| 0    | 1        | yes     |     41.20 ms|      1x | Keep it simple and obviously correct ...
| 3    | 4        | yes     |      182 us |    226x | Use a sieve of Eratosthenes over a boolean array.
...
```

## Measuring the speedup

`symbiont/benches/vllm/` has a concurrency sweep that quantifies what the batch
saves against a real vLLM server, and decomposes it into inference time, build
time and prefix-cache hit rate.
