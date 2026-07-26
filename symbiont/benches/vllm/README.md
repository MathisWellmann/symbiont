# Batched-inference benchmark

Measures what `Runtime::evolve_batch` actually buys, by running the same
eight-lane batch at in-flight limits 1, 2, 4 and 8 against a real vLLM server.
Limit 1 is the serial loop the API replaces; limit 8 puts every lane in flight
at once so the server's continuous batcher can merge them into shared forward
passes.

## Why the gain exists

Two effects compound, and the benchmark reports them separately so you can see
which one you are getting:

- **Decode is memory-bandwidth-bound.** At batch 1 the model weights are read
  from HBM once per generated token. At batch *n* they are read once for *n*
  tokens. This is why eight lanes cost far less than eight times one lane even
  though no individual request gets faster.
- **The lanes share a prompt prefix.** Everything before the per-lane strategy
  hint is byte-identical, so a server with prefix caching prefills it once. The
  effect is largest for hosts that pass their crate name to `init_agent`, since
  that embeds the rustdoc-derived API surface into the shared part.

  Measuring it is provider-dependent. symbiont's
  `LLM_TOKENS{kind="cached_input"}` comes from the OpenAI-compatible
  `usage.prompt_tokens_details.cached_tokens` field, and **vLLM does not
  populate it** — verified by sending an identical 923-token prompt twice and
  getting `prompt_tokens_details: null` both times. Its prefix cache is
  working regardless; it is only reported on `/metrics`, as
  `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`. The
  benchmark scrapes those and falls back to the provider-reported number for
  backends that do fill the field in.

The sweep keeps the lane count fixed at 8 and varies only the in-flight limit,
so every level does the same inference work and only the overlap differs.

## Running it

Requires an NVIDIA GPU with roughly 20 GB free (the 8B model at
`--gpu-memory-utilization=0.85` on a 24 GB card; lower it for smaller cards).
The first start downloads ~16 GB of weights into `~/.cache/huggingface`.

```sh
cd symbiont/benches/vllm
docker compose up -d

# Weights download + CUDA graph capture on a cold start; a warm start is ~1 min.
until curl -fsS http://127.0.0.1:8231/health >/dev/null; do sleep 5; done

cd ../../..
BASE_URL=http://127.0.0.1:8231/v1 MODEL=local \
  cargo bench --bench batch_throughput

cd symbiont/benches/vllm && docker compose down
```

The benchmark exits 0 with a message when `BASE_URL`/`MODEL` are unset or
nothing is listening, so `cargo bench` stays green without a server, and CI's
`cargo bench --no-run` only ever compiles it.

**Do not run other cargo commands while the sweep is in flight.** Each lane
shells out to `cargo build` for its dylib, and cargo serializes on a global
`$CARGO_HOME/.package-cache` lock, so a concurrent `cargo clippy` in another
terminal blocks the lanes' builds. It shows up as a level that is *slower* than
the serial baseline, with `slot wait` and `compile` absorbing the difference.

## The model

`prism-ml/Bonsai-8B-unpacked` — the safetensors base that
`prism-ml/Bonsai-8B-gguf` was quantized from, which is what `devenv.nix` serves
and what CI runs the examples against. Same weights, in the format vLLM loads
natively.

`llama-server` batches too, given `--parallel n` (devenv.nix sets 8), so you can
point the benchmark at the dev shell instead. vLLM is used for the published
numbers because it is the reference implementation of continuous batching and
paged prefix caching, and because `llama-server` splits `--ctx-size` across its
slots rather than sharing a pool. Both bind 8231, so run one or the other.

## GPU passthrough

`compose.yaml` requests the GPU as a CDI device (`nvidia.com/gpu=all`) rather
than through `deploy.resources.reservations.devices`. On hosts where the NVIDIA
container runtime is not registered with the Docker daemon — the common NixOS
setup, where `nvidia-container-toolkit` only drops a spec into `/var/run/cdi`
— `--gpus all` fails with a misleading error while the CDI device resolves
fine. Check the spec exists:

```sh
ls /var/run/cdi/    # expect nvidia-container-toolkit.json
```

If vLLM exits with `Free memory on device cuda:0 (...) is less than desired GPU
memory utilization`, something else is holding the card — including a
`llama-server` left running from `devenv up`.

## Measured result

vLLM v0.26.0 serving `prism-ml/Bonsai-8B-unpacked` on an RTX PRO 6000 Blackwell,
via the `compose.yaml` in this directory (prefix caching on, `--max-num-seqs=32`),
32 lanes, dylibs built in the debug profile, server restarted cold beforehand:

```
| in flight | wall    | s/candidate | speedup | dec tok/s | tok/s gain | lanes ok | built |
|-----------|---------|-------------|---------|-----------|------------|----------|-------|
|         1 |  407.1s |       12.7s |   1.00x |        72 |      1.00x |    18/32 |    17 |
|         2 |  679.6s |       21.2s |   0.60x |        88 |      1.22x |    18/32 |    16 |  <-- outlier, see below
|         4 |  149.8s |        4.7s |   2.72x |       196 |      2.71x |    18/32 |    13 |
|         8 |  102.3s |        3.2s |   3.98x |       300 |      4.15x |    16/32 |     9 |
|        16 |   57.2s |        1.8s |   7.12x |       475 |      6.59x |    16/32 |    10 |
|        32 |   40.8s |        1.3s |   9.97x |       656 |      9.10x |    17/32 |     5 |

| in flight | llm (sum) | compile | slot wait | prompt tok | output tok | prefix hit |
|-----------|-----------|---------|-----------|------------|------------|------------|
|         1 |    394.8s |    4.5s |       0ms |     363278 |      29380 |        74% |
|         2 |    840.3s |    4.0s |     353ms |     365529 |      59687 |        75% |
|         4 |    410.5s |    3.3s |     605ms |     405417 |      29311 |        74% |
|         8 |    461.0s |    2.3s |      1.3s |     406015 |      30640 |        75% |
|        16 |    458.8s |    2.5s |      2.8s |     362446 |      27184 |        76% |
|        32 |    578.3s |    1.3s |      5.8s |     387498 |      26794 |        75% |
```

**A population round costs about a tenth of the sequential loop.** 32
candidates in 41s against 407s one at a time, at 9.1x the decode throughput.

**No saturation through 32, but clear diminishing returns.** 16 → 32 doubled
concurrency for 1.4x wall clock. `llm (sum)` climbing from ~460s to 578s at the
top limit is per-request latency rising under batch pressure — the aggregate
still improves, but the marginal lane buys less.

**Level 16 is the most trustworthy row in the table**, at 56.4s, 58.4s and
57.2s across three separate runs (two single-threaded, one multi-threaded).

**The build slot is starting to show.** `slot wait` grows monotonically with
concurrency, 0ms → 5.8s, reaching ~14% of wall clock at limit 32 — up from
under 4% at limit 8. Still not the bottleneck, but this is the column that says
when splitting the crate directory per lane starts to pay. Total `compile`
*falls* with concurrency (4.5s → 1.3s) because dedup collapses more lanes at
higher limits, visible in `built` dropping 17 → 5.

**Prefix caching is doing real work**: a steady 74-76% of prompt tokens served
from cache. This comes from vLLM's `/metrics`; the provider-reported
`cached_input_tokens` was 0 for every single request in every run.

### Retry variance dominates — check `output tok` before trusting a row

The limit-2 row above reads 0.60x, *slower than serial*. It is not a
concurrency effect: its `output tok` is 59687, almost exactly double every
other level's ~29000. That level drew a set of lanes that ground through their
full retry budgets.

The asymmetry matters. Retries within a lane are sequential, so they lengthen
that lane's critical path, and a level's wall clock is its slowest lane —
concurrency cannot recover any of it. A level with 2x the tokens is far worse
than 2x slower. The same thing hit limit 32 in an earlier run (57124 output
tokens, 429.9s, 0.90x) while two other runs put it at 40.8s and 47.3s.

This is the workload, not the harness: Bonsai-8B converges on 16-20 of 32
lanes, with the sieve-family prompts mostly dying on `u32`-vs-`usize` indexing
against the `fn(n: u32) -> u32` signature. Run the sweep more than once and
discard rows whose `output tok` departs from the rest.

One further caveat: `built` falls below `lanes ok` because candidates
deduplicate against revisions earlier levels registered. The prompts carry a
per-level tag, but dedup keys on generated source, which the tag never reaches.

## Reading the output

`s/candidate` is the headline: it should fall as the limit rises. Below it, the
decomposition separates what batching helps with from what it cannot touch:

| Column      | What it tells you |
|-------------|-------------------|
| `llm (sum)` | Summed per-lane inference time. Should stay roughly flat across levels — batching overlaps requests, it does not shorten them. If it *grows* with the limit, the server is queueing rather than batching (`--max-num-seqs` too low, or `--parallel` unset on `llama-server`). |
| `compile`   | Summed `cargo build` time. Independent of the limit. |
| `slot wait` | Time lanes spent queued behind the single build slot. Near zero means the batch is inference-bound, which is the regime the one-slot design assumes. If it rivals `llm`, the shared crate directory has become the bottleneck and wants splitting per lane — or another cargo process is holding the package-cache lock. |
| `prefix hit` | Prefix-cache hit rate, from the server's own counters where it exposes them. Low means prefix caching is off, or the prompts vary too early to share a prefix. `n/a` means neither the server nor the provider reported anything. |
| `built`     | Distinct revisions the level added. Below `lanes ok` when lanes converged on identical source and deduplicated — a signal the prompt variants are not diversifying the output. |
