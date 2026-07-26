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
via the `compose.yaml` in this directory (prefix caching on, `--max-num-seqs=16`),
dylibs built in the debug profile:

```
| in flight | wall    | s/candidate | speedup | lanes ok | built |
|-----------|---------|-------------|---------|----------|-------|
|         1 |   74.0s |        9.2s |   1.00x |      5/8 |     5 |
|         2 |   37.3s |        4.7s |   1.98x |      7/8 |     6 |
|         4 |   21.6s |        2.7s |   3.43x |      6/8 |     2 |
|         8 |   21.3s |        2.7s |   3.47x |      5/8 |     3 |

| in flight | llm (sum) | compile | slot wait | prompt tok | output tok | prefix hit |
|-----------|-----------|---------|-----------|------------|------------|------------|
|         1 |     71.2s |    1.3s |       0ms |      81245 |       5653 |        72% |
|         2 |     50.0s |    1.6s |     142ms |      46287 |       3819 |        78% |
|         4 |     63.0s |   524ms |     239ms |      79247 |       4576 |        75% |
|         8 |     71.1s |   776ms |     873ms |      81274 |       5052 |        72% |
```

**A population round costs about what three sequential evolutions cost.**
Eight candidates in 21s against 74s for the same eight run one at a time.

**Decode throughput is the work-normalized view, and it does not saturate.**
Wall-clock speedup flattens between 4 and 8 (3.43x → 3.47x), but output
tokens ÷ wall clock keeps climbing: **76 → 102 → 212 → 237 tok/s**, a 3.1x
gain. The flattening is a fixed serial tail, not the batcher giving up — each
lane still parses, validates, compiles and loads on its own, and the lanes that
exhaust their retry budget run the full ten attempts regardless of how many
siblings are in flight.

`llm (sum)` staying in the 50-71s band rather than growing with the limit is
the check that the server is batching rather than queueing. Dividing it by
wall clock gives the concurrency actually achieved — 0.96, 1.34, 2.92, 3.34 —
consistently below the limit, because lanes spend part of their life in the
local pipeline instead of in flight.

**The build slot behaves as designed.** `slot wait` grows with concurrency
exactly as predicted (0ms → 873ms) but stays under 4% of wall clock even at
eight lanes, while total `compile` never exceeds 1.6s. The batch is
inference-bound throughout, which is the regime the single-slot design assumes.

**Prefix caching is doing real work**: 72-78% of prompt tokens served from
cache. Note this figure comes from vLLM's `/metrics`; the provider-reported
`cached_input_tokens` was 0 for every one of these requests.

Two caveats on the numbers:

- **Levels do unequal work.** A lane that exhausts its budget spends ten
  requests, so a level with more failures does more inference — visible in the
  `prompt tok` column (level 2 did 46k tokens against level 1's 81k). Bonsai-8B
  gets 5-7 of 8 lanes to compiling code; the sieve-family prompts mostly die on
  `u32`-vs-`usize` indexing against the `fn(n: u32) -> u32` signature. Treat the
  speedup column as indicative, and prefer the tok/s figures.
- **`built` is below `lanes ok` at limits 4 and 8** because candidates
  deduplicated against revisions earlier levels had already registered. The
  prompts carry a per-level tag, but dedup keys on generated source, which the
  tag does not reach. It saves those levels roughly a second of compile time —
  immaterial against a 74s → 21s change, but it is why their `compile` column
  is lower.

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
