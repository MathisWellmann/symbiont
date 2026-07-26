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
