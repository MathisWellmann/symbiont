# CUDA Softmax — Evolving a GPU Kernel Against a Measured Roofline

An LLM writes **CUDA kernels**, the host compiles them with NVRTC, checks them
against a CPU oracle, times them on the GPU, and feeds the achieved memory
throughput back into the next round. The best kernel is hot-swapped in; the
rest stay registered as revisions.

```bash
MODEL=... cargo run -p cuda-softmax-example --release
```

The evolvable function is not the kernel itself — it is the thing that *emits*
one:

```rust
fn plan(rows: usize, cols: usize) -> KernelPlan   // { source, grid, block, shared_bytes }
```

The agent controls both halves, because both matter: the same source at the
wrong block size is an order of magnitude slower.

## Why a GPU kernel is a good subject for an evolution loop

- **The fitness function is not a matter of taste.** A row-wise softmax must
  read `rows * cols` floats and write as many, and nothing else. The example
  measures the ceiling instead of quoting a spec sheet: a copy kernel over the
  *same buffers* on the *same card*, so a candidate's score is a percentage of
  something real. (On a card with a large L2 the working set may be cache
  resident — which is exactly why the ceiling is measured under the same
  conditions as the candidates rather than taken from a datasheet.)
- **The gap is enormous and comes from strategy, not micro-edits.** The naive
  kernel the run starts from is one thread per row: the 32 threads of a warp
  touch addresses 4 KiB apart, so every load pulls a full cache line to use
  four bytes of it, and the row is re-read from memory three times. Closing the
  gap means coalescing, block-per-row reductions, warp shuffles, vectorized
  `float4` loads, fast-math intrinsics — different *programs*, which is what
  `evolve_batch` is for.
- **Correctness is checkable and the trap is real.** The benchmark input has
  per-row biases up to ~90, so `expf(x)` overflows to `inf` in f32 unless the
  row maximum is subtracted first. Deleting that pass is a tempting way to
  drop a third of the memory traffic; the oracle catches it.

## A real run

RTX PRO 6000 Blackwell (188 SMs), 4096x1024 f32, one round of two lanes
against a local `Qwen3.6-35B-A3B` at Q4. One lane produced a kernel with an
incomplete reduction and was rejected by the oracle; the other:

| kernel | time | throughput | % of ceiling | vs baseline |
|---|---|---|---|---|
| naive, one thread per row (the starting point) | 523 us | 64 GB/s | 1.1% | 1x |
| **evolved: one warp per row, pure warp shuffles** | **12.7 us** | **2640 GB/s** | **47%** | **41x** |
| hand-written reference: block per row, `float4` in registers | 7.1 us | 4730 GB/s | 84% | 74x |

```cuda
// what the agent came up with, verbatim
extern "C" __global__ void softmax(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int stride = cols >> 5;
    int base = row * cols;

    float local_max = -INFINITY;
    for (int i = 0; i < stride; ++i)
        local_max = fmaxf(local_max, input[base + tid + (i << 5)]);
    for (int m = 16; m > 0; m >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, m));
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    float local_sum = 0.0f;
    for (int i = 0; i < stride; ++i)
        local_sum += expf(input[base + tid + (i << 5)] - row_max);
    for (int m = 16; m > 0; m >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, m);
    float row_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);

    float inv_sum = 1.0f / row_sum;
    for (int i = 0; i < stride; ++i)
        output[base + tid + (i << 5)] = expf(input[base + tid + (i << 5)] - row_max) * inv_sum;
}
```

It sidestepped the cross-warp reduction entirely by giving each row a single
warp — 32 threads, `__shfl_down_sync` only, no shared memory, no
`__syncthreads()`, and therefore none of the bugs that killed the other lane.
It is also *not* the best kernel available: at 32 threads per block it leaves
about half the bandwidth on the table, which the last row shows. The harness
reports that honestly rather than declaring victory — which is the point of
scoring against a measured ceiling instead of against the previous attempt.

Whether a given model gets further is a property of the model. Everything the
loop needs to keep pushing — the incumbent's source, its throughput, and the
exact reason each rejected candidate was rejected — goes into the next round's
prompt.

## The interesting part: a GPU kernel cannot be `catch_unwind`-ed

Symbiont contains a misbehaving CPU implementation *in process*. Generated
functions are wrapped in `catch_unwind` inside the dylib, a panic becomes a
default return value, and the message is fed back to the agent. The loop never
stops.

None of that is available on a GPU. An illegal memory access is a **sticky**
CUDA error: it does not merely fail the launch, it invalidates the context —
and, as this example's own test asserts, afterwards *no* context can be created
in that process at all. `cuDevicePrimaryCtxRetain` hands back the poisoned one
and `cuCtxCreate` fails too, even after every handle has been dropped:

```
fault:        CUDA_ERROR_ILLEGAL_ADDRESS
primary:      Err(CUDA_ERROR_ILLEGAL_ADDRESS)
non-primary:  Err(CUDA_ERROR_ILLEGAL_ADDRESS)
```

So the process boundary *is* the GPU's `catch_unwind`. The parent never
launches agent-written kernels: it re-executes itself once per candidate with
`CUDA_SOFTMAX_EVAL_PLAN` set, and that child compiles, verifies, times, and
writes a one-line report. A candidate that faults the device, hangs, or
segfaults costs one child process and one table row — the search continues.
`tests/isolation.rs` asserts exactly that, end to end.

## What the loop feeds back

Each rejection is classified, because different failures need different
nudges:

| kind | source | fed back as |
|---|---|---|
| `bad geometry` | host-side check | "block (2048,1,1) is 2048 threads, the device allows at most 1024" |
| `nvrtc error` | NVRTC log | the compiler diagnostics, with line numbers pointing at the agent's own source |
| `missing symbol` | module lookup | "the kernel must be declared `extern \"C\"`" |
| `wrong output` | CPU oracle | "output[0][0] = 6.38e-5, expected 2.04e-3 (tolerance 1e-6)" |
| `launch fault` | dead child | "the evaluation process exited without reporting" |

Two details worth stealing:

- NVRTC compiles without the CUDA toolkit headers, so `INFINITY`, `NAN` and
  `FLT_MAX` are simply undefined — the single most common way for an otherwise
  fine kernel to fail to compile. The host prepends guarded definitions plus a
  `#line 1` directive, so NVRTC's diagnostics still point at the agent's line
  numbers.
- The output buffer is filled with `NaN` before the correctness run, so a
  kernel that leaves elements untouched fails instead of silently passing on
  the previous candidate's results.

## Structure

- `src/lib.rs` — `KernelPlan`, the naive starting kernel, the deterministic
  benchmark input and the `f64` CPU oracle.
- `src/gpu.rs` — every `unsafe` block in the example: NVRTC, launch,
  verification, timing, and the measured copy ceiling.
- `src/isolate.rs` — the parent/child protocol.
- `src/main.rs` — the `evolvable!` declaration, the prompts, and the search.

Agent code stays inside symbiont's policy the whole time: no `unsafe`, no
statics, no FFI. It emits *text* and typed launch parameters, and the host
decides whether that text is even allowed near the device. The generated dylib
is compiled in **debug** on purpose — the Rust it contains just builds a
string, and all the performance lives in the CUDA source.

## Knobs

| env | default | meaning |
|---|---|---|
| `MODEL` | required | model slug served at `BASE_URL` |
| `ROUNDS` | 3 | search rounds |
| `LANES` | 4 | candidates per round (8 strategy hints available, then they cycle) |
| `STRICT` | unset | fail instead of skipping when no CUDA device is present |

Without an NVIDIA GPU the example prints why it is skipping and exits
successfully. It still *builds* anywhere: `cudarc`'s default `dynamic-loading`
resolves `libcuda`/`libnvrtc` with `dlopen` at runtime, so no CUDA toolkit is
needed at build time.
