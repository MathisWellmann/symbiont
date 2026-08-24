# Token Budget — Design Proposal

Goal: symbiont should **never send a request the inference engine will reject**, and the combined
in-flight requests of all lanes should **never exceed a configured token budget**. Today the
harness discovers both conditions the same way — by being told `no` after the round trip
(`is_context_size_error`, `symbiont/src/utils.rs:213`) — and recovers by wiping a lane's chat
history. That is reactive, wastes a round trip, and, for the llama.cpp `500` variants, punishes
lanes that did nothing wrong (§1.3).

This is a design proposal; no code changes are implied yet.

---

## 1. Ground truth: what actually causes a rejection

Engine behaviour differs enough that "prompt + `max_tokens` > context" is *not* a portable
predicate. Verified against llama.cpp `master` (2026-08-25, `tools/server/`).

### 1.1 llama.cpp

| Condition | Where | Status | Body |
|---|---|---|---|
| `task.n_tokens >= slot.n_ctx` | `server-context.cpp`, `update_slots()` ~3096 | **400** `exceed_context_size_error` | "request (N tokens) exceeds the available context size (M tokens), try increasing it" |
| `task.n_tokens > slot.n_ctx`, non-causal | same, ~3086 | **400** `exceed_context_size_error` | "input (N tokens) is larger than the max context size (M tokens). skipping" |
| `task.n_tokens > n_ubatch` | same, ~3070 | **500** `server_error` | "input (N tokens) is too large to process. increase the physical batch size (current batch size: B)" |
| generation reaches `n_ctx`, `--no-context-shift` | `pre_decode()`, ~2801 | **500** `server_error` | "context shift is disabled" |
| KV cache cannot fit a single-token decode batch | `decode()`, ~3579 | **500** `server_error` | "Context size has been exceeded." |

`ERROR_TYPE_SERVER` → 500, `ERROR_TYPE_EXCEED_CONTEXT_SIZE` → 400 (`server-common.cpp:35-54`).
All five are matched today by `CONTEXT_SIZE_MARKERS` / `LLAMA_CPP_500_MARKERS`
(`symbiont/src/utils.rs:243`, `:258`) — *after the fact*.

Three consequences the budget arithmetic must respect:

1. **The 400 check looks at the prompt only.** `max_tokens` is not part of it. Output overflow is
   discovered later, during generation, as one of the 500s.
2. **`n_ubatch` is an independent and much smaller threshold** than `n_ctx` (typically 512–2048 vs
   32k+). A prompt can be far below the context limit and still be rejected.
3. **The last row is collateral.** The error is sent to *every* processing slot
   (`decode()` loop ~3593-3600) and re-broadcast by `abort_all_slots("decode() failed: " + err)`
   (`server-context.cpp:2638`, caught at ~2776). With `max_in_flight > 1`, one oversized lane makes
   every concurrent lane wipe its history and spend one of `MAX_CONTEXT_RESETS`
   (`symbiont/src/runtime.rs:237`). Preventing the send is worth much more than classifying the
   error.

### 1.2 Other engines

- **vLLM / OpenAI-compatible**: validates `prompt_tokens + max_tokens > context_length` up front →
  400 `context_length_exceeded` / "maximum context length". Stricter than llama.cpp's 400.
- **Anthropic**: "prompt is too long: N tokens > M maximum", prompt-only.

The portable predicate is therefore the **conservative union**: `prompt + max_output <= cap`, plus
the separate `prompt <= n_ubatch` constraint where it is known.

### 1.3 The cap is per slot, not per server

llama.cpp gives each of `--parallel n` slots `n_ctx / n` tokens, unless `--kv-unified` makes it one
shared pool. A "combined lanes" budget is only meaningful if it says which regime it models.
`GET /props` reports `total_slots` and `default_generation_settings.n_ctx` (the per-slot value), so
this is discoverable rather than guessed.

---

## 2. What is already available

| Item | Where it lives today |
|---|---|
| Serialized outbound body, before any I/O | `measured()` in `symbiont/src/metered_http.rs:220`; the `Bytes` exist at `:131` and `:172` |
| Per-request body size metric | `REQUEST_BODY_BYTES` (`symbiont/src/observability.rs:138`) |
| Per-run actual input/output tokens | `AgentRun.usage`, folded in `evolve_no_backpressure` (`symbiont/src/runtime.rs:401-409`) |
| Bytes-per-token ratio of the deployed tokenizer | Already documented as `REQUEST_BODY_BYTES ÷ LLM_RUN_INPUT_TOKENS` (`observability.rs:126-137`) — the calibration loop needs **no new inputs** |
| Request admission point | `MeteredHttpClient::send` / `send_streaming` (`metered_http.rs:117`, `:164`), which reads `GateScope::current()` synchronously and holds the permit until the body/stream completes (`:144`, `:209`) |
| Concurrency limiter | `InferenceGate`, a priority semaphore over **request count** (`inference_gate.rs:96` `State { capacity, in_flight }`) |
| Overflow recovery | `Runtime::evolve` ladder, `runtime.rs:1207` (context reset) before `:1292` (transient retry) |
| Where `additional_params` are set | `inference.rs:136`, from `ThinkingLevel::to_additional_params` (`thinking_level.rs:44`) |

**`max_tokens` is never set.** The only occurrence is the *reasoning* budget
(`thinking_level.rs:55`, inside `"reasoning"`). Unset, llama.cpp generates until the context ends,
so the output half of the budget is unbounded and no arithmetic is possible. Setting it explicitly
is a prerequisite, not an optimization.

---

## 3. Token estimation

Exact counting is impossible without the model's tokenizer. Three options:

| Approach | Exact | Cost | Applies to |
|---|---|---|---|
| Calibrated bytes→tokens EWMA | no, ±10–15% | free (metrics already collected) | every provider |
| llama.cpp `POST /tokenize` preflight | yes | one extra local round trip per request | llama.cpp only |
| `tokenizers` crate + `tokenizer.json` | yes | heavy dependency, per-model asset | local models only |

Proposal: a `TokenEstimator` trait with the EWMA as the default implementation, so hosts that need
exactness can plug in `/tokenize` or a real tokenizer without symbiont growing a dependency.

The default estimator updates from every completed run: it has the request's byte length and the
provider's reported `usage.input_tokens`. Seed it conservatively (e.g. 2.5 bytes/token), converge
from observation, and apply a safety factor to the **reservation**, not to the preflight cap —
inflating the cap rejects valid work, inflating the reservation only costs a little throughput.

---

## 4. Design

### 4.1 Configuration

```
TokenBudget {
    max_output: u32,      // written into the request AND into every estimate
    per_request: u32,     // hard cap: est_prompt + max_output must fit
    per_request_batch: u32, // optional: n_ubatch, prompt-only constraint
    total: u32,           // fleet-wide, all lanes combined
}
```

Lives on `Runtime` beside `inference_gate` (`runtime.rs:215`), with `unlimited()` as the default so
existing hosts see no behaviour change. An optional best-effort `GET /props` probe at agent
construction defaults `per_request` / `per_request_batch` to the server's real numbers instead of
asking the user to guess them.

### 4.2 Enforcement point A — per-request preflight

In `MeteredHttpClient::send` the serialized body already exists before any I/O (`metered_http.rs:131`).
If `estimate(body) + max_output > per_request` (or `estimate(body) > per_request_batch`), fail
locally with a new `Error::ContextBudgetExceeded { estimated, cap }`, which `Runtime::evolve` treats
exactly as `is_context_size_error` does today (`runtime.rs:1207`): discard history, restart from the
base prompt.

Two wins: no round trip is wasted, and the oversized request is never sent, so it can never abort
another lane's slot (§1.1, consequence 3).

### 4.3 Enforcement point B — fleet budget as a weighted gate

`InferenceGate`'s `State { capacity: u16, in_flight: u16 }` (`inference_gate.rs:96`) gains a token
dimension: a request reserves `est_prompt + max_output` and is admitted only when
`in_flight_tokens + weight <= total`. Permit lifetime is unchanged — released when the body or
stream completes (`metered_http.rs:144`, `:209`), which is when the request stops occupying the
endpoint.

Hazards, each of which needs a test:

- **`weight > total` deadlocks the queue.** Must be unreachable, which §4.2 guarantees as long as
  `per_request <= total`. Assert that invariant at configuration time.
- **Head-of-line starvation of large requests.** With a naive "does it fit?" test, small requests
  overtake a large one forever. The head waiter must *reserve* capacity as it frees, rather than
  letting later arrivals consume it.
- **The request-count cap stays.** `max_in_flight` (`runtime.rs:997`) is what fills the server's
  continuous batch; the token budget is a second, independent constraint, not a replacement.

### 4.4 Proactive trimming

Once an estimate exists, the evolve loop can trim the oldest attempt pairs when the next request
crosses a soft threshold (~80% of `per_request`), instead of waiting for an overflow and wiping the
whole history. `MAX_CONTEXT_RESETS` becomes a rare fallback rather than the primary mechanism.

### 4.5 Metrics

- gauge: budget, in-flight tokens, current bytes-per-token ratio
- histogram: **`estimated ÷ actual` input tokens** — the tuning signal for the safety factor, and
  the gate for enabling enforcement at all
- counter: preflight rejections, by cause (`per_request` / `n_ubatch` / fleet)

---

## 5. Phasing

| Phase | Content | Risk |
|---|---|---|
| 1 | `max_output` plumbed into `additional_params`; `TokenBudget` config; optional `/props` probe | none (no enforcement) |
| 2 | `TokenEstimator` + EWMA calibration + metrics, **observe only** | none |
| 3 | Per-request preflight (§4.2) + `Error::ContextBudgetExceeded` | rejects work if the estimator is optimistic |
| 4 | Weighted gate (§4.3) | throughput regression if the safety factor is too large |
| 5 | Proactive trimming (§4.4); optional exact `/tokenize` estimator | prompt-quality regression if trimming is too eager |

**Phase 2 is not optional.** An estimator that is 15% optimistic and allowed to reject requests is
worse than the reactive handling we have now. Ship it observe-only, read the `estimated ÷ actual`
histogram from a real run, then enable Phase 3.

---

## 6. Known holes and non-goals

- **In-flight tokens understate KV occupancy.** With prompt caching a lane's KV stays resident in
  its slot between requests; llama.cpp reclaims it under pressure (`try_clear_idle_slots`). The
  budget models active requests only. This is a deliberate approximation and the reason the
  reactive 500 handling in `is_context_size_error` must stay as a backstop.
- **Unaccounted requests**: `send_multipart` is unsized by design (`metered_http.rs:154`), and any
  request issued from a `tokio::spawn` inside a lane bypasses the task-local gate entirely
  (documented at `inference_gate.rs:233`).
- **Not a cost budget.** This proposal bounds context pressure, not spend. A spend cap would key on
  cumulative `Usage`, not on in-flight reservations, and is out of scope.
- **The estimator is per model, not per request.** Tool-heavy payloads and code-heavy payloads have
  different ratios; a single EWMA smears them. Acceptable while the safety factor covers the
  spread; revisit if the `estimated ÷ actual` histogram turns out to be bimodal.

---

## 7. Open questions

1. Should Phase 3 reject with a dedicated error, or synthesize an error that the existing
   `is_context_size_error` path already matches? A dedicated variant is cleaner but touches the
   public `Error` enum.
2. Should `per_request` default to *discovered* (`/props`) or stay unlimited unless the host opts
   in? Discovery is friendlier but makes agent construction depend on a reachable endpoint.
3. Is the fleet budget worth it at all once §4.2 is in place, given §6's caching caveat? Phase 3
   alone may remove the overwhelming majority of overflow events; Phase 4 should be justified by
   measurement rather than assumed.
