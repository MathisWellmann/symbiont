# Fractal Studio — Interactive Shader Evolution

An interactive egui window whose **per-pixel shader is written by an LLM agent**
and hot-swapped into the running binary as optimized native code.

Type a prompt — *"an animated Julia set, c orbiting the main cardioid, with a
glowing sunset palette"* — and the agent implements:

```rust
fn shade(x: f64, y: f64, t: f64) -> u32
```

The harness validates the signature, compiles the code with `--release`, and
hot-swaps the dylib. The live animation morphs in place, no restart.

## Why this showcases symbiont

- **Bare-metal performance where it matters**: `shade` is called once per pixel
  (~0.5M calls/frame at 960×540, more on a larger window), parallelized over all cores with rayon, with
  fractal workloads running hundreds of iterations per pixel. The ~1.6 ns
  dispatch overhead makes the hot-swap abstraction effectively free — an
  interpreted agent-code loop would be orders of magnitude too slow to animate.
- **Human-in-the-loop evolution**: the user is the evaluator. The runtime keeps
  the chat history, so follow-up prompts refine the current shader.
- **Constrained generation, visibly**: the side panel shows the exact code that
  is currently running; parse/signature/compiler errors are fed back to the
  agent automatically until the code is valid.
- **Panic containment**: if the agent code panics, the harness catches it
  inside the dylib (rendered as black pixels), and the message is shown in the
  UI and fed back into the next evolution prompt.

## Architecture

Three threads, coordinated around the feedback-loop contract
(*no evolvable call may be in flight while the dispatch pointers are swapped*):

- **egui UI** (main thread): canvas, prompt box, telemetry (ms/frame, Mpix/s),
  and a syntax-highlighted view of the live agent code.
- **render thread**: tight frame loop calling `shade` for every pixel via
  rayon, at whatever size the UI last reported (capped at ~1080p worth of
  pixels, then upscaled). Parks at a frame boundary only for the revision
  swap.

The canvas is re-rendered at the window's aspect ratio instead of being fitted
into it, so resizing never letterboxes — which means `aspect` is not a
constant, and the evolution prompt tells the agent as much.
- **evolution worker**: runs a single-lane `Runtime::evolve_batch` on a tokio
  runtime, then drains the render gate and calls
  `Runtime::activate_revision`.

### The animation keeps running while the agent works

`evolve_batch` compiles and *registers* the candidate without touching the
dispatch pointers, which is why it is exempt from the feedback-loop contract:
a retained-but-inactive revision is invisible to running calls. The render
thread therefore keeps calling the **current** revision's function pointer for
the whole generate → validate → compile round — seconds of LLM inference plus a
`cargo build --release` — and the canvas never freezes.

Only the commit is gated. `activate_revision` republishes function pointers
that were resolved when the dylib was loaded, so it is a handful of atomic
stores: the render thread parks at a frame boundary, the swap happens, and
rendering resumes. The side panel reports that park time in microseconds next
to the multi-second evolution time.

The one visible effect during an evolution is a frame-rate dip: the nested
`cargo build` competes for the same cores rayon renders on.

## Running

```bash
# Requires API_KEY, BASE_URL, and MODEL env vars (or a local llama-cpp server).
cargo run -p fractal-studio-example --release
```

## Prompt ideas

- "A Mandelbrot zoom into the seahorse valley with smooth iteration coloring."
- "An animated Julia set whose parameter orbits the main cardioid."
- "A Newton fractal of z^3 - 1 with basin coloring and shading by convergence speed."
- "Burning ship fractal, fiery palette, slow camera drift."
- "An orbit-trap fractal that looks like glowing stained glass."
- "Now make the palette cycle with time." (follow-up refinement)
