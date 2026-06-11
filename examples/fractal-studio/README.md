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
  (~0.5M calls/frame at 960×540), parallelized over all cores with rayon, with
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
(*no evolvable call may be in flight while the dylib is swapped*):

- **egui UI** (main thread): canvas, prompt box, telemetry (ms/frame, Mpix/s),
  and a syntax-highlighted view of the live agent code.
- **render thread**: tight frame loop calling `shade` for every pixel via
  rayon. Parks at a frame boundary when an evolution is requested.
- **evolution worker**: drains the render gate, runs `Runtime::evolve` on a
  tokio runtime, publishes the new code, resumes rendering.

The animation freezes (showing the last frame) while the agent generates and
compiles, then resumes with the new shader — that pause *is* the contract.

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
