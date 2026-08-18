// SPDX-License-Identifier: MPL-2.0
//! Fractal Studio: an interactive egui canvas whose per-pixel shader is written
//! by an LLM agent and hot-swapped into the running binary as native code.
//!
//! Type a prompt ("an animated Julia set with a sunset palette"), the agent
//! implements `shade`, the harness validates + compiles + hot-swaps the dylib,
//! and the live animation morphs in place. Every pixel of every frame runs
//! bare-metal compiled Rust (~1.6 ns dispatch overhead), parallelized across
//! all cores with rayon — the kind of workload where interpreted agent code
//! would be orders of magnitude too slow.
//!
//! Architecture (three threads):
//! - egui UI (main thread): displays the latest framebuffer, the prompt box,
//!   live telemetry, and the current agent-generated code.
//! - render thread: tight loop calling the evolvable `shade` for every pixel
//!   via `rayon`, pushing finished frames to the UI.
//! - evolution worker: receives user prompts and runs `Runtime::evolve_batch`
//!   with a single lane, which generates, validates, compiles and *registers*
//!   the candidate without touching the dispatch pointers. Rendering keeps
//!   running off the active revision the whole time; only the final
//!   `activate_revision` needs the render thread parked, and that costs a
//!   handful of atomic stores at a frame boundary instead of the multi-second
//!   generate-and-compile round.

use std::{
    sync::{
        Arc,
        Condvar,
        Mutex,
        atomic::{
            AtomicU64,
            Ordering,
        },
        mpsc::{
            Receiver,
            Sender,
            channel,
        },
    },
    time::{
        Duration,
        Instant,
    },
};

use eframe::egui;
use rayon::prelude::*;
use symbiont::Runtime;
use tracing::{
    info,
    warn,
};

symbiont::evolvable! {
    /// Compute the color of a single pixel of the canvas.
    ///
    /// # Coordinates
    /// - `x`, `y`: canvas coordinates with `(0.0, 0.0)` at the center.
    ///   `y` spans `[-1.0, 1.0]` (positive is up); `x` spans
    ///   `[-aspect, +aspect]` where `aspect = width / height`. The canvas is
    ///   re-rendered at the window's aspect ratio whenever it is resized, so
    ///   `aspect` is not a constant: keep the composition centered instead of
    ///   assuming a particular width.
    /// - `t`: seconds since program start — use it for smooth animation
    ///   (palette cycling, zooming, morphing parameters, ...).
    ///
    /// # Returns
    /// The pixel color packed as `0x00_RR_GG_BB` (alpha is implied opaque).
    ///
    /// # Constraints
    /// The function must be pure: no allocation, no I/O, no statics, no
    /// `unsafe`. It is called once per pixel — millions of times per frame —
    /// and parallelized across all cores by the host, so per-call cost must
    /// stay bounded (cap iteration counts).
    fn shade(x: f64, y: f64, t: f64) -> u32 {
        // Default implementation: a Julia set that morphs and cycles color,
        // so the canvas is alive before the first evolution.
        //
        // `c` crawls along the boundary of the Mandelbrot set's main cardioid
        // (pulled 0.5% inwards, where the sets stay connected but stringy), so
        // the shape continuously grows and folds new filaments. The exterior
        // is colored by the *smooth* (fractional) escape count and faded to
        // black away from the set; the interior — flat black in the textbook
        // rendering — is colored by an orbit trap on the closest the orbit
        // ever came to the origin, which lights it up as glowing nebulae with
        // contour rings. Both go through the same cosine palette, whose phase
        // drifts with `t`.
        const MAX_ITER: u32 = 192;
        // A large bailout is what makes the smooth iteration count smooth:
        // the fractional part converges as the escape radius grows.
        const ESCAPE: f64 = 256.0;

        /// Cosine palette: cheap, periodic, saturated everywhere.
        fn palette(s: f64) -> (f64, f64, f64) {
            use std::f64::consts::TAU;
            (
                0.55 + 0.45 * (TAU * (s + 0.00)).cos(),
                0.45 + 0.40 * (TAU * (s + 0.28)).cos(),
                0.55 + 0.45 * (TAU * (s + 0.62)).cos(),
            )
        }

        /// Pack floats into `0x00_RR_GG_BB`, gamma corrected.
        fn pack(r: f64, g: f64, b: f64) -> u32 {
            let q = |v: f64| (v.clamp(0.0, 1.0).sqrt() * 255.0) as u32;
            (q(r) << 16) | (q(g) << 8) | q(b)
        }

        // Slow breathing zoom keeps the composition from feeling static.
        let zoom = 1.35 + 0.10 * (t * 0.19).sin();
        let (mut zx, mut zy) = (x * zoom, y * zoom);

        // Cardioid boundary: c(th) = e^(i*th)/2 - e^(2i*th)/4. The offset
        // start angle opens on a filigreed set rather than a round blob.
        let th = 0.8 + t * 0.09;
        let (sin_th, cos_th) = th.sin_cos();
        let (sin_2th, cos_2th) = (2.0 * th).sin_cos();
        let cx = 0.995 * (0.5 * cos_th - 0.25 * cos_2th);
        let cy = 0.995 * (0.5 * sin_th - 0.25 * sin_2th);

        let mut m = zx * zx + zy * zy;
        let mut trap = m;
        let mut i = 0_u32;
        while m <= ESCAPE && i < MAX_ITER {
            let next_zx = zx * zx - zy * zy + cx;
            zy = 2.0 * zx * zy + cy;
            zx = next_zx;
            m = zx * zx + zy * zy;
            if m < trap {
                trap = m;
            }
            i += 1;
        }

        if i == MAX_ITER {
            // Interior: glow by how tightly the orbit hugged the origin.
            let d = trap.sqrt();
            let glow = (-2.0 * d).exp();
            let band = 0.82 + 0.18 * (d * 26.0 - t).sin();
            let (r, g, b) = palette(0.30 + 0.55 * d - 0.04 * t);
            let k = (0.10 + 0.90 * glow) * band;
            return pack(r * k, g * k, b * k);
        }

        // Exterior: continuous escape count, so no iteration banding.
        let smooth = f64::from(i) + 1.0 - (0.5 * m.ln()).ln() / std::f64::consts::LN_2;
        let (r, g, b) = palette(0.11 * smooth.max(0.0).sqrt() + 0.05 * t);
        // Fade the fast-escaping far field to black to frame the set.
        let v = (smooth / 22.0).min(1.0).powf(1.6);
        pack(r * v, g * v, b * v)
    }
}

/// Canvas size before the UI has reported how much space it has.
const INITIAL_SIZE: (usize, usize) = (960, 540);
/// Smallest canvas edge in physical pixels, so a collapsed panel cannot
/// produce a zero-sized (or degenerate one-pixel) render target.
const MIN_EDGE: usize = 64;
/// Upper bound on the number of pixels rendered per frame (~1080p).
///
/// Beyond this the per-frame shader cost grows faster than the visible gain;
/// the canvas keeps the window's aspect ratio and is upscaled by the GPU, so
/// it still fills the panel edge to edge — a maximized 4K window just renders
/// slightly softer instead of dropping to a few frames per second.
const MAX_PIXELS: usize = 1920 * 1080;
/// Canvas edges are rounded down to a multiple of this, so dragging the
/// window does not reallocate and re-render on every sub-pixel change.
const SIZE_QUANTUM: usize = 8;
/// Frame pacing target (~60 fps). Rendering faster than this just burns CPU.
const TARGET_FRAME_TIME: Duration = Duration::from_millis(16);

/// The canvas size the UI wants, in physical pixels, packed as
/// `(width << 32) | height`.
///
/// Written by the UI thread whenever the panel is laid out and read by the
/// render thread at every frame boundary. A relaxed atomic rather than a
/// mutex: the two sides never need to agree on *when* a resize takes effect,
/// only that the render thread eventually picks the latest value up.
#[derive(Debug)]
struct CanvasSize(AtomicU64);

impl CanvasSize {
    /// Start at [`INITIAL_SIZE`] until the UI reports its available space.
    fn new() -> Self {
        let (width, height) = INITIAL_SIZE;
        Self(AtomicU64::new(((width as u64) << 32) | height as u64))
    }

    /// Record the size the UI has room for, in physical pixels.
    ///
    /// The request is quantized, clamped to [`MIN_EDGE`], and scaled down to
    /// [`MAX_PIXELS`] while preserving the requested aspect ratio — matching
    /// that aspect ratio is what keeps the canvas free of letterbox bars.
    fn request(&self, width: f32, height: f32) {
        let (mut width, mut height) = (f64::from(width).max(1.0), f64::from(height).max(1.0));
        let pixels = width * height;
        let budget = MAX_PIXELS as f64;
        if pixels > budget {
            let scale = (budget / pixels).sqrt();
            width *= scale;
            height *= scale;
        }
        let quantize = |v: f64| {
            let v = v as usize / SIZE_QUANTUM * SIZE_QUANTUM;
            v.max(MIN_EDGE)
        };
        let packed = ((quantize(width) as u64) << 32) | quantize(height) as u64;
        self.0.store(packed, Ordering::Relaxed);
    }

    /// The current canvas size in physical pixels.
    fn get(&self) -> (usize, usize) {
        let packed = self.0.load(Ordering::Relaxed);
        ((packed >> 32) as usize, (packed & 0xFFFF_FFFF) as usize)
    }
}

#[cfg(test)]
mod canvas_size_tests {
    use super::*;

    #[test]
    fn packs_and_unpacks_both_edges() {
        let size = CanvasSize::new();
        assert_eq!(size.get(), INITIAL_SIZE);
        size.request(1280.0, 720.0);
        assert_eq!(size.get(), (1280, 720));
    }

    #[test]
    fn quantizes_and_clamps_to_min_edge() {
        let size = CanvasSize::new();
        size.request(1283.0, 727.0);
        assert_eq!(size.get(), (1280, 720));
        size.request(0.0, -5.0);
        assert_eq!(size.get(), (MIN_EDGE, MIN_EDGE));
    }

    #[test]
    fn scales_oversized_requests_down_keeping_the_aspect_ratio() {
        let size = CanvasSize::new();
        size.request(3840.0, 2160.0);
        let (width, height) = size.get();
        assert!(width * height <= MAX_PIXELS, "{width}x{height}");
        // The aspect ratio is what keeps the canvas free of letterbox bars.
        let aspect = width as f64 / height as f64;
        assert!((aspect - 3840.0 / 2160.0).abs() < 0.01, "aspect {aspect}");
    }
}

/// Render one full frame of `width` x `height` into an RGB byte buffer
/// (3 bytes per pixel) by calling the hot-swappable `shade` function for
/// every pixel, parallelized over rows with rayon.
///
/// The aspect ratio is derived from the frame size rather than fixed, so the
/// coordinate system `shade` sees always matches the shape of the window and
/// the result can be blitted edge to edge.
fn render_frame(t: f64, rgb: &mut [u8], width: usize, height: usize) {
    let aspect = width as f64 / height as f64;
    rgb.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(py, row)| {
            // `y` points up: top row maps to +1, bottom row to -1.
            let y = 1.0 - 2.0 * (py as f64 / (height - 1) as f64);
            for (px, pixel) in row.chunks_exact_mut(3).enumerate() {
                let x = (2.0 * (px as f64 / (width - 1) as f64) - 1.0) * aspect;
                // Bare-metal call into the hot-loaded native dylib.
                let c = shade(x, y, t);
                pixel[0] = ((c >> 16) & 0xFF) as u8;
                pixel[1] = ((c >> 8) & 0xFF) as u8;
                pixel[2] = (c & 0xFF) as u8;
            }
        });
}

/// State of the render gate, coordinating the render thread with evolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateState {
    /// The render thread renders frames freely.
    Run,
    /// The evolution worker asked the render thread to park at the next frame
    /// boundary.
    DrainRequested,
    /// The render thread is parked; no evolvable calls are in flight.
    Paused,
}

/// Synchronizes the render thread with the evolution worker so that no
/// evolvable function call is in flight while
/// [`Runtime::activate_revision`] republishes the dispatch pointers (the
/// feedback-loop contract).
///
/// Only the pointer swap is gated — generation and compilation happen while
/// the render thread runs freely against the previous revision.
#[derive(Debug)]
struct Gate {
    /// Current gate state.
    state: Mutex<GateState>,
    /// Signals state transitions to whichever side is waiting.
    cvar: Condvar,
}

impl Gate {
    /// Create a new gate in the [`GateState::Run`] state.
    fn new() -> Self {
        Self {
            state: Mutex::new(GateState::Run),
            cvar: Condvar::new(),
        }
    }

    /// Called by the render thread between frames. Acknowledges a pending
    /// drain request and blocks while the gate is paused.
    fn frame_boundary(&self) {
        let mut state = self.state.lock().expect("gate mutex is not poisoned");
        if *state == GateState::DrainRequested {
            *state = GateState::Paused;
            self.cvar.notify_all();
        }
        while *state == GateState::Paused {
            state = self.cvar.wait(state).expect("gate mutex is not poisoned");
        }
    }

    /// Called by the evolution worker right before the swap. Blocks until the
    /// render thread has parked at a frame boundary, guaranteeing no in-flight
    /// `shade` calls. Held for the duration of one pointer store, so the
    /// animation misses at most a frame.
    fn drain(&self) {
        let mut state = self.state.lock().expect("gate mutex is not poisoned");
        if *state == GateState::Run {
            *state = GateState::DrainRequested;
        }
        while *state != GateState::Paused {
            state = self.cvar.wait(state).expect("gate mutex is not poisoned");
        }
    }

    /// Called by the evolution worker after the swap to resume rendering.
    fn resume(&self) {
        *self.state.lock().expect("gate mutex is not poisoned") = GateState::Run;
        self.cvar.notify_all();
    }
}

/// State shared between the render thread, the evolution worker and the UI.
#[derive(Debug, Clone)]
struct SharedUi {
    /// True while the evolution worker is generating / compiling / swapping.
    /// Rendering continues throughout; only the swap parks the render thread.
    evolving: bool,
    /// The current agent-generated code running in the dylib.
    code: String,
    /// Message of the last panic caught inside the agent code, if any.
    panic_msg: Option<String>,
    /// Error message of the last failed evolution, if any.
    evolve_error: Option<String>,
    /// Size of the most recently rendered frame, in physical pixels.
    canvas: (usize, usize),
    /// Most recent frame time in milliseconds.
    frame_ms: f64,
    /// Most recent throughput in megapixels per second.
    mpix_per_s: f64,
    /// Number of successful evolutions so far.
    evolutions: usize,
    /// Wall-clock duration of the last successful evolution in seconds.
    last_evolve_secs: Option<f64>,
    /// How long the render thread was parked for the last revision swap, in
    /// microseconds — the only part of an evolution that stalls the animation.
    last_swap_us: Option<f64>,
}

/// Slot holding the most recently rendered frame for the UI to pick up.
type FrameSlot = Arc<Mutex<Option<egui::ColorImage>>>;

/// Spawn the render thread: an endless loop of frame rendering, telemetry
/// updates and panic collection, parking at the gate during evolutions.
fn spawn_render_thread(
    gate: Arc<Gate>,
    shared: Arc<Mutex<SharedUi>>,
    frame_slot: FrameSlot,
    canvas_size: Arc<CanvasSize>,
    runtime: &'static Runtime,
    ctx: egui::Context,
) {
    std::thread::Builder::new()
        .name("symbiont-render".to_string())
        .spawn(move || {
            let start = Instant::now();
            let mut rgb = Vec::new();
            loop {
                gate.frame_boundary();

                // Adopt the size the UI last asked for. Resizing between
                // frames (never during one) keeps `shade`'s coordinate system
                // consistent across a single image.
                let (width, height) = canvas_size.get();
                rgb.resize(width * height * 3, 0);

                let frame_start = Instant::now();
                render_frame(start.elapsed().as_secs_f64(), &mut rgb, width, height);
                let frame_time = frame_start.elapsed();

                // Panics inside the agent code are caught in the dylib and
                // rendered as zeroed (black) pixels; surface the message.
                if let Some(msg) = runtime.take_panic() {
                    warn!("Agent code panicked: {msg}");
                    shared
                        .lock()
                        .expect("shared state mutex is not poisoned")
                        .panic_msg = Some(msg);
                }

                *frame_slot.lock().expect("frame slot mutex is not poisoned") =
                    Some(egui::ColorImage::from_rgb([width, height], &rgb));
                {
                    let mut s = shared.lock().expect("shared state mutex is not poisoned");
                    s.canvas = (width, height);
                    s.frame_ms = frame_time.as_secs_f64() * 1e3;
                    s.mpix_per_s = (width * height) as f64 / frame_time.as_secs_f64() / 1e6;
                }
                ctx.request_repaint();

                if frame_time < TARGET_FRAME_TIME {
                    std::thread::sleep(TARGET_FRAME_TIME - frame_time);
                }
            }
        })
        .expect("can spawn the render thread");
}

/// Build the evolution prompt from the user request and live telemetry.
///
/// Deliberately prompts with only the bare function *signature* (never the
/// current or default implementation) so the agent is not anchored to an
/// existing algorithm and new creative programs can emerge. The canvas
/// conventions the agent needs are spelled out as text instead.
fn evolution_prompt(
    fn_sig: &str,
    user_prompt: &str,
    canvas: (usize, usize),
    frame_ms: f64,
    mpix_per_s: f64,
    panic_msg: Option<String>,
) -> String {
    let (width, height) = canvas;
    let aspect = width as f64 / height as f64;
    let panic_feedback = panic_msg.map_or_else(String::new, |msg| {
        format!(
            "The previous implementation panicked at runtime: \"{msg}\". Avoid that failure mode.\n"
        )
    });
    format!(
        "Implement this per-pixel shader function:\n```rust\n{fn_sig}\n```\n\
         The user wants the canvas to show: {user_prompt}\n\
         Canvas conventions: `(x, y)` is the pixel position with `(0, 0)` at \
         the center; `y` spans [-1, 1] (positive is up) and `x` spans \
         [-aspect, +aspect] where aspect is the window's aspect ratio, \
         currently {aspect:.2} but it changes when the user resizes the \
         window — do not hard-code it, and keep the composition centered. \
         `t` is seconds since program start — use it for smooth animation. \
         Return the color packed as `0x00_RR_GG_BB`.\n\
         Telemetry of the previous implementation: {frame_ms:.1} ms/frame at \
         {width}x{height} ({mpix_per_s:.1} Mpix/s).\n\
         {panic_feedback}\
         Hard constraints: keep the exact signature. The function must be pure \
         (no allocation, no I/O, no statics, no unsafe). It is called once per \
         pixel and parallelized by the host, so keep the per-call cost bounded \
         (cap iteration counts). Pick whatever algorithm best fits the request \
         — be creative. Respond with Rust code only."
    )
}

/// Spawn the evolution worker: for each user prompt it runs a single-lane
/// [`Runtime::evolve_batch`] on the tokio runtime, which registers the new
/// revision *without* activating it, then parks the render thread just long
/// enough to [`Runtime::activate_revision`].
///
/// This is what keeps the canvas alive during an evolution: `evolve_batch` is
/// explicitly exempt from the feedback-loop contract, because a
/// retained-but-inactive revision is invisible to running calls. So the render
/// thread keeps hammering the previous revision's function pointer for the
/// entire generate → validate → compile round, and the animation only pauses
/// for the atomic stores that commit the winner.
fn spawn_evolution_worker(
    prompt_rx: Receiver<String>,
    gate: Arc<Gate>,
    shared: Arc<Mutex<SharedUi>>,
    runtime: &'static Runtime,
    agent: symbiont::Agent,
    tokio_handle: tokio::runtime::Handle,
    ctx: egui::Context,
) {
    std::thread::Builder::new()
        .name("symbiont-evolution".to_string())
        .spawn(move || {
            // Only the bare signature — no default/current body — so the
            // agent is free to invent a fresh algorithm each evolution.
            let fn_sig = runtime.fn_sigs()[0].clone();
            while let Ok(user_prompt) = prompt_rx.recv() {
                let (canvas, frame_ms, mpix_per_s, panic_msg) = {
                    let mut s = shared.lock().expect("shared state mutex is not poisoned");
                    s.evolving = true;
                    s.evolve_error = None;
                    (s.canvas, s.frame_ms, s.mpix_per_s, s.panic_msg.take())
                };
                ctx.request_repaint();

                let prompt = evolution_prompt(
                    &fn_sig,
                    &user_prompt,
                    canvas,
                    frame_ms,
                    mpix_per_s,
                    panic_msg,
                );

                // One lane, one candidate. Unlike `evolve`, this registers the
                // revision without publishing it, so the render thread needs
                // no gating here and the animation keeps running.
                let evolve_start = Instant::now();
                let result = tokio_handle
                    .block_on(runtime.evolve_batch(&agent, std::slice::from_ref(&prompt)))
                    .pop()
                    .expect("one result per prompt");
                let evolve_secs = evolve_start.elapsed().as_secs_f64();

                // The candidate is compiled and loaded; committing to it is
                // the only step bound by the feedback-loop contract. Park the
                // render thread at a frame boundary, swap, resume.
                let swap = result.and_then(|info| {
                    let revision = info.revision();
                    let swap_start = Instant::now();
                    gate.drain();
                    let activated = runtime.activate_revision(revision);
                    gate.resume();
                    activated.map(|()| (revision, swap_start.elapsed()))
                });

                {
                    let mut s = shared.lock().expect("shared state mutex is not poisoned");
                    match swap {
                        Ok((revision, swap_time)) => {
                            s.code = runtime.current_code();
                            s.evolutions += 1;
                            s.last_evolve_secs = Some(evolve_secs);
                            s.last_swap_us = Some(swap_time.as_secs_f64() * 1e6);
                            s.panic_msg = None;
                            info!(
                                "Evolution #{} hot-swapped successfully (revision {revision}, \
                                 render thread parked for {:.0} us).",
                                s.evolutions,
                                swap_time.as_secs_f64() * 1e6
                            );
                        }
                        Err(e) => {
                            warn!("Evolution failed: {e}");
                            s.evolve_error = Some(e.to_string());
                        }
                    }
                    s.evolving = false;
                }
                ctx.request_repaint();
            }
        })
        .expect("can spawn the evolution worker thread");
}

/// The egui application: canvas, prompt box, telemetry and agent code view.
struct FractalApp {
    /// State shared with the render thread and evolution worker.
    shared: Arc<Mutex<SharedUi>>,
    /// Latest rendered frame, produced by the render thread.
    frame_slot: FrameSlot,
    /// Canvas size requested from the render thread, updated on every layout.
    canvas_size: Arc<CanvasSize>,
    /// GPU texture holding the current frame.
    texture: Option<egui::TextureHandle>,
    /// Contents of the prompt input box.
    prompt_input: String,
    /// Channel to the evolution worker.
    prompt_tx: Sender<String>,
}

impl FractalApp {
    /// Telemetry, evolution status and error/panic banners.
    fn status_section(ui: &mut egui::Ui, s: &SharedUi) {
        ui.monospace(format!(
            "frame time {:7.2} ms   throughput {:7.1} Mpix/s",
            s.frame_ms, s.mpix_per_s
        ));
        ui.monospace(format!(
            "canvas     {}x{}   evolutions {}",
            s.canvas.0, s.canvas.1, s.evolutions
        ));
        if let Some(secs) = s.last_evolve_secs {
            ui.monospace(format!("last evolution took {secs:.1} s"));
        }
        if let Some(us) = s.last_swap_us {
            ui.monospace(format!("of which the canvas was parked {us:.0} us"));
        }
        if s.evolving {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    "evolving: generating → validating → compiling ... \
                     (canvas keeps animating on the current revision)",
                );
            });
        }
        if let Some(err) = &s.evolve_error {
            ui.colored_label(egui::Color32::RED, format!("evolution failed: {err}"));
        }
        if let Some(panic_msg) = &s.panic_msg {
            ui.colored_label(
                egui::Color32::RED,
                format!("agent code panicked (rendered as black pixels): {panic_msg}"),
            );
        }
    }

    /// Prompt input box and the evolve button.
    fn prompt_section(&mut self, ui: &mut egui::Ui, s: &SharedUi) {
        ui.label("Describe what the canvas should show:");
        ui.add(
            egui::TextEdit::multiline(&mut self.prompt_input)
                .desired_rows(4)
                .desired_width(f32::INFINITY)
                .hint_text(
                    "e.g. \"An animated Julia set, c orbiting the main cardioid, \
                     with a glowing sunset palette and smooth iteration coloring\"",
                ),
        );
        let can_send = !s.evolving && !self.prompt_input.trim().is_empty();
        if ui
            .add_enabled(can_send, egui::Button::new("Evolve"))
            .clicked()
        {
            self.prompt_tx
                .send(self.prompt_input.trim().to_owned())
                .expect("the evolution worker outlives the UI");
        }
        ui.small(
            "Follow-up prompts refine the current shader — the chat history \
             is kept by the symbiont runtime.",
        );
    }

    /// Scrollable, syntax-highlighted view of the live agent code.
    fn code_section(ui: &mut egui::Ui, s: &SharedUi) {
        ui.label("Agent code currently running (hot-swapped native dylib):");
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let theme =
                    egui_extras::syntax_highlighting::CodeTheme::from_memory(ui.ctx(), ui.style());
                egui_extras::syntax_highlighting::code_view_ui(ui, &theme, &s.code, "rs");
            });
    }

    /// The right-hand control panel.
    fn side_panel(&mut self, ui: &mut egui::Ui) {
        let snapshot = self
            .shared
            .lock()
            .expect("shared state mutex is not poisoned")
            .clone();
        egui::Panel::right("controls")
            .resizable(true)
            .default_size(460.0)
            .show_inside(ui, |ui| {
                ui.add_space(8.0);
                ui.heading("Symbiont Fractal Studio");
                ui.label(
                    "Prompt the agent to implement the per-pixel shader as native \
                     Rust. The harness validates, compiles and hot-swaps it into \
                     the live render loop without a restart.",
                );
                ui.separator();
                Self::status_section(ui, &snapshot);
                ui.separator();
                self.prompt_section(ui, &snapshot);
                ui.separator();
                Self::code_section(ui, &snapshot);
            });
        if snapshot.evolving {
            // The render thread requests repaints on its own, but the cargo
            // build competes for every core: keep the spinner ticking even if
            // frames get sparse.
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }

    /// The central canvas, filling all the space the side panel leaves.
    ///
    /// Rather than fitting a fixed-resolution image into the panel — which
    /// letterboxes as soon as the window's aspect ratio differs from the
    /// render target's — the panel size is reported back to the render thread,
    /// which renders the *next* frame at exactly that aspect ratio. The image
    /// is then drawn at the full available size, so there are no bars.
    fn canvas(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show_inside(ui, |ui| {
                let avail = ui.available_size();
                let points_to_pixels = ui.ctx().pixels_per_point();
                self.canvas_size
                    .request(avail.x * points_to_pixels, avail.y * points_to_pixels);

                let Some(texture) = &self.texture else {
                    ui.centered_and_justified(|ui| ui.spinner());
                    return;
                };
                // While a resize is in flight the last frame still has the
                // previous aspect ratio; stretching it for the frame or two
                // until the render thread catches up reads better than bars
                // appearing and disappearing during the drag.
                ui.image((texture.id(), avail));
            });
    }
}

impl eframe::App for FractalApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Upload the latest frame from the render thread to the GPU.
        if let Some(image) = self
            .frame_slot
            .lock()
            .expect("frame slot mutex is not poisoned")
            .take()
        {
            let options = egui::TextureOptions::LINEAR;
            match &mut self.texture {
                Some(texture) => texture.set(image, options),
                None => {
                    self.texture = Some(ui.ctx().load_texture("fractal-canvas", image, options));
                }
            }
        }
        self.side_panel(ui);
        self.canvas(ui);
    }
}

fn main() -> eframe::Result<()> {
    symbiont::init_tracing();

    // Tokio runtime for the symbiont harness (LLM calls, evolution). It lives
    // for the duration of the app; the evolution worker drives futures on it
    // through its handle.
    let tokio_rt = tokio::runtime::Runtime::new().expect("can build the tokio runtime");

    // The shader is compute-bound: compile the agent dylib with optimizations.
    let runtime = tokio_rt
        .block_on(Runtime::new(
            SYMBIONT_DECLS,
            SYMBIONT_PRELUDE,
            symbiont::Profile::Release,
        ))
        .expect("can initialize the symbiont runtime");
    info!("fn_sigs: {:?}", runtime.fn_sigs());

    let model =
        std::env::var("MODEL").expect("the MODEL env var names the model slug to evolve with");
    let agent = tokio_rt
        .block_on(symbiont::init_agent_from_env(None, &model, false))
        .expect("can initialize the agent; check the API_KEY and BASE_URL env vars");
    let tokio_handle = tokio_rt.handle().clone();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 760.0])
            .with_title("Symbiont Fractal Studio"),
        ..Default::default()
    };

    eframe::run_native(
        "symbiont-fractal-studio",
        options,
        Box::new(move |cc| {
            let gate = Arc::new(Gate::new());
            let shared = Arc::new(Mutex::new(SharedUi {
                evolving: false,
                code: runtime.current_code(),
                panic_msg: None,
                evolve_error: None,
                canvas: INITIAL_SIZE,
                frame_ms: 0.0,
                mpix_per_s: 0.0,
                evolutions: 0,
                last_evolve_secs: None,
                last_swap_us: None,
            }));
            let frame_slot: FrameSlot = Arc::new(Mutex::new(None));
            let canvas_size = Arc::new(CanvasSize::new());
            let (prompt_tx, prompt_rx) = channel();

            spawn_render_thread(
                Arc::clone(&gate),
                Arc::clone(&shared),
                Arc::clone(&frame_slot),
                Arc::clone(&canvas_size),
                runtime,
                cc.egui_ctx.clone(),
            );
            spawn_evolution_worker(
                prompt_rx,
                gate,
                Arc::clone(&shared),
                runtime,
                agent,
                tokio_handle,
                cc.egui_ctx.clone(),
            );

            Ok(Box::new(FractalApp {
                shared,
                frame_slot,
                canvas_size,
                texture: None,
                prompt_input: String::new(),
                prompt_tx,
            }))
        }),
    )
}
