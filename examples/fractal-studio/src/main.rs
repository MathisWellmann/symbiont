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
//! - evolution worker: receives user prompts, parks the render thread at a
//!   frame boundary (the feedback-loop contract: no evolvable calls may be in
//!   flight during `Runtime::evolve`), evolves, then resumes rendering.

use std::{
    sync::{
        Arc,
        Condvar,
        Mutex,
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
    ///   `[-aspect, +aspect]` where `aspect = width / height` (~1.78).
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
        // Default implementation: a gently pulsing grayscale Mandelbrot set,
        // so the canvas shows something before the first evolution.
        const MAX_ITER: u32 = 256;
        let cx = x * 0.95 - 0.6;
        let cy = y * 0.95;
        let (mut zx, mut zy) = (0.0_f64, 0.0_f64);
        let mut i = 0_u32;
        while zx * zx + zy * zy <= 4.0 && i < MAX_ITER {
            let next_zx = zx * zx - zy * zy + cx;
            zy = 2.0 * zx * zy + cy;
            zx = next_zx;
            i += 1;
        }
        if i == MAX_ITER {
            return 0x000000;
        }
        let pulse = 0.75 + 0.25 * (t * 0.8).sin();
        let v = ((f64::from(i) / f64::from(MAX_ITER)).sqrt() * 255.0 * pulse) as u32;
        (v << 16) | (v << 8) | v
    }
}

/// Fixed render resolution; the canvas is scaled to fit the window.
const WIDTH: usize = 960;
/// Fixed render resolution; the canvas is scaled to fit the window.
const HEIGHT: usize = 540;
/// Aspect ratio used to scale the `x` coordinate passed to `shade`.
const ASPECT: f64 = WIDTH as f64 / HEIGHT as f64;
/// Frame pacing target (~60 fps). Rendering faster than this just burns CPU.
const TARGET_FRAME_TIME: Duration = Duration::from_millis(16);

/// Render one full frame into an RGB byte buffer (3 bytes per pixel) by
/// calling the hot-swappable `shade` function for every pixel, parallelized
/// over rows with rayon.
fn render_frame(t: f64, rgb: &mut [u8]) {
    rgb.par_chunks_mut(WIDTH * 3)
        .enumerate()
        .for_each(|(py, row)| {
            // `y` points up: top row maps to +1, bottom row to -1.
            let y = 1.0 - 2.0 * (py as f64 / (HEIGHT - 1) as f64);
            for (px, pixel) in row.chunks_exact_mut(3).enumerate() {
                let x = (2.0 * (px as f64 / (WIDTH - 1) as f64) - 1.0) * ASPECT;
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
/// evolvable function call is in flight while [`Runtime::evolve`] hot-swaps
/// the dylib (the feedback-loop contract).
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

    /// Called by the evolution worker. Blocks until the render thread has
    /// parked at a frame boundary, guaranteeing no in-flight `shade` calls.
    fn drain(&self) {
        let mut state = self.state.lock().expect("gate mutex is not poisoned");
        if *state == GateState::Run {
            *state = GateState::DrainRequested;
        }
        while *state != GateState::Paused {
            state = self.cvar.wait(state).expect("gate mutex is not poisoned");
        }
    }

    /// Called by the evolution worker after the hot-swap to resume rendering.
    fn resume(&self) {
        *self.state.lock().expect("gate mutex is not poisoned") = GateState::Run;
        self.cvar.notify_all();
    }
}

/// State shared between the render thread, the evolution worker and the UI.
#[derive(Debug, Clone)]
struct SharedUi {
    /// True while the evolution worker is generating / compiling / swapping.
    evolving: bool,
    /// The current agent-generated code running in the dylib.
    code: String,
    /// Message of the last panic caught inside the agent code, if any.
    panic_msg: Option<String>,
    /// Error message of the last failed evolution, if any.
    evolve_error: Option<String>,
    /// Most recent frame time in milliseconds.
    frame_ms: f64,
    /// Most recent throughput in megapixels per second.
    mpix_per_s: f64,
    /// Number of successful evolutions so far.
    evolutions: usize,
    /// Wall-clock duration of the last successful evolution in seconds.
    last_evolve_secs: Option<f64>,
}

/// Slot holding the most recently rendered frame for the UI to pick up.
type FrameSlot = Arc<Mutex<Option<egui::ColorImage>>>;

/// Spawn the render thread: an endless loop of frame rendering, telemetry
/// updates and panic collection, parking at the gate during evolutions.
fn spawn_render_thread(
    gate: Arc<Gate>,
    shared: Arc<Mutex<SharedUi>>,
    frame_slot: FrameSlot,
    runtime: &'static Runtime,
    ctx: egui::Context,
) {
    std::thread::Builder::new()
        .name("symbiont-render".to_string())
        .spawn(move || {
            let start = Instant::now();
            let mut rgb = vec![0_u8; WIDTH * HEIGHT * 3];
            loop {
                gate.frame_boundary();

                let frame_start = Instant::now();
                render_frame(start.elapsed().as_secs_f64(), &mut rgb);
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
                    Some(egui::ColorImage::from_rgb([WIDTH, HEIGHT], &rgb));
                {
                    let mut s = shared.lock().expect("shared state mutex is not poisoned");
                    s.frame_ms = frame_time.as_secs_f64() * 1e3;
                    s.mpix_per_s = (WIDTH * HEIGHT) as f64 / frame_time.as_secs_f64() / 1e6;
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
    frame_ms: f64,
    mpix_per_s: f64,
    panic_msg: Option<String>,
) -> String {
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
         [-{ASPECT:.2}, {ASPECT:.2}]. `t` is seconds since program start — use \
         it for smooth animation. Return the color packed as `0x00_RR_GG_BB`.\n\
         Telemetry of the previous implementation: {frame_ms:.1} ms/frame at \
         {WIDTH}x{HEIGHT} ({mpix_per_s:.1} Mpix/s).\n\
         {panic_feedback}\
         Hard constraints: keep the exact signature. The function must be pure \
         (no allocation, no I/O, no statics, no unsafe). It is called once per \
         pixel and parallelized by the host, so keep the per-call cost bounded \
         (cap iteration counts). Pick whatever algorithm best fits the request \
         — be creative. Respond with Rust code only."
    )
}

/// Spawn the evolution worker: for each user prompt it parks the render
/// thread (feedback-loop contract), runs [`Runtime::evolve`] on the tokio
/// runtime, publishes the new agent code to the UI, and resumes rendering.
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
                let (frame_ms, mpix_per_s, panic_msg) = {
                    let mut s = shared.lock().expect("shared state mutex is not poisoned");
                    s.evolving = true;
                    s.evolve_error = None;
                    (s.frame_ms, s.mpix_per_s, s.panic_msg.take())
                };
                ctx.request_repaint();

                let prompt =
                    evolution_prompt(&fn_sig, &user_prompt, frame_ms, mpix_per_s, panic_msg);

                // Feedback-loop contract: park the render thread so no
                // evolvable call is in flight while the dylib is swapped.
                gate.drain();
                let evolve_start = Instant::now();
                let result = tokio_handle.block_on(runtime.evolve(&agent, &prompt));
                {
                    let mut s = shared.lock().expect("shared state mutex is not poisoned");
                    match result {
                        Ok(revision) => {
                            s.code = runtime.current_code();
                            s.evolutions += 1;
                            s.last_evolve_secs = Some(evolve_start.elapsed().as_secs_f64());
                            s.panic_msg = None;
                            info!(
                                "Evolution #{} hot-swapped successfully (revision {revision}).",
                                s.evolutions
                            );
                        }
                        Err(e) => {
                            warn!("Evolution failed: {e}");
                            s.evolve_error = Some(e.to_string());
                        }
                    }
                    s.evolving = false;
                }
                gate.resume();
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
            "canvas     {WIDTH}x{HEIGHT}   evolutions {}",
            s.evolutions
        ));
        if let Some(secs) = s.last_evolve_secs {
            ui.monospace(format!("last evolution took {secs:.1} s"));
        }
        if s.evolving {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("evolving: generating → validating → compiling → hot-swapping ...");
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
            // Keep the spinner animated while the render thread is parked.
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }

    /// The central canvas, scaled to fit while preserving aspect ratio.
    fn canvas(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show_inside(ui, |ui| {
                let Some(texture) = &self.texture else {
                    ui.centered_and_justified(|ui| ui.spinner());
                    return;
                };
                let avail = ui.available_size();
                let scale = (avail.x / WIDTH as f32).min(avail.y / HEIGHT as f32);
                let size = egui::vec2(WIDTH as f32 * scale, HEIGHT as f32 * scale);
                ui.centered_and_justified(|ui| ui.image((texture.id(), size)));
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

    let agent = tokio_rt
        .block_on(symbiont::init_agent(None))
        .expect("can initialize the agent; check the API_KEY, BASE_URL and MODEL env vars");
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
                frame_ms: 0.0,
                mpix_per_s: 0.0,
                evolutions: 0,
                last_evolve_secs: None,
            }));
            let frame_slot: FrameSlot = Arc::new(Mutex::new(None));
            let (prompt_tx, prompt_rx) = channel();

            spawn_render_thread(
                Arc::clone(&gate),
                Arc::clone(&shared),
                Arc::clone(&frame_slot),
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
                texture: None,
                prompt_input: String::new(),
                prompt_tx,
            }))
        }),
    )
}
