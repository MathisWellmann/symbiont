// SPDX-License-Identifier: MPL-2.0
//! Running candidate kernels in a child process.
//!
//! Symbiont contains a misbehaving *CPU* implementation inside the dylib: the
//! generated code is wrapped in `catch_unwind`, a panic is turned into a
//! default return value, and the message is fed back to the agent. The loop
//! keeps running in the same process.
//!
//! A GPU kernel cannot be contained that way. An illegal memory access is a
//! **sticky** CUDA error: it does not just fail the launch, it invalidates the
//! context — and, as this example's own probe showed, `cuCtxCreate` and
//! `cuDevicePrimaryCtxRetain` then keep returning `CUDA_ERROR_ILLEGAL_ADDRESS`
//! for the rest of the process. Dropping every handle first does not help.
//! There is no in-process recovery; the process is what has to be replaced.
//!
//! So the parent process never launches agent-written kernels. It re-executes
//! itself once per candidate with [`EVAL_ENV`] set, and that child does the
//! compiling, verifying and timing. A candidate that faults the device, or
//! hangs, or segfaults the process outright, costs one child and one table
//! row — the search continues.

use std::{
    io,
    path::{
        Path,
        PathBuf,
    },
    process::Command,
    time::{
        Duration,
        Instant,
    },
};

use crate::{
    COLS,
    Gpu,
    KernelFailure,
    KernelPlan,
    Measurement,
    ROWS,
};

/// Set on the child process; names the file holding the plan to evaluate.
/// The child writes its verdict to the same path with `.report` appended.
pub const EVAL_ENV: &str = "CUDA_SOFTMAX_EVAL_PLAN";
/// Set on the child process; the copy ceiling every candidate is scored
/// against, in GB/s.
///
/// Measured once by the parent and handed down, so that every percentage in a
/// run refers to the same number. Letting each child measure its own ceiling
/// made the reports incoherent — a kernel came back as "2613 GB/s, 46% of a
/// 636 GB/s ceiling" because the two processes had caught the GPU at
/// different clock states.
pub const ROOFLINE_ENV: &str = "CUDA_SOFTMAX_ROOFLINE";
/// Appended to the plan path to get the report path.
const REPORT_SUFFIX: &str = ".report";
/// How long a candidate may take before the child is killed. Generous: it
/// covers CUDA init, NVRTC, and the timed runs. A kernel that exceeds it is
/// almost certainly stuck in an unterminated loop.
const EVAL_TIMEOUT: Duration = Duration::from_secs(90);

/// Spawns child processes to evaluate candidate kernels.
#[derive(Debug)]
pub struct Isolated {
    exe: PathBuf,
    dir: PathBuf,
    next: std::cell::Cell<u64>,
    roofline_gb_per_s: Option<f64>,
}

impl Isolated {
    /// Prepare a scratch directory for plan/report files.
    ///
    /// # Errors
    ///
    /// If the running executable cannot be located or the scratch directory
    /// cannot be created.
    pub fn new() -> io::Result<Self> {
        Self::with_executable(std::env::current_exe()?)
    }

    /// [`Isolated::new`] with an explicit evaluator binary.
    ///
    /// The binary must call [`evaluator_main`] before doing anything else.
    /// Tests use this to drive the real child protocol against the example's
    /// binary, which `current_exe` would not point at.
    ///
    /// # Errors
    ///
    /// If the scratch directory cannot be created.
    pub fn with_executable(exe: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("cuda-softmax-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            exe: exe.into(),
            dir,
            next: std::cell::Cell::new(0),
            roofline_gb_per_s: None,
        })
    }

    /// Score every candidate against `gb_per_s` instead of letting each child
    /// measure its own ceiling. See [`ROOFLINE_ENV`].
    #[must_use]
    pub fn with_roofline(mut self, gb_per_s: f64) -> Self {
        self.roofline_gb_per_s = Some(gb_per_s);
        self
    }

    /// Compile, verify and time `plan` in a child process.
    ///
    /// # Errors
    ///
    /// The [`KernelFailure`] the child reported, or a synthesized one when the
    /// child died, timed out, or produced no report — all of which mean the
    /// candidate took the CUDA context down with it.
    pub fn evaluate(&self, plan: &KernelPlan) -> Result<Measurement, KernelFailure> {
        let id = self.next.get();
        self.next.set(id + 1);
        let plan_path = self.dir.join(format!("plan-{id}.cu"));
        let report_path = report_path_of(&plan_path);

        write_plan(&plan_path, plan)
            .map_err(|err| KernelFailure::Launch(format!("could not stage the plan: {err}")))?;

        let status = self.run_child(&plan_path)?;
        let report = std::fs::read_to_string(&report_path).map_err(|_| {
            KernelFailure::Launch(format!(
                "the evaluation process exited with {status} without reporting — the kernel \
                 faulted the CUDA context or crashed the process"
            ))
        })?;
        parse_report(&report)
    }

    /// Run one child to completion, killing it if it overruns.
    fn run_child(&self, plan_path: &Path) -> Result<String, KernelFailure> {
        let spawn_failed =
            |err: &dyn std::fmt::Display| KernelFailure::Launch(format!("child process: {err}"));
        let mut command = Command::new(&self.exe);
        command
            .env(EVAL_ENV, plan_path)
            // The child is a measurement harness, not a participant in the
            // evolution loop: keep its logs out of the parent's report.
            .env("RUST_LOG", "error");
        if let Some(roofline) = self.roofline_gb_per_s {
            command.env(ROOFLINE_ENV, roofline.to_string());
        }
        let mut child = command.spawn().map_err(|err| spawn_failed(&err))?;

        let deadline = Instant::now() + EVAL_TIMEOUT;
        loop {
            match child.try_wait().map_err(|err| spawn_failed(&err))? {
                Some(status) => return Ok(status.to_string()),
                None if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(KernelFailure::Launch(format!(
                        "the kernel did not finish within {}s and was killed — check for an \
                         unterminated loop",
                        EVAL_TIMEOUT.as_secs()
                    )));
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
}

/// The child half: evaluate the staged plan and write the report.
///
/// Returns `None` in the parent process (where [`EVAL_ENV`] is unset), so
/// `main` can call it first thing and carry on. Returns `Some(exit_code)` in a
/// child, which should exit with it immediately.
#[must_use]
pub fn evaluator_main() -> Option<i32> {
    let plan_path = PathBuf::from(std::env::var_os(EVAL_ENV)?);
    let report_path = report_path_of(&plan_path);

    let report = match read_plan(&plan_path) {
        Ok(plan) => match Gpu::new(ROWS, COLS) {
            // Anything past this point may kill the process; that is the
            // reason this code is in a child at all.
            Ok(mut gpu) => match gpu.evaluate(&plan) {
                Ok(measurement) => format_ok(&measurement),
                Err(failure) => format_err(&failure),
            },
            Err(err) => format!("err\tdevice\t{}", escape(&err.to_string())),
        },
        Err(err) => format!("err\tplan\t{}", escape(&err.to_string())),
    };

    // Only reached when the candidate did not take the process down with it.
    if let Err(err) = std::fs::write(&report_path, report) {
        eprintln!("could not write {}: {err}", report_path.display());
        return Some(2);
    }
    Some(0)
}

fn report_path_of(plan_path: &Path) -> PathBuf {
    let mut path = plan_path.as_os_str().to_owned();
    path.push(REPORT_SUFFIX);
    PathBuf::from(path)
}

/// Plan file format: three header lines, then the CUDA source verbatim.
fn write_plan(path: &Path, plan: &KernelPlan) -> io::Result<()> {
    let (gx, gy, gz) = plan.grid;
    let (bx, by, bz) = plan.block;
    std::fs::write(
        path,
        format!(
            "grid {gx} {gy} {gz}\nblock {bx} {by} {bz}\nshared {}\nsource\n{}",
            plan.shared_bytes, plan.source
        ),
    )
}

fn read_plan(path: &Path) -> io::Result<KernelPlan> {
    let raw = std::fs::read_to_string(path)?;
    let (header, source) = raw
        .split_once("source\n")
        .ok_or_else(|| io::Error::other("malformed plan: no source marker"))?;
    let mut plan = KernelPlan {
        source: source.to_string(),
        ..KernelPlan::default()
    };
    for line in header.lines() {
        let mut parts = line.split_whitespace();
        let field = parts.next().unwrap_or_default();
        let mut number = || {
            parts
                .next()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0)
        };
        match field {
            "grid" => plan.grid = (number(), number(), number()),
            "block" => plan.block = (number(), number(), number()),
            "shared" => plan.shared_bytes = number(),
            _ => return Err(io::Error::other(format!("malformed plan line: {line}"))),
        }
    }
    Ok(plan)
}

fn format_ok(m: &Measurement) -> String {
    format!(
        "ok\t{}\t{}\t{}\t{}",
        m.micros, m.gb_per_s, m.pct_of_roofline, m.max_abs_err
    )
}

fn format_err(failure: &KernelFailure) -> String {
    let max_abs_err = match failure {
        KernelFailure::Wrong { max_abs_err, .. } => max_abs_err.to_string(),
        _ => "-".to_string(),
    };
    format!(
        "err\t{}\t{max_abs_err}\t{}",
        failure.kind(),
        escape(failure.detail())
    )
}

/// Reports are one line, so embedded newlines are escaped.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('\n', "\\n")
}

fn unescape(text: &str) -> String {
    text.replace("\\n", "\n").replace("\\\\", "\\")
}

fn parse_report(report: &str) -> Result<Measurement, KernelFailure> {
    let mut fields = report.trim_end().split('\t');
    match fields.next() {
        Some("ok") => {
            let mut number = || fields.next().and_then(|v| v.parse::<f64>().ok());
            let (Some(micros), Some(gb_per_s), Some(pct_of_roofline), Some(max_abs_err)) =
                (number(), number(), number(), number())
            else {
                return Err(KernelFailure::Launch(format!(
                    "unparseable report: {report}"
                )));
            };
            Ok(Measurement {
                micros,
                gb_per_s,
                pct_of_roofline,
                max_abs_err: max_abs_err as f32,
            })
        }
        Some("err") => {
            let kind = fields.next().unwrap_or("unknown");
            let max_abs_err = fields
                .next()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(f32::NAN);
            let message = unescape(fields.next().unwrap_or_default());
            Err(rehydrate(kind, max_abs_err, message))
        }
        _ => Err(KernelFailure::Launch(format!(
            "unparseable report: {report}"
        ))),
    }
}

/// Turn a report line back into the variant the child produced, so the parent
/// can classify it the same way in its table.
fn rehydrate(kind: &str, max_abs_err: f32, message: String) -> KernelFailure {
    match kind {
        "bad geometry" => KernelFailure::Geometry(message),
        "nvrtc error" => KernelFailure::Compile(message),
        "missing symbol" => KernelFailure::MissingSymbol(message),
        "wrong output" => KernelFailure::Wrong {
            max_abs_err,
            detail: message,
        },
        _ => KernelFailure::Launch(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_round_trip_through_the_staging_file() {
        let dir = std::env::temp_dir().join("cuda-softmax-roundtrip");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("plan.cu");
        let plan = KernelPlan {
            source: "extern \"C\" __global__ void softmax() {\n  // source\n}\n".to_string(),
            grid: (7, 2, 1),
            block: (128, 1, 1),
            shared_bytes: 4096,
        };
        write_plan(&path, &plan).expect("write");
        assert_eq!(read_plan(&path).expect("read"), plan);
    }

    #[test]
    fn measurements_round_trip_through_the_report() {
        let measurement = Measurement {
            micros: 12.5,
            gb_per_s: 1234.0,
            pct_of_roofline: 88.5,
            max_abs_err: 1e-7,
        };
        let parsed = parse_report(&format_ok(&measurement)).expect("ok report");
        assert!((parsed.micros - 12.5).abs() < 1e-9);
        assert!((parsed.pct_of_roofline - 88.5).abs() < 1e-9);
    }

    #[test]
    fn failures_keep_their_kind_and_multi_line_message() {
        let failure = KernelFailure::Compile("line 1\nline 2".to_string());
        let parsed = parse_report(&format_err(&failure)).expect_err("err report");
        assert_eq!(parsed.kind(), "nvrtc error");
        assert!(parsed.to_string().contains("line 2"), "{parsed}");
    }

    #[test]
    fn a_wrong_answer_keeps_its_error_magnitude_and_is_labelled_once() {
        let failure = KernelFailure::Wrong {
            max_abs_err: 9.25e-3,
            detail: "output[0][0] = 1, expected 2".to_string(),
        };
        let parsed = parse_report(&format_err(&failure)).expect_err("err report");
        let text = parsed.to_string();
        assert_eq!(text.matches("wrong output").count(), 1, "{text}");
        assert!(text.contains("9.250e-3"), "{text}");
    }

    #[test]
    fn a_report_that_never_arrived_is_a_launch_fault() {
        let parsed = parse_report("").expect_err("empty report");
        assert!(parsed.poisons_context());
    }
}
