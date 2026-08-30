// SPDX-License-Identifier: MPL-2.0
//! Concurrency sweep for [`symbiont::Runtime::evolve_batch`] against a live
//! inference server.
//!
//! Runs the same batch of [`LANES`] candidates at several in-flight limits and
//! reports what each level cost. At limit 1 the lanes are fully serialized.
//! At the top limit they are all in flight,
//! which is what lets the server's continuous batcher merge them into shared forward passes.
//!
//! # Running it
//!
//! Needs a server; see `benches/vllm/README.md` for a containerized vLLM
//! serving the same model CI uses. Then:
//!
//! ```sh
//! BASE_URL=http://127.0.0.1:8231/v1 MODEL=local \
//!   cargo bench --bench batch_throughput
//! ```
//!
//! With no reachable server it prints why it is skipping and exits 0, so
//! `cargo bench` (and CI's `cargo bench --no-run`) stay green on machines
//! without one.
//!
//! # Reading the output
//!
//! `s/candidate` is the number that should fall as the limit rises, and
//! `speedup` is against the serial level. Prefer `dec tok/s` / `tok/s gain`
//! when levels differ in how many retries they burned — wall clock rewards a
//! level that got lucky with the model, generated-tokens-per-second does not.
//!
//! # Retry variance dominates, and it is not symmetric
//!
//! Check `output tok` before believing any single row. Levels are supposed to
//! do equal work, but whether a lane converges is stochastic, and a lane that
//! spends all [`Runtime::MAX_EVOLVE_ATTEMPTS`] attempts costs roughly ten
//! times one that succeeds immediately. Observed in practice: individual
//! levels landing at exactly double the usual `output tok` and running 4-8x
//! slower than their neighbours — including one that came out *slower than
//! serial*.
//!
//! The asymmetry is the important part. Retries within a lane are sequential,
//! so they extend that lane's critical path, and a level's wall clock is its
//! slowest lane. Extra concurrency cannot recover any of it. A level with 2x
//! the tokens is therefore far worse than 2x slower.
//!
//! So: run the sweep more than once, and discard levels whose `output tok`
//! departs from the others. Reducing [`Runtime::MAX_EVOLVE_ATTEMPTS`] or using
//! prompts the model reliably satisfies would shrink the variance at the cost
//! of measuring a less realistic workload.
//! Expect wall clock to flatten before throughput does: every lane still
//! parses, validates, compiles and loads on its own, and that tail does not
//! shrink with concurrency.
//!
//! The decomposition below separates what batching can help with from what it
//! cannot:
//!
//! - **llm** is summed per-lane inference time. It stays roughly flat in total
//!   across levels — batching does not make any single request faster, it
//!   overlaps them — so watching it *not* grow is the check that the server is
//!   really batching rather than queueing.
//! - **compile** and **slot wait** are the local build pipeline, serialized
//!   behind one build slot. If slot wait starts to rival the inference time,
//!   the batch has stopped being inference-bound and the shared crate
//!   directory is the next thing to fix.
//! - **prefix hit** is the share of the prompt served from the server's prefix
//!   cache. All lanes share every token before the trailing hint, so it should
//!   be high on any server with prefix caching on. Read from vLLM's
//!   `/metrics` when available, because vLLM does not fill in the
//!   OpenAI-compatible `usage.prompt_tokens_details.cached_tokens` field that
//!   [`crate::observability::LLM_TOKENS`]`{kind="cached_input"}` is derived
//!   from — that field stays zero there even on a guaranteed cache hit.
#![expect(
    unused_crate_dependencies,
    reason = "benches don't need every dev-dependency"
)]

use std::{
    env,
    net::{
        SocketAddr,
        TcpStream,
        ToSocketAddrs,
    },
    time::{
        Duration,
        Instant,
    },
};

use metrics_util::debugging::{
    DebugValue,
    DebuggingRecorder,
    Snapshotter,
};
use symbiont::{
    Profile,
    Runtime,
    observability,
};

/// In-flight limits to sweep.
const LEVELS: &[u16] = &[1, 2, 4, 8, 16, 32];
/// Lanes per batch. Constant across levels so every level does the same amount
/// of inference work and only the overlap differs — which also means it must
/// be at least the highest limit, or that limit is silently clamped and two
/// rows of the table measure the same thing.
const LANES: u16 = 32;
/// How long to wait for a TCP connection to the inference endpoint.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
/// Output cap per lane.
const MAX_OUTPUT_TOKENS: u64 = 1536;

// A task with enough substance that the model emits a few hundred tokens.
symbiont::evolvable! {
    fn bench_count_primes(n: u32) -> u32 {
        let mut count = 0;
        let mut candidate = 2;
        while candidate < n {
            let mut is_prime = true;
            let mut divisor = 2;
            while divisor * divisor <= candidate {
                if candidate % divisor == 0 {
                    is_prime = false;
                    break;
                }
                divisor += 1;
            }
            if is_prime {
                count += 1;
            }
            candidate += 1;
        }
        count
    }
}

/// One distinct hint per lane, so a lane is a different program rather than a
/// paraphrase of its neighbour. There must be at least [`LANES`] of them.
///
/// Difficulty is deliberately mixed. The hard ones (Atkin, Legendre) mostly
/// fail on a small model, and that is fine for a throughput benchmark: a lane
/// that burns its retry budget still generates tokens, which is what is being
/// measured. It does mean levels do unequal work when their failure counts
/// differ — see the `dec tok/s` column, which normalizes for that.
const STRATEGIES: &[&str] = &[
    "a direct trial-division loop",
    "trial division bounded by the square root of the candidate",
    "trial division that skips even candidates",
    "trial division against only the primes found so far",
    "trial division against a hardcoded table of small primes",
    "a sieve of Eratosthenes over a boolean array",
    "a sieve of Eratosthenes that starts marking at i*i",
    "a sieve of Eratosthenes with bit-packed flags",
    "an odd-only sieve that stores no slots for even numbers",
    "a segmented sieve sized to fit in cache",
    "a block-wise sieve that processes one cache line at a time",
    "a sieve stored as u64 words, counting with count_ones",
    "a two-pass approach: sieve the small primes first, then the rest",
    "an incremental sieve that grows a list of primes",
    "the sieve of Sundaram",
    "the sieve of Atkin",
    "wheel factorization modulo 6",
    "wheel factorization modulo 30",
    "a deterministic Miller-Rabin test applied to each candidate",
    "a Fermat primality test with a few small bases",
    "Legendre's prime-counting recurrence",
    "an allocation-free strategy",
    "a fixed-size stack array instead of a heap allocation",
    "a branch-free inner marking loop",
    "an unrolled inner marking loop",
    "iterator chains rather than explicit indexing",
    "a single flat loop with no helper functions",
    "the fewest memory writes you can manage",
    "u64 arithmetic internally to keep overflow impossible",
    "an approach tuned for small n",
    "an approach tuned for large n",
    "the most readable implementation you can write",
];

/// Build the [`LANES`] prompts for one level.
///
/// The shared part comes first and the varying hint last, which is the layout
/// prefix caching rewards. `level` is folded into the hint so a level does not
/// ride on revisions an earlier level registered — though only partly: dedup
/// keys on generated source, which the tag never reaches, so lanes that emit
/// identical code across levels still share a revision and skip its build.
fn prompts_for(signature: &str, level: u16) -> Vec<String> {
    Vec::from_iter(STRATEGIES.iter().take(usize::from(LANES)).map(|strategy| {
        format!(
            "Implement this function, which returns how many prime numbers are \
             strictly less than `n`:\n\
             ```\n{signature}\n```\n\n\
             Rules:\n\
             - Implement the algorithm from scratch.\n\
             - `bench_count_primes(0)`, `(1)` and `(2)` must all return 0.\n\
             - The function must never panic or overflow.\n\n\
             Use {strategy}. Name this attempt v{level}.\n\
             Code only."
        )
    }))
}

/// What one level of the sweep cost.
struct LevelResult {
    limit: u16,
    wall: Duration,
    /// Distinct revisions this level added. Below `ok_lanes` when lanes
    /// converged on identical source and deduplicated.
    built: u64,
    ok_lanes: usize,
    /// Summed per-lane inference time. Exceeds `wall` when lanes overlap.
    llm: Duration,
    compile: Duration,
    slot_wait: Duration,
    input_tokens: u64,
    cached_tokens: u64,
    output_tokens: u64,
    /// Prefix-cache hit rate scraped from the server, when it exposes one.
    server_cache_rate: Option<f64>,
}

impl LevelResult {
    fn per_candidate(&self) -> Duration {
        self.wall / u32::from(LANES)
    }

    /// Generated tokens per second of wall clock.
    ///
    /// The work-normalized view of the same run, and the one to trust when
    /// levels differ in how many retries they burned: wall clock rewards a
    /// level that happened to get lucky with the model, this does not. It is
    /// also the quantity batching actually improves — decode is
    /// memory-bandwidth-bound, so overlapping requests raises tokens/s even
    /// where it can no longer lower wall clock.
    fn decode_throughput(&self) -> f64 {
        if self.wall.is_zero() {
            return 0.0;
        }
        self.output_tokens as f64 / self.wall.as_secs_f64()
    }

    /// Share of the prompt served from the server's prefix cache.
    ///
    /// Prefers the server's own counters, since the provider-reported
    /// `cached_input_tokens` is zero on backends that do not fill in
    /// `prompt_tokens_details` — vLLM among them.
    fn cache_hit_rate(&self) -> Option<f64> {
        self.server_cache_rate.or_else(|| {
            if self.input_tokens == 0 {
                None
            } else {
                Some(self.cached_tokens as f64 / self.input_tokens as f64)
            }
        })
    }
}

/// One entry of a metrics snapshot: key, unit, description, value.
type SnapshotEntry = (
    metrics_util::CompositeKey,
    Option<metrics::Unit>,
    Option<metrics::SharedString>,
    DebugValue,
);

/// Take the snapshot **once**.
///
/// `Snapshotter::snapshot` drains the recorded histogram buckets, so calling it
/// per column silently zeroes every column after the first. Every extraction
/// below therefore reads from one captured `Vec` rather than re-snapshotting.
fn take_snapshot(snapshotter: &Snapshotter) -> Vec<SnapshotEntry> {
    snapshotter.snapshot().into_vec()
}

/// Entries of one metric, optionally narrowed to a single label value.
fn series<'a>(
    snapshot: &'a [SnapshotEntry],
    name: &str,
    label: Option<(&str, &str)>,
) -> Vec<&'a SnapshotEntry> {
    snapshot
        .iter()
        .filter(|(key, _, _, _)| key.key().name() == name)
        .filter(|(key, _, _, _)| {
            label.is_none_or(|(k, v)| key.key().labels().any(|l| l.key() == k && l.value() == v))
        })
        .collect()
}

/// Sum a histogram's samples, as a duration.
fn histogram_total(
    snapshot: &[SnapshotEntry],
    name: &str,
    label: Option<(&str, &str)>,
) -> Duration {
    let secs: f64 = series(snapshot, name, label)
        .iter()
        .filter_map(|(_, _, _, value)| match value {
            DebugValue::Histogram(samples) => {
                Some(samples.iter().map(|s| s.into_inner()).sum::<f64>())
            }
            _ => None,
        })
        .sum();
    Duration::from_secs_f64(secs)
}

/// Read a counter, summed over matching series.
fn counter_total(snapshot: &[SnapshotEntry], name: &str, label: Option<(&str, &str)>) -> u64 {
    series(snapshot, name, label)
        .iter()
        .filter_map(|(_, _, _, value)| match value {
            DebugValue::Counter(c) => Some(*c),
            _ => None,
        })
        .sum()
}

/// vLLM's prefix-cache counters, scraped from its Prometheus endpoint.
///
/// The OpenAI-compatible `usage.prompt_tokens_details.cached_tokens` field —
/// which is what rig surfaces as `cached_input_tokens`, and the only
/// provider-agnostic signal symbiont has — is **not** populated by vLLM, even
/// on a guaranteed cache hit. Its prefix cache is nevertheless real and
/// counted; it is just only visible on `/metrics`. So the benchmark reads it
/// from there and falls back to the provider-reported number for servers that
/// do report it.
#[derive(Clone, Copy, Default)]
struct PrefixCache {
    hits: f64,
    queries: f64,
}

impl PrefixCache {
    /// Hit rate between two scrapes, or `None` if nothing was queried.
    fn rate_since(self, before: Self) -> Option<f64> {
        let queries = self.queries - before.queries;
        if queries <= 0.0 {
            return None;
        }
        Some((self.hits - before.hits) / queries)
    }
}

/// Minimal HTTP/1.0 GET, so the benchmark can read a Prometheus endpoint
/// without pulling in an HTTP client. 1.0 rather than 1.1 to avoid having to
/// deal with chunked framing: the server closes the connection at the end of
/// the body.
fn http_get(endpoint: &str, path: &str) -> Option<String> {
    use std::io::{
        Read,
        Write,
    };

    let addr = endpoint.to_socket_addrs().ok()?.next()?;
    let mut stream = TcpStream::connect_timeout(&addr, PROBE_TIMEOUT).ok()?;
    stream.set_read_timeout(Some(PROBE_TIMEOUT)).ok()?;
    let host = endpoint.split(':').next().unwrap_or(endpoint);
    write!(stream, "GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n").ok()?;

    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let (_, body) = response.split_once("\r\n\r\n")?;
    Some(body.to_string())
}

/// Value of a Prometheus counter, summed over label sets.
fn prometheus_counter(body: &str, name: &str) -> f64 {
    body.lines()
        .filter(|line| !line.starts_with('#'))
        .filter(|line| line.starts_with(name))
        // Only `name` or `name{labels...}`, never `name_something_else`.
        .filter(|line| line[name.len()..].starts_with(['{', ' ']))
        .filter_map(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
        .sum()
}

/// Scrape vLLM's prefix-cache counters, if the endpoint exposes them.
fn scrape_prefix_cache(base_url: &str) -> Option<PrefixCache> {
    let body = http_get(&endpoint_of(base_url)?, "/metrics")?;
    // Presence of the series, not a non-zero value, is what says the server
    // exposes this. A freshly started server reports a legitimate zero, and
    // treating that as "unavailable" would discard the *baseline* of the very
    // first level's diff and silently fall back to the provider-reported
    // figure — which is always zero on vLLM, so the level would report a
    // confident 0% instead of its real hit rate.
    if !body
        .lines()
        .any(|line| !line.starts_with('#') && line.starts_with("vllm:prefix_cache_queries_total"))
    {
        return None;
    }
    Some(PrefixCache {
        hits: prometheus_counter(&body, "vllm:prefix_cache_hits_total"),
        queries: prometheus_counter(&body, "vllm:prefix_cache_queries_total"),
    })
}

/// Host and port of an `http(s)://host[:port]/path` URL.
fn endpoint_of(base_url: &str) -> Option<String> {
    let rest = base_url
        .strip_prefix("http://")
        .or_else(|| base_url.strip_prefix("https://"))?;
    let default_port = if base_url.starts_with("https://") {
        443
    } else {
        80
    };
    let authority = rest.split('/').next()?;
    if authority.is_empty() {
        return None;
    }
    if authority.contains(':') {
        Some(authority.to_string())
    } else {
        Some(format!("{authority}:{default_port}"))
    }
}

/// Whether a TCP connection to the inference endpoint succeeds.
///
/// A cheap probe so the bench can skip cleanly instead of burning the
/// self-healing retry ladder against a closed port.
fn server_reachable(base_url: &str) -> bool {
    let Some(endpoint) = endpoint_of(base_url) else {
        return false;
    };
    let Ok(addrs) = endpoint.to_socket_addrs() else {
        return false;
    };
    let addrs: Vec<SocketAddr> = addrs.collect();
    addrs
        .iter()
        .any(|addr| TcpStream::connect_timeout(addr, PROBE_TIMEOUT).is_ok())
}

fn format_duration(d: Duration) -> String {
    if d.as_secs() >= 1 {
        format!("{:.1}s", d.as_secs_f64())
    } else {
        format!("{}ms", d.as_millis())
    }
}

fn print_results(results: &[LevelResult]) {
    let baseline = results.first().map(|r| r.wall);

    let baseline_throughput = results.first().map(LevelResult::decode_throughput);

    println!(
        "\n| in flight | wall    | s/candidate | speedup | dec tok/s | tok/s gain | lanes ok | built |"
    );
    println!(
        "|-----------|---------|-------------|---------|-----------|------------|----------|-------|"
    );
    for r in results {
        let speedup = baseline.map_or_else(
            || "-".to_string(),
            |b| format!("{:.2}x", b.as_secs_f64() / r.wall.as_secs_f64()),
        );
        let gain = baseline_throughput.filter(|b| *b > 0.0).map_or_else(
            || "-".to_string(),
            |b| format!("{:.2}x", r.decode_throughput() / b),
        );
        println!(
            "| {:>9} | {:>7} | {:>11} | {:>7} | {:>9.0} | {:>10} | {:>8} | {:>5} |",
            r.limit,
            format_duration(r.wall),
            format_duration(r.per_candidate()),
            speedup,
            r.decode_throughput(),
            gain,
            format!("{}/{LANES}", r.ok_lanes),
            r.built,
        );
    }

    println!(
        "\n| in flight | llm (sum) | compile | slot wait | prompt tok | output tok | prefix hit |"
    );
    println!(
        "|-----------|-----------|---------|-----------|------------|------------|------------|"
    );
    for r in results {
        println!(
            "| {:>9} | {:>9} | {:>7} | {:>9} | {:>10} | {:>10} | {:>10} |",
            r.limit,
            format_duration(r.llm),
            format_duration(r.compile),
            format_duration(r.slot_wait),
            r.input_tokens,
            r.output_tokens,
            r.cache_hit_rate()
                .map_or_else(|| "n/a".to_string(), |rate| format!("{:.0}%", rate * 100.0)),
        );
    }
}

/// One level: eight lanes at the given in-flight limit, measured in isolation.
///
/// One shared global recorder, snapshotted once per level. `snapshot()` drains
/// what it reports, so each snapshot contains exactly the level that just ran —
/// no diffing, provided nothing else snapshots in between.
///
/// A global rather than a per-level thread-local recorder, because the sweep
/// runs on a multi-threaded runtime: lanes are polled on whichever worker is
/// free, and a thread-local recorder would silently miss everything recorded
/// off the installing thread.
async fn run_level<A>(
    runtime: &'static Runtime,
    agent: &A,
    signature: &str,
    limit: u16,
    base_url: &str,
    snapshotter: &Snapshotter,
) -> LevelResult
where
    A: symbiont::EvolutionAgent + Sync,
{
    let cache_before = scrape_prefix_cache(base_url);
    // Drain anything left from the previous level so this level starts clean.
    let _ = take_snapshot(snapshotter);

    let revisions_before = runtime.revision_count();
    let prompts = prompts_for(signature, limit);
    let started = Instant::now();
    runtime.set_max_in_flight(limit);
    let results = runtime.evolve_batch(agent, &prompts).await;
    let wall = started.elapsed();
    let built = runtime.revision_count() - revisions_before;

    for (lane, result) in results.iter().enumerate() {
        if let Err(e) = result {
            eprintln!("  lane {lane} failed: {e}");
        }
    }

    let server_cache_rate = cache_before
        .zip(scrape_prefix_cache(base_url))
        .and_then(|(before, after)| after.rate_since(before));

    let snapshot = take_snapshot(snapshotter);
    let stage = observability::PIPELINE_STAGE_DURATION;
    let tokens = observability::LLM_TOKENS;

    LevelResult {
        limit,
        wall,
        built,
        ok_lanes: results.iter().filter(|r| r.is_ok()).count(),
        llm: histogram_total(&snapshot, stage, Some(("stage", "llm"))),
        compile: histogram_total(&snapshot, stage, Some(("stage", "compile"))),
        slot_wait: histogram_total(&snapshot, observability::BUILD_SLOT_WAIT, None),
        input_tokens: counter_total(&snapshot, tokens, Some(("kind", "input"))),
        cached_tokens: counter_total(&snapshot, tokens, Some(("kind", "cached_input"))),
        output_tokens: counter_total(&snapshot, tokens, Some(("kind", "output"))),
        server_cache_rate,
    }
}

// Current-thread: the lanes are I/O bound and the metrics recorder is
// thread-local, so a single worker both suffices and keeps the measurement
// attributable.
// Multi-threaded, because every lane does real CPU work off the wire —
// deserializing the response, parsing it with `syn`, re-rendering it with
// `prettyplease`, validating signatures, then `dlopen`ing the result — and on
// one worker that serializes behind itself.
//
// Measured, it turns out not to matter much here: limit 16 came in at 56.4s,
// 58.4s and 57.2s across two single-threaded runs and one multi-threaded one.
// The wild outliers that prompted the switch (a level landing 4-8x slower than
// its neighbours) were not the runtime at all — see the note on retry variance
// in the module docs. Multi-threaded is kept as the honest default for a
// benchmark at these widths, not as a fix.
#[tokio::main]
async fn main() -> symbiont::Result<()> {
    // A level above `LANES` admits every lane at once, duplicating the `LANES`
    // row instead of measuring anything new.
    assert!(
        LEVELS.iter().all(|&level| level <= LANES),
        "every level must be <= LANES ({LANES}), else it duplicates the LANES row"
    );
    assert!(
        STRATEGIES.len() >= usize::from(LANES),
        "need at least one distinct strategy per lane ({LANES}), have {}",
        STRATEGIES.len()
    );

    let base_url = env::var("BASE_URL").unwrap_or_default();
    let model = env::var("MODEL").unwrap_or_default();

    if base_url.is_empty() || model.is_empty() {
        println!(
            "skipping batch_throughput: BASE_URL and MODEL must both be set \
             (see benches/vllm/README.md)"
        );
        return Ok(());
    }
    if !server_reachable(&base_url) {
        println!("skipping batch_throughput: no server reachable at {base_url}");
        return Ok(());
    }

    // Debug profile: this benchmark measures the evolution pipeline, not the
    // speed of the generated code, and a release build per candidate would add
    // seconds of cargo to every lane for no measurement value.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder)
        .expect("no other recorder is installed in this benchmark");

    let runtime = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug).await?;
    let signature = runtime.fn_sigs()[0].clone();
    let agent = symbiont::Agent::new(
        symbiont::agent_builder_from_env(None, symbiont::DocMode::default(), &model, false)
            .await?
            .max_tokens(MAX_OUTPUT_TOKENS)
            .build(),
        base_url.clone(),
    );

    println!("Sweeping {LANES} lanes at in-flight limits {LEVELS:?} against {base_url} ({model}).");

    let mut results = Vec::with_capacity(LEVELS.len());
    for &limit in LEVELS {
        println!("\n=== in flight: {limit} ===");
        let result = run_level(runtime, &agent, &signature, limit, &base_url, &snapshotter).await;
        println!(
            "  {} wall, {} per candidate, {}/{LANES} lanes ok",
            format_duration(result.wall),
            format_duration(result.per_candidate()),
            result.ok_lanes,
        );
        results.push(result);
    }

    print_results(&results);
    Ok(())
}
