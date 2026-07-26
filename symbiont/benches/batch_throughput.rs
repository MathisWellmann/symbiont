// SPDX-License-Identifier: MPL-2.0
//! Concurrency sweep for [`symbiont::Runtime::evolve_batch`] against a live
//! inference server.
//!
//! Runs the same eight-lane batch at several in-flight limits and reports what
//! each level cost. At limit 1 the lanes are fully serialized — the loop this
//! API replaces. At limit 8 they are all in flight, which is what lets the
//! server's continuous batcher merge them into shared forward passes.
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
//! `s/candidate` is the number that should fall as the limit rises; the
//! `speedup` column is against the serial level. The decomposition below each
//! row separates what batching can help with from what it cannot:
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

/// In-flight limits to sweep. 1 is the serial baseline this API replaces.
const LEVELS: &[usize] = &[1, 2, 4, 8];
/// Lanes per batch. Constant across levels so every level does the same amount
/// of inference work and only the overlap differs.
const LANES: usize = 8;
/// How long to wait for a TCP connection to the inference endpoint.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
/// Output cap per lane.
///
/// Not a detail — without it the sweep measures the wrong thing. Left
/// unbounded, a chat-completions request may generate until it runs out of
/// context, and a single lane that starts rambling takes minutes while its
/// siblings finish in seconds. Because a level's wall clock is its *slowest*
/// lane, one such lane swamps the level and the comparison between levels
/// becomes noise. A few hundred tokens is ample for these implementations;
/// this leaves generous headroom while still bounding the tail.
const MAX_OUTPUT_TOKENS: u64 = 1536;

// A task with enough substance that the model emits a few hundred tokens —
// decode is what batching accelerates, so a one-liner would measure nothing.
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

const STRATEGIES: &[&str] = &[
    "a direct trial-division loop",
    "trial division bounded by the square root of the candidate",
    "trial division that skips even candidates",
    "a sieve of Eratosthenes over a boolean array",
    "a sieve of Eratosthenes with bit-packed flags",
    "a segmented sieve sized to fit in cache",
    "wheel factorization modulo 6",
    "an allocation-free strategy",
];

/// Build the eight prompts for one level.
///
/// The shared part comes first and the varying hint last, which is the layout
/// prefix caching rewards. `level` is folded into the hint so that a level does
/// not silently ride on revisions an earlier level already registered: an
/// identical candidate is deduplicated and skips its build, which would make a
/// later level look faster than it is.
fn prompts_for(signature: &str, level: usize) -> Vec<String> {
    Vec::from_iter(STRATEGIES.iter().map(|strategy| {
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
    limit: usize,
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
        self.wall / u32::try_from(LANES).expect("lane count fits in u32")
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
    let queries = prometheus_counter(&body, "vllm:prefix_cache_queries_total");
    if queries == 0.0 {
        return None;
    }
    Some(PrefixCache {
        hits: prometheus_counter(&body, "vllm:prefix_cache_hits_total"),
        queries,
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

    println!("\n| in flight | wall    | s/candidate | speedup | lanes ok | built |");
    println!("|-----------|---------|-------------|---------|----------|-------|");
    for r in results {
        let speedup = baseline.map_or_else(
            || "-".to_string(),
            |b| format!("{:.2}x", b.as_secs_f64() / r.wall.as_secs_f64()),
        );
        println!(
            "| {:>9} | {:>7} | {:>11} | {:>7} | {:>8} | {:>5} |",
            r.limit,
            format_duration(r.wall),
            format_duration(r.per_candidate()),
            speedup,
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
/// Each level installs its own recorder, so the metrics read back afterwards
/// belong to that level alone rather than to the whole sweep.
async fn run_level<A>(
    runtime: &'static Runtime,
    agent: &A,
    signature: &str,
    limit: usize,
    base_url: &str,
) -> LevelResult
where
    A: symbiont::EvolutionAgent + Sync,
{
    let cache_before = scrape_prefix_cache(base_url);
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    // A local recorder rather than a global one: the bench runs on a
    // current-thread runtime, so every lane records on this thread. That the
    // lanes still overlap on one thread is the point — they are I/O bound, and
    // since `compile_dylib` awaits cargo rather than blocking, even the builds
    // do not stall the others.
    let _guard = metrics::set_default_local_recorder(&recorder);

    let revisions_before = runtime.revision_count();
    let prompts = prompts_for(signature, limit);
    let started = Instant::now();
    let results = runtime.evolve_batch_limited(agent, &prompts, limit).await;
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

    let snapshot = take_snapshot(&snapshotter);
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
#[tokio::main(flavor = "current_thread")]
async fn main() -> symbiont::Result<()> {
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
    let runtime = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug).await?;
    let signature = runtime.fn_sigs()[0].clone();
    let agent = symbiont::agent_builder_from_env(None, &model)
        .await?
        .max_tokens(MAX_OUTPUT_TOKENS)
        .build();

    println!("Sweeping {LANES} lanes at in-flight limits {LEVELS:?} against {base_url} ({model}).");

    let mut results = Vec::with_capacity(LEVELS.len());
    for &limit in LEVELS {
        println!("\n=== in flight: {limit} ===");
        let result = run_level(runtime, &agent, &signature, limit, &base_url).await;
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
