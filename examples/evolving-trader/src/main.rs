// SPDX-License-Identifier: MPL-2.0
//! Trading strategy evolution: the LLM evolves a futures trading strategy
//! against a simulated exchange.
//!
//! Raw BitMEX XBTUSD trades are aggregated into **volume candles** using
//! [`trade_aggregation`](https://crates.io/crates/trade_aggregation)'s
//! information-driven rules, and executions are simulated with the leveraged
//! futures exchange [`lfest`](https://crates.io/crates/lfest), including
//! taker fees, spread and margin requirements.
//!
//! The evolvable `decide` function receives a sliding window of candles plus
//! the current account state and returns an [`Action`] (hold / market buy /
//! market sell). Each round the harness backtests the evolved strategy on the
//! training segment of the data and feeds the performance report (return,
//! drawdown, Sharpe, fees, rejected orders) back to the LLM. After the search,
//! the best revision is re-activated — its dylib stayed loaded, so no
//! recompilation — and only then evaluated on the held-out test segment.
//! Finally, the top revisions run side by side as an equal-weight **ensemble**
//! on the test segment via typed `RevisionFn` handles: several evolved
//! strategies executing concurrently as compiled code in one process.
//!
//! This showcases symbiont evolving **quantitative reasoning** through code:
//! the LLM must discover features (momentum, volatility, order flow), position
//! sizing and fee-awareness — expressed as compiled Rust running against a
//! realistic exchange simulation.

use std::num::{
    NonZeroU16,
    NonZeroU32,
};

use evolving_trader_example::prelude::*;
use lfest::prelude::{
    BaseCurrency,
    Config,
    ContractSpecification,
    Exchange,
    Fee,
    MarketOrder,
    NoUserOrderId,
    OrderRateLimits,
    PriceFilter,
    QuantityFilter,
    QuoteCurrency,
    Side,
    Zero as _,
    const_decimal::Decimal,
    decimal_from_f64,
    leverage,
};
use plotters::prelude::*;
use symbiont::{
    DylibConfig,
    Revision,
    Runtime,
};
use tracing::{
    info,
    warn,
};
use trade_aggregation::{
    By,
    CandleComponent,
    CandleComponentUpdate,
    GenericAggregator,
    ModularCandle,
    Trade,
    VolumeRule,
    aggregate_all_trades,
    candle_components::{
        Close,
        DirectionalVolumeRatio,
        High,
        Low,
        Open,
        Volume,
    },
    load_trades_from_csv,
};

/// Decimal precision of all prices and quantities.
const DECIMALS: u8 = 5;
/// Number of candles in the sliding window passed to the strategy.
const WINDOW: usize = 50;
/// Number of volume candles to aggregate the raw trades into.
const TARGET_NUM_CANDLES: f64 = 2000.0;
/// Starting account balance in USD.
const STARTING_BALANCE: i64 = 100_000;
/// Half of the assumed bid-ask spread in USD.
const HALF_SPREAD: f64 = 0.25;
/// Taker fee as a fraction (6 basis points, BitMEX-style).
const TAKER_FEE: f64 = 0.0006;
/// Fraction of candles used for the training segment; the rest is held out.
const TRAIN_FRACTION: f64 = 0.7;
/// Number of LLM evolution rounds.
const MAX_ROUNDS: usize = 10;
/// Number of top strategies (by training return) combined into the ensemble.
const TOP_K: usize = 3;

/// Default location of the raw trade data csv file.
const DEFAULT_DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/Bitmex_XBTUSD_1M.csv");
/// Where to fetch the raw trade data from if it is missing.
const DATA_URL: &str = "https://raw.githubusercontent.com/MathisWellmann/trade_aggregation-rs/master/data/Bitmex_XBTUSD_1M.csv";

// The default implementation never trades — a perfectly safe strategy with
// exactly zero return. The LLM must evolve it into something profitable.
symbiont::evolvable! {
    /// Decide the next trading action given the most recent `candles`
    /// (oldest first, `candles[candles.len() - 1]` is the current candle)
    /// and the current `account` state.
    ///
    /// Execution model: the returned action is executed immediately as a
    /// market order at the close of the current candle (plus spread and
    /// taker fee). Position notional is limited to 1x leverage.
    fn decide(candles: &[Candle], account: &AccountState) -> Action {
        let _ = (candles, account);
        Action::Hold
    }
}

// -- Data loading & candle aggregation ----------------------------------------

/// The intermediate candle used during trade aggregation, built from modular
/// [`trade_aggregation`] components.
#[derive(Debug, Default, Clone, trade_aggregation::Candle)]
struct AggCandle {
    open: Open,
    high: High,
    low: Low,
    close: Close,
    volume: Volume,
    buy_volume_ratio: DirectionalVolumeRatio,
}

/// Load raw trades from disk and aggregate them into volume candles.
fn load_candles() -> Vec<Candle> {
    let path = std::env::var("TRADES_CSV").unwrap_or_else(|_| DEFAULT_DATA_PATH.to_string());
    assert!(
        std::path::Path::new(&path).exists(),
        "Trade data csv not found at `{path}`. Download it first:\n\
         curl -L --create-dirs -o {DEFAULT_DATA_PATH} \\\n  {DATA_URL}\n\
         or point the TRADES_CSV env var at an existing file with `timestamp,price,size` columns."
    );
    let trades = load_trades_from_csv(&path).expect("Can load trades from the csv file");
    info!("Loaded {} raw trades from {path}", trades.len());

    // Information-driven aggregation: a new candle is created every time a
    // fixed amount of volume has been traded, rather than every N minutes.
    let total_volume: f64 = trades.iter().map(|t| t.size.abs()).sum();
    let threshold = total_volume / TARGET_NUM_CANDLES;
    let rule = VolumeRule::new(threshold, By::Quote).expect("Volume threshold is positive");
    let mut aggregator = GenericAggregator::<AggCandle, VolumeRule, Trade>::new(rule, true);

    Vec::from_iter(
        aggregate_all_trades(&trades, &mut aggregator)
            .iter()
            .map(|c| Candle {
                open: c.open(),
                high: c.high(),
                low: c.low(),
                close: c.close(),
                volume: c.volume(),
                buy_volume_ratio: c.buy_volume_ratio(),
            }),
    )
}

// -- Exchange setup ------------------------------------------------------------

/// Linear BTCUSD futures: quantities in BTC (base), margin and PnL in USD (quote).
type Exch = Exchange<i64, DECIMALS, BaseCurrency<i64, DECIMALS>, NoUserOrderId>;

/// Create a fresh exchange for one backtest run.
fn new_exchange() -> Exch {
    let contract_spec = ContractSpecification::new(
        leverage!(1),
        Decimal::try_from_scaled(5, 1).expect("0.5 is a valid maintenance margin fraction"),
        PriceFilter::default(),
        QuantityFilter::new(None, None, BaseCurrency::new(1, 4))
            .expect("0.0001 BTC is a valid quantity tick"),
        Fee::from(Decimal::try_from_scaled(2, 4).expect("2 bps maker fee")),
        Fee::from(Decimal::try_from_scaled(6, 4).expect("6 bps taker fee")),
    )
    .expect("Contract specification is valid");
    let config = Config::new(
        QuoteCurrency::new(STARTING_BALANCE, 0),
        NonZeroU16::new(10).expect("is non-zero"),
        contract_spec,
        // Timestamps don't advance when driving the exchange purely through
        // `set_best_bid_and_ask`, so disable order rate limiting.
        OrderRateLimits::new(NonZeroU32::MAX),
    )
    .expect("Config is valid");
    Exchange::new(config)
}

/// Convert an `f64` price into the exchange's decimal quote currency.
fn quote(price: f64) -> QuoteCurrency<i64, DECIMALS> {
    QuoteCurrency::from(
        decimal_from_f64::<i64, DECIMALS>(price).expect("Price fits into the decimal type"),
    )
}

// -- Backtest ------------------------------------------------------------------

/// Outcome of backtesting one strategy implementation over a candle slice.
struct EvalResult {
    /// Strategy return over the run, in percent.
    ret_pct: f64,
    /// Buy & hold return over the same candles, in percent.
    buy_hold_ret_pct: f64,
    /// Maximum peak-to-trough drawdown of the equity curve, in percent.
    max_dd_pct: f64,
    /// Mean / std of per-candle returns, scaled by sqrt(num candles).
    sharpe: f64,
    /// Number of filled market orders.
    fills: usize,
    /// Number of rejected orders (invalid qty or not enough margin).
    rejected: usize,
    /// Total fees paid in USD.
    fees: f64,
    /// Final mark-to-market equity in USD.
    final_equity: f64,
    /// Panic message if the evolved function panicked during the run.
    panic: Option<String>,
    /// Mark-to-market equity after each candle.
    equity_curve: Vec<f64>,
}

impl EvalResult {
    /// Scalar score used to rank strategies (training return %).
    fn score(&self) -> f64 {
        if self.panic.is_some() {
            f64::NEG_INFINITY
        } else {
            self.ret_pct
        }
    }
}

impl std::fmt::Display for EvalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "| Metric          | Value      |\n\
             |-----------------|------------|"
        )?;
        writeln!(f, "| Return          | {:>9.2}% |", self.ret_pct)?;
        writeln!(f, "| Buy & hold      | {:>9.2}% |", self.buy_hold_ret_pct)?;
        writeln!(f, "| Max drawdown    | {:>9.2}% |", self.max_dd_pct)?;
        writeln!(f, "| Sharpe          | {:>10.2} |", self.sharpe)?;
        writeln!(f, "| Fills           | {:>10} |", self.fills)?;
        writeln!(f, "| Rejected orders | {:>10} |", self.rejected)?;
        writeln!(f, "| Fees paid       | {:>9.2}$ |", self.fees)?;
        writeln!(f, "| Final equity    | {:>9.2}$ |", self.final_equity)?;
        if let Some(msg) = &self.panic {
            writeln!(f, "\nWARNING: the strategy panicked during the run: {msg}")?;
        }
        Ok(())
    }
}

fn max_drawdown_pct(curve: &[f64]) -> f64 {
    let mut peak = f64::MIN;
    let mut max_dd = 0.0_f64;
    for &v in curve {
        peak = peak.max(v);
        if peak > 0.0 {
            max_dd = max_dd.max((peak - v) / peak);
        }
    }
    max_dd * 100.0
}

fn sharpe(curve: &[f64]) -> f64 {
    if curve.len() < 2 {
        return 0.0;
    }
    let rets = Vec::from_iter(curve.windows(2).map(|w| (w[1] - w[0]) / w[0]));
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
    let std = var.sqrt();
    if std == 0.0 {
        0.0
    } else {
        mean / std * (rets.len() as f64).sqrt()
    }
}

/// Backtest the currently active `decide` implementation over `candles`.
fn run_backtest(runtime: &Runtime, candles: &[Candle]) -> EvalResult {
    run_backtest_with(decide, || runtime.take_panic(), candles)
}

/// Backtest an arbitrary decision function over `candles`.
///
/// `decide_impl` is called once per candle and `take_panic` is checked after
/// each call, so panics caught inside a dylib abort the run. Taking the
/// decision as a closure is what allows backtesting [`symbiont::RevisionFn`]
/// handles of retained revisions — including several combined into an
/// ensemble — without touching the active revision.
///
/// Prices are fed to the exchange via [`Exchange::set_best_bid_and_ask`] from
/// each candle close — a reasonable execution proxy when only market orders
/// are used: buys fill at the ask, sells at the bid, both pay the taker fee.
fn run_backtest_with(
    decide_impl: impl Fn(&[Candle], &AccountState) -> Action,
    take_panic: impl Fn() -> Option<String>,
    candles: &[Candle],
) -> EvalResult {
    assert!(candles.len() > WINDOW, "Not enough candles for the window");
    let mut exchange = new_exchange();
    let mut equity_curve = Vec::with_capacity(candles.len() - WINDOW + 2);
    let mut fills: usize = 0;
    let mut rejected: usize = 0;
    let mut panic_msg: Option<String> = None;

    let first_close = candles[WINDOW - 1].close;
    let last_close = candles[candles.len() - 1].close;
    let buy_hold_ret_pct = (last_close / first_close - 1.0) * 100.0;

    for i in WINDOW..=candles.len() {
        let window = &candles[i - WINDOW..i];
        let close = window[WINDOW - 1].close;
        let bid = quote(close - HALF_SPREAD);
        let ask = quote(close + HALF_SPREAD);
        exchange.set_best_bid_and_ask(bid, ask);

        let account = exchange.account();
        let position = account.position();
        let unrealized_pnl: f64 = if position.quantity().is_zero() {
            0.0
        } else {
            position.unrealized_pnl(bid, ask).into()
        };
        let state = AccountState {
            equity: account.balances().equity().into(),
            available_balance: account.available_balance().into(),
            position_qty: position.quantity().into(),
            entry_price: position.entry_price().into(),
            unrealized_pnl,
        };
        equity_curve.push(state.equity + unrealized_pnl);

        let action = decide_impl(window, &state);
        if let Some(msg) = take_panic() {
            panic_msg = Some(msg);
            break;
        }

        let (side, qty) = match action {
            Action::Hold => continue,
            Action::Buy { qty } => (Side::Buy, qty),
            Action::Sell { qty } => (Side::Sell, qty),
        };
        if !qty.is_finite() || qty <= 0.0 {
            rejected += 1;
            continue;
        }
        let Some(decimal_qty) = decimal_from_f64::<i64, DECIMALS>(qty) else {
            rejected += 1;
            continue;
        };
        match MarketOrder::new(side, BaseCurrency::from(decimal_qty)) {
            Ok(order) => match exchange.submit_market_order(order) {
                Ok(_) => fills += 1,
                Err(_) => rejected += 1,
            },
            Err(_) => rejected += 1,
        }
    }

    // Final mark-to-market equity at the last close.
    let bid = quote(last_close - HALF_SPREAD);
    let ask = quote(last_close + HALF_SPREAD);
    let account = exchange.account();
    let position = account.position();
    let unrealized_pnl: f64 = if position.quantity().is_zero() {
        0.0
    } else {
        position.unrealized_pnl(bid, ask).into()
    };
    let equity: f64 = account.balances().equity().into();
    let final_equity = equity + unrealized_pnl;
    equity_curve.push(final_equity);

    EvalResult {
        ret_pct: (final_equity / STARTING_BALANCE as f64 - 1.0) * 100.0,
        buy_hold_ret_pct,
        max_dd_pct: max_drawdown_pct(&equity_curve),
        sharpe: sharpe(&equity_curve),
        fills,
        rejected,
        fees: account.balances().total_fees_paid().into(),
        final_equity,
        panic: panic_msg,
        equity_curve,
    }
}

// -- Prompting -----------------------------------------------------------------

/// A strategy that earned a seat in the top-K list during the search.
///
/// Only training-segment information is kept during the search; the held-out
/// test segment is evaluated once at the end, after re-activating `rev` (and
/// running the top-K seats side by side as an ensemble).
struct BestStrategy {
    /// The registered revision of the strategy's compiled dylib. Revisions
    /// stay loaded, so any of them can be re-run after the search
    /// without recompiling anything.
    rev: Revision,
    /// The strategy's code, fed back into later prompts as "best so far".
    code: String,
    /// Training-segment report, fed back into later prompts.
    train_report: String,
    /// Training-segment score used to rank strategies.
    score: f64,
}

fn build_prompt(
    task: &str,
    last_code: &str,
    last_result: &EvalResult,
    best: Option<&BestStrategy>,
) -> String {
    if let Some(panic_msg) = &last_result.panic {
        return format!(
            "{task}\
             ## Last attempt\n```rust\n{last_code}\n```\n\n\
             ## Runtime panic\n\
             The strategy panicked during the backtest: {panic_msg}\n\n\
             Fix the panic. Mind that `candles` has exactly {WINDOW} elements. Rust code only.",
        );
    }

    let last_attempt = if last_code.is_empty() {
        String::new()
    } else {
        format!("## Last attempt\n```rust\n{last_code}\n```\nResult:\n{last_result}\n")
    };
    let best_so_far = match best {
        Some(b) => format!(
            "## Best so far\n```rust\n{}\n```\nResult:\n{}\n",
            b.code, b.train_report
        ),
        None => {
            String::from("## Best so far\nNone yet — the default `Hold` strategy returns 0%.\n\n")
        }
    };

    format!(
        "{task}\
         {last_attempt}\
         {best_so_far}\
         ## Direction\n\
         Compute features from the candle window (momentum, volatility, position of \
         close within the recent range, order flow via `buy_volume_ratio`) and trade \
         only when the edge plausibly exceeds the round-trip cost. Size positions \
         deliberately from `account.available_balance` and manage open positions \
         (take profit / cut losses / flip). Beat the best strategy so far on return \
         while keeping drawdown reasonable.\n\
         No external crates — only `std`. Rust code only."
    )
}

// -- Plotting ------------------------------------------------------------------

/// Render the equity curve of the best strategy vs buy & hold and display it.
fn plot_equity_curves(
    strategy: &[f64],
    buy_hold: &[f64],
    split_curve_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let y_min = strategy
        .iter()
        .chain(buy_hold)
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = strategy
        .iter()
        .chain(buy_hold)
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let pad = (y_max - y_min).max(1.0) * 0.05;
    let n = strategy.len().max(buy_hold.len());

    let path = std::env::temp_dir().join("evolving_trader_equity.png");
    {
        let root = BitMapBackend::new(&path, (1600, 900)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Evolved strategy vs buy & hold (full dataset)",
                ("sans-serif", 24).into_font().with_color(BLACK),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(0.0..n as f64, (y_min - pad)..(y_max + pad))?;

        chart
            .configure_mesh()
            .x_desc("Volume candle #")
            .y_desc("Equity (USD)")
            .draw()?;

        let strategy_color = RGBColor(233, 69, 96);
        chart
            .draw_series(LineSeries::new(
                strategy.iter().enumerate().map(|(i, &v)| (i as f64, v)),
                strategy_color.stroke_width(2),
            ))?
            .label("evolved strategy")
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 16, y)], strategy_color.stroke_width(2))
            });

        let bh_color = RGBColor(15, 52, 96);
        chart
            .draw_series(LineSeries::new(
                buy_hold.iter().enumerate().map(|(i, &v)| (i as f64, v)),
                bh_color.stroke_width(2),
            ))?
            .label("buy & hold")
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 16, y)], bh_color.stroke_width(2))
            });

        chart
            .draw_series(LineSeries::new(
                vec![
                    (split_curve_idx as f64, y_min - pad),
                    (split_curve_idx as f64, y_max + pad),
                ],
                BLACK.stroke_width(1),
            ))?
            .label("train/test split")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], BLACK));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperLeft)
            .draw()?;

        root.present()?;
    }
    println!("Equity curve plot saved to: {}", path.display());

    let conf = viuer::Config {
        width: Some(80),
        absolute_offset: false,
        ..Default::default()
    };
    viuer::print_from_file(&path, &conf)?;

    Ok(())
}

/// Buy & hold equity curve aligned with the backtest equity curve indices.
fn buy_hold_curve(candles: &[Candle]) -> Vec<f64> {
    let start_close = candles[WINDOW - 1].close;
    let mut curve = Vec::from_iter(
        (WINDOW..=candles.len())
            .map(|i| STARTING_BALANCE as f64 * candles[i - 1].close / start_close),
    );
    if let Some(&last) = curve.last() {
        // Align with the final mark-to-market point of the strategy curve.
        curve.push(last);
    }
    curve
}

// -- Main ----------------------------------------------------------------------

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    symbiont::init_tracing();

    let candles = load_candles();
    let split = (candles.len() as f64 * TRAIN_FRACTION) as usize;
    let (train, test) = candles.split_at(split);
    info!(
        "Aggregated {} volume candles (~{:.0}$ traded per candle): {} train / {} test",
        candles.len(),
        candles[0].volume,
        train.len(),
        test.len(),
    );

    let host_crate = env!("CARGO_PKG_NAME");
    let runtime = Runtime::new(
        SYMBIONT_DECLS,
        SYMBIONT_PRELUDE,
        DylibConfig::host_package(
            symbiont::Profile::Debug,
            host_crate,
            env!("CARGO_MANIFEST_DIR"),
        ),
    )
    .await?;
    // Include the host crate's documented API (Candle, AccountState, Action)
    // in the system prompt. Cap generation so small local models that fail to
    // stop cannot overflow the inference server's context window.
    let agent = symbiont::agent_builder(Some(host_crate))
        .await?
        .max_tokens(4096)
        .build();

    let fn_source = runtime.fn_full_sources();
    let fn_prelude = runtime.fn_prelude();
    let task = format!(
        "## Task\n\
         Implement this evolvable trading strategy function:\n\
         ```rust\n{:#?}\n{:#?}\n```\n\n\
         ## Market & execution model\n\
         - Data: ~1M BitMEX XBTUSD trades aggregated into volume candles \
         (each candle contains the same traded volume, not the same time span).\n\
         - `candles` holds the {WINDOW} most recent candles, oldest first; \
         `candles[candles.len() - 1]` is the current candle.\n\
         - Your action executes as a market order at the current close \
         ± {HALF_SPREAD}$ spread, paying {fee_pct:.3}% taker fee. A round trip \
         costs roughly {round_trip_pct:.2}% of traded notional — overtrading loses money.\n\
         - Leverage is 1x: additional position notional (qty * price) is limited by \
         `account.available_balance`. Order quantities are in BTC (min step 0.0001).\n\
         - Non-finite, non-positive or unaffordable quantities are rejected and waste the candle.\n\
         - The function is called once per candle; it must never panic.\n\n",
        fn_source[0],
        fn_prelude[0],
        fee_pct = TAKER_FEE * 100.0,
        round_trip_pct = 2.0 * (TAKER_FEE + HALF_SPREAD / candles[0].close) * 100.0,
    );

    // -- Round 0: the default Hold strategy -----------------------------------
    println!("\n=== Round 0: default implementation (always Hold) ===");
    let mut result = run_backtest(runtime, train);
    println!("{result}");

    let default_score = result.score();
    let mut best_score = default_score;
    // Top strategies so far, sorted by descending training score.
    let mut top: Vec<BestStrategy> = Vec::new();
    let mut last_code = String::new();

    // -- Evolution loop --------------------------------------------------------
    for round in 1..=MAX_ROUNDS {
        println!("\n=== Round {round}: evolving via LLM ===");

        let prompt = build_prompt(&task, &last_code, &result, top.first());
        let rev = match runtime.evolve(&agent, &prompt).await {
            Ok(rev) => rev,
            Err(e) => {
                warn!("Evolution failed: {e} — retrying next round.");
                continue;
            }
        };
        last_code = runtime.current_code();

        result = run_backtest(runtime, train);
        println!("{result}");

        let score = result.score();
        if score > default_score {
            // Any strategy that beats the round-0 default competes for one of
            // the TOP_K ensemble seats.
            top.push(BestStrategy {
                rev,
                code: last_code.clone(),
                train_report: result.to_string(),
                score,
            });
            top.sort_by(|a, b| b.score.total_cmp(&a.score));
            top.truncate(TOP_K);
        }
        if score > best_score {
            best_score = score;
            info!("New best strategy (revision {rev}) with {best_score:.2}% training return");
        }
    }

    // -- Final report: evaluate the best strategy out-of-sample ----------------
    if top.is_empty() {
        println!("\nNo evolution improved on the default Hold strategy after {MAX_ROUNDS} rounds.");
    } else {
        report_best(runtime, &top, &candles, test, split)?;
    }

    Ok(())
}

/// Re-activate the single best revision, evaluate it on the held-out test
/// segment and the full dataset, print + plot the report, and pit it against
/// an equal-weight ensemble of all top revisions.
fn report_best(
    runtime: &Runtime,
    top: &[BestStrategy],
    candles: &[Candle],
    test: &[Candle],
    split: usize,
) -> symbiont::Result<()> {
    // Re-activate the best revision — its dylib is still loaded, so this is
    // a pointer swap without recompilation — and only now run the held-out
    // test segment: out-of-sample results never influence the search.
    let best = &top[0];
    runtime.activate_revision(best.rev)?;
    let test_result = run_backtest(runtime, test);
    let full_result = run_backtest(runtime, candles);

    println!(
        "\n=== Best strategy (revision {}, training return: {:.2}%) ===",
        best.rev, best.score
    );
    println!("```rust\n{}\n```", best.code);
    println!("\nIn-sample (train):\n{}", best.train_report);
    println!("Out-of-sample (test):\n{test_result}");

    report_ensemble(top, test);

    let bh_curve = buy_hold_curve(candles);
    let split_curve_idx = split.saturating_sub(WINDOW - 1);
    if let Err(e) = plot_equity_curves(&full_result.equity_curve, &bh_curve, split_curve_idx) {
        warn!("Failed to render equity curve plot: {e}");
    }
    Ok(())
}

/// Evaluate an equal-weight ensemble of the top revisions on the held-out
/// test segment.
///
/// `decide_fn(rev)` (generated by `evolvable!`) returns a typed
/// [`symbiont::RevisionFn`] handle pinning its revision's dylib: all members
/// stay callable side by side as compiled code, independent of which revision
/// is currently active. Each candle, every member votes and the signed
/// quantities are averaged into one net market order.
fn report_ensemble(top: &[BestStrategy], test: &[Candle]) {
    if top.len() < 2 {
        println!("(only one strategy beat the default — no ensemble to run)");
        return;
    }
    let handles = Vec::from_iter(
        top.iter()
            .map(|b| decide_fn(b.rev).expect("top revisions are retained by the registry")),
    );
    // Hoist the bare fn pointers once; the per-candle calls below are plain
    // indirect calls into each member's dylib.
    let members = Vec::from_iter(handles.iter().map(symbiont::RevisionFn::get));

    let result = run_backtest_with(
        |window, state| combine_actions(members.iter().map(|f| f(window, state)), members.len()),
        || handles.iter().find_map(|h| h.take_panic()),
        test,
    );

    let revisions = Vec::from_iter(top.iter().map(|b| b.rev.to_string()));
    println!(
        "Equal-weight ensemble of revisions [{}] out-of-sample (test):\n{result}",
        revisions.join(", ")
    );
}

/// Combine the ensemble members' actions into one equal-weight order:
/// average the signed quantities (buy = `+qty`, sell = `-qty`, hold = `0`)
/// and emit the net as a single market order. Nets below the exchange's
/// 0.0001 BTC quantity step become [`Action::Hold`].
fn combine_actions(actions: impl Iterator<Item = Action>, members: usize) -> Action {
    const MIN_QTY: f64 = 0.0001;
    let mut net = 0.0;
    for action in actions {
        match action {
            Action::Buy { qty } => net += qty,
            Action::Sell { qty } => net -= qty,
            Action::Hold => {}
        }
    }
    let avg = net / members as f64;
    if avg >= MIN_QTY {
        Action::Buy { qty: avg }
    } else if avg <= -MIN_QTY {
        Action::Sell { qty: -avg }
    } else {
        Action::Hold
    }
}
