# Evolving Trader — Trading Strategy Evolution Example

This example challenges an LLM to evolve a **futures trading strategy** purely through code,
combining two real-world crates:

- [`trade_aggregation`](https://crates.io/crates/trade_aggregation): aggregates ~1M raw BitMEX
  XBTUSD trades into **volume candles** (information-driven aggregation — every candle contains
  the same traded volume, not the same time span).
- [`lfest`](https://crates.io/crates/lfest): a leveraged futures exchange simulator that models
  taker fees, bid-ask spread, margin requirements and order filters.

An evolvable

```rust
fn decide(candles: &[Candle], account: &AccountState) -> Action
```

function receives a sliding window of the 50 most recent candles (OHLC, volume and the
buy-volume ratio as an order-flow feature) plus the current account state, and returns an
`Action` (`Hold`, `Buy { qty }` or `Sell { qty }`). Actions execute as **market orders** against
the simulated exchange, whose top of book is driven via `Exchange::set_best_bid_and_ask` from
each candle close ± half spread — a reasonable execution proxy for market orders: buys fill at
the ask, sells at the bid, both pay the taker fee.

The default implementation never trades (0% return). Each round the harness backtests the
evolved strategy on the **training segment** (70% of the candles) and feeds a performance
report — return, buy & hold benchmark, max drawdown, Sharpe, fills, rejected orders, fees —
back to the LLM. Runtime panics in the evolved code are caught and reported as targeted
feedback. The best strategy is finally evaluated on the **held-out test segment** (never shown
to the LLM, to avoid leakage) and its equity curve is plotted against buy & hold.

This showcases symbiont evolving **quantitative reasoning** through code: the LLM must discover
features (momentum, volatility, order flow), position sizing and fee-awareness — expressed as
compiled Rust running against a realistic exchange simulation.

## Data

The example expects raw trade data (`timestamp,price,size` csv) at `data/Bitmex_XBTUSD_1M.csv`
(gitignored). Download it once:

```bash
curl -L --create-dirs -o examples/evolving-trader/data/Bitmex_XBTUSD_1M.csv \
  https://raw.githubusercontent.com/MathisWellmann/trade_aggregation-rs/master/data/Bitmex_XBTUSD_1M.csv
```

Alternatively, point the `TRADES_CSV` env var at any csv file with the same format.

## Running

```bash
# Requires API_KEY, BASE_URL, and MODEL env vars (or a local llama-cpp server).
cargo run -p evolving-trader-example
```

The example runs for up to 10 evolution rounds and prints the best strategy's source code,
its in-sample and out-of-sample reports, and renders the equity curve plot in the terminal.
Press `Ctrl+C` to stop early.
