// SPDX-License-Identifier: MPL-2.0
//! Shared boundary types between the backtest harness (host binary) and the
//! LLM-evolved trading strategy (hot-swapped dylib).
//!
//! These types cross the dylib boundary, so they live in the package's library
//! target which both the binary and the generated dylib depend on
//! (see [`symbiont::DylibConfig::host_package`]).

#![allow(
    unused_crate_dependencies,
    reason = "Dependencies are used by this package's binary target."
)]

/// A price candle aggregated from raw trades by *traded volume*
/// (an information-driven aggregation rule), not by wall-clock time.
///
/// Every candle contains roughly the same amount of market activity,
/// so quiet periods are compressed and busy periods are stretched out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candle {
    /// Price of the first trade in the candle.
    pub open: f64,
    /// Highest traded price in the candle.
    pub high: f64,
    /// Lowest traded price in the candle.
    pub low: f64,
    /// Price of the last trade in the candle.
    pub close: f64,
    /// Total traded volume in quote currency (USD). Roughly constant per
    /// candle by construction of the volume aggregation rule.
    pub volume: f64,
    /// Fraction of the volume that came from aggressive buyers, in `[0, 1]`.
    /// Values above 0.5 indicate net buying pressure (order-flow information).
    pub buy_volume_ratio: f64,
}

/// A snapshot of the trading account at decision time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccountState {
    /// Realized wallet equity in USD (excludes unrealized PnL).
    pub equity: f64,
    /// Balance available as margin for new positions, in USD.
    /// At 1x leverage the additional position notional (qty * price) is
    /// limited by this value.
    pub available_balance: f64,
    /// Signed position size in BTC. Positive = long, negative = short, 0 = flat.
    pub position_qty: f64,
    /// Average position entry price in USD (0 if flat).
    pub entry_price: f64,
    /// Unrealized PnL of the open position in USD, marked to the current market.
    pub unrealized_pnl: f64,
}

/// The trading decision returned by the evolved strategy.
/// Only market orders are available; they fill immediately at the current
/// bid/ask and pay taker fees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    /// Do nothing this candle.
    Hold,
    /// Submit a market buy order for `qty` BTC.
    /// Increases long exposure or reduces/flips a short position.
    Buy {
        /// Order quantity in BTC. Must be finite and positive.
        qty: f64,
    },
    /// Submit a market sell order for `qty` BTC.
    /// Reduces a long position or opens/increases a short position.
    Sell {
        /// Order quantity in BTC. Must be finite and positive.
        qty: f64,
    },
}

/// Prelude imported by the generated dylib through [`symbiont::DylibConfig`].
pub mod prelude {
    pub use crate::{
        AccountState,
        Action,
        Candle,
    };
}
