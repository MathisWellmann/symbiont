// SPDX-License-Identifier: MPL-2.0
//! Rust/serde mirror of the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
//! (`dsh`) session log, and the container it is stored in.
//!
//! The harness records one session as JSON Lines: a [`SessionHeaderLine`]
//! (`type: "session"`) first, then one [`Event`] per line, discriminated by
//! `type`. [`LogLine`] is the sum of every record the harness itself writes.
//!
//! On disk the lines are packed into a `session.jsonl.zstd` frame container;
//! see [`container`]. Writing that container needs the `zstd` feature.
//!
//! This crate contains no agent: it is the vocabulary an agent harness uses
//! to write a log the `dsh` viewer can open, or to read one back.
//!
//! # Events the harness does not know
//!
//! `LogLine` is closed on purpose: an unrecognized `type` must reject a log
//! unless the record carries `ignorable: true` ([`Event::ignorable`]). A
//! producer with its own event types wraps `LogLine` in an
//! `#[serde(untagged)]` enum beside its own tagged enum and marks every custom
//! event ignorable.

pub mod container;
pub mod types;

pub use types::*;
