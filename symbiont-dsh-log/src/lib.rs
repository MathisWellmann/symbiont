// SPDX-License-Identifier: MPL-2.0
//! Rust/serde mirror of the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
//! (`dsh`) session log, generation v3 (the harness's current format), and the
//! container it is stored in.
//!
//! The harness records one session as JSON Lines: a [`SessionHeaderLine`]
//! (`type: "session"`, `version: 3`) first, then one [`Event`] per line,
//! discriminated by `type`. [`LogLine`] is the sum of every record the
//! harness itself writes.
//!
//! On disk the lines are packed into a `session.v3.jsonl.zstd` frame
//! container; see [`container`]. Writing that container needs the `zstd`
//! feature.
//!
//! This crate contains no agent: it is the vocabulary an agent harness uses
//! to write a log the `dsh` viewer can open, or to read one back.
//!
//! # Events the harness does not know
//!
//! `LogLine` is closed on purpose: a producer with its own event types wraps
//! `LogLine` in an `#[serde(untagged)]` enum beside its own tagged enum and
//! marks every custom event `ignorable: true` ([`Event::ignorable`]). The
//! harness honors that marker in current-format (v3) logs; older
//! generations refuse unknown historical events regardless.

pub mod container;
pub mod types;

pub use types::*;
