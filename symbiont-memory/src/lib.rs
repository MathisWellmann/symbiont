// SPDX-License-Identifier: MPL-2.0
//! Memory across [`symbiont`](https://docs.rs/symbiont) `evolve` calls.
//!
//! The symbiont runtime scopes chat history to one `evolve` call on purpose:
//! request sizes stay bounded in long-lived processes. A host that wants the
//! agent to learn from earlier rounds renders that knowledge into the next
//! prompt itself. This crate is that rendering, plus the bookkeeping behind
//! it, and it is entirely optional: the runtime does not know about it.
//!
//! # The archive
//!
//! [`Archive`] holds every evaluated attempt: the source, the host's
//! [`Score`] and a report written for the model. [`Memory::render`] turns the
//! best and the most recent attempts into a prompt section within a token
//! budget. Bad recent attempts are shown on purpose: they keep the next lane
//! from repeating them.
//!
//! # Known pitfalls
//!
//! [`Failures`] counts the rejected attempts in every lane's
//! [`EvolutionTrace`](symbiont::EvolutionTrace) and the panics the host
//! caught. A failure that several lanes hit is listed in the shared part of
//! the prompt, so the next lane does not invent the same method or index out
//! of bounds again. Counting needs no model call.
//!
//! # Ownership and determinism
//!
//! A [`Memory`] is plain data. It is owned by one task, mutated through
//! `&mut self` and serializable, so a host can checkpoint it next to its
//! population. Nothing in it reads the clock or a random number generator,
//! and rendering is a pure function of the state: the same state renders the
//! same bytes, which keeps a server's prefix cache warm and makes a recorded
//! run replayable.
//!
//! # Example
//!
//! ```
//! use symbiont_memory::{Evaluated, Memory, MemoryConfig, Score};
//!
//! let mut memory = Memory::<u64>::new(MemoryConfig::default());
//! // After the host evaluated a candidate:
//! let _ = memory.commit(
//!     Evaluated::builder()
//!         .id(0xfeed)
//!         .source("fn step(x: f64) -> f64 { x * 2.0 }")
//!         .score(Score::new(1.25).expect("finite"))
//!         .report("return 1.25%, max drawdown 3.1%")
//!         .build(),
//! );
//! // Before the next evolve call:
//! let rendered = memory.render(&[]);
//! let prompt = format!("Improve the function.\n\n{}", rendered.text());
//! assert!(prompt.contains("return 1.25%"));
//! ```

mod archive;
mod failures;
mod render;

use serde::{
    Deserialize,
    Serialize,
};

pub use crate::{
    archive::{
        Archive,
        ArchiveConfig,
        Committed,
        Entry,
        EntrySeq,
        Evaluated,
        NonFiniteScore,
        Score,
    },
    failures::{
        FailureConfig,
        Failures,
        Pitfall,
    },
    render::{
        RenderConfig,
        estimate_tokens,
    },
};

/// A configuration that cannot work.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// The archive could never evict: its capacity is zero or does not exceed
    /// the protected recent tail.
    #[error(
        "archive capacity {capacity} must be positive and exceed protect_recent {protect_recent}"
    )]
    ArchiveCapacity {
        /// The requested capacity.
        capacity: usize,
        /// The requested protected tail.
        protect_recent: usize,
    },
}

/// The configuration of a [`Memory`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// The archive limits.
    pub archive: ArchiveConfig,
    /// What the archive section of a prompt shows.
    pub render: RenderConfig,
    /// How recurring failures are kept and shown.
    pub failures: FailureConfig,
}

/// The memory of one optimization problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory<I> {
    config: MemoryConfig,
    archive: Archive<I>,
    failures: Failures,
}

/// The prompt text of a [`Memory`], see [`Memory::render`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[must_use]
pub struct Rendered {
    shared: String,
    lane: String,
    entries: Vec<EntrySeq>,
}

impl Rendered {
    /// The whole memory section, ready to go into a prompt: the
    /// [`Self::shared`] part, then the [`Self::lane`] part.
    #[must_use]
    pub fn text(&self) -> String {
        let mut text = String::with_capacity(self.shared.len() + self.lane.len());
        text.push_str(&self.shared);
        text.push_str(&self.lane);
        text
    }

    /// The part every lane sees alike, such as the known pitfalls. It changes
    /// rarely, so a host that cares about a server's prefix cache puts it
    /// before anything lane-specific in the prompt.
    #[must_use]
    pub fn shared(&self) -> &str {
        &self.shared
    }

    /// The part that depends on the lane and changes with every commit: the
    /// archive of attempts.
    #[must_use]
    pub fn lane(&self) -> &str {
        &self.lane
    }

    /// The archive entries the text shows. A host that records its run as a
    /// discovery tree (see `symbiont-dream-rsi`) adds these as context edges
    /// of the attempt, so a replay never credits a policy with information
    /// the attempt had not seen.
    #[must_use]
    pub fn entries(&self) -> &[EntrySeq] {
        &self.entries
    }
}

impl<I: PartialEq + Clone> Memory<I> {
    /// An empty memory.
    #[must_use]
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            archive: Archive::new(config.archive),
            failures: Failures::new(config.failures),
            config,
        }
    }

    /// Record an evaluated attempt. See [`Archive::commit`].
    pub fn commit(&mut self, evaluated: Evaluated<I>) -> Committed {
        self.archive.commit(evaluated)
    }

    /// Count the rejected attempts of one lane, see [`Failures::commit_trace`].
    /// Call it for every lane, successful or not: a lane that succeeded on
    /// its third attempt still hit two failures.
    pub fn commit_trace(&mut self, trace: &symbiont::EvolutionTrace) {
        self.failures.commit_trace(trace);
    }

    /// Count a panic the host caught while it evaluated a candidate.
    pub fn commit_panic(&mut self, message: &str) {
        self.failures.commit_panic(message);
    }

    /// The memory as prompt text for the next lane.
    ///
    /// `exclude` names candidates the host already shows in its prompt, such
    /// as the parents of a mutation, so they are not repeated.
    pub fn render(&self, exclude: &[I]) -> Rendered {
        let (lane, entries) = render::render_archive(&self.archive, exclude, &self.config.render);
        Rendered {
            shared: self.failures.render(),
            lane,
            entries,
        }
    }

    /// The archive.
    #[must_use]
    pub fn archive(&self) -> &Archive<I> {
        &self.archive
    }

    /// The archive, for rescoring and removing entries.
    pub fn archive_mut(&mut self) -> &mut Archive<I> {
        &mut self.archive
    }

    /// The recurring failures.
    #[must_use]
    pub fn failures(&self) -> &Failures {
        &self.failures
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_part_comes_before_the_lane_part() {
        let mut memory = Memory::<u64>::new(MemoryConfig::default());
        assert_eq!(memory.render(&[]), Rendered::default());
        let _ = memory.commit(
            Evaluated::builder()
                .id(1)
                .source("fn f() {}")
                .score(Score::new(0.5).expect("finite"))
                .report("slow")
                .build(),
        );
        memory.commit_panic("attempt to divide by zero");
        memory.commit_panic("attempt to divide by zero");
        let rendered = memory.render(&[]);
        assert!(
            rendered
                .shared()
                .contains("- panic: attempt to divide by zero")
        );
        assert!(rendered.lane().contains("slow"));
        assert!(!rendered.lane().contains("divide by zero"));
        assert_eq!(
            rendered.text(),
            format!("{}{}", rendered.shared(), rendered.lane())
        );
        assert_eq!(rendered.entries().len(), 1);
    }
}
