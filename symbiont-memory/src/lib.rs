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
//! # The playbook
//!
//! [`Playbook`] holds lessons as itemized bullets with helpful and harmful
//! counters, as in ACE (*Agentic Context Engineering*,
//! [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)). Bullets are only
//! ever added, tagged or removed by deterministic code, never rewritten as a
//! whole, which keeps detail from eroding over many rounds. A host can seed
//! it with lessons it already knows through [`Memory::playbook_mut`].
//!
//! # Reflection
//!
//! Once a round of attempts is evaluated, [`Memory::reflection_request`]
//! builds one prompt that reviews them: their scores against their parents,
//! the reports, the code, and the bullets each lane was shown. The host sends
//! it to a model of its choice and hands the answer to
//! [`Memory::apply_reflection`], which tags the shown bullets helpful,
//! harmful or neutral, prunes refuted ones and adds the new lessons that cite
//! a reviewed attempt as evidence. The memory never performs I/O itself.
//!
//! ```
//! use symbiont_memory::{Evaluated, Memory, MemoryConfig, Score};
//!
//! let mut memory = Memory::<u64>::new(MemoryConfig::default());
//! let rendered = memory.render(&[]);
//! let _ = memory.commit(
//!     Evaluated::builder()
//!         .id(1)
//!         .source("fn f(x: f64) -> f64 { 1.0 / x }")
//!         .score(Score::new(-1.0).expect("finite"))
//!         .report("panicked on x = 0")
//!         .bullets_shown(rendered.bullets().to_vec())
//!         .build(),
//! );
//! let request = memory
//!     .reflection_request("Minimize f on [-1, 1].")
//!     .expect("one attempt awaits review");
//! // let answer = model.complete(request.system_prompt(), request.user_prompt()).await?;
//! let answer = r#"{"operations": [{"type": "ADD", "section": "mistakes",
//!     "content": "Guard the division: x can be 0.", "evidence": [0]}]}"#;
//! let report = memory.apply_reflection(&request, answer).expect("valid answer");
//! assert_eq!(report.added().len(), 1);
//! assert!(memory.render(&[]).shared().contains("Guard the division"));
//! ```
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
mod playbook;
mod reflection;
mod render;

use serde::{
    Deserialize,
    Serialize,
};

use crate::reflection::Pending;
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
    playbook::{
        Bullet,
        BulletId,
        Playbook,
        PlaybookConfig,
        Rejection,
        Section,
        Tag,
    },
    reflection::{
        CurationReport,
        ReflectionConfig,
        ReflectionError,
        ReflectionRequest,
        RequestId,
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
    /// The playbook limits.
    pub playbook: PlaybookConfig,
    /// When and how much the reflection step reviews.
    pub reflection: ReflectionConfig,
}

/// The memory of one optimization problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory<I> {
    config: MemoryConfig,
    archive: Archive<I>,
    failures: Failures,
    playbook: Playbook,
    /// Attempts committed since they were last handed to a reflection
    /// request, oldest first.
    pending: Vec<Pending>,
    /// Requests handed out and neither applied nor abandoned.
    outstanding: Vec<RequestId>,
    next_request: u32,
}

/// The prompt text of a [`Memory`], see [`Memory::render`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[must_use]
pub struct Rendered {
    shared: String,
    lane: String,
    entries: Vec<EntrySeq>,
    bullets: Vec<BulletId>,
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

    /// The part every lane sees alike: the playbook and the known pitfalls.
    /// It changes rarely, so a host that cares about a server's prefix cache puts it
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

    /// The playbook bullets the text shows.
    #[must_use]
    pub fn bullets(&self) -> &[BulletId] {
        &self.bullets
    }
}

impl<I: PartialEq + Clone> Memory<I> {
    /// An empty memory.
    #[must_use]
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            archive: Archive::new(config.archive),
            failures: Failures::new(config.failures),
            playbook: Playbook::new(config.playbook),
            pending: Vec::new(),
            outstanding: Vec::new(),
            next_request: 0,
            config,
        }
    }

    /// Record an evaluated attempt. See [`Archive::commit`].
    ///
    /// A new attempt also awaits review by the next reflection request. A
    /// re-evaluation of a known id does not: it is not a new attempt.
    pub fn commit(&mut self, evaluated: Evaluated<I>) -> Committed {
        let bullets = evaluated.bullets_shown().to_vec();
        let committed = self.archive.commit(evaluated);
        if let Committed::Inserted { seq, evicted } = committed {
            if let Some(evicted) = evicted {
                self.pending.retain(|p| p.seq() != evicted);
            }
            self.pending.push(Pending::new(seq, bullets));
        }
        committed
    }

    /// Whether [`ReflectionConfig::reflect_every`] attempts await review.
    #[must_use]
    pub fn reflection_due(&self) -> bool {
        self.pending.len() >= self.config.reflection.reflect_every()
    }

    /// The prompts that review the attempts awaiting review, oldest first, at
    /// most [`ReflectionConfig::max_reviewed`] of them. `None` if no attempt
    /// awaits review.
    ///
    /// `task` describes the optimization problem to the reviewer, typically
    /// the task part of the host's evolve prompt.
    ///
    /// The reviewed attempts leave the queue. Apply the answer with
    /// [`Self::apply_reflection`]; if the model call fails, hand the request
    /// back with [`Self::abandon_reflection`] so the attempts are reviewed
    /// later.
    pub fn reflection_request(&mut self, task: &str) -> Option<ReflectionRequest> {
        let archive = &self.archive;
        self.pending.retain(|p| archive.by_seq(p.seq()).is_some());
        if self.pending.is_empty() {
            return None;
        }
        let take = self
            .pending
            .len()
            .min(self.config.reflection.max_reviewed());
        let pending = Vec::from_iter(self.pending.drain(..take));
        let id = RequestId::new(self.next_request);
        self.next_request += 1;
        self.outstanding.push(id);
        let request = reflection::request(
            id,
            reflection::PromptInput::new(
                task,
                &self.archive,
                &self.playbook,
                &self.failures,
                pending,
                &self.config.reflection,
            ),
        );
        assert_eq!(request.reviewed().count(), take);
        Some(request)
    }

    /// Merge the model's answer to `request` into the playbook.
    ///
    /// # Errors
    ///
    /// [`ReflectionError::NotOutstanding`] if the request was already applied
    /// or abandoned, so an answer never counts twice.
    /// [`ReflectionError::NoJson`] if the answer holds no reflection object;
    /// the request stays outstanding.
    pub fn apply_reflection(
        &mut self,
        request: &ReflectionRequest,
        answer: &str,
    ) -> Result<CurationReport, ReflectionError> {
        if !self.outstanding.contains(&request.id()) {
            return Err(ReflectionError::NotOutstanding(request.id()));
        }
        let report = reflection::apply(request, answer, &mut self.playbook, &self.archive)?;
        self.outstanding.retain(|id| *id != request.id());
        Ok(report)
    }

    /// Return the attempts of an unanswered request to the front of the
    /// review queue. A request that is not outstanding is dropped.
    pub fn abandon_reflection(&mut self, request: ReflectionRequest) {
        let id = request.id();
        if !self.outstanding.contains(&id) {
            return;
        }
        self.outstanding.retain(|r| *r != id);
        let mut queue = request.into_pending();
        queue.append(&mut self.pending);
        self.pending = queue;
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
        let (mut shared, bullets) = self.playbook.render();
        shared.push_str(&self.failures.render());
        Rendered {
            shared,
            lane,
            entries,
            bullets,
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

    /// The playbook.
    #[must_use]
    pub fn playbook(&self) -> &Playbook {
        &self.playbook
    }

    /// The playbook, for seeding it with lessons the host already knows.
    pub fn playbook_mut(&mut self) -> &mut Playbook {
        &mut self.playbook
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
        assert!(rendered.bullets().is_empty());

        let seq = rendered.entries()[0];
        let id = memory
            .playbook_mut()
            .add(Section::Mistakes, "Guard every division.", vec![seq])
            .expect("added");
        let rendered = memory.render(&[]);
        assert_eq!(rendered.bullets(), [id]);
        let playbook = rendered.shared().find("## Playbook").expect("present");
        let pitfalls = rendered
            .shared()
            .find("## Known pitfalls")
            .expect("present");
        assert!(playbook < pitfalls);
    }

    fn attempt(id: u64, parents: Vec<u64>, score: f64, bullets: &[BulletId]) -> Evaluated<u64> {
        Evaluated::builder()
            .id(id)
            .parents(parents)
            .source(format!("fn f() -> u64 {{ {id} }}"))
            .score(Score::new(score).expect("finite"))
            .report(format!("report {id}"))
            .bullets_shown(bullets.to_vec())
            .build()
    }

    #[test]
    fn reflection_round_trip() {
        let mut memory = Memory::<u64>::new(MemoryConfig::default());
        assert!(memory.reflection_request("task").is_none());
        let shown = memory
            .playbook_mut()
            .add(Section::Strategies, "Prefer a sieve.", vec![])
            .err();
        assert_eq!(shown, Some(Rejection::NoEvidence));

        let _ = memory.commit(attempt(1, vec![], 1.0, &[]));
        let seed = memory
            .playbook_mut()
            .add(
                Section::Strategies,
                "Prefer a sieve.",
                vec![EntrySeq::from_raw(0)],
            )
            .expect("added");
        let other = memory
            .playbook_mut()
            .add(
                Section::Mistakes,
                "Never allocate per call.",
                vec![EntrySeq::from_raw(0)],
            )
            .expect("added");
        let _ = memory.commit(attempt(2, vec![1], 3.0, &[seed]));
        // A re-evaluation is not a new attempt.
        let _ = memory.commit(attempt(2, vec![1], 3.5, &[seed]));
        assert!(!memory.reflection_due());

        let request = memory
            .reflection_request("Count primes.")
            .expect("two pending");
        assert_eq!(
            Vec::from_iter(request.reviewed()),
            vec![EntrySeq::from_raw(0), EntrySeq::from_raw(1)]
        );
        assert!(request.system_prompt().contains("\"bullet_tags\""));
        let user = request.user_prompt();
        assert!(user.starts_with("## Problem\n\nCount primes."));
        assert!(user.contains("### #1 (score 3.5; parent #0 scored 1, delta +2.5)"));
        assert!(user.contains("Bullets shown: str-00001"));
        assert!(user.contains("## Bullets to tag\n\n[str-00001] :: Prefer a sieve."));
        assert!(
            !user.contains("[err-00002] ::"),
            "only shown bullets are tagged"
        );
        assert!(
            memory.reflection_request("again").is_none(),
            "the queue was drained"
        );

        assert_eq!(
            memory.apply_reflection(&request, "no json"),
            Err(ReflectionError::NoJson)
        );
        let answer = r##"{"reasoning": "the sieve helped",
            "bullet_tags": [{"id": "str-00001", "tag": "helpful"},
                            {"id": "err-00002", "tag": "harmful"},
                            {"id": "str-00001", "tag": "harmful"}],
            "operations": [{"type": "ADD", "section": "heuristics",
                            "content": "Bit-pack the sieve for cache locality.", "evidence": ["#1"]},
                           {"type": "ADD", "section": "heuristics",
                            "content": "Unfounded rule.", "evidence": [99]},
                           {"type": "DELETE", "id": "err-00002"}]}"##;
        let report = memory.apply_reflection(&request, answer).expect("applies");
        assert_eq!(report.reasoning(), "the sieve helped");
        assert_eq!(report.helpful(), 1);
        assert_eq!(report.ignored_tags(), 2, "not shown, and repeated");
        assert_eq!(report.added().len(), 1);
        assert_eq!(report.rejected().len(), 1);
        assert_eq!(report.rejected()[0].1, Rejection::NoEvidence);
        assert_eq!(report.unsupported(), 1);
        assert_eq!(memory.playbook().get(seed).expect("kept").helpful(), 1);
        assert_eq!(memory.playbook().get(other).expect("kept").harmful(), 0);
        let added = memory.playbook().get(report.added()[0]).expect("added");
        assert_eq!(added.evidence(), &[EntrySeq::from_raw(1)]);
        assert_eq!(added.section(), Section::Heuristics);
        assert_eq!(
            memory.apply_reflection(&request, answer),
            Err(ReflectionError::NotOutstanding(request.id()))
        );
    }

    #[test]
    fn abandoned_requests_requeue_their_attempts() {
        let config = MemoryConfig {
            reflection: ReflectionConfig::new(2, 1, 1_000, 0),
            ..MemoryConfig::default()
        };
        let mut memory = Memory::<u64>::new(config);
        let _ = memory.commit(attempt(1, vec![], 1.0, &[]));
        let _ = memory.commit(attempt(2, vec![], 2.0, &[]));
        assert!(memory.reflection_due());
        let first = memory.reflection_request("t").expect("pending");
        assert_eq!(
            Vec::from_iter(first.reviewed()),
            vec![EntrySeq::from_raw(0)]
        );
        assert!(!memory.reflection_due());
        memory.abandon_reflection(first);
        assert!(memory.reflection_due());
        let again = memory.reflection_request("t").expect("pending");
        assert_eq!(
            Vec::from_iter(again.reviewed()),
            vec![EntrySeq::from_raw(0)]
        );
    }

    #[test]
    fn removed_attempts_leave_the_review_queue() {
        let config = MemoryConfig {
            archive: ArchiveConfig::new(2, 1).expect("valid"),
            ..MemoryConfig::default()
        };
        let mut memory = Memory::<u64>::new(config);
        let _ = memory.commit(attempt(1, vec![], 1.0, &[]));
        let _ = memory.commit(attempt(2, vec![], 0.0, &[]));
        // Evicts #1 (the worst unprotected entry).
        let _ = memory.commit(attempt(3, vec![], 2.0, &[]));
        let _ = memory.archive_mut().remove(&1);
        let request = memory.reflection_request("t").expect("pending");
        assert_eq!(
            Vec::from_iter(request.reviewed()),
            vec![EntrySeq::from_raw(2)]
        );
    }
}
