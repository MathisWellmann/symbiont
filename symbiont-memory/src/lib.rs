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
//! use symbiont_memory::{Evaluated, IslandId, Memory, MemoryConfig, Score};
//!
//! let mut memory = Memory::<u64>::new(MemoryConfig::default());
//! let rendered = memory.render(IslandId::default(), &[]);
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
//!     .reflection_request(IslandId::default(), "Minimize f on [-1, 1].")
//!     .expect("one attempt awaits review");
//! // let answer = model.complete(request.system_prompt(), request.user_prompt()).await?;
//! let answer = r#"{"operations": [{"type": "ADD", "section": "mistakes",
//!     "content": "Guard the division: x can be 0.", "evidence": [0]}]}"#;
//! let report = memory.apply_reflection(&request, answer).expect("valid answer");
//! assert_eq!(report.added().len(), 1);
//! assert!(memory.render(IslandId::default(), &[]).shared().contains("Guard the division"));
//! ```
//!
//! # Islands
//!
//! With [`IslandConfig::count`] above one, every attempt belongs to the
//! island of the lane that wrote it, each island keeps its own archive
//! members and lessons, and [`Memory::migrate`] moves the best attempts
//! around a ring and promotes proven island lessons to the global playbook.
//! This keeps one shared memory from collapsing the whole population onto
//! one program. [`Memory::novelty`] flags candidates that nearly copy an
//! archived attempt, before the host pays for their evaluation.
//!
//! # Ownership and determinism
//!
//! A [`Memory`] is plain data. It is owned by one task, mutated through
//! `&mut self` and serializable, so a host can checkpoint it next to its
//! population: [`Memory::snapshot`] writes a versioned snapshot and
//! [`Memory::restore`] checks one before it reads it back. Nothing in it reads the clock or a random number generator,
//! and rendering is a pure function of the state: the same state renders the
//! same bytes, which keeps a server's prefix cache warm and makes a recorded
//! run replayable.
//!
//! # Watching it work
//!
//! Every state change logs a `tracing` event at `debug` level: commits and
//! evictions, pitfalls that start to be shown, the size of every rendered
//! memory section, reflection requests and what each answer changed,
//! migrations, resets and near-copy verdicts. They are verbose, one or more
//! per lane and round, so they stay off at the default level. An operator
//! turns them on with `RUST_LOG=symbiont_memory=debug` (with
//! `symbiont::init_tracing` or any `EnvFilter` subscriber).
//!
//! # Example
//!
//! ```
//! use symbiont_memory::{Evaluated, IslandId, Memory, MemoryConfig, Score};
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
//! let rendered = memory.render(IslandId::default(), &[]);
//! let prompt = format!("Improve the function.\n\n{}", rendered.text());
//! assert!(prompt.contains("return 1.25%"));
//! ```

mod archive;
mod failures;
mod island;
mod playbook;
mod reflection;
mod render;
mod snapshot;

use serde::{
    Deserialize,
    Serialize,
};
use tracing::debug;

use crate::reflection::Pending;
pub use crate::{
    archive::{
        Archive,
        ArchiveConfig,
        Committed,
        Entry,
        EntrySeq,
        Evaluated,
        EvaluatedBuilder,
        NonFiniteScore,
        Score,
    },
    failures::{
        FailureConfig,
        Failures,
        Pitfall,
    },
    island::{
        IslandConfig,
        IslandId,
        Migration,
        Nearest,
        Scope,
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
        REFLECTION_SYSTEM_PROMPT,
        ReflectionConfig,
        ReflectionError,
        ReflectionRequest,
        RequestId,
    },
    render::{
        RenderConfig,
        estimate_tokens,
    },
    snapshot::{
        RestoreError,
        SNAPSHOT_VERSION,
        SnapshotError,
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
    /// There must be at least one island.
    #[error("the island count must be positive")]
    NoIslands,
    /// A limit outside the range it works in.
    #[error("{field} {reason}")]
    Range {
        /// The config field, e.g. `playbook.max_bullets`.
        field: &'static str,
        /// What the value must be.
        reason: &'static str,
    },
}

/// The configuration of a [`Memory`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
// Missing fields take their defaults, so a config file names only what it
// changes. Deserializing skips the constructors: check with
// `MemoryConfig::validate`.
#[serde(default)]
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
    /// The islands.
    pub islands: IslandConfig,
}

impl MemoryConfig {
    /// Check the limits that the constructors of the parts enforce, for a
    /// config that came from a file: deserializing bypasses them.
    ///
    /// # Errors
    ///
    /// The first [`ConfigError`] found.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let archive = self.archive;
        let _ = ArchiveConfig::new(archive.capacity(), archive.protect_recent())?;
        let islands = self.islands;
        let _ = IslandConfig::new(
            islands.count(),
            islands.migrants(),
            islands.promote_net(),
            islands.novelty_percent(),
        )?;
        let checks = [
            (
                self.reflection.reflect_every() == 0,
                "reflection.reflect_every",
                "must be positive",
            ),
            (
                self.reflection.max_reviewed() == 0,
                "reflection.max_reviewed",
                "must be positive",
            ),
            (
                self.playbook.max_bullets() == 0,
                "playbook.max_bullets",
                "must be positive",
            ),
            (
                self.playbook.prune_margin() == 0,
                "playbook.prune_margin",
                "must be positive",
            ),
            (
                self.playbook.duplicate_percent() > 100,
                "playbook.duplicate_percent",
                "must be at most 100",
            ),
            (
                self.failures.min_occurrences() == 0,
                "failures.min_occurrences",
                "must be positive",
            ),
            (
                self.failures.max_tracked() < self.failures.max_shown().max(1),
                "failures.max_tracked",
                "must be positive and at least failures.max_shown",
            ),
            (
                islands.promote_net() == 0,
                "islands.promote_net",
                "must be positive",
            ),
            (
                islands.novelty_percent() > 100,
                "islands.novelty_percent",
                "must be at most 100",
            ),
        ];
        match checks.into_iter().find(|(bad, _, _)| *bad) {
            Some((_, field, reason)) => Err(ConfigError::Range { field, reason }),
            None => Ok(()),
        }
    }
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
    island: String,
    lane: String,
    entries: Vec<EntrySeq>,
    bullets: Vec<BulletId>,
}

impl Rendered {
    /// The whole memory section, ready to go into a prompt: the
    /// [`Self::shared`], [`Self::island`] and [`Self::lane`] parts in that
    /// order.
    #[must_use]
    pub fn text(&self) -> String {
        let mut text =
            String::with_capacity(self.shared.len() + self.island.len() + self.lane.len());
        text.push_str(&self.shared);
        text.push_str(&self.island);
        text.push_str(&self.lane);
        text
    }

    /// The part every lane sees alike: the global playbook and the known
    /// pitfalls. It changes rarely, so a host that cares about a server's
    /// prefix cache puts it before anything lane-specific in the prompt.
    #[must_use]
    pub fn shared(&self) -> &str {
        &self.shared
    }

    /// The lessons of the lane's island; empty with a single island. Lanes of
    /// one island share it.
    #[must_use]
    pub fn island(&self) -> &str {
        &self.island
    }

    /// The part that changes with every commit: the island's archive of
    /// attempts.
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

    /// The playbook bullets the text shows. Pass them on with
    /// [`EvaluatedBuilder`]'s `bullets_shown`, so the reflection step knows
    /// which lessons the attempt was given.
    #[must_use]
    pub fn bullets(&self) -> &[BulletId] {
        &self.bullets
    }
}

impl<I: PartialEq + Clone> Memory<I> {
    /// An empty memory.
    ///
    /// # Panics
    ///
    /// If [`MemoryConfig::validate`] rejects `config`: a host that reads the
    /// config from a file validates it first.
    #[must_use]
    pub fn new(config: MemoryConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("invalid memory config: {e}");
        }
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

    fn assert_island(&self, island: IslandId) {
        assert!(
            island.get() < self.config.islands.count(),
            "{island} out of range for {} islands",
            self.config.islands.count()
        );
    }

    /// Record an evaluated attempt. See [`Archive::commit`].
    ///
    /// A new attempt also awaits review by the next reflection request of
    /// its island. A re-evaluation of a known id does not: it is not a new
    /// attempt.
    ///
    /// # Panics
    ///
    /// If the attempt's island is not below [`IslandConfig::count`].
    pub fn commit(&mut self, evaluated: Evaluated<I>) -> Committed {
        let island = evaluated.island();
        self.assert_island(island);
        let bullets = evaluated.bullets_shown().to_vec();
        let shown = bullets.len();
        let committed = self.archive.commit(evaluated);
        match committed {
            Committed::Inserted { seq, evicted } => {
                self.pending.push(Pending::new(seq, island, bullets));
                let entry = self
                    .archive
                    .by_seq(seq)
                    .expect("just inserted and protected");
                debug!(
                    %seq,
                    %island,
                    score = entry.score().get(),
                    bullets_shown = shown,
                    evicted = ?evicted.map(|e| e.to_string()),
                    awaiting_review = self.pending.len(),
                    archive_len = self.archive.len(),
                    "memory: committed a new attempt"
                );
            }
            Committed::Updated { seq } => {
                let entry = self.archive.by_seq(seq).expect("just updated");
                debug!(
                    %seq,
                    %island,
                    score = entry.score().get(),
                    islands = ?entry.islands(),
                    "memory: re-scored a known attempt"
                );
            }
        }
        committed
    }

    /// Whether [`ReflectionConfig::reflect_every`] attempts of `island`
    /// await review.
    #[must_use]
    pub fn reflection_due(&self, island: IslandId) -> bool {
        let waiting = self.pending.iter().filter(|p| p.island() == island).count();
        waiting >= self.config.reflection.reflect_every()
    }

    /// The prompts that review the attempts of `island` awaiting review,
    /// oldest first, at most [`ReflectionConfig::max_reviewed`] of them.
    /// `None` if none awaits review.
    ///
    /// `task` describes the optimization problem to the reviewer, typically
    /// the task part of the host's evolve prompt.
    ///
    /// The reviewed attempts leave the queue. Apply the answer with
    /// [`Self::apply_reflection`]; if the model call fails, hand the request
    /// back with [`Self::abandon_reflection`] so the attempts are reviewed
    /// later. With several islands the new lessons belong to `island`; they
    /// reach the other islands once [`Self::migrate`] promotes them.
    ///
    /// # Panics
    ///
    /// If `island` is not below [`IslandConfig::count`].
    pub fn reflection_request(
        &mut self,
        island: IslandId,
        task: &str,
    ) -> Option<ReflectionRequest> {
        self.assert_island(island);
        let archive = &self.archive;
        self.pending.retain(|p| archive.by_seq(p.seq()).is_some());
        let max = self.config.reflection.max_reviewed();
        let mut pending = Vec::with_capacity(max);
        self.pending.retain(|p| {
            let take = p.island() == island && pending.len() < max;
            if take {
                pending.push(p.clone());
            }
            !take
        });
        if pending.is_empty() {
            return None;
        }
        let id = RequestId::new(self.next_request);
        self.next_request += 1;
        self.outstanding.push(id);
        let scope = if self.config.islands.count() > 1 {
            Scope::Island(island)
        } else {
            Scope::Global
        };
        let taken = pending.len();
        let request = reflection::request(id, self, island, scope, task, pending);
        assert_eq!(request.reviewed().count(), taken);
        debug!(
            request = id.get(),
            %island,
            ?scope,
            reviewed = taken,
            still_waiting = self.pending.len(),
            prompt_tokens = estimate_tokens(request.user_prompt()),
            "memory: built a reflection request"
        );
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
        let report = match reflection::apply(request, answer, &mut self.playbook, &self.archive) {
            Ok(report) => report,
            Err(e) => {
                debug!(
                    request = request.id().get(),
                    answer_bytes = answer.len(),
                    "memory: reflection answer holds no reflection JSON, request stays outstanding"
                );
                return Err(e);
            }
        };
        self.outstanding.retain(|id| *id != request.id());
        self.log_curation(request, &report);
        Ok(report)
    }

    fn log_curation(&self, request: &ReflectionRequest, report: &CurationReport) {
        debug!(
            request = request.id().get(),
            island = %request.island(),
            added = report.added().len(),
            rejected = report.rejected().len(),
            helpful = report.helpful(),
            harmful = report.harmful(),
            neutral = report.neutral(),
            ignored_tags = report.ignored_tags(),
            unsupported_ops = report.unsupported(),
            pruned = report.pruned().len(),
            playbook_len = self.playbook.bullets().len(),
            key_insight = report.key_insight().as_str(),
            "memory: applied a reflection"
        );
        for id in report.added() {
            let bullet = self.playbook.get(*id).expect("just added");
            debug!(
                bullet = bullet.label().as_str(),
                scope = ?bullet.scope(),
                evidence = ?bullet.evidence(),
                content = bullet.content().as_str(),
                "memory: playbook lesson added"
            );
        }
        for (content, reason) in report.rejected() {
            debug!(%reason, content = content.as_str(), "memory: playbook lesson rejected");
        }
        for id in report.pruned() {
            debug!(bullet = id.get(), "memory: refuted playbook lesson pruned");
        }
    }

    /// Return the attempts of an unanswered request to the front of the
    /// review queue. A request that is not outstanding is dropped.
    pub fn abandon_reflection(&mut self, request: ReflectionRequest) {
        let id = request.id();
        if !self.outstanding.contains(&id) {
            return;
        }
        self.outstanding.retain(|r| *r != id);
        let island = request.island();
        let mut queue = request.into_pending();
        let requeued = queue.len();
        queue.append(&mut self.pending);
        self.pending = queue;
        debug!(
            request = id.get(),
            %island,
            requeued,
            awaiting_review = self.pending.len(),
            "memory: abandoned a reflection request, its attempts wait for the next one"
        );
    }

    /// Count the rejected attempts of one lane, see [`Failures::commit_trace`].
    /// Call it for every lane, successful or not: a lane that succeeded on
    /// its third attempt still hit two failures. Pitfalls are global: a
    /// method the host type lacks is missing on every island.
    pub fn commit_trace(&mut self, trace: &symbiont::EvolutionTrace) {
        self.failures.commit_trace(trace);
        debug!(
            lane = %trace.lane(),
            attempts = trace.attempts().len(),
            tracked_failures = self.failures.pitfalls().len(),
            shown_pitfalls = self.failures.shown().len(),
            "memory: counted a lane's failures"
        );
    }

    /// Count a panic the host caught while it evaluated a candidate.
    pub fn commit_panic(&mut self, message: &str) {
        self.failures.commit_panic(message);
    }

    /// The memory as prompt text for the next lane on `island`.
    ///
    /// `exclude` names candidates the host already shows in its prompt, such
    /// as the parents of a mutation, so they are not repeated.
    ///
    /// # Panics
    ///
    /// If `island` is not below [`IslandConfig::count`].
    pub fn render(&self, island: IslandId, exclude: &[I]) -> Rendered {
        self.assert_island(island);
        let (lane, entries) =
            render::render_archive(&self.archive, island, exclude, &self.config.render);
        let (mut shared, mut bullets) = self.playbook.render(Scope::Global);
        shared.push_str(&self.failures.render());
        let (island_text, island_bullets) = self.playbook.render(Scope::Island(island));
        bullets.extend(island_bullets);
        debug!(
            %island,
            shared_tokens = estimate_tokens(&shared),
            island_tokens = estimate_tokens(&island_text),
            lane_tokens = estimate_tokens(&lane),
            entries = entries.len(),
            bullets = bullets.len(),
            pitfalls = self.failures.shown().len(),
            "memory: rendered the memory for a lane"
        );
        Rendered {
            shared,
            island: island_text,
            lane,
            entries,
            bullets,
        }
    }

    /// Exchange knowledge between islands: every island sends its
    /// [`IslandConfig::migrants`] best attempts to the next island in the
    /// ring, and island lessons whose [`Bullet::net`] reached
    /// [`IslandConfig::promote_net`] move to the global playbook (unless a
    /// global bullet already says the same). Does nothing with one island.
    ///
    /// The migrants are chosen before any of them moves, so an attempt
    /// travels one island per call. Call it every few rounds; how often is
    /// the classic trade-off between diversity and spreading discoveries.
    pub fn migrate(&mut self) -> Migration {
        let islands = self.config.islands;
        if islands.count() < 2 {
            return Migration::default();
        }
        let moves = Vec::from_iter(islands.ids().flat_map(|from| {
            let to = islands.next(from);
            let best = self.archive.ranked(from);
            Vec::from_iter(
                best.into_iter()
                    .take(islands.migrants())
                    .map(move |e| (e.seq(), to)),
            )
        }));
        let mut migration = Migration::default();
        for (seq, to) in moves {
            if self.archive.migrate(seq, to) {
                migration.migrated.push((seq, to));
            }
        }
        let ripe = Vec::from_iter(
            self.playbook
                .bullets()
                .iter()
                .filter(|b| b.scope() != Scope::Global)
                .filter(|b| b.net() >= i64::from(islands.promote_net()))
                .map(Bullet::id),
        );
        for id in ripe {
            match self.playbook.promote(id) {
                Ok(()) => migration.promoted.push(id),
                Err(reason) => {
                    debug!(bullet = id.get(), %reason, "memory: island lesson not promoted");
                }
            }
        }
        for (seq, to) in &migration.migrated {
            debug!(%seq, to = %to, "memory: attempt migrated");
        }
        for id in &migration.promoted {
            let bullet = self.playbook.get(*id).expect("just promoted");
            debug!(
                bullet = bullet.label().as_str(),
                net = bullet.net(),
                content = bullet.content().as_str(),
                "memory: island lesson promoted to the global playbook"
            );
        }
        debug!(
            migrated = migration.migrated.len(),
            promoted = migration.promoted.len(),
            "memory: migration done"
        );
        migration
    }

    /// Forget an island: its archive membership, its lessons and its queued
    /// attempts. Use it when the host resets a stagnant sub-population; the
    /// next [`Self::migrate`] re-seeds it from its neighbour. An outstanding
    /// reflection request for the island still applies afterwards, adding
    /// fresh lessons to the empty island.
    ///
    /// # Panics
    ///
    /// If `island` is not below [`IslandConfig::count`].
    pub fn reset_island(&mut self, island: IslandId) {
        self.assert_island(island);
        self.archive.clear_island(island);
        self.playbook.clear_island(island);
        self.pending.retain(|p| p.island() != island);
        assert_eq!(self.archive.members(island).count(), 0);
        debug!(
            %island,
            archive_len = self.archive.len(),
            playbook_len = self.playbook.bullets().len(),
            "memory: island reset"
        );
    }

    /// The attempt on `island` most similar to `source`, by token trigrams.
    /// `None` if the island holds no attempt.
    ///
    /// Call it before an expensive evaluation and skip candidates that are
    /// not [`Nearest::novel`]: a near copy of an archived attempt would
    /// mostly re-measure a known score.
    ///
    /// # Panics
    ///
    /// If `island` is not below [`IslandConfig::count`].
    #[must_use]
    pub fn novelty(&self, island: IslandId, source: &str) -> Option<Nearest> {
        self.assert_island(island);
        let candidate = island::shingles(source);
        let (seq, percent) = self
            .archive
            .members(island)
            .map(|e| {
                (
                    e.seq(),
                    island::similarity(&candidate, &island::shingles(e.source())),
                )
            })
            .max_by_key(|&(seq, percent)| (percent, std::cmp::Reverse(seq)))?;
        assert!(percent <= 100);
        let novel = percent < self.config.islands.novelty_percent();
        debug!(%island, nearest = %seq, percent, novel, "memory: novelty check");
        Some(Nearest {
            seq,
            percent,
            novel,
        })
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
        assert_eq!(memory.render(IslandId::default(), &[]), Rendered::default());
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
        let rendered = memory.render(IslandId::default(), &[]);
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
            .add(
                Scope::Global,
                Section::Mistakes,
                "Guard every division.",
                vec![seq],
            )
            .expect("added");
        let rendered = memory.render(IslandId::default(), &[]);
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
        assert!(
            memory
                .reflection_request(IslandId::default(), "task")
                .is_none()
        );
        let shown = memory
            .playbook_mut()
            .add(
                Scope::Global,
                Section::Strategies,
                "Prefer a sieve.",
                vec![],
            )
            .err();
        assert_eq!(shown, Some(Rejection::NoEvidence));

        let _ = memory.commit(attempt(1, vec![], 1.0, &[]));
        let seed = memory
            .playbook_mut()
            .add(
                Scope::Global,
                Section::Strategies,
                "Prefer a sieve.",
                vec![EntrySeq::from_raw(0)],
            )
            .expect("added");
        let other = memory
            .playbook_mut()
            .add(
                Scope::Global,
                Section::Mistakes,
                "Never allocate per call.",
                vec![EntrySeq::from_raw(0)],
            )
            .expect("added");
        let _ = memory.commit(attempt(2, vec![1], 3.0, &[seed]));
        // A re-evaluation is not a new attempt.
        let _ = memory.commit(attempt(2, vec![1], 3.5, &[seed]));
        assert!(!memory.reflection_due(IslandId::default()));

        let request = memory
            .reflection_request(IslandId::default(), "Count primes.")
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
            memory
                .reflection_request(IslandId::default(), "again")
                .is_none(),
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
        assert!(memory.reflection_due(IslandId::default()));
        let first = memory
            .reflection_request(IslandId::default(), "t")
            .expect("pending");
        assert_eq!(
            Vec::from_iter(first.reviewed()),
            vec![EntrySeq::from_raw(0)]
        );
        assert!(!memory.reflection_due(IslandId::default()));
        memory.abandon_reflection(first);
        assert!(memory.reflection_due(IslandId::default()));
        let again = memory
            .reflection_request(IslandId::default(), "t")
            .expect("pending");
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
        let request = memory
            .reflection_request(IslandId::default(), "t")
            .expect("pending");
        assert_eq!(
            Vec::from_iter(request.reviewed()),
            vec![EntrySeq::from_raw(2)]
        );
    }

    fn islands(count: u16) -> MemoryConfig {
        MemoryConfig {
            archive: ArchiveConfig::new(8, 1).expect("valid"),
            reflection: ReflectionConfig::new(1, 8, 1_000, 1),
            islands: IslandConfig::new(count, 1, 2, 90).expect("valid"),
            ..MemoryConfig::default()
        }
    }

    fn on(island: u16, id: u64, score: f64, source: &str) -> Evaluated<u64> {
        Evaluated::builder()
            .id(id)
            .source(source)
            .score(Score::new(score).expect("finite"))
            .report(format!("report {id}"))
            .island(IslandId::new(island))
            .build()
    }

    #[test]
    fn islands_keep_separate_archives_and_lessons() {
        let mut memory = Memory::<u64>::new(islands(2));
        let (a, b) = (IslandId::new(0), IslandId::new(1));
        let _ = memory.commit(on(0, 1, 5.0, "fn a() {}"));
        let _ = memory.commit(on(1, 2, 1.0, "fn b() {}"));
        assert!(memory.render(a, &[]).lane().contains("report 1"));
        assert!(!memory.render(a, &[]).lane().contains("report 2"));
        assert!(memory.reflection_due(b));

        let request = memory.reflection_request(b, "t").expect("pending on b");
        assert_eq!(request.scope(), Scope::Island(b));
        assert_eq!(
            Vec::from_iter(request.reviewed()),
            vec![EntrySeq::from_raw(1)]
        );
        assert!(
            !request.user_prompt().contains("report 1"),
            "a's attempts stay on a"
        );
        let answer = r#"{"operations": [{"type": "ADD", "section": "others",
            "content": "Island b lesson.", "evidence": [1]}]}"#;
        let report = memory.apply_reflection(&request, answer).expect("applies");
        let lesson = report.added()[0];
        assert!(memory.render(b, &[]).island().contains("Island b lesson."));
        assert!(memory.render(b, &[]).bullets().contains(&lesson));
        assert!(memory.render(a, &[]).island().is_empty());
        assert!(memory.render(a, &[]).shared().is_empty());

        // Promotion needs a net of 2.
        assert!(memory.migrate().promoted.is_empty());
        for _ in 0..2 {
            assert!(memory.playbook_mut().tag(lesson, Tag::Helpful));
        }
        let migration = memory.migrate();
        assert_eq!(migration.promoted, vec![lesson]);
        assert!(memory.render(a, &[]).shared().contains("Island b lesson."));
        // Every island sent its best attempt on; the first call already did.
        assert!(
            memory
                .archive()
                .get(&1)
                .expect("known")
                .islands()
                .contains(&b)
        );
        assert!(
            memory
                .archive()
                .get(&2)
                .expect("known")
                .islands()
                .contains(&a)
        );
    }

    #[test]
    fn migration_moves_one_island_per_call() {
        let mut memory = Memory::<u64>::new(islands(3));
        let _ = memory.commit(on(0, 1, 5.0, "fn a() {}"));
        let migration = memory.migrate();
        assert_eq!(
            migration.migrated,
            vec![(EntrySeq::from_raw(0), IslandId::new(1))]
        );
        let islands = memory.archive().get(&1).expect("known").islands().clone();
        assert_eq!(islands, vec![IslandId::new(0), IslandId::new(1)]);
        let _ = memory.migrate();
        assert_eq!(memory.archive().get(&1).expect("known").islands().len(), 3);
        assert_eq!(
            Memory::<u64>::new(MemoryConfig::default()).migrate(),
            Migration::default()
        );
    }

    #[test]
    fn reset_forgets_an_island() {
        let mut memory = Memory::<u64>::new(islands(2));
        let (a, b) = (IslandId::new(0), IslandId::new(1));
        let _ = memory.commit(on(0, 1, 5.0, "fn a() {}"));
        let _ = memory.commit(on(1, 2, 1.0, "fn b() {}"));
        let seq = EntrySeq::from_raw(1);
        let _ = memory
            .playbook_mut()
            .add(Scope::Island(b), Section::Others, "b only", vec![seq])
            .expect("added");
        memory.reset_island(b);
        assert!(memory.render(b, &[]).text().is_empty());
        assert!(memory.archive().get(&2).is_none());
        assert!(memory.reflection_request(b, "t").is_none());
        assert!(memory.reflection_request(a, "t").is_some());
    }

    #[test]
    fn novelty_flags_near_copies() {
        let mut memory = Memory::<u64>::new(islands(2));
        let island = IslandId::new(0);
        assert_eq!(memory.novelty(island, "fn f() {}"), None);
        let source = "fn f(xs: &[f64]) -> f64 { xs.iter().map(|x| x * x).sum::<f64>().sqrt() }";
        let _ = memory.commit(on(0, 1, 1.0, source));
        let reformatted = source.replace("{ ", "{\n    // norm\n    ");
        let copy = memory.novelty(island, &reformatted).expect("one member");
        assert_eq!(copy.seq, EntrySeq::from_raw(0));
        assert_eq!(copy.percent, 100);
        assert!(!copy.novel);
        let other = memory
            .novelty(
                island,
                "fn f(xs: &[f64]) -> f64 { xs.iter().fold(0.0, f64::max) }",
            )
            .expect("one member");
        assert!(other.novel);
        assert_eq!(
            memory.novelty(IslandId::new(1), source),
            None,
            "other island"
        );
    }

    #[test]
    #[should_panic(expected = "island 2 out of range for 2 islands")]
    fn unknown_islands_are_rejected() {
        let memory = Memory::<u64>::new(islands(2));
        let _ = memory.render(IslandId::new(2), &[]);
    }

    #[test]
    #[tracing_test::traced_test]
    fn state_changes_are_logged_at_debug() {
        let mut memory = Memory::<u64>::new(MemoryConfig::default());
        let island = IslandId::default();
        let _ = memory.commit(attempt(1, vec![], 1.0, &[]));
        let _ = memory.render(island, &[]);
        let request = memory.reflection_request(island, "t").expect("pending");
        let answer = r#"{"operations": [{"type": "ADD", "section": "others",
            "content": "A logged lesson.", "evidence": [0]}]}"#;
        let _ = memory.apply_reflection(&request, answer).expect("applies");
        assert!(logs_contain("memory: committed a new attempt"));
        assert!(logs_contain("memory: rendered the memory for a lane"));
        assert!(logs_contain("memory: built a reflection request"));
        assert!(logs_contain("memory: applied a reflection"));
        assert!(logs_contain("A logged lesson."));
    }

    #[test]
    fn configs_fill_missing_fields_and_validate() {
        let config: MemoryConfig =
            serde_json::from_str(r#"{"islands": {"count": 3}}"#).expect("partial config parses");
        assert_eq!(config.islands.count(), 3);
        assert_eq!(
            config.islands.migrants(),
            IslandConfig::default().migrants()
        );
        assert_eq!(config.archive, ArchiveConfig::default());
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(MemoryConfig::default().validate(), Ok(()));

        let no_islands: MemoryConfig =
            serde_json::from_str(r#"{"islands": {"count": 0}}"#).expect("parses");
        assert_eq!(no_islands.validate(), Err(ConfigError::NoIslands));
        let full: MemoryConfig =
            serde_json::from_str(r#"{"archive": {"capacity": 4, "protect_recent": 4}}"#)
                .expect("parses");
        assert!(matches!(
            full.validate(),
            Err(ConfigError::ArchiveCapacity { .. })
        ));
        let zero: MemoryConfig =
            serde_json::from_str(r#"{"playbook": {"max_bullets": 0}}"#).expect("parses");
        assert_eq!(
            zero.validate(),
            Err(ConfigError::Range {
                field: "playbook.max_bullets",
                reason: "must be positive",
            })
        );
    }

    #[test]
    #[should_panic(expected = "invalid memory config")]
    fn new_rejects_an_invalid_config() {
        let config: MemoryConfig =
            serde_json::from_str(r#"{"reflection": {"max_reviewed": 0}}"#).expect("parses");
        let _ = Memory::<u64>::new(config);
    }
}
