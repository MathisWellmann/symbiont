// SPDX-License-Identifier: MPL-2.0
//! A recorded tree as a simulator: a policy walks it, the replay reveals the
//! recorded continuations, nothing is generated or evaluated.

use std::num::NonZeroUsize;

use getset::{
    CopyGetters,
    Getters,
    MutGetters,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    policy::{
        Action,
        ExpansionRule,
        Policy,
        View,
    },
    tree::{
        DiscoveryTree,
        NodeId,
    },
};

/// How an [`Action`] is matched against recorded children in replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchRule {
    /// Match on `from` only (paper semantics). The child's own context must
    /// be revealed, but need not equal the action's context.
    #[default]
    Primary,
    /// Match on `from` and require the child's context to equal the action's
    /// context as a set. Stricter, so fewer recorded nodes are in support.
    ExactContext,
}

/// Parameters of one replay.
///
/// The counts are at least 1: a limit of 0 rounds or 0 workers would end
/// every replay before its first decision, so the builders raise 0 to 1.
/// Defaults: 1 worker, 1000 rounds, stall limit 4, [`ExpansionRule::LeavesOnly`],
/// [`MatchRule::Primary`], not strict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Maximum batch size `W`.
    workers: NonZeroUsize,
    /// Round limit `K₂`: replay stops after this many non-empty batches.
    max_rounds: NonZeroUsize,
    /// Consecutive non-empty batches that reveal nothing before the replay
    /// gives up with [`Termination::Stalled`].
    stall_limit: NonZeroUsize,
    /// Which revealed nodes may be continued from.
    expansion: ExpansionRule,
    /// How actions are matched to recorded children.
    matching: MatchRule,
    /// Terminate on the first batch that contains an illegal, repeated or
    /// over-budget action instead of dropping the offenders.
    strict: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            workers: at_least_one(1),
            max_rounds: at_least_one(1_000),
            stall_limit: at_least_one(4),
            expansion: ExpansionRule::default(),
            matching: MatchRule::default(),
            strict: false,
        }
    }
}

const fn at_least_one(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(n) => n,
        None => NonZeroUsize::MIN,
    }
}

impl ReplayConfig {
    /// Default configuration with `workers` parallel slots (at least 1).
    #[must_use]
    pub fn with_workers(workers: usize) -> Self {
        Self {
            workers: at_least_one(workers),
            ..Self::default()
        }
    }

    /// Set the round limit (at least 1).
    #[must_use]
    pub const fn max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = at_least_one(max_rounds);
        self
    }

    /// Set the stall limit (at least 1).
    #[must_use]
    pub const fn stall_limit(mut self, stall_limit: usize) -> Self {
        self.stall_limit = at_least_one(stall_limit);
        self
    }

    /// Set the expansion rule.
    #[must_use]
    pub const fn expansion(mut self, expansion: ExpansionRule) -> Self {
        self.expansion = expansion;
        self
    }

    /// Set the matching rule.
    #[must_use]
    pub const fn matching(mut self, matching: MatchRule) -> Self {
        self.matching = matching;
        self
    }

    /// Set strict mode.
    #[must_use]
    pub const fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

/// Why a rollout ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Termination {
    /// The policy returned an empty batch.
    PolicyStopped,
    /// [`ReplayConfig::max_rounds`] was reached.
    RoundLimit,
    /// Every recorded node is revealed.
    Exhausted,
    /// [`ReplayConfig::stall_limit`] non-empty batches revealed nothing.
    Stalled,
    /// Strict mode and the batch was not legal.
    IllegalBatch,
    /// The consumer ended a live run (budget, time, user).
    External,
}

/// What happened in one decision round.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Getters, MutGetters)]
pub struct RoundRecord {
    /// Actions that were accepted and executed. Empty when the policy
    /// submitted a batch and every action of it was rejected; the round still
    /// counts, see [`Trajectory::rounds`].
    #[getset(get = "pub", get_mut = "pub(crate)")]
    batch: Vec<Action>,

    /// Nodes revealed (replay) or recorded (live) by this batch.
    #[getset(get = "pub", get_mut = "pub(crate)")]
    revealed: Vec<NodeId>,

    /// Actions dropped as illegal, duplicate or beyond the worker budget.
    #[getset(get = "pub")]
    rejected: Vec<Action>,
}

impl RoundRecord {
    pub(crate) const fn empty() -> Self {
        Self {
            batch: Vec::new(),
            revealed: Vec::new(),
            rejected: Vec::new(),
        }
    }
}

/// The rounds of one rollout and how it ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Trajectory {
    /// Decision rounds in order, one per non-empty batch the policy
    /// *submitted*. A batch whose actions were all rejected is still a
    /// decision the policy spent: it counts towards `K₂` and the stall
    /// limit, deflates the parallelism term, and keeps its `rejected` list
    /// for diagnosis.
    #[getset(get = "pub")]
    rounds: Vec<RoundRecord>,

    /// Why the rollout ended.
    #[getset(get_copy = "pub")]
    termination: Termination,
}

impl Trajectory {
    pub(crate) fn new(rounds: Vec<RoundRecord>, termination: Termination) -> Self {
        Self {
            rounds,
            termination,
        }
    }

    /// Number of decision rounds `k*`.
    #[must_use]
    pub fn round_count(&self) -> usize {
        self.rounds.len()
    }

    /// Every node revealed over the rollout, in reveal order. The root is not
    /// included.
    pub fn revealed(&self) -> impl Iterator<Item = NodeId> {
        self.rounds.iter().flat_map(|r| r.revealed.iter().copied())
    }

    /// Number of revealed non-root nodes `N`.
    #[must_use]
    pub fn revealed_count(&self) -> usize {
        self.rounds.iter().map(|r| r.revealed.len()).sum()
    }

    /// Number of actions dropped over the rollout.
    #[must_use]
    pub fn rejected_count(&self) -> usize {
        self.rounds.iter().map(|r| r.rejected.len()).sum()
    }
}

/// A step-by-step replay over one recorded tree.
///
/// Use [`replay`] to drive a [`Policy`] to termination, or step manually
/// with [`Replay::view`] and [`Replay::step`].
pub struct Replay<'a, O> {
    tree: &'a DiscoveryTree<O>,
    revealed: Vec<bool>,
    config: ReplayConfig,
    rounds: Vec<RoundRecord>,
    stalled: usize,
    termination: Option<Termination>,
}

impl<'a, O> Replay<'a, O> {
    /// Start a replay with only the root revealed.
    #[must_use]
    pub fn new(tree: &'a DiscoveryTree<O>, config: ReplayConfig) -> Self {
        let mut revealed = vec![false; tree.node_count()];
        revealed[0] = true;
        Self {
            tree,
            revealed,
            config,
            rounds: Vec::new(),
            stalled: 0,
            termination: None,
        }
    }

    /// The prefix the policy may observe now.
    #[must_use]
    pub fn view(&self) -> View<'_, O> {
        View::new(
            self.tree,
            &self.revealed,
            self.config.expansion,
            self.config.workers.get(),
            self.rounds.len(),
        )
    }

    /// Why the replay ended, once it has.
    #[must_use]
    pub const fn termination(&self) -> Option<Termination> {
        self.termination
    }

    /// Apply one batch. Returns the termination if the replay is over.
    ///
    /// Illegal and over-budget actions are dropped (or, in strict mode, end
    /// the replay), as are repeated non-root `from` ids under
    /// [`ExpansionRule::LeavesOnly`]; a batch rejected in full is recorded as
    /// a round that revealed nothing. Each accepted action reveals at most
    /// one recorded child: the earliest unrevealed child of `from` whose
    /// context is fully revealed. Repeated actions from the same node reveal
    /// successive children.
    pub fn step(&mut self, batch: Vec<Action>) -> Option<Termination> {
        if let Some(t) = self.termination {
            return Some(t);
        }
        if batch.is_empty() {
            return self.end(Termination::PolicyStopped);
        }

        let (accepted, rejected) = self.sanitize(batch);
        let revealed = self.matches(&accepted);
        for id in &revealed {
            self.revealed[id.index()] = true;
        }
        if revealed.is_empty() {
            self.stalled += 1;
        } else {
            self.stalled = 0;
        }
        let illegal = self.config.strict && !rejected.is_empty();
        self.rounds.push(RoundRecord {
            batch: accepted,
            revealed,
            rejected,
        });

        if illegal {
            self.end(Termination::IllegalBatch)
        } else if self.revealed.iter().all(|&r| r) {
            self.end(Termination::Exhausted)
        } else if self.rounds.len() >= self.config.max_rounds.get() {
            self.end(Termination::RoundLimit)
        } else if self.stalled >= self.config.stall_limit.get() {
            self.end(Termination::Stalled)
        } else {
            None
        }
    }

    /// Consume the replay. A replay that has not terminated is recorded as
    /// [`Termination::External`].
    #[must_use]
    pub fn finish(self) -> Trajectory {
        Trajectory {
            rounds: self.rounds,
            termination: self.termination.unwrap_or(Termination::External),
        }
    }

    fn end(&mut self, termination: Termination) -> Option<Termination> {
        self.termination = Some(termination);
        Some(termination)
    }

    fn sanitize(&self, batch: Vec<Action>) -> (Vec<Action>, Vec<Action>) {
        let view = self.view();
        let mut accepted: Vec<Action> = Vec::new();
        let mut rejected = Vec::new();
        let reject_repeats = self.config.expansion == ExpansionRule::LeavesOnly;
        for action in batch {
            let repeat = reject_repeats
                && !action.from().is_root()
                && accepted.iter().any(|a| a.from() == action.from());
            if repeat
                || !view.is_legal(action.from())
                || accepted.len() >= self.config.workers.get()
            {
                rejected.push(action);
            } else {
                accepted.push(action);
            }
        }
        (accepted, rejected)
    }

    /// Children revealed by `accepted`, matched against the pre-round prefix
    /// so that batch members cannot depend on each other.
    fn matches(&self, accepted: &[Action]) -> Vec<NodeId> {
        let mut taken: Vec<NodeId> = Vec::with_capacity(accepted.len());
        for action in accepted {
            let next = self
                .tree
                .children(action.from())
                .filter(|c| !self.revealed[c.id().index()] && !taken.contains(&c.id()))
                .filter(|c| c.context().iter().all(|&d| self.revealed[d.index()]))
                .find(|c| match self.config.matching {
                    MatchRule::Primary => true,
                    MatchRule::ExactContext => same_set(c.context(), action.context()),
                });
            if let Some(child) = next {
                taken.push(child.id());
            }
        }
        taken
    }
}

fn same_set(a: &[NodeId], b: &[NodeId]) -> bool {
    a.iter().all(|x| b.contains(x)) && b.iter().all(|x| a.contains(x))
}

/// Run `policy` over `tree` to termination and return its trajectory.
///
/// The policy is [`reset`](Policy::reset) first.
pub fn replay<O, P>(tree: &DiscoveryTree<O>, policy: &mut P, config: &ReplayConfig) -> Trajectory
where
    P: Policy<O> + ?Sized,
{
    policy.reset();
    let mut sim = Replay::new(tree, config.clone());
    loop {
        let batch = policy.select_batch(&sim.view());
        if sim.step(batch).is_some() {
            return sim.finish();
        }
    }
}
