// SPDX-License-Identifier: MPL-2.0
//! A recorded tree as a simulator: a policy walks it, the replay reveals the
//! recorded continuations, nothing is generated or evaluated.

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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Maximum batch size `W`.
    pub workers: usize,
    /// Round limit `K₂`: replay stops after this many non-empty batches.
    pub max_rounds: usize,
    /// Consecutive non-empty batches that reveal nothing before the replay
    /// gives up with [`Termination::Stalled`].
    pub stall_limit: usize,
    /// Which revealed nodes may be continued from.
    pub expansion: ExpansionRule,
    /// How actions are matched to recorded children.
    pub matching: MatchRule,
    /// Terminate on the first batch that contains an illegal, duplicate or
    /// over-budget action instead of dropping the offenders.
    pub strict: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            max_rounds: 1_000,
            stall_limit: 4,
            expansion: ExpansionRule::default(),
            matching: MatchRule::default(),
            strict: false,
        }
    }
}

impl ReplayConfig {
    /// Default configuration with `workers` parallel slots.
    #[must_use]
    pub fn with_workers(workers: usize) -> Self {
        Self {
            workers,
            ..Self::default()
        }
    }

    /// Set the round limit.
    #[must_use]
    pub const fn max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Set the stall limit.
    #[must_use]
    pub const fn stall_limit(mut self, stall_limit: usize) -> Self {
        self.stall_limit = stall_limit;
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundRecord {
    /// Actions that were accepted and executed. Empty when the policy
    /// submitted a batch and every action of it was rejected; the round still
    /// counts, see [`Trajectory::rounds`].
    pub batch: Vec<Action>,
    /// Nodes revealed (replay) or recorded (live) by this batch.
    pub revealed: Vec<NodeId>,
    /// Actions dropped as illegal, duplicate or beyond the worker budget.
    pub rejected: Vec<Action>,
}

/// The rounds of one rollout and how it ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trajectory {
    /// Decision rounds in order, one per non-empty batch the policy
    /// *submitted*. A batch whose actions were all rejected is still a
    /// decision the policy spent: it counts towards `K₂` and the stall
    /// limit, deflates the parallelism term, and keeps its `rejected` list
    /// for diagnosis.
    pub rounds: Vec<RoundRecord>,
    /// Why the rollout ended.
    pub termination: Termination,
}

impl Trajectory {
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
            self.config.workers,
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
    /// Illegal, duplicate and over-budget actions are dropped (or, in strict
    /// mode, end the replay); a batch rejected in full is recorded as a round
    /// that revealed nothing. Each accepted action reveals at most one
    /// recorded child: the earliest unrevealed child of `from` whose context
    /// is fully revealed. Repeated root actions open successive branches.
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
        } else if self.rounds.len() >= self.config.max_rounds {
            self.end(Termination::RoundLimit)
        } else if self.stalled >= self.config.stall_limit {
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
        for action in batch {
            let duplicate =
                !action.from.is_root() && accepted.iter().any(|a| a.from == action.from);
            if duplicate || !view.is_legal(action.from) || accepted.len() >= self.config.workers {
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
                .children(action.from)
                .filter(|c| !self.revealed[c.id().index()] && !taken.contains(&c.id()))
                .filter(|c| c.context().iter().all(|&d| self.revealed[d.index()]))
                .find(|c| match self.config.matching {
                    MatchRule::Primary => true,
                    MatchRule::ExactContext => same_set(&c.context(), &action.context),
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
