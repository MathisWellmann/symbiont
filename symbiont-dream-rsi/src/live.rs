// SPDX-License-Identifier: MPL-2.0
//! Recording a live run through the same decision interface the replay
//! uses, so the resulting tree is a valid replay world.

use crate::{
    policy::{
        Action,
        ExpansionRule,
        View,
    },
    replay::{
        RoundRecord,
        Termination,
        Trajectory,
    },
    tree::{
        DiscoveryTree,
        Error,
        NodeId,
    },
};

/// An online rollout under construction.
///
/// The consumer asks the policy for a batch through [`Live::view`], runs the
/// discovery agent and evaluator for every action, and records the outcomes
/// with [`Live::commit_round`]. Every attempt should be recorded, failures
/// included: they cost budget and they are what a policy learns to avoid.
///
/// ```
/// use symbiont_dream_rsi::{Action, Live, NodeId, Policy, Termination};
///
/// let mut live = Live::new(0.0_f64, 2);
/// let mut policy = |view: &symbiont_dream_rsi::View<'_, f64>| -> Vec<Action> {
///     if view.round() == 0 { vec![Action::expand(NodeId::ROOT); 2] } else { Vec::new() }
/// };
/// loop {
///     let batch = policy.select_batch(&live.view());
///     if batch.is_empty() {
///         break;
///     }
///     // run the agent for each action, then record what it produced:
///     let outcomes = batch.into_iter().map(|a| (a, 1.0));
///     live.commit_round(outcomes).expect("actions reference known nodes");
/// }
/// let (tree, trajectory) = live.finish(Termination::PolicyStopped);
/// assert_eq!(tree.node_count(), 3);
/// assert_eq!(trajectory.round_count(), 1);
/// ```
pub struct Live<O> {
    tree: DiscoveryTree<O>,
    revealed: Vec<bool>,
    rounds: Vec<RoundRecord>,
    workers: usize,
    expansion: ExpansionRule,
}

impl<O> Live<O> {
    /// Start a run from the baseline observation `root` with `workers`
    /// parallel slots.
    #[must_use]
    pub fn new(root: O, workers: usize) -> Self {
        Self {
            tree: DiscoveryTree::new(root),
            revealed: vec![true],
            rounds: Vec::new(),
            workers,
            expansion: ExpansionRule::default(),
        }
    }

    /// Set the expansion rule the policy is offered.
    #[must_use]
    pub const fn expansion(mut self, expansion: ExpansionRule) -> Self {
        self.expansion = expansion;
        self
    }

    /// The whole tree so far: live, everything is revealed.
    #[must_use]
    pub fn view(&self) -> View<'_, O> {
        View::new(
            &self.tree,
            &self.revealed,
            self.expansion,
            self.workers,
            self.rounds.len(),
        )
    }

    /// The tree so far.
    #[must_use]
    pub const fn tree(&self) -> &DiscoveryTree<O> {
        &self.tree
    }

    /// Decision rounds committed so far.
    #[must_use]
    pub fn round_count(&self) -> usize {
        self.rounds.len()
    }

    /// Record the outcomes of one batch as new nodes. Returns their ids in
    /// input order.
    ///
    /// The batch is recorded as executed; legality is the policy's job and
    /// an outcome that was paid for is never discarded here. A round that
    /// records no node (empty `outcomes`, or an error on the first action)
    /// is not added to the trajectory.
    ///
    /// # Errors
    /// [`Error::UnknownNode`] if an action references a node that is not in
    /// the tree. Nodes recorded before the offending action stay recorded, as
    /// a partial round.
    pub fn commit_round<I>(&mut self, outcomes: I) -> Result<Vec<NodeId>, Error>
    where
        I: IntoIterator<Item = (Action, O)>,
    {
        let mut record = RoundRecord::default();
        for (action, observation) in outcomes {
            let id = match self
                .tree
                .push(action.from(), action.context().clone(), observation)
            {
                Ok(id) => id,
                Err(e) => {
                    self.push_round(record);
                    return Err(e);
                }
            };
            self.revealed.push(true);
            record.batch_mut().push(action);
            record.revealed_mut().push(id);
        }
        let ids = record.revealed().clone();
        self.push_round(record);
        Ok(ids)
    }

    fn push_round(&mut self, record: RoundRecord) {
        if !record.revealed().is_empty() {
            self.rounds.push(record);
        }
    }

    /// End the run. The trajectory carries the same round structure a replay
    /// produces, so the live run can be scored by the same objective.
    #[must_use]
    pub fn finish(self, termination: Termination) -> (DiscoveryTree<O>, Trajectory) {
        (self.tree, Trajectory::new(self.rounds, termination))
    }
}
