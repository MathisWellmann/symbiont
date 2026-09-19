// SPDX-License-Identifier: MPL-2.0
//! The replay objective (eq. 1 of the paper), parameterised by how the
//! consumer reads quality and cost out of an observation.

use getset::CopyGetters;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    replay::Trajectory,
    tree::{
        DiscoveryTree,
        Error,
    },
};

type Measure<'f, O> = Box<dyn Fn(&O) -> f64 + Send + Sync + 'f>;

/// `V = max quality − β₁·Σ cost + β₂·N / max(1, k)`.
///
/// * quality: the best value of `quality` over the root and every revealed
///   node (larger is better);
/// * cost: the sum of `cost` over revealed nodes, one per node by default;
/// * parallelism: revealed nodes per decision round, rewarding batched
///   continuations over serial ones.
pub struct Objective<'f, O> {
    quality: Measure<'f, O>,
    cost: Measure<'f, O>,
    beta_cost: f64,
    beta_parallel: f64,
}

impl<'f, O> Objective<'f, O> {
    /// An objective that reads quality with `quality`, charges one unit per
    /// revealed node, and weights both penalties at zero.
    pub fn new(quality: impl Fn(&O) -> f64 + Send + Sync + 'f) -> Self {
        Self {
            quality: Box::new(quality),
            cost: Box::new(|_| 1.0),
            beta_cost: 0.0,
            beta_parallel: 0.0,
        }
    }

    /// Read the cost of one revealed node from its observation (agent
    /// calls, tokens, seconds, money).
    #[must_use]
    pub fn cost(mut self, cost: impl Fn(&O) -> f64 + Send + Sync + 'f) -> Self {
        self.cost = Box::new(cost);
        self
    }

    /// Weight `β₁` of the cost term.
    #[must_use]
    pub const fn cost_weight(mut self, beta: f64) -> Self {
        self.beta_cost = beta;
        self
    }

    /// Weight `β₂` of the parallelism term.
    #[must_use]
    pub const fn parallelism_weight(mut self, beta: f64) -> Self {
        self.beta_parallel = beta;
        self
    }

    fn quality_of(&self, observation: &O) -> f64 {
        (self.quality)(observation)
    }

    fn cost_of(&self, observation: &O) -> f64 {
        (self.cost)(observation)
    }

    /// Score `trajectory`, which must have been produced over `tree`.
    ///
    /// # Errors
    /// [`Error::UnknownNode`] if the trajectory references a node that is
    /// not in `tree`.
    pub fn score(
        &self,
        tree: &DiscoveryTree<O>,
        trajectory: &Trajectory,
    ) -> Result<ReplayScore, Error> {
        let mut best_quality = self.quality_of(tree.root().observation());
        let mut total_cost = 0.0;
        let mut revealed = 0_usize;
        for id in trajectory.revealed() {
            let node = tree.get(id).ok_or(Error::UnknownNode(id))?;
            best_quality = best_quality.max(self.quality_of(node.observation()));
            total_cost += self.cost_of(node.observation());
            revealed += 1;
        }
        let rounds = trajectory.round_count();
        let parallelism = revealed as f64 / rounds.max(1) as f64;
        let value = best_quality - self.beta_cost * total_cost + self.beta_parallel * parallelism;
        Ok(ReplayScore {
            best_quality,
            total_cost,
            revealed,
            rounds,
            parallelism,
            value,
        })
    }
}

/// The terms of one scored trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, CopyGetters)]
pub struct ReplayScore {
    /// Best quality among the root and the revealed nodes.
    #[getset(get_copy = "pub")]
    best_quality: f64,

    /// Summed cost of the revealed nodes.
    #[getset(get_copy = "pub")]
    total_cost: f64,

    /// Revealed non-root nodes `N`.
    #[getset(get_copy = "pub")]
    revealed: usize,

    /// Decision rounds `k*`.
    #[getset(get_copy = "pub")]
    rounds: usize,

    /// `N / max(1, k*)`.
    #[getset(get_copy = "pub")]
    parallelism: f64,

    /// The combined value `V`.
    #[getset(get_copy = "pub")]
    value: f64,
}
