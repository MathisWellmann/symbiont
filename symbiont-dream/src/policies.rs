// SPDX-License-Identifier: MPL-2.0
//! Reference policies.

use crate::{
    policy::{
        Action,
        Policy,
        View,
    },
    tree::NodeId,
};

/// The paper's initial policy: open `branches` independent branches from the
/// root and refine each of them `refinements` times.
///
/// Stateless; it reads the branch count and depths from the revealed prefix,
/// so it replays faithfully. Refinements are scheduled before new branches
/// and the batch is cut to [`View::workers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelRefining {
    /// Branches to open from the root.
    pub branches: usize,
    /// Refinement steps after the first attempt of every branch.
    pub refinements: usize,
}

impl<O> Policy<O> for ParallelRefining {
    fn select_batch(&mut self, view: &View<'_, O>) -> Vec<Action> {
        let mut batch: Vec<Action> = view
            .frontier()
            .into_iter()
            .filter(|&id| view.depth(id).is_some_and(|d| d <= self.refinements))
            .map(Action::expand)
            .collect();
        let opened = view.children(NodeId::ROOT).count();
        batch.extend((opened..self.branches).map(|_| Action::expand(NodeId::ROOT)));
        batch.truncate(view.workers());
        batch
    }
}
