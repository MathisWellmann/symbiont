// SPDX-License-Identifier: MPL-2.0
//! The decision interface shared by online exploration and offline replay:
//! what a policy may see ([`View`]) and what it may do ([`Action`]).

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::tree::{
    DiscoveryTree,
    Node,
    NodeId,
};

/// One request to the discovery agent: continue from `from`, also showing
/// `context`.
///
/// Online the consumer turns this into a prompt. In replay only `from` is
/// matched against recorded children by default; see
/// [`MatchRule`](crate::MatchRule).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Action {
    /// The node whose state the attempt resumes.
    #[getset(get_copy = "pub")]
    from: NodeId,

    /// Other revealed nodes the attempt is shown.
    #[getset(get = "pub")]
    context: Vec<NodeId>,
}

impl Action {
    /// Continue from `from` without extra context.
    #[must_use]
    pub fn expand(from: NodeId) -> Self {
        Self {
            from,
            context: Vec::new(),
        }
    }

    /// Continue from `from`, also showing `context`.
    #[must_use]
    pub fn with_context(from: NodeId, context: impl IntoIterator<Item = NodeId>) -> Self {
        Self {
            from,
            context: context.into_iter().collect(),
        }
    }
}

impl From<NodeId> for Action {
    fn from(from: NodeId) -> Self {
        Self::expand(from)
    }
}

/// Which revealed nodes a policy may continue from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpansionRule {
    /// The root and the revealed leaves (paper semantics: every branch is a
    /// chain and can only grow at its tip). A batch may name a non-root node
    /// once; only the root may repeat.
    #[default]
    LeavesOnly,
    /// Any revealed node, any number of times per batch. Needed when several
    /// attempts fan out from one parent, as with a batch of lanes seeded from
    /// the same revision.
    AnyRevealed,
}

/// The prefix a policy is allowed to observe.
///
/// A `View` exposes only revealed nodes. Everything reachable from it is
/// derived from those, so a policy written against `View` is prefix-only by
/// construction and cannot read unrevealed outcomes.
#[derive(Clone, Copy)]
pub struct View<'a, O> {
    tree: &'a DiscoveryTree<O>,
    revealed: &'a [bool],
    rule: ExpansionRule,
    workers: usize,
    round: usize,
}

impl<'a, O> View<'a, O> {
    pub(crate) fn new(
        tree: &'a DiscoveryTree<O>,
        revealed: &'a [bool],
        rule: ExpansionRule,
        workers: usize,
        round: usize,
    ) -> Self {
        debug_assert_eq!(tree.node_count(), revealed.len());
        debug_assert!(
            revealed.first().copied().unwrap_or(false),
            "root is always revealed"
        );
        Self {
            tree,
            revealed,
            rule,
            workers,
            round,
        }
    }

    /// The root node.
    #[must_use]
    pub fn root(&self) -> &'a Node<O> {
        self.tree.root()
    }

    /// The node with `id`, if it has been revealed.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&'a Node<O>> {
        if self.is_revealed(id) {
            self.tree.get(id)
        } else {
            None
        }
    }

    /// `true` if `id` is revealed.
    #[must_use]
    pub fn is_revealed(&self, id: NodeId) -> bool {
        self.revealed.get(id.index()).copied().unwrap_or(false)
    }

    /// Every revealed node, root first, in creation order.
    pub fn observed(&self) -> impl Iterator<Item = &'a Node<O>> {
        let revealed = self.revealed;
        self.tree
            .nodes()
            .iter()
            .filter(move |n| revealed.get(n.id().index()).copied().unwrap_or(false))
    }

    /// Number of revealed nodes, not counting the root.
    #[must_use]
    pub fn revealed_count(&self) -> usize {
        self.revealed
            .iter()
            .filter(|&&r| r)
            .count()
            .saturating_sub(1)
    }

    /// Revealed nodes whose primary parent is `id`, in creation order.
    pub fn children(&self, id: NodeId) -> impl Iterator<Item = &'a Node<O>> {
        let revealed = self.revealed;
        self.tree
            .children(id)
            .filter(move |n| revealed.get(n.id().index()).copied().unwrap_or(false))
    }

    /// `true` if a revealed node continues from `id`.
    #[must_use]
    pub fn has_children(&self, id: NodeId) -> bool {
        self.children(id).next().is_some()
    }

    /// Revealed non-root nodes that nothing revealed continues from: the tips
    /// of the observed branches.
    #[must_use]
    pub fn frontier(&self) -> Vec<NodeId> {
        self.observed()
            .filter(|n| !n.is_root() && !self.has_children(n.id()))
            .map(|n| n.id())
            .collect()
    }

    /// Nodes a batch may continue from under the current [`ExpansionRule`].
    /// The root is always included.
    #[must_use]
    pub fn legal_actions(&self) -> Vec<NodeId> {
        match self.rule {
            ExpansionRule::LeavesOnly => {
                let mut legal = vec![NodeId::ROOT];
                legal.extend(self.frontier());
                legal
            }
            ExpansionRule::AnyRevealed => self.observed().map(|n| n.id()).collect(),
        }
    }

    /// `true` if a batch may continue from `id`.
    #[must_use]
    pub fn is_legal(&self, id: NodeId) -> bool {
        self.is_revealed(id)
            && match self.rule {
                ExpansionRule::LeavesOnly => id.is_root() || !self.has_children(id),
                ExpansionRule::AnyRevealed => true,
            }
    }

    /// Primary-edge distance of a revealed node to the root.
    #[must_use]
    pub fn depth(&self, id: NodeId) -> Option<usize> {
        self.is_revealed(id).then(|| self.tree.depth(id)).flatten()
    }

    /// Primary chain from a revealed node up to the root. Every node on it is
    /// revealed: a child is only ever revealed after its primary parent.
    #[must_use]
    pub fn lineage(&self, id: NodeId) -> Option<Vec<NodeId>> {
        self.is_revealed(id)
            .then(|| self.tree.lineage(id))
            .flatten()
    }

    /// The expansion rule in force.
    #[must_use]
    pub const fn rule(&self) -> ExpansionRule {
        self.rule
    }

    /// Maximum batch size `W`.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }

    /// Decision rounds completed so far.
    #[must_use]
    pub const fn round(&self) -> usize {
        self.round
    }
}

/// An exploration policy: given the revealed prefix, choose where to continue.
///
/// The same policy drives the live run and the replay; only the transition
/// after a batch differs. An empty batch means *stop*.
pub trait Policy<O> {
    /// Clear per-rollout state. Called once before every live run or replay.
    fn reset(&mut self) {}

    /// Select up to [`View::workers`] actions from [`View::legal_actions`].
    ///
    /// Each occurrence of a `from` id continues one more attempt from that
    /// node. Under [`ExpansionRule::LeavesOnly`] only the root may occur more
    /// than once; see [`View::rule`].
    fn select_batch(&mut self, view: &View<'_, O>) -> Vec<Action>;
}

impl<O, F> Policy<O> for F
where
    F: FnMut(&View<'_, O>) -> Vec<Action>,
{
    fn select_batch(&mut self, view: &View<'_, O>) -> Vec<Action> {
        self(view)
    }
}
