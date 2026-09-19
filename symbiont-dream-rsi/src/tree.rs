// SPDX-License-Identifier: MPL-2.0
//! The recorded discovery tree: node identifiers, nodes, and structural
//! queries. Nothing in here interprets the observation `O`.

use std::fmt;

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};

/// Identifies one node of a [`DiscoveryTree`].
///
/// Ids are dense and allocated in creation order: `a < b` means `a` was
/// recorded before `b`. Replay uses that order to pick the *earliest* recorded
/// continuation of a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(u32);

impl NodeId {
    /// The root: the initial state before any attempt.
    pub const ROOT: Self = Self(0);

    /// Position of the node in [`DiscoveryTree::nodes`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// `true` for [`NodeId::ROOT`].
    #[must_use]
    pub const fn is_root(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Errors of the structural layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A referenced node is not part of the tree.
    #[error("node {0} is not part of the tree.")]
    UnknownNode(NodeId),
    /// The tree holds `u32::MAX` nodes already.
    #[error("the tree cannot hold more nodes.")]
    Capacity,
    /// A deserialized tree has no nodes; the root must exist.
    #[error("the tree has no root.")]
    Empty,
    /// A deserialized node's id does not match its position.
    #[error("node {id} is stored at position {position}.")]
    MisplacedId {
        /// Where the node is stored.
        position: usize,
        /// The id it claims.
        id: NodeId,
    },
    /// The deserialized root has a primary parent.
    #[error("the root must not have a primary parent.")]
    RootWithParent,
    /// A deserialized non-root node has no primary parent.
    #[error("node {0} has no primary parent.")]
    Orphan(NodeId),
    /// A deserialized edge points at the node itself or a later node.
    #[error("node {node} references {target}, which is not an earlier node.")]
    ForwardEdge {
        /// The node holding the edge.
        node: NodeId,
        /// Where the edge points.
        target: NodeId,
    },
}

/// One recorded generate–evaluate attempt.
///
/// The crate reads only the edges. `observation` is whatever the consumer
/// measured: a score, a report, a failure class, a cost, all of them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Node<O> {
    /// This node's id.
    #[getset(get_copy = "pub")]
    id: NodeId,
    /// The node this attempt resumed from. `None` only for the root.
    #[getset(get_copy = "pub")]
    primary: Option<NodeId>,
    /// Other nodes the attempt was shown besides `primary`.
    ///
    /// Replay reveals this node only once every context node is revealed,
    /// so a policy is never credited with an outcome that depended on
    /// information it had not yet paid for.
    #[getset(get = "pub")]
    context: Vec<NodeId>,
    /// What the consumer recorded for this attempt.
    #[getset(get = "pub")]
    observation: O,
}

impl<O> Node<O> {
    /// `true` for the root.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.primary.is_none()
    }
}

/// One completed discovery run: the root plus every attempt made from it, in
/// creation order.
///
/// Trees are append-only. Queries are linear scans; runs have hundreds of
/// nodes, not millions.
///
/// Every tree upholds these invariants, deserialized ones included: the root
/// is at position 0 and has no primary parent, every other node has one,
/// ids equal positions, and every edge points at an earlier node. So there
/// are no cycles and no dangling references.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawTree<O>")]
pub struct DiscoveryTree<O> {
    nodes: Vec<Node<O>>,
}

/// The wire form of a tree before validation.
#[derive(Deserialize)]
struct RawTree<O> {
    nodes: Vec<Node<O>>,
}

impl<O> TryFrom<RawTree<O>> for DiscoveryTree<O> {
    type Error = Error;

    fn try_from(raw: RawTree<O>) -> Result<Self, Error> {
        let nodes = raw.nodes;
        if nodes.is_empty() {
            return Err(Error::Empty);
        }
        for (position, node) in nodes.iter().enumerate() {
            if node.id.index() != position {
                return Err(Error::MisplacedId {
                    position,
                    id: node.id,
                });
            }
            let earlier = |target: NodeId| {
                if target < node.id {
                    Ok(())
                } else {
                    Err(Error::ForwardEdge {
                        node: node.id,
                        target,
                    })
                }
            };
            match (position, node.primary) {
                (0, None) => {}
                (0, Some(_)) => return Err(Error::RootWithParent),
                (_, None) => return Err(Error::Orphan(node.id)),
                (_, Some(primary)) => earlier(primary)?,
            }
            for &dep in &node.context {
                earlier(dep)?;
            }
        }
        Ok(Self { nodes })
    }
}

impl<O> DiscoveryTree<O> {
    /// A tree holding only the root, whose observation is the state before
    /// any attempt (a baseline evaluation, typically).
    #[must_use]
    pub fn new(root: O) -> Self {
        Self {
            nodes: vec![Node {
                id: NodeId::ROOT,
                primary: None,
                context: Vec::new(),
                observation: root,
            }],
        }
    }

    /// The root node.
    #[must_use]
    pub fn root(&self) -> &Node<O> {
        &self.nodes[0]
    }

    /// Number of nodes including the root.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// All nodes in creation order; the root comes first.
    #[must_use]
    pub fn nodes(&self) -> &[Node<O>] {
        &self.nodes
    }

    /// The node with `id`, if it exists.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Node<O>> {
        self.nodes.get(id.index())
    }

    /// `true` if `id` is part of the tree.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        id.index() < self.nodes.len()
    }

    /// Record an attempt that resumed from `primary` and was also shown
    /// `context`.
    ///
    /// # Errors
    /// [`Error::UnknownNode`] if `primary` or a context node does not exist;
    /// [`Error::Capacity`] if the id space is exhausted.
    pub fn push(
        &mut self,
        primary: NodeId,
        context: Vec<NodeId>,
        observation: O,
    ) -> Result<NodeId, Error> {
        self.check(primary)?;
        for &dep in &context {
            self.check(dep)?;
        }
        let id = NodeId(u32::try_from(self.nodes.len()).map_err(|_| Error::Capacity)?);
        self.nodes.push(Node {
            id,
            primary: Some(primary),
            context,
            observation,
        });
        Ok(id)
    }

    /// Nodes whose primary parent is `id`, in creation order.
    pub fn children(&self, id: NodeId) -> impl Iterator<Item = &Node<O>> {
        self.nodes.iter().filter(move |n| n.primary == Some(id))
    }

    /// Number of primary edges between `id` and the root; the root has depth 0.
    #[must_use]
    pub fn depth(&self, id: NodeId) -> Option<usize> {
        self.lineage(id).map(|path| path.len() - 1)
    }

    /// The primary chain from `id` up to and including the root, or `None`
    /// if `id` is unknown.
    #[must_use]
    pub fn lineage(&self, id: NodeId) -> Option<Vec<NodeId>> {
        let mut path = vec![id];
        let mut cur = self.get(id)?;
        while let Some(parent) = cur.primary {
            path.push(parent);
            cur = self.get(parent)?;
        }
        Some(path)
    }

    /// Project every observation through `f`, keeping the structure.
    ///
    /// Useful to strip a rich record down to what a policy may see, or to
    /// derive per-node statistics (deltas to the parent, ranks) once.
    #[must_use]
    pub fn map<P>(self, mut f: impl FnMut(&Node<O>) -> P) -> DiscoveryTree<P> {
        let nodes = self
            .nodes
            .iter()
            .map(|n| Node {
                id: n.id,
                primary: n.primary,
                context: n.context.clone(),
                observation: f(n),
            })
            .collect();
        DiscoveryTree { nodes }
    }

    fn check(&self, id: NodeId) -> Result<(), Error> {
        if self.contains(id) {
            Ok(())
        } else {
            Err(Error::UnknownNode(id))
        }
    }
}
