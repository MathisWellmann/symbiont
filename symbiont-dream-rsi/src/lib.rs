// SPDX-License-Identifier: MPL-2.0
//! Discovery history as a replay simulator — the core mechanism of
//! *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*
//! (Zheng et al., [arXiv:2609.14858](https://arxiv.org/abs/2609.14858)),
//! with nothing domain-specific in it.
//!
//! # The idea
//!
//! A discovery loop (propose → evaluate → refine, many times) is driven by an
//! *exploration policy*: which candidate to continue from, how many attempts
//! to run in parallel, when to open a fresh branch, when to stop. Evaluating a
//! policy online is expensive because it means running the whole loop again.
//!
//! But a completed run already is a tree of attempts with their measured
//! outcomes. A different policy can walk that recorded tree: it selects nodes
//! to continue from, the tree reveals the continuations it actually recorded,
//! and no agent or evaluator runs. One paid-for run supports thousands of
//! free off-policy evaluations ("dreaming"). The best-scoring policy is then
//! deployed for the next live run, which grows the pool of replay worlds.
//!
//! # What this crate is agnostic about
//!
//! * **The observation.** Every node carries an `O` the consumer chose — a
//!   return, a drawdown, a compile failure, a token count. The crate never
//!   reads it; [`Objective`] takes closures that do.
//! * **What parents mean.** A node has one *primary* edge (the state the
//!   attempt resumed from) and any number of *context* edges (what else it
//!   was shown). The crate uses them for one purpose only: in replay a node
//!   becomes revealable once its primary and all context nodes are revealed,
//!   so a policy is never credited with an outcome that depended on
//!   information it had not yet paid for.
//!
//! # The pieces
//!
//! | | |
//! |---|---|
//! | [`DiscoveryTree`], [`Node`], [`NodeId`] | the recorded run |
//! | [`Policy`], [`View`], [`Action`] | the decision interface, shared by live run and replay; a `View` only exposes revealed nodes |
//! | [`Live`] | records a live run through that interface, producing a tree and a [`Trajectory`] |
//! | [`Replay`], [`replay()`] | walks a recorded tree with a policy, revealing recorded continuations |
//! | [`Objective`], [`ReplayScore`] | eq. 1 of the paper: best quality − β₁·cost + β₂·parallelism |
//! | [`History`], [`evaluate`], [`select_best`] | replay candidates over every world, average, pick the incumbent-or-better |
//! | [`ParallelRefining`] | the paper's initial policy, as a reference implementation |
//!
//! # Example
//!
//! ```
//! use symbiont_dream_rsi::{
//!     Action, History, Live, NodeId, Objective, ParallelRefining, Policy, ReplayConfig,
//!     Termination, View, evaluate, replay, select_best,
//! };
//!
//! // The consumer's observation: here just a score and a cost.
//! #[derive(Clone)]
//! struct Obs { score: f64, agent_calls: u32 }
//!
//! // A stand-in for "run the agent and evaluate": refining adds 1, fresh
//! // branches start at 5 and get weaker.
//! fn agent(view: &View<'_, Obs>, a: &Action) -> Obs {
//!     let parent = view.get(a.from).expect("actions come from the view");
//!     let score = if a.from.is_root() {
//!         5.0 - view.children(NodeId::ROOT).count() as f64
//!     } else {
//!         parent.observation().score + 1.0
//!     };
//!     Obs { score, agent_calls: 1 }
//! }
//!
//! // 1. Live run under the incumbent policy, recorded as a tree.
//! let workers = 3;
//! let mut incumbent = ParallelRefining { branches: 3, refinements: 2 };
//! let mut live = Live::new(Obs { score: 0.0, agent_calls: 0 }, workers);
//! loop {
//!     let batch = incumbent.select_batch(&live.view());
//!     if batch.is_empty() {
//!         break;
//!     }
//!     let outcomes: Vec<(Action, Obs)> =
//!         batch.into_iter().map(|a| { let o = agent(&live.view(), &a); (a, o) }).collect();
//!     live.commit_round(outcomes).expect("live actions reference live nodes");
//! }
//! let (tree, _) = live.finish(Termination::PolicyStopped);
//! assert_eq!(tree.node_count(), 1 + 3 * 3);
//!
//! let mut history = History::new();
//! history.push(tree);
//!
//! // 2. Dream: replay candidate policies over the history, no agent involved.
//! let objective = Objective::new(|o: &Obs| o.score)
//!     .cost(|o: &Obs| f64::from(o.agent_calls))
//!     .cost_weight(0.5)
//!     .parallelism_weight(0.1);
//! let config = ReplayConfig::with_workers(workers);
//!
//! // A cheaper candidate: only ever deepen the single best branch.
//! let mut greedy = |view: &View<'_, Obs>| -> Vec<Action> {
//!     if view.round() == 0 {
//!         return vec![Action::expand(NodeId::ROOT)];
//!     }
//!     view.frontier()
//!         .into_iter()
//!         .max_by(|&a, &b| {
//!             let q = |id: NodeId| view.get(id).map_or(f64::MIN, |n| n.observation().score);
//!             q(a).total_cmp(&q(b))
//!         })
//!         .map(|id| vec![Action::expand(id)])
//!         .unwrap_or_default()
//! };
//!
//! let evals = vec![
//!     history.evaluate(&mut incumbent, &objective, &config).expect("valid history"),
//!     history.evaluate(&mut greedy, &objective, &config).expect("valid history"),
//! ];
//! // Same best score (7.0) for 3 calls instead of 9: the greedy candidate wins.
//! assert_eq!(select_best(&evals), Some(1));
//! assert_eq!(evals[1].worlds[0].score.best_quality, 7.0);
//! assert_eq!(evals[1].worlds[0].score.revealed, 3);
//!
//! // 3. Deploy the winner for the next live run; its tree joins the history.
//! ```
//!
//! # Caveats the paper shares
//!
//! Replay covers only realised branches: a policy earns nothing for a choice
//! whose outcome was never recorded, and the agent's stochasticity is
//! ignored. The pool is empty before the first live run, so the first round
//! always runs the hand-written policy. The mechanism pays off over repeated
//! rounds on the same or related tasks.

// The lib-test target links every dev-dependency; only the integration tests use this one.
#[cfg(test)]
use serde_json as _;

pub mod dream;
pub mod live;
pub mod objective;
pub mod policies;
pub mod policy;
pub mod replay;
pub mod tree;

pub use dream::{
    Evaluation,
    History,
    WorldEvaluation,
    evaluate,
    select_best,
};
pub use live::Live;
pub use objective::{
    Objective,
    ReplayScore,
};
pub use policies::ParallelRefining;
pub use policy::{
    Action,
    ExpansionRule,
    Policy,
    View,
};
pub use replay::{
    MatchRule,
    Replay,
    ReplayConfig,
    RoundRecord,
    Termination,
    Trajectory,
    replay,
};
pub use tree::{
    DiscoveryTree,
    Error,
    Node,
    NodeId,
};
