// SPDX-License-Identifier: MPL-2.0
//! End-to-end behaviour of the replay world: recording, prefix-only views,
//! reveal rules, context gating, the objective, and policy selection.

use symbiont_dream::{
    Action,
    DiscoveryTree,
    Error,
    ExpansionRule,
    History,
    Live,
    MatchRule,
    NodeId,
    Objective,
    ParallelRefining,
    Policy,
    ReplayConfig,
    Termination,
    View,
    replay,
    select_best,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Obs {
    score: f64,
    calls: u32,
}

fn obs(score: f64) -> Obs {
    Obs { score, calls: 1 }
}

/// Deterministic stand-in agent: fresh branches score `5 - #branches`,
/// refinements add one.
fn agent(view: &View<'_, Obs>, action: &Action) -> Obs {
    if action.from.is_root() {
        obs(5.0 - view.children(NodeId::ROOT).count() as f64)
    } else {
        let parent = view.get(action.from).expect("action from the view");
        obs(parent.observation.score + 1.0)
    }
}

fn record<P: Policy<Obs>>(policy: &mut P, workers: usize) -> DiscoveryTree<Obs> {
    let mut live = Live::new(obs(0.0), workers);
    policy.reset();
    loop {
        let batch = policy.select_batch(&live.view());
        if batch.is_empty() {
            break;
        }
        let outcomes: Vec<(Action, Obs)> = batch
            .into_iter()
            .map(|a| {
                let o = agent(&live.view(), &a);
                (a, o)
            })
            .collect();
        live.commit_round(outcomes).expect("live actions are valid");
    }
    live.finish(Termination::PolicyStopped).0
}

fn quality() -> Objective<'static, Obs> {
    Objective::new(|o: &Obs| o.score).cost(|o: &Obs| f64::from(o.calls))
}

#[test]
fn live_run_records_parallel_refining_grid() {
    let tree = record(
        &mut ParallelRefining {
            branches: 3,
            refinements: 2,
        },
        3,
    );
    assert_eq!(tree.node_count(), 1 + 3 * 3);
    assert_eq!(tree.children(NodeId::ROOT).count(), 3);
    let deepest = tree
        .nodes()
        .iter()
        .filter_map(|n| tree.depth(n.id))
        .max()
        .expect("nodes exist");
    assert_eq!(deepest, 3);
}

#[test]
fn replaying_the_recording_policy_reveals_everything() {
    let mut policy = ParallelRefining {
        branches: 3,
        refinements: 2,
    };
    let tree = record(&mut policy, 3);
    let trajectory = replay(&tree, &mut policy, &ReplayConfig::with_workers(3));
    assert_eq!(trajectory.termination, Termination::Exhausted);
    assert_eq!(trajectory.revealed_count(), 9);
    assert_eq!(trajectory.round_count(), 3);
    assert_eq!(trajectory.rejected_count(), 0);
}

#[test]
fn view_is_prefix_only() {
    let tree = record(
        &mut ParallelRefining {
            branches: 2,
            refinements: 1,
        },
        2,
    );
    let mut sim = symbiont_dream::Replay::new(&tree, ReplayConfig::with_workers(2));
    let view = sim.view();
    assert_eq!(view.observed().count(), 1);
    assert_eq!(view.legal_actions(), vec![NodeId::ROOT]);
    assert!(view.get(NodeId::ROOT).is_some());
    let first_child = tree
        .children(NodeId::ROOT)
        .next()
        .expect("root has children")
        .id;
    assert!(view.get(first_child).is_none());
    assert!(!view.is_legal(first_child));
    assert_eq!(view.depth(first_child), None);

    // Two root actions open two branches in one round.
    sim.step(vec![Action::expand(NodeId::ROOT); 2]);
    let view = sim.view();
    assert_eq!(view.revealed_count(), 2);
    assert_eq!(view.frontier().len(), 2);
    assert!(view.get(first_child).is_some());
    assert_eq!(view.depth(first_child), Some(1));
}

#[test]
fn leaves_only_rejects_interior_nodes_and_duplicates() {
    let tree = record(
        &mut ParallelRefining {
            branches: 1,
            refinements: 2,
        },
        1,
    );
    let mut sim = symbiont_dream::Replay::new(&tree, ReplayConfig::with_workers(2));
    sim.step(vec![Action::expand(NodeId::ROOT)]);
    let leaf = sim.view().frontier()[0];
    sim.step(vec![Action::expand(leaf)]);
    // `leaf` now has a revealed child: illegal under LeavesOnly. The
    // duplicate of its child is dropped too.
    let child = sim.view().frontier()[0];
    let term = sim.step(vec![
        Action::expand(leaf),
        Action::expand(child),
        Action::expand(child),
    ]);
    assert_eq!(term, Some(Termination::Exhausted));
    let trajectory = sim.finish();
    let last = trajectory.rounds.last().expect("three rounds");
    assert_eq!(last.batch.len(), 1);
    assert_eq!(last.rejected.len(), 2);
}

#[test]
fn strict_mode_terminates_on_illegal_batch() {
    let tree = record(
        &mut ParallelRefining {
            branches: 1,
            refinements: 1,
        },
        1,
    );
    let config = ReplayConfig::with_workers(1).strict(true);
    let mut sim = symbiont_dream::Replay::new(&tree, config);
    let unrevealed = NodeId::ROOT;
    let child = tree.children(unrevealed).next().expect("child").id;
    assert_eq!(
        sim.step(vec![Action::expand(child)]),
        Some(Termination::IllegalBatch)
    );
}

#[test]
fn any_revealed_allows_fan_out_from_one_parent() {
    // Record two attempts from the same parent in one round.
    let mut live = Live::new(obs(0.0), 2).expansion(ExpansionRule::AnyRevealed);
    let [a] = live
        .commit_round([(Action::expand(NodeId::ROOT), obs(1.0))])
        .expect("root exists")[..]
    else {
        panic!("one node")
    };
    live.commit_round([(Action::expand(a), obs(2.0)), (Action::expand(a), obs(3.0))])
        .expect("a exists");
    let (tree, _) = live.finish(Termination::PolicyStopped);

    // Under LeavesOnly the second child of `a` is unreachable: `a` stops
    // being legal once one child is revealed.
    let mut everything = |view: &View<'_, Obs>| -> Vec<Action> {
        view.legal_actions()
            .into_iter()
            .map(Action::expand)
            .collect()
    };
    let leaves = replay(&tree, &mut everything, &ReplayConfig::with_workers(4));
    assert_eq!(leaves.revealed_count(), 2);
    assert_eq!(leaves.termination, Termination::Stalled);

    let config = ReplayConfig::with_workers(4).expansion(ExpansionRule::AnyRevealed);
    let any = replay(&tree, &mut everything, &config);
    assert_eq!(any.revealed_count(), 3);
    assert_eq!(any.termination, Termination::Exhausted);
}

#[test]
fn context_gates_reveal_until_dependencies_are_revealed() {
    // root -> a, root -> b, a -> c (also shown b).
    let mut tree = DiscoveryTree::new(obs(0.0));
    let a = tree.push(NodeId::ROOT, vec![], obs(1.0)).expect("root");
    let b = tree.push(NodeId::ROOT, vec![], obs(2.0)).expect("root");
    let c = tree.push(a, vec![b], obs(9.0)).expect("a and b");

    let mut sim = symbiont_dream::Replay::new(&tree, ReplayConfig::with_workers(2));
    sim.step(vec![Action::expand(NodeId::ROOT)]);
    assert!(sim.view().is_revealed(a));
    // `b` is not revealed, so `c` is out of support.
    sim.step(vec![Action::expand(a)]);
    assert!(!sim.view().is_revealed(c));
    // Reveal `b`, then `c` becomes reachable from `a`.
    sim.step(vec![Action::expand(NodeId::ROOT)]);
    assert!(sim.view().is_revealed(b));
    let term = sim.step(vec![Action::expand(a)]);
    assert!(sim.view().is_revealed(c));
    assert_eq!(term, Some(Termination::Exhausted));
}

#[test]
fn exact_context_matching_requires_the_same_set() {
    let mut tree = DiscoveryTree::new(obs(0.0));
    let a = tree.push(NodeId::ROOT, vec![], obs(1.0)).expect("root");
    let b = tree.push(NodeId::ROOT, vec![], obs(2.0)).expect("root");
    tree.push(a, vec![b], obs(9.0)).expect("a and b");

    let config = ReplayConfig::with_workers(2).matching(MatchRule::ExactContext);
    let mut sim = symbiont_dream::Replay::new(&tree, config);
    sim.step(vec![Action::expand(NodeId::ROOT); 2]);
    // Plain refinement of `a` does not match the recorded crossover.
    sim.step(vec![Action::expand(a)]);
    assert_eq!(sim.view().revealed_count(), 2);
    // The crossover action does.
    let term = sim.step(vec![Action::with_context(a, [b])]);
    assert_eq!(term, Some(Termination::Exhausted));
}

#[test]
fn objective_combines_quality_cost_and_parallelism() {
    let mut policy = ParallelRefining {
        branches: 2,
        refinements: 1,
    };
    let tree = record(&mut policy, 2);
    let trajectory = replay(&tree, &mut policy, &ReplayConfig::with_workers(2));
    let objective = quality().cost_weight(0.5).parallelism_weight(1.0);
    let score = objective.score(&tree, &trajectory).expect("same tree");
    // Branch scores 5 and 4, refined to 6 and 5. Four nodes over two rounds.
    assert_eq!(score.best_quality, 6.0);
    assert_eq!(score.total_cost, 4.0);
    assert_eq!(score.revealed, 4);
    assert_eq!(score.rounds, 2);
    assert_eq!(score.parallelism, 2.0);
    assert_eq!(score.value, 6.0 - 0.5 * 4.0 + 2.0);

    // The root alone scores the baseline.
    let empty = replay(
        &tree,
        &mut |_: &View<'_, Obs>| Vec::<Action>::new(),
        &ReplayConfig::default(),
    );
    assert_eq!(empty.termination, Termination::PolicyStopped);
    let score = objective.score(&tree, &empty).expect("same tree");
    assert_eq!(score.best_quality, 0.0);
    assert_eq!(score.value, 0.0);
}

#[test]
fn objective_rejects_foreign_trajectory() {
    let big = record(
        &mut ParallelRefining {
            branches: 2,
            refinements: 2,
        },
        2,
    );
    let small = DiscoveryTree::new(obs(0.0));
    let mut policy = ParallelRefining {
        branches: 2,
        refinements: 2,
    };
    let trajectory = replay(&big, &mut policy, &ReplayConfig::with_workers(2));
    assert!(matches!(
        quality().score(&small, &trajectory),
        Err(Error::UnknownNode(_))
    ));
}

#[test]
fn selection_never_regresses_from_the_incumbent() {
    let mut incumbent = ParallelRefining {
        branches: 3,
        refinements: 2,
    };
    let mut history = History::new();
    history.push(record(&mut incumbent, 3));
    history.push(record(&mut incumbent, 3));
    assert_eq!(history.len(), 2);

    let objective = quality().cost_weight(0.5);
    let config = ReplayConfig::with_workers(3);

    // Candidate: deepen only the best branch — same best score, fewer calls.
    let mut greedy = |view: &View<'_, Obs>| -> Vec<Action> {
        if view.round() == 0 {
            return vec![Action::expand(NodeId::ROOT)];
        }
        let q = |id| view.get(id).map_or(f64::MIN, |n| n.observation.score);
        view.frontier()
            .into_iter()
            .max_by(|&a, &b| q(a).total_cmp(&q(b)))
            .map(|id| vec![Action::expand(id)])
            .unwrap_or_default()
    };
    // Candidate: stop immediately — worst.
    let mut idle = |_: &View<'_, Obs>| Vec::<Action>::new();

    let evals = vec![
        history
            .evaluate(&mut incumbent, &objective, &config)
            .expect("valid"),
        history
            .evaluate(&mut greedy, &objective, &config)
            .expect("valid"),
        history
            .evaluate(&mut idle, &objective, &config)
            .expect("valid"),
    ];
    assert_eq!(evals[0].worlds.len(), 2);
    assert_eq!(evals[0].worlds[0].score.best_quality, 7.0);
    assert_eq!(evals[1].worlds[0].score.best_quality, 7.0);
    assert_eq!(evals[1].worlds[0].score.revealed, 3);
    assert!(evals[1].mean_value > evals[0].mean_value);
    assert_eq!(select_best(&evals), Some(1));

    // Identical candidates tie towards the incumbent.
    let mut twin = incumbent;
    let evals = vec![
        history
            .evaluate(&mut incumbent, &objective, &config)
            .expect("valid"),
        history
            .evaluate(&mut twin, &objective, &config)
            .expect("valid"),
    ];
    assert_eq!(select_best(&evals), Some(0));
    assert_eq!(select_best(&[]), None);
}

#[test]
fn tree_and_trajectory_round_trip_through_serde() {
    let mut policy = ParallelRefining {
        branches: 2,
        refinements: 1,
    };
    let tree = record(&mut policy, 2);
    let trajectory = replay(&tree, &mut policy, &ReplayConfig::with_workers(2));

    let json = serde_json::to_string(&tree).expect("serialize tree");
    let back: DiscoveryTree<Obs> = serde_json::from_str(&json).expect("deserialize tree");
    assert_eq!(back, tree);

    let json = serde_json::to_string(&trajectory).expect("serialize trajectory");
    let back: symbiont_dream::Trajectory =
        serde_json::from_str(&json).expect("deserialize trajectory");
    assert_eq!(back, trajectory);
}

#[test]
fn map_projects_observations_and_keeps_structure() {
    let tree = record(
        &mut ParallelRefining {
            branches: 2,
            refinements: 1,
        },
        2,
    );
    let scores = tree.clone().map(|n| n.observation.score);
    assert_eq!(scores.node_count(), tree.node_count());
    for (a, b) in scores.nodes().iter().zip(tree.nodes()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.primary, b.primary);
        assert_eq!(a.observation, b.observation.score);
    }
}
