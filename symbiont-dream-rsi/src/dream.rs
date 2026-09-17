// SPDX-License-Identifier: MPL-2.0
//! Dreaming: evaluate candidate policies over the recorded history and pick
//! the one to deploy next.

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    objective::{
        Objective,
        ReplayScore,
    },
    policy::Policy,
    replay::{
        ReplayConfig,
        Trajectory,
        replay,
    },
    tree::{
        DiscoveryTree,
        Error,
    },
};

/// One policy replayed over one world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Getters)]
pub struct WorldEvaluation {
    /// What the policy did.
    #[getset(get = "pub")]
    trajectory: Trajectory,

    /// What it was worth.
    #[getset(get = "pub")]
    score: ReplayScore,
}

/// One policy replayed over every world of the history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Evaluation {
    /// Per-world results, in history order.
    #[getset(get = "pub")]
    worlds: Vec<WorldEvaluation>,

    /// Mean of [`ReplayScore::value`] over the worlds; `NaN` for an empty
    /// history.
    #[getset(get_copy = "pub")]
    mean_value: f64,
}

/// Replay `policy` over every world in `history` and average the objective.
///
/// # Errors
/// Propagates [`Objective::score`] errors.
pub fn evaluate<O, P>(
    history: &[DiscoveryTree<O>],
    policy: &mut P,
    objective: &Objective<'_, O>,
    config: &ReplayConfig,
) -> Result<Evaluation, Error>
where
    P: Policy<O> + ?Sized,
{
    let mut worlds = Vec::with_capacity(history.len());
    for world in history {
        let trajectory = replay(world, policy, config);
        let score = objective.score(world, &trajectory)?;
        worlds.push(WorldEvaluation { trajectory, score });
    }
    let mean_value = if worlds.is_empty() {
        f64::NAN
    } else {
        worlds.iter().map(|w| w.score.value()).sum::<f64>() / worlds.len() as f64
    };
    Ok(Evaluation { worlds, mean_value })
}

/// Index of the evaluation with the highest mean value.
///
/// Ties go to the earliest index and `NaN` never wins, so placing the
/// incumbent policy at index 0 guarantees the selection is never worse than
/// it on the fixed history.
#[must_use]
pub fn select_best(evaluations: &[Evaluation]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, e) in evaluations.iter().enumerate() {
        if e.mean_value.is_nan() {
            continue;
        }
        match best {
            Some((_, v)) if v >= e.mean_value => {}
            _ => best = Some((i, e.mean_value)),
        }
    }
    best.map(|(i, _)| i)
}

/// The growing pool of replay worlds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct History<O> {
    worlds: Vec<DiscoveryTree<O>>,
}

impl<O> Default for History<O> {
    fn default() -> Self {
        Self { worlds: Vec::new() }
    }
}

impl<O> History<O> {
    /// An empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a completed run.
    pub fn push(&mut self, world: DiscoveryTree<O>) {
        self.worlds.push(world);
    }

    /// The recorded runs, oldest first.
    #[must_use]
    pub fn worlds(&self) -> &[DiscoveryTree<O>] {
        &self.worlds
    }

    /// Number of recorded runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// `true` if nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// [`evaluate`] over this history.
    ///
    /// # Errors
    /// Propagates [`Objective::score`] errors.
    pub fn evaluate<P>(
        &self,
        policy: &mut P,
        objective: &Objective<'_, O>,
        config: &ReplayConfig,
    ) -> Result<Evaluation, Error>
    where
        P: Policy<O> + ?Sized,
    {
        evaluate(&self.worlds, policy, objective, config)
    }
}
