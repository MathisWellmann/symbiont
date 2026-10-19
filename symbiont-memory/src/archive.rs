// SPDX-License-Identifier: MPL-2.0
//! The episodic store: every evaluated attempt with its score and report.
//!
//! Entries are kept in commit order. The archive is small by design (a few
//! hundred entries) because every entry is a candidate for the prompt, so a
//! linear scan is cheaper than an index and keeps the serialized form plain:
//! a checkpoint written with a positional format such as bincode needs no map
//! with non-string keys.

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};
use typed_builder::TypedBuilder;

use crate::{
    BulletId,
    ConfigError,
};

/// A finite fitness value. Higher is better.
///
/// The memory ranks attempts by this one number. A host with several
/// objectives collapses them into a scalar for ranking and puts the full
/// picture into the [`Entry::report`], which the model reads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Score(f64);

/// The error of [`Score::new`]: a NaN or an infinity cannot be ranked.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("a score must be finite, got {0}")]
pub struct NonFiniteScore(f64);

impl Score {
    /// A score from a finite value.
    ///
    /// # Errors
    ///
    /// Returns [`NonFiniteScore`] for NaN and the infinities.
    pub fn new(value: f64) -> Result<Self, NonFiniteScore> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(NonFiniteScore(value))
        }
    }

    /// The value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Score {
    type Error = NonFiniteScore;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Score> for f64 {
    fn from(score: Score) -> Self {
        score.0
    }
}

// Equality and order agree with `total_cmp`, so `Score` can be `Ord`. The
// constructor rejects NaN, which leaves only the `-0.0 < 0.0` difference.
impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The commit number of an [`Entry`], unique within one [`Archive`] and never
/// reused. It is the short handle the prompts use (`#12`), because a host id
/// such as a 64 bit hash is noise to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntrySeq(u64);

impl EntrySeq {
    /// The handle with number `n`, for parsing a `#n` the model wrote.
    pub(crate) const fn from_raw(n: u64) -> Self {
        Self(n)
    }

    /// The number.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EntrySeq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// One evaluated attempt, as the host hands it to [`Archive::commit`].
///
/// `I` is the host's identity of a candidate. Use a content hash of the
/// source, not a [`symbiont::Revision`](https://docs.rs/symbiont): revision
/// ids are local to one process.
#[derive(Debug, Clone, TypedBuilder)]
pub struct Evaluated<I> {
    /// The host's id of the candidate.
    id: I,
    /// The candidates this one was derived from: none for a fresh seed, one
    /// for a mutation, two for a crossover.
    #[builder(default)]
    parents: Vec<I>,
    /// The source the agent wrote.
    #[builder(setter(into))]
    source: String,
    /// The scalar fitness the archive ranks by.
    score: Score,
    /// What the evaluation found, written for the model: metrics, failures,
    /// panics. It goes into later prompts as is.
    #[builder(default, setter(into))]
    report: String,
    /// The playbook bullets the lane that wrote this attempt was shown, from
    /// [`crate::Rendered::bullets`]. The reflection step tags exactly these.
    #[builder(default)]
    bullets_shown: Vec<BulletId>,
}

impl<I> Evaluated<I> {
    pub(crate) fn bullets_shown(&self) -> &[BulletId] {
        &self.bullets_shown
    }
}

/// An attempt in the [`Archive`].
#[derive(Debug, Clone, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Entry<I> {
    /// The commit number.
    #[getset(get_copy = "pub")]
    seq: EntrySeq,
    /// The host's id of the candidate.
    #[getset(get = "pub")]
    id: I,
    /// The candidates this one was derived from.
    #[getset(get = "pub")]
    parents: Vec<I>,
    /// The source the agent wrote.
    #[getset(get = "pub")]
    source: String,
    /// The latest score.
    #[getset(get_copy = "pub")]
    score: Score,
    /// The latest evaluation report.
    #[getset(get = "pub")]
    report: String,
}

/// The size limits of an [`Archive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CopyGetters)]
pub struct ArchiveConfig {
    /// The most entries the archive holds.
    #[getset(get_copy = "pub")]
    capacity: usize,
    /// The newest entries that eviction never removes, so the record of what
    /// was just tried survives even when it scored badly. A bad recent result
    /// is what keeps the next lane from trying the same thing again.
    #[getset(get_copy = "pub")]
    protect_recent: usize,
}

impl ArchiveConfig {
    /// A configuration with the given limits.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::ArchiveCapacity`] when `capacity` is zero or
    /// does not exceed `protect_recent`: eviction would have no entry to
    /// choose from.
    pub fn new(capacity: usize, protect_recent: usize) -> Result<Self, ConfigError> {
        if capacity == 0 || capacity <= protect_recent {
            return Err(ConfigError::ArchiveCapacity {
                capacity,
                protect_recent,
            });
        }
        Ok(Self {
            capacity,
            protect_recent,
        })
    }
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self::new(256, 16).expect("the default limits are valid")
    }
}

/// What [`Archive::commit`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Committed {
    /// The id was new. The entry got `seq`, and committing it pushed the
    /// archive over capacity by one, so the `evicted` entry was dropped.
    Inserted {
        /// The commit number of the new entry.
        seq: EntrySeq,
        /// The entry eviction removed, if the archive was full.
        evicted: Option<EntrySeq>,
    },
    /// The id was already in the archive, for example a re-evaluation after
    /// the data window moved. Its score and report were replaced; it keeps
    /// its commit number.
    Updated {
        /// The commit number of the existing entry.
        seq: EntrySeq,
    },
}

impl Committed {
    /// The commit number of the entry that holds the committed attempt.
    #[must_use]
    pub fn seq(self) -> EntrySeq {
        match self {
            Self::Inserted { seq, .. } | Self::Updated { seq } => seq,
        }
    }
}

/// Every evaluated attempt, up to [`ArchiveConfig::capacity`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Archive<I> {
    /// Ascending by `seq`.
    entries: Vec<Entry<I>>,
    next_seq: u64,
    config: ArchiveConfig,
}

impl<I: PartialEq + Clone> Archive<I> {
    /// An empty archive.
    #[must_use]
    pub fn new(config: ArchiveConfig) -> Self {
        assert!(config.capacity > config.protect_recent);
        Self {
            entries: Vec::with_capacity(config.capacity + 1),
            next_seq: 0,
            config,
        }
    }

    /// Record an evaluated attempt.
    ///
    /// A known id replaces the score and report of its entry (see
    /// [`Committed::Updated`]); the source and parents stay, because the same
    /// id is the same code.
    pub fn commit(&mut self, evaluated: Evaluated<I>) -> Committed {
        let Evaluated {
            id,
            parents,
            source,
            score,
            report,
            bullets_shown: _,
        } = evaluated;
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.score = score;
            entry.report = report;
            return Committed::Updated { seq: entry.seq };
        }

        let seq = EntrySeq(self.next_seq);
        self.next_seq += 1;
        self.entries.push(Entry {
            seq,
            id,
            parents,
            source,
            score,
            report,
        });
        let evicted = (self.entries.len() > self.config.capacity).then(|| self.evict());
        assert!(self.entries.len() <= self.config.capacity);
        if let Some(evicted) = evicted {
            assert_ne!(evicted, seq, "the new entry is protected");
        }
        Committed::Inserted { seq, evicted }
    }

    /// Remove the worst entry outside the protected recent tail. Ties go to
    /// the older entry.
    fn evict(&mut self) -> EntrySeq {
        let open = self.entries.len() - self.config.protect_recent;
        assert!(open > 0, "capacity exceeds protect_recent");
        let (idx, _) = self.entries[..open]
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| (e.score, e.seq))
            .expect("at least one unprotected entry");
        self.entries.remove(idx).seq
    }

    /// Replace the score and report of a known candidate. Returns `false` if
    /// the archive does not hold `id`.
    pub fn rescore(&mut self, id: &I, score: Score, report: impl Into<String>) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == *id) else {
            return false;
        };
        entry.score = score;
        entry.report = report.into();
        true
    }

    /// Remove a candidate, for example one the host found to be invalid.
    pub fn remove(&mut self, id: &I) -> Option<Entry<I>> {
        let idx = self.entries.iter().position(|e| e.id == *id)?;
        Some(self.entries.remove(idx))
    }

    /// Keep only the entries `keep` accepts.
    pub fn retain(&mut self, keep: impl FnMut(&Entry<I>) -> bool) {
        self.entries.retain(keep);
    }

    /// The entry of a candidate.
    #[must_use]
    pub fn get(&self, id: &I) -> Option<&Entry<I>> {
        self.entries.iter().find(|e| e.id == *id)
    }

    /// The entry with a commit number.
    #[must_use]
    pub fn by_seq(&self, seq: EntrySeq) -> Option<&Entry<I>> {
        self.entries
            .binary_search_by_key(&seq, |e| e.seq)
            .ok()
            .map(|idx| &self.entries[idx])
    }

    /// Every entry, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &Entry<I>> {
        self.entries.iter()
    }

    /// Every entry, best first. Ties go to the older entry, so the order is
    /// stable from one call to the next.
    #[must_use]
    pub fn ranked(&self) -> Vec<&Entry<I>> {
        let mut ranked = Vec::from_iter(self.entries.iter());
        ranked.sort_by(|a, b| b.score.cmp(&a.score).then(a.seq.cmp(&b.seq)));
        ranked
    }

    /// Every entry, newest first.
    pub fn recent(&self) -> impl Iterator<Item = &Entry<I>> {
        self.entries.iter().rev()
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(id: u64, score: f64) -> Evaluated<u64> {
        Evaluated::builder()
            .id(id)
            .source(format!("fn f() {{ {id} }}"))
            .score(Score::new(score).expect("finite"))
            .report(format!("score {score}"))
            .build()
    }

    #[test]
    fn score_rejects_non_finite() {
        assert!(Score::new(f64::NAN).is_err());
        assert!(Score::new(f64::INFINITY).is_err());
        assert!(Score::new(f64::NEG_INFINITY).is_err());
        assert_eq!(Score::new(1.5).expect("finite").get(), 1.5);
        // `try_from = "f64"` routes deserialization through this conversion.
        assert!(Score::try_from(f64::NAN).is_err());
    }

    #[test]
    fn config_rejects_unusable_limits() {
        assert!(ArchiveConfig::new(0, 0).is_err());
        assert!(ArchiveConfig::new(4, 4).is_err());
        assert!(ArchiveConfig::new(4, 5).is_err());
        assert!(ArchiveConfig::new(5, 4).is_ok());
    }

    #[test]
    fn commit_assigns_dense_seqs_and_updates_known_ids() {
        let mut archive = Archive::new(ArchiveConfig::default());
        assert_eq!(archive.commit(eval(10, 1.0)).seq(), EntrySeq(0));
        assert_eq!(archive.commit(eval(11, 2.0)).seq(), EntrySeq(1));
        let again = archive.commit(eval(10, 3.0));
        assert_eq!(again, Committed::Updated { seq: EntrySeq(0) });
        assert_eq!(archive.len(), 2);
        assert_eq!(archive.get(&10).expect("known").score().get(), 3.0);
        assert_eq!(archive.by_seq(EntrySeq(1)).expect("known").id(), &11);
        assert!(archive.by_seq(EntrySeq(2)).is_none());
    }

    #[test]
    fn eviction_drops_the_worst_unprotected_entry() {
        let mut archive = Archive::new(ArchiveConfig::new(3, 1).expect("valid"));
        let _ = archive.commit(eval(1, 5.0));
        let _ = archive.commit(eval(2, 1.0));
        let _ = archive.commit(eval(3, 3.0));
        // The newest entry is protected even though it is the worst.
        let committed = archive.commit(eval(4, 0.0));
        assert_eq!(
            committed,
            Committed::Inserted {
                seq: EntrySeq(3),
                evicted: Some(EntrySeq(1)),
            }
        );
        assert_eq!(archive.len(), 3);
        assert!(archive.get(&2).is_none());
        assert!(archive.get(&4).is_some());
    }

    #[test]
    fn eviction_ties_go_to_the_older_entry() {
        let mut archive = Archive::new(ArchiveConfig::new(2, 0).expect("valid"));
        let _ = archive.commit(eval(1, 1.0));
        let _ = archive.commit(eval(2, 1.0));
        let committed = archive.commit(eval(3, 2.0));
        assert_eq!(
            committed,
            Committed::Inserted {
                seq: EntrySeq(2),
                evicted: Some(EntrySeq(0)),
            }
        );
    }

    #[test]
    fn ranked_is_best_first_and_stable() {
        let mut archive = Archive::new(ArchiveConfig::default());
        for (id, score) in [(1, 1.0), (2, 3.0), (3, 3.0), (4, -1.0)] {
            let _ = archive.commit(eval(id, score));
        }
        let ids = Vec::from_iter(archive.ranked().into_iter().map(|e| *e.id()));
        assert_eq!(ids, vec![2, 3, 1, 4]);
        let recent = Vec::from_iter(archive.recent().map(|e| *e.id()));
        assert_eq!(recent, vec![4, 3, 2, 1]);
    }

    #[test]
    fn rescore_remove_and_retain() {
        let mut archive = Archive::new(ArchiveConfig::default());
        let _ = archive.commit(eval(1, 1.0));
        let _ = archive.commit(eval(2, 2.0));
        assert!(archive.rescore(&1, Score::new(9.0).expect("finite"), "new"));
        assert!(!archive.rescore(&7, Score::new(9.0).expect("finite"), "new"));
        assert_eq!(archive.get(&1).expect("known").report(), "new");
        assert_eq!(archive.remove(&2).expect("known").seq(), EntrySeq(1));
        assert!(archive.remove(&2).is_none());
        archive.retain(|e| e.score().get() < 5.0);
        assert!(archive.is_empty());
    }
}
