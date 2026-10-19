// SPDX-License-Identifier: MPL-2.0
//! Islands: separate memories for sub-populations, with migration between
//! them, and a code novelty check.
//!
//! A memory that every lane shares pulls the population together: the lanes
//! read the same best attempt and the same lessons and converge on the same
//! program (symbiont's `REVISION_DEDUP_HITS` counts the collapse). Islands are
//! the classic counter from FunSearch and OpenEvolve. Each island has its own
//! archive members and its own lessons; every so often [`crate::Memory::migrate`]
//! copies the best attempts of each island to the next one in a ring and
//! promotes island lessons that proved themselves to the global part of the
//! playbook. A stagnant island can be wiped with
//! [`crate::Memory::reset_island`] and re-seeded by the next migration.
//!
//! The novelty check follows ShinkaEvolve's rejection sampling: a candidate
//! that is nearly a copy of an archived attempt is not worth an expensive
//! evaluation. ShinkaEvolve compares embeddings; this crate compares token
//! trigrams, which needs no model and catches the common case of a lane that
//! returns its parent with cosmetic edits.

use std::collections::BTreeSet;

use getset::CopyGetters;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    BulletId,
    ConfigError,
    EntrySeq,
};

/// An island. Islands are numbered `0..count`.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct IslandId(u16);

impl IslandId {
    /// The island with number `n`.
    #[must_use]
    pub const fn new(n: u16) -> Self {
        Self(n)
    }

    /// The number.
    #[must_use]
    pub fn get(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for IslandId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "island {}", self.0)
    }
}

/// Where a playbook bullet applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// Every island sees it.
    Global,
    /// Only this island sees it.
    Island(IslandId),
}

impl Scope {
    /// Whether a lane on `island` sees a bullet of this scope.
    #[must_use]
    pub fn visible_on(self, island: IslandId) -> bool {
        match self {
            Self::Global => true,
            Self::Island(own) => own == island,
        }
    }
}

/// How many islands there are and how they exchange knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CopyGetters)]
// Missing fields take their defaults, so a config file names only what it
// changes. Deserializing skips the constructors: check with
// `MemoryConfig::validate`.
#[serde(default)]
#[getset(get_copy = "pub")]
pub struct IslandConfig {
    /// The number of islands. With one island, every lesson is global and
    /// migration does nothing.
    count: u16,
    /// How many of its best attempts an island sends to the next island in
    /// the ring on [`crate::Memory::migrate`].
    migrants: usize,
    /// The [`crate::Bullet::net`] (`helpful - harmful`) at which an island
    /// lesson is promoted to the global playbook on migration.
    promote_net: u32,
    /// A candidate whose token trigrams overlap an archived attempt by at
    /// least this much (Jaccard, in percent) is not novel, see
    /// [`crate::Memory::novelty`].
    novelty_percent: u8,
}

impl IslandConfig {
    /// A configuration with the given parameters.
    ///
    /// # Errors
    ///
    /// [`ConfigError::NoIslands`] for `count == 0`.
    pub fn new(
        count: u16,
        migrants: usize,
        promote_net: u32,
        novelty_percent: u8,
    ) -> Result<Self, ConfigError> {
        if count == 0 {
            return Err(ConfigError::NoIslands);
        }
        Ok(Self {
            count,
            migrants,
            promote_net: promote_net.max(1),
            novelty_percent: novelty_percent.min(100),
        })
    }

    /// Every island, in order.
    pub fn ids(self) -> impl Iterator<Item = IslandId> {
        (0..self.count).map(IslandId)
    }

    /// The island that `island` sends its migrants to.
    #[must_use]
    pub fn next(self, island: IslandId) -> IslandId {
        assert!(island.0 < self.count, "{island} out of range");
        IslandId((island.0 + 1) % self.count)
    }
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self::new(1, 2, 2, 95).expect("one island is valid")
    }
}

/// What [`crate::Memory::migrate`] moved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[must_use]
pub struct Migration {
    /// `(entry, destination)` for every attempt that joined another island.
    pub migrated: Vec<(EntrySeq, IslandId)>,
    /// Island lessons promoted to the global playbook.
    pub promoted: Vec<BulletId>,
}

/// The archived attempt most similar to a candidate, see
/// [`crate::Memory::novelty`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nearest {
    /// The archived attempt.
    pub seq: EntrySeq,
    /// The trigram overlap with the candidate (Jaccard, in percent).
    pub percent: u8,
    /// Whether the overlap is below [`IslandConfig::novelty_percent`].
    pub novel: bool,
}

/// The token trigrams of a source: identifiers, numbers and single
/// punctuation characters, with whitespace and `//` comments ignored, so
/// reformatting and commenting do not count as change.
pub(crate) fn shingles(source: &str) -> BTreeSet<[&str; 3]> {
    let mut tokens = Vec::new();
    for line in source.lines() {
        let code = line.split_once("//").map_or(line, |(code, _)| code);
        let mut start = None;
        for (idx, c) in code.char_indices() {
            let word = c.is_alphanumeric() || c == '_';
            match (word, start) {
                (true, None) => start = Some(idx),
                (true, Some(_)) => {}
                (false, Some(s)) => {
                    tokens.push(&code[s..idx]);
                    start = None;
                }
                (false, None) => {}
            }
            if !word && !c.is_whitespace() {
                tokens.push(&code[idx..idx + c.len_utf8()]);
            }
        }
        if let Some(s) = start {
            tokens.push(&code[s..]);
        }
    }
    BTreeSet::from_iter(tokens.windows(3).map(|w| [w[0], w[1], w[2]]))
}

/// The Jaccard overlap of two trigram sets, in percent. Two empty sets are
/// identical.
pub(crate) fn similarity(a: &BTreeSet<[&str; 3]>, b: &BTreeSet<[&str; 3]>) -> u8 {
    let union = a.union(b).count();
    if union == 0 {
        return 100;
    }
    let shared = a.intersection(b).count();
    u8::try_from(shared * 100 / union).expect("a ratio of at most one, in percent")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validates_and_rings() {
        assert_eq!(IslandConfig::new(0, 1, 1, 90), Err(ConfigError::NoIslands));
        let config = IslandConfig::new(3, 1, 0, 120).expect("valid");
        assert_eq!(config.promote_net(), 1);
        assert_eq!(config.novelty_percent(), 100);
        assert_eq!(config.next(IslandId::new(0)), IslandId::new(1));
        assert_eq!(config.next(IslandId::new(2)), IslandId::new(0));
        assert_eq!(config.ids().count(), 3);
    }

    #[test]
    #[should_panic(expected = "island 3 out of range")]
    fn next_rejects_unknown_islands() {
        let _ = IslandConfig::new(3, 1, 1, 90)
            .expect("valid")
            .next(IslandId::new(3));
    }

    #[test]
    fn scope_visibility() {
        let a = IslandId::new(0);
        let b = IslandId::new(1);
        assert!(Scope::Global.visible_on(a));
        assert!(Scope::Island(a).visible_on(a));
        assert!(!Scope::Island(a).visible_on(b));
    }

    #[test]
    fn shingles_ignore_layout_and_comments() {
        let a = shingles("fn f(x: u32) -> u32 { x + 1 }");
        let b = shingles("fn f(x: u32) -> u32 {\n    // add one\n    x + 1\n}\n");
        assert_eq!(a, b);
        assert_eq!(similarity(&a, &b), 100);
        let c = shingles("fn g(v: &[f64]) -> f64 { v.iter().sum() }");
        assert!(similarity(&a, &c) < 20);
        assert_eq!(similarity(&shingles(""), &shingles("")), 100);
        assert_eq!(similarity(&shingles("a b c"), &shingles("")), 0);
    }

    #[test]
    fn shingles_handle_unicode() {
        let s = shingles("let größe = \"é\";");
        assert!(s.contains(&["let", "größe", "="]));
    }
}
