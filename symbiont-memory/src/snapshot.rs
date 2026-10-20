// SPDX-License-Identifier: MPL-2.0
//! A versioned, self-describing snapshot of a [`Memory`], for a host that
//! checkpoints its run and resumes it in a later process.
//!
//! The format belongs to this crate, not to the host: a host cannot know
//! when a new release of this crate changes the shape of a [`Memory`], so the
//! snapshot carries [`SNAPSHOT_VERSION`] and [`Memory::restore`] refuses any
//! other. The bytes are a JSON object `{"version": .., "memory": ..}`: JSON
//! names every field, so a reader never misreads a changed layout the way a
//! positional format can, and a host that stores the snapshot in a database
//! can still query the playbook out of it. The cost is size; snapshots
//! compress well.
//!
//! Restoring checks the snapshot before it is used: it must hold a valid
//! config equal to the one the host runs with (a host whose config reads
//! "the defaults" would otherwise resume under the defaults of an older
//! release), no reflection request may be outstanding (the host held the
//! request, so its attempts would never be reviewed), and every island,
//! archive and playbook limit must hold, so a later call does not panic on a
//! corrupt snapshot.

use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
};

use crate::{
    ConfigError,
    IslandId,
    Memory,
    MemoryConfig,
    Scope,
};

/// The snapshot format this build writes and reads. Bump it whenever the
/// serialized shape of a [`Memory`] changes in a way an older snapshot does
/// not decode into faithfully.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Why [`Memory::snapshot`] failed.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    /// A reflection request is still out: its attempts live in the request
    /// the host holds, so they would not survive the snapshot. Apply or
    /// abandon it first.
    #[error("{count} reflection request(s) still outstanding")]
    Outstanding {
        /// How many requests are out.
        count: usize,
    },
    /// The host's candidate id type does not serialize to JSON.
    #[error("the memory does not encode: {0}")]
    Encode(#[source] serde_json::Error),
}

/// Why [`Memory::restore`] refused a snapshot.
#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    /// The bytes are not a snapshot this build can read.
    #[error("the snapshot does not decode: {0}")]
    Decode(#[source] serde_json::Error),
    /// Written by a build with another [`SNAPSHOT_VERSION`].
    #[error("snapshot version {found} is not readable by this build (version {expected})")]
    Version {
        /// The snapshot's version.
        found: u32,
        /// [`SNAPSHOT_VERSION`].
        expected: u32,
    },
    /// The snapshot's config cannot work.
    #[error("the snapshot's config is invalid: {0}")]
    Config(#[from] ConfigError),
    /// The snapshot was taken under another config than the host runs with.
    #[error("the snapshot was taken with another memory config than the expected one")]
    ConfigMismatch,
    /// A reflection request was outstanding; [`Memory::snapshot`] never
    /// writes one.
    #[error("{count} reflection request(s) outstanding in the snapshot")]
    Outstanding {
        /// How many requests were out.
        count: usize,
    },
    /// A part refers to an island the config does not have.
    #[error("{island} out of range for {count} islands")]
    IslandOutOfRange {
        /// The offending island.
        island: IslandId,
        /// [`crate::IslandConfig::count`].
        count: u16,
    },
    /// An island holds more archive members than the archive capacity.
    #[error("{island} holds {members} archive members, over the capacity {capacity}")]
    ArchiveOverCapacity {
        /// The island.
        island: IslandId,
        /// Its member count.
        members: usize,
        /// [`crate::ArchiveConfig::capacity`].
        capacity: usize,
    },
    /// A playbook scope holds more bullets than the playbook allows.
    #[error("the {scope:?} playbook holds {bullets} bullets, over the limit {max}")]
    PlaybookOverCapacity {
        /// The scope.
        scope: Scope,
        /// Its bullet count.
        bullets: usize,
        /// [`crate::PlaybookConfig::max_bullets`].
        max: usize,
    },
}

#[derive(Serialize)]
struct EnvelopeRef<'a, I> {
    version: u32,
    memory: &'a Memory<I>,
}

#[derive(Deserialize)]
struct Header {
    version: u32,
}

#[derive(Deserialize)]
struct Envelope<I> {
    memory: Memory<I>,
}

impl<I: PartialEq + Clone + Serialize> Memory<I> {
    /// The whole memory as snapshot bytes: a JSON object
    /// `{"version": SNAPSHOT_VERSION, "memory": ..}`. JSON names every field,
    /// so a changed layout fails to decode instead of being misread.
    ///
    /// Take it between rounds, when no reflection request is outstanding.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::Outstanding`] if a reflection request is out,
    /// [`SnapshotError::Encode`] if `I` does not serialize to JSON.
    pub fn snapshot(&self) -> Result<Vec<u8>, SnapshotError> {
        if !self.outstanding.is_empty() {
            return Err(SnapshotError::Outstanding {
                count: self.outstanding.len(),
            });
        }
        let bytes = serde_json::to_vec(&EnvelopeRef {
            version: SNAPSHOT_VERSION,
            memory: self,
        })
        .map_err(SnapshotError::Encode)?;
        assert!(!bytes.is_empty());
        Ok(bytes)
    }
}

impl<I: PartialEq + Clone + DeserializeOwned> Memory<I> {
    /// The memory a [`Self::snapshot`] holds, if it was taken by this
    /// [`SNAPSHOT_VERSION`] under the config `expected`, and is consistent.
    ///
    /// # Errors
    ///
    /// See [`RestoreError`]; nothing is restored then.
    pub fn restore(bytes: &[u8], expected: &MemoryConfig) -> Result<Self, RestoreError> {
        let header: Header = serde_json::from_slice(bytes).map_err(RestoreError::Decode)?;
        if header.version != SNAPSHOT_VERSION {
            return Err(RestoreError::Version {
                found: header.version,
                expected: SNAPSHOT_VERSION,
            });
        }
        let envelope: Envelope<I> = serde_json::from_slice(bytes).map_err(RestoreError::Decode)?;
        let memory = envelope.memory;
        memory.check_restorable(expected)?;
        Ok(memory)
    }

    fn check_restorable(&self, expected: &MemoryConfig) -> Result<(), RestoreError> {
        self.config.validate()?;
        if self.config != *expected {
            return Err(RestoreError::ConfigMismatch);
        }
        if !self.outstanding.is_empty() {
            return Err(RestoreError::Outstanding {
                count: self.outstanding.len(),
            });
        }
        let islands = self.config.islands;
        let count = islands.count();
        let in_range = |island: IslandId| {
            if island.get() < count {
                Ok(())
            } else {
                Err(RestoreError::IslandOutOfRange { island, count })
            }
        };
        for entry in self.archive.iter() {
            entry.islands().iter().copied().try_for_each(in_range)?;
        }
        self.pending
            .iter()
            .map(|p| p.island())
            .try_for_each(in_range)?;
        for bullet in self.playbook.bullets() {
            if let Scope::Island(island) = bullet.scope() {
                in_range(island)?;
            }
        }
        let capacity = self.config.archive.capacity();
        let max = self.config.playbook.max_bullets();
        for scope in std::iter::once(Scope::Global).chain(islands.ids().map(Scope::Island)) {
            let bullets = self
                .playbook
                .bullets()
                .iter()
                .filter(|b| b.scope() == scope)
                .count();
            if bullets > max {
                return Err(RestoreError::PlaybookOverCapacity {
                    scope,
                    bullets,
                    max,
                });
            }
            let Scope::Island(island) = scope else {
                continue;
            };
            let members = self.archive.members(island).count();
            if members > capacity {
                return Err(RestoreError::ArchiveOverCapacity {
                    island,
                    members,
                    capacity,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Evaluated,
        IslandConfig,
        Score,
        Section,
    };

    fn config(islands: u16) -> MemoryConfig {
        MemoryConfig {
            islands: IslandConfig::new(islands, 1, 2, 95).expect("valid"),
            ..MemoryConfig::default()
        }
    }

    fn attempt(id: u64, island: u16, score: f64) -> Evaluated<u64> {
        Evaluated::builder()
            .id(id)
            .source(format!("fn f() -> u64 {{ {id} }}"))
            .score(Score::new(score).expect("finite"))
            .report(format!("report {id}"))
            .island(IslandId::new(island))
            .build()
    }

    /// A memory with something in every part: attempts on two islands, a
    /// pitfall, a lesson and attempts awaiting review.
    fn filled() -> Memory<u64> {
        let mut memory = Memory::new(config(2));
        for (id, island, score) in [(1, 0, 0.5), (2, 1, -0.25), (3, 0, 1.5)] {
            let _ = memory.commit(attempt(id, island, score));
        }
        memory.commit_panic("index out of bounds");
        memory.commit_panic("index out of bounds");
        let seq = memory.archive().get(&3).expect("committed").seq();
        let _ = memory
            .playbook_mut()
            .add(
                Scope::Island(IslandId::new(0)),
                Section::Strategies,
                "Clamp the index.",
                vec![seq],
            )
            .expect("added");
        memory
    }

    #[test]
    fn a_snapshot_restores_the_same_memory() {
        let memory = filled();
        let bytes = memory.snapshot().expect("nothing outstanding");
        let back = Memory::<u64>::restore(&bytes, &config(2)).expect("restorable");
        for island in [IslandId::new(0), IslandId::new(1)] {
            assert_eq!(
                back.render(island, &[]),
                memory.render(island, &[]),
                "rendering is a pure function of the state"
            );
        }
        assert_eq!(back.archive().len(), 3);
        assert_eq!(back.playbook().bullets(), memory.playbook().bullets());
        assert_eq!(back.failures(), memory.failures());
        assert_eq!(back.pending.len(), memory.pending.len());
        assert_eq!(back.next_request, memory.next_request);
        assert_eq!(
            back.snapshot().expect("restored"),
            bytes,
            "a restored memory snapshots to the same bytes"
        );
    }

    #[test]
    fn the_snapshot_names_its_version_and_parts() {
        let bytes = filled().snapshot().expect("nothing outstanding");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        assert_eq!(value["version"], SNAPSHOT_VERSION);
        assert!(value["memory"]["playbook"]["bullets"].is_array());
        assert!(value["memory"].get("archive").is_some());
    }

    #[test]
    fn an_outstanding_request_blocks_the_snapshot_until_it_is_resolved() {
        let mut memory = filled();
        let request = memory
            .reflection_request(IslandId::new(0), "task")
            .expect("island 0 has attempts awaiting review");
        assert!(matches!(
            memory.snapshot(),
            Err(SnapshotError::Outstanding { count: 1 })
        ));
        memory.abandon_reflection(request);
        assert!(memory.snapshot().is_ok());
    }

    #[test]
    fn another_version_is_refused() {
        let bytes = filled().snapshot().expect("nothing outstanding");
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        value["version"] = serde_json::json!(SNAPSHOT_VERSION + 1);
        let bytes = serde_json::to_vec(&value).expect("JSON");
        assert!(matches!(
            Memory::<u64>::restore(&bytes, &config(2)),
            Err(RestoreError::Version { found, expected: SNAPSHOT_VERSION }) if found == SNAPSHOT_VERSION + 1
        ));
    }

    #[test]
    fn garbage_and_a_bare_memory_are_refused() {
        assert!(matches!(
            Memory::<u64>::restore(b"", &config(2)),
            Err(RestoreError::Decode(_))
        ));
        assert!(matches!(
            Memory::<u64>::restore(&[0, 1, 2, 3], &config(2)),
            Err(RestoreError::Decode(_))
        ));
        let bare = serde_json::to_vec(&filled()).expect("JSON");
        assert!(
            matches!(
                Memory::<u64>::restore(&bare, &config(2)),
                Err(RestoreError::Decode(_))
            ),
            "a memory without the envelope has no version"
        );
    }

    #[test]
    fn another_config_is_refused() {
        let bytes = filled().snapshot().expect("nothing outstanding");
        assert!(matches!(
            Memory::<u64>::restore(&bytes, &config(3)),
            Err(RestoreError::ConfigMismatch)
        ));
        assert!(matches!(
            Memory::<u64>::restore(&bytes, &MemoryConfig::default()),
            Err(RestoreError::ConfigMismatch)
        ));
    }

    /// Edit one field of a valid snapshot's JSON and restore it.
    fn restore_edited(edit: impl FnOnce(&mut serde_json::Value)) -> Result<(), RestoreError> {
        let bytes = filled().snapshot().expect("nothing outstanding");
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        edit(&mut value["memory"]);
        let bytes = serde_json::to_vec(&value).expect("JSON");
        Memory::<u64>::restore(&bytes, &config(2)).map(|_| ())
    }

    #[test]
    fn an_inconsistent_snapshot_is_refused() {
        assert!(
            restore_edited(|_| ()).is_ok(),
            "the unedited snapshot restores"
        );
        assert!(matches!(
            restore_edited(|m| m["outstanding"] = serde_json::json!([0])),
            Err(RestoreError::Outstanding { count: 1 })
        ));
        assert!(matches!(
            restore_edited(|m| m["archive"]["entries"][0]["islands"] = serde_json::json!([2])),
            Err(RestoreError::IslandOutOfRange { count: 2, .. })
        ));
        assert!(matches!(
            restore_edited(|m| m["pending"][0]["island"] = serde_json::json!(5)),
            Err(RestoreError::IslandOutOfRange { count: 2, .. })
        ));
        assert!(matches!(
            restore_edited(|m| m["archive"]["entries"][1]["score"] = serde_json::Value::Null),
            Err(RestoreError::Decode(_))
        ));
    }

    /// Shrink a limit in a valid snapshot's config, edit its parts, and
    /// restore it with the shrunk config as the expected one.
    fn restore_shrunk(
        shrink: impl FnOnce(&mut serde_json::Value),
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> Result<(), RestoreError> {
        let bytes = filled().snapshot().expect("nothing outstanding");
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        shrink(&mut value["memory"]["config"]);
        edit(&mut value["memory"]);
        let expected: MemoryConfig =
            serde_json::from_value(value["memory"]["config"].clone()).expect("a config");
        let bytes = serde_json::to_vec(&value).expect("JSON");
        Memory::<u64>::restore(&bytes, &expected).map(|_| ())
    }

    #[test]
    fn limits_over_the_config_are_refused() {
        let one_bullet = |c: &mut serde_json::Value| c["playbook"]["max_bullets"] = 1.into();
        assert!(
            restore_shrunk(one_bullet, |_| ()).is_ok(),
            "one bullet fits"
        );
        let result = restore_shrunk(one_bullet, |m| {
            let bullets = m["playbook"]["bullets"].as_array_mut().expect("an array");
            let mut second = bullets[0].clone();
            second["id"] = 99.into();
            bullets.push(second);
        });
        assert!(
            matches!(
                result,
                Err(RestoreError::PlaybookOverCapacity {
                    bullets: 2,
                    max: 1,
                    ..
                })
            ),
            "{result:?}"
        );

        let result = restore_shrunk(
            |c| {
                c["archive"]["capacity"] = 1.into();
                c["archive"]["protect_recent"] = 0.into();
            },
            |_| (),
        );
        assert!(
            matches!(
                result,
                Err(RestoreError::ArchiveOverCapacity {
                    members: 2,
                    capacity: 1,
                    ..
                })
            ),
            "island 0 holds two attempts: {result:?}"
        );
    }
}
