// SPDX-License-Identifier: MPL-2.0
//! Shared API for the struct support example.

#![allow(
    unused_crate_dependencies,
    reason = "Dependencies are used by this package's binary target."
)]

use std::collections::HashMap;

/// A 2D game state, with just the x and y coordinates.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    /// The x coordinate. Range of 0..100
    pub x: usize,
    /// The y coordinate. Range of 0..250
    pub y: usize,
    /// Extra metadata demonstrating that nested dependency/std types are
    /// compiled in the shared crate instead of source-copied into the dylib.
    pub metadata: HashMap<String, String>,
}

impl GameState {
    /// Move the state by `(dx, dy)`, clamping to the documented bounds.
    pub fn move_by(&mut self, dx: usize, dy: usize) {
        self.x = (self.x + dx).min(100);
        self.y = (self.y + dy).min(250);
    }

    /// Add a metadata key-value pair to the state.
    pub fn insert_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Prelude imported by the generated dylib through [`symbiont::DylibConfig`].
pub mod prelude {
    pub use crate::GameState;
}
