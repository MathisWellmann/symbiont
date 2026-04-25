//! Symbiont: Agent harness for hot-reloadable function evolution in Rust.
//!
//! Declare functions with [`evolvable!`] and let an LLM rewrite their
//! implementations at runtime. The library manages a temporary dylib crate,
//! compilation, loading, and hot-swapping transparently.
//!
//! # Example
//!
//! ```rust,ignore
//! symbiont::evolvable! {
//!     fn step(counter: &mut usize) {
//!         *counter += 1;
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> symbiont::Result<()> {
//!     let runtime = symbiont::Runtime::init(SYMBIONT_DECLS).await?;
//!
//!     let mut counter = 0;
//!     loop {
//!         step(&mut counter);
//!         println!("counter: {counter}");
//!         std::thread::sleep(std::time::Duration::from_secs(1));
//!         // TODO: show the actual function evolution once the API is nicer.
//!     }
//! }
//! ```

pub mod error;
pub mod inference;
pub mod runtime;

mod compiler;
mod decl;
mod parser;
mod utils;
mod validation;

// Re-export the proc macro.
// Re-export key types.
pub use decl::EvolvableDecl;
pub use error::{
    Error,
    Result,
};
// Re-export libloading so macro-generated code can reference it
// without the user adding it to their own Cargo.toml.
#[doc(hidden)]
pub use libloading;
pub use runtime::Runtime;
pub use symbiont_macros::evolvable;

/// Internal module for macro-generated dispatch code.
///
/// Not part of the public API — used by `evolvable!` expansion.
#[doc(hidden)]
pub mod __internal {
    pub use crate::runtime::lib_read_lock;
}
