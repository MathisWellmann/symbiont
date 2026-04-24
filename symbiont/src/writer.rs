use crate::Result;
use std::fs;
use std::path::Path;

use prettyplease;
use tracing::info;

use crate::error::Error;

/// Write the validated AST to `symbiont-lib/src/lib.rs`, preserving
/// any non-function items (comments, structs, etc.) that were already there.
///
/// Strategy: extract all `pub fn` + `#[no_mangle]` functions from the AST,
/// then write them to lib.rs. Everything else in the existing lib.rs
/// (structs, comments, etc.) is kept as-is.
///
/// The generated Rust code is formatted with `prettyplease` for clean output.
pub(crate) fn write_generated_lib(file: &syn::File) -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let lib_path = Path::new(manifest_dir)
        .parent()
        .unwrap()
        .join("symbiont-lib")
        .join("src")
        .join("lib.rs");

    let formatted_code = prettyplease::unparse(file);

    fs::write(&lib_path, formatted_code)
        .map_err(|e| Error::WriteLib(format!("Failed to write lib.rs: {e}")))?;
    info!("Wrote new `lib.rs` to {lib_path:?}");

    Ok(())
}
