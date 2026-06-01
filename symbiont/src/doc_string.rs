//! The module contains code to document the dependencies of the dylib
//! and provide a doc string to the LLM in the system prompt.

use std::fmt::Write;

use cargo_doc_md::convert_json_string;
use tokio::{
    fs::File,
    io::AsyncReadExt,
    process::Command,
};
use tracing::trace;

use crate::{
    Error,
    Result,
};

/// Document the prelude crate to give the LLM context about available types and methods.
pub(crate) async fn write_prelude_doc_string(s: &mut String, crate_name: &str) -> Result<()> {
    let args = [
        "rustdoc",
        "-p",
        crate_name,
        "--",
        "--output-format=json",
        "-Z",
        "unstable-options",
    ];
    let output = Command::new("cargo").args(args).output().await?;
    trace!("output: {output:?}");
    if !output.status.success() {
        return Err(Error::CargoDoc);
    }

    let filename = format!("target/doc/{}.json", crate_name.replace("-", "_"));
    let mut json_file = File::open(&filename).await?;
    let mut json_str = String::with_capacity(10_000);
    json_file.read_to_string(&mut json_str).await?;
    const INCLUDE_PRIVATE: bool = false;
    let md = convert_json_string(&json_str, INCLUDE_PRIVATE).map_err(|_| Error::MdDoc)?;
    trace!("md: {md}");

    write!(s, "{}", md)?;

    Ok(())
}
