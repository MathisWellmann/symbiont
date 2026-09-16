// SPDX-License-Identifier: MPL-2.0
//! Everything related to exporting `zstd` compressed `.jsonl` files that `deepseek-harness` expects

use std::io;

use dsh_log::container::SessionArtifact;

use crate::{
    DshSession,
    EvolutionTrace,
    dsh::export::dsh_lines,
};

/// Write `trace` into `sessions_root` as a zstd session artifact, under the
/// directory layout the harness's JSONL backend expects
/// (`<root>/<project-key>/<session-id>/session.v3.jsonl.zstd`), and return the
/// path written.
///
/// `sessions_root` is `$DSH_HOME/sessions`, which is `~/.dsh/sessions` by
/// default.
///
/// The artifact is a frame container, not a compressed file: the header goes
/// into its own frame and the events into a second one, because the harness's
/// session listing decodes only the first frame of every artifact. See
/// [`dsh_log::container`] for the layout and why a plain `session.jsonl` in
/// that root is wrong.
///
/// # Errors
///
/// Returns the directory-creation, serialization and write errors.
pub fn export_dsh_session(
    trace: &EvolutionTrace,
    session: &DshSession<'_>,
    sessions_root: &std::path::Path,
) -> io::Result<std::path::PathBuf> {
    let artifact =
        SessionArtifact::locate(sessions_root, session.cwd(), &session.resolved_id(trace));

    let lines = dsh_lines(trace, session);
    let (header, events) = lines.split_at(usize::from(!lines.is_empty()));
    if let Some(header) = header.first() {
        artifact.create(header)?;
    }
    artifact.append(events)?;
    Ok(artifact.path().to_path_buf())
}

#[cfg(test)]
mod tests {
    use dsh_log::LogLine;

    use super::*;
    use crate::dsh::tests::sample_trace;

    /// The artifact is a frame container, and the harness's session listing
    /// decodes only the **first** frame of every session it finds, demanding
    /// exactly one header line from it.
    ///
    /// This is a regression test with teeth: a log compressed as one frame
    /// round-trips perfectly through `zstd -d`, passes every other check here,
    /// and still stops `dsh` from booting, because the listing walks the whole
    /// sessions root. Decoding the whole file is exactly the check that misses
    /// it, so this one decodes the first frame alone.
    #[test]
    fn the_first_zstd_frame_holds_only_the_header() {
        let root =
            std::env::temp_dir().join(format!("symbiont-dsh-export-frames-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        let session = DshSession::builder().cwd("/tmp/project").build();
        let path = export_dsh_session(&sample_trace(), &session, &root).expect("the export writes");
        assert!(path.starts_with(root.join("--tmp-project--")));
        let artifact = SessionArtifact::in_dir(path.parent().expect("a session dir"));

        let first: Vec<LogLine> = artifact
            .read_first_frame()
            .expect("the first frame decodes on its own");
        assert_eq!(
            first.len(),
            1,
            "the first frame must be exactly the header line"
        );
        assert!(matches!(first[0], LogLine::Session(_)));

        // The events follow in their own frame, and the container still reads
        // back as one contiguous JSONL stream.
        let whole: Vec<LogLine> = artifact.read().expect("the container decodes");
        assert!(whole.len() > 1, "the events must follow the header");
        assert!(
            matches!(whole[0], LogLine::Session(_)),
            "the header comes first"
        );
        assert!(!matches!(whole[1], LogLine::Session(_)));

        std::fs::remove_dir_all(&root).expect("the test cleans up after itself");
    }
}
