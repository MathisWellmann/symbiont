// SPDX-License-Identifier: MPL-2.0
//! Everything related to exporting `zstd` compressed `.jsonl` files that `deepseek-harness` expects

use std::{
    fmt::Write as _,
    io::{
        self,
        Write,
    },
};

use crate::{
    DshSession,
    EvolutionTrace,
    write_dsh_session,
};

/// Compression level for a session artifact. Traces are text and write once,
/// so the default level is the right trade.
const ZSTD_LEVEL: i32 = zstd::DEFAULT_COMPRESSION_LEVEL;

/// Write `trace` into `sessions_root` as a zstd session artifact, under the
/// directory layout the harness's JSONL backend expects
/// (`<root>/<project-key>/<session-id>/session.jsonl.zstd`), and return the
/// path written.
///
/// `sessions_root` is `$DSH_HOME/sessions`, which is `~/.dsh/sessions` by
/// default.
///
/// # The artifact is a frame container, not a compressed file
///
/// A `.jsonl.zstd` session is a **concatenation of independently decodable
/// zstd frames**, which is what lets the harness append a batch without
/// rewriting the file.
/// So this writes the header as its own frame and the events as a second one.
///
/// # Compression is not optional either
///
/// The backend refuses to load a root that mixes encodings: it walks every
/// session directory and errors out if it finds an artifact with the suffix
/// it is not configured for. Dropping a plain `session.jsonl` there is wrong.
///
/// # Errors
///
/// Returns the directory-creation, serialization and write errors.
pub fn export_dsh_session(
    trace: &EvolutionTrace,
    session: &DshSession<'_>,
    sessions_root: &std::path::Path,
) -> io::Result<std::path::PathBuf> {
    let dir = sessions_root
        .join(project_key(session.cwd()))
        .join(encode_segment(&session.resolved_id(trace)));
    std::fs::create_dir_all(&dir)?;

    let mut jsonl = Vec::new();
    write_dsh_session(trace, session, &mut jsonl)?;

    // The header is the first line and JSON, so it holds no raw newline: the
    // first `\n` in the stream always ends it.
    let header_end = jsonl
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(jsonl.len(), |index| index + 1);
    let (header, events) = jsonl.split_at(header_end);

    let mut artifact = zstd_frame(header)?;
    if !events.is_empty() {
        artifact.extend_from_slice(&zstd_frame(events)?);
    }

    let path = dir.join("session.jsonl.zstd");
    std::fs::write(&path, artifact)?;
    Ok(path)
}

/// Compress `input` into one complete, checksummed zstd frame — the unit the
/// harness's container is built from.
fn zstd_frame(input: &[u8]) -> io::Result<Vec<u8>> {
    let mut encoder = zstd::stream::raw::Encoder::new(ZSTD_LEVEL)?;
    // The harness compresses every frame with `ZSTD_c_checksumFlag` set, and
    // its frame scanner reads the flag out of each frame header to find the
    // next boundary. Matching it keeps a written frame byte-comparable with a
    // harness-written one.
    encoder.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))?;

    let mut writer = zstd::stream::write::Encoder::with_encoder(Vec::new(), encoder);
    writer.write_all(input)?;
    writer.finish()
}

/// The harness's project-directory name for `cwd`: separator runs collapse to
/// `-`, anything outside `[A-Za-z0-9._-]` becomes `~XXXX` over UTF-16 code
/// units, and the result is wrapped in `--`.
fn project_key(cwd: Option<&str>) -> String {
    let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) else {
        return "_no-cwd".to_string();
    };

    let mut readable = String::new();
    let mut separator_run = false;
    for unit in cwd.encode_utf16() {
        match char::from_u32(u32::from(unit)) {
            Some('/' | '\\' | ':') => {
                if !separator_run {
                    readable.push('-');
                }
                separator_run = true;
            }
            Some(ch) if is_safe_segment_char(ch) => {
                readable.push(ch);
                separator_run = false;
            }
            _ => {
                let _ = write!(readable, "~{unit:04X}");
                separator_run = false;
            }
        }
    }

    let trimmed: String = readable
        .trim_start_matches('-')
        .encode_utf16()
        .take(251)
        .collect::<Vec<u16>>()
        .iter()
        .filter_map(|unit| char::from_u32(u32::from(*unit)))
        .collect();
    let body = if trimmed.is_empty() { "root" } else { &trimmed };
    format!("--{body}--")
}

/// The harness's injective single-path-segment encoding of a session id.
fn encode_segment(raw: &str) -> String {
    match raw {
        "" => "_".to_string(),
        "." => "~002E".to_string(),
        ".." => "~002E~002E".to_string(),
        _ => raw
            .encode_utf16()
            .map(|unit| match char::from_u32(u32::from(unit)) {
                Some(ch) if is_safe_segment_char(ch) => ch.to_string(),
                _ => format!("~{unit:04X}"),
            })
            .collect(),
    }
}

/// The harness's literal-in-a-path-segment character class. `~` is excluded:
/// it introduces an escape.
fn is_safe_segment_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::dsh::tests::sample_trace;

    /// The project directory and the session directory follow the harness's
    /// own encoding, or the picker files the session somewhere it never looks.
    #[test]
    fn paths_match_the_harness_encoding() {
        assert_eq!(
            project_key(Some("/home/m/MathisWellmann/symbiont")),
            "--home-m-MathisWellmann-symbiont--",
        );
        assert_eq!(project_key(None), "_no-cwd");
        assert_eq!(project_key(Some("/")), "--root--");
        // A space is not in the literal class, so it escapes to its UTF-16
        // code unit.
        assert_eq!(
            project_key(Some("/tmp/my project")),
            "--tmp-my~0020project--"
        );
        assert_eq!(encode_segment("session-abc_1.2"), "session-abc_1.2");
        assert_eq!(encode_segment("a/b"), "a~002Fb");
        assert_eq!(encode_segment(".."), "~002E~002E");
    }
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
        use std::io::Read as _;

        let root =
            std::env::temp_dir().join(format!("symbiont-dsh-export-frames-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        let session = DshSession::builder().cwd("/tmp/project").build();
        let path = export_dsh_session(&sample_trace(), &session, &root).expect("the export writes");
        let artifact = std::fs::read(&path).expect("the artifact is readable");

        let mut first_frame = Vec::new();
        zstd::stream::read::Decoder::new(artifact.as_slice())
            .expect("a zstd stream")
            .single_frame()
            .read_to_end(&mut first_frame)
            .expect("the first frame decodes on its own");
        let first_frame = String::from_utf8(first_frame).expect("valid utf-8");

        assert!(
            first_frame.ends_with('\n') && first_frame.matches('\n').count() == 1,
            "the first frame must be exactly the header line, got {} line(s)",
            first_frame.lines().count(),
        );
        let header: Value =
            serde_json::from_str(first_frame.trim_end()).expect("the header line is JSON");
        assert_eq!(header["type"], "session");

        // The events follow in their own frame, and the container still reads
        // back as one contiguous JSONL stream.
        let whole = zstd::decode_all(artifact.as_slice()).expect("the container decodes");
        let whole = String::from_utf8(whole).expect("valid utf-8");
        assert!(
            whole.lines().count() > 1,
            "the events must follow the header",
        );
        assert!(whole.starts_with(&first_frame), "the header comes first");
        assert!(whole.ends_with('\n'), "every record is newline-terminated");

        std::fs::remove_dir_all(&root).expect("the test cleans up after itself");
    }
}
