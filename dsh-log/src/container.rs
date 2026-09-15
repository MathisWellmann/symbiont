// SPDX-License-Identifier: MPL-2.0
//! The on-disk container of a session log and the harness's path layout.
//!
//! A session lives at `<sessions_root>/<project-key>/<session-id>/session.jsonl.zstd`.
//! The file is not a compressed file but a **concatenation of independently
//! decodable, checksummed zstd frames**: the first frame holds the header line
//! alone, every later frame holds one append batch of JSONL records. That is
//! what lets the harness (and this crate) append a batch without rewriting the
//! file, and the harness's session listing decodes only the first frame of
//! each artifact to read its header.
//!
//! The harness refuses to load a sessions root that mixes encodings, so a
//! plain `session.jsonl` must never be dropped there.

use std::{
    fmt::Write as _,
    io,
    path::{
        Path,
        PathBuf,
    },
};

use serde::{
    Serialize,
    de::DeserializeOwned,
};

/// File name of a session artifact inside its session directory.
pub const SESSION_FILE_NAME: &str = "session.jsonl.zstd";

/// The harness's project-directory name for `cwd`: separator runs collapse to
/// `-`, anything outside `[A-Za-z0-9._-]` becomes `~XXXX` over UTF-16 code
/// units, and the result is wrapped in `--`.
#[must_use]
pub fn project_key(cwd: Option<&str>) -> String {
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
#[must_use]
pub fn encode_segment(raw: &str) -> String {
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

/// The directory the harness expects the session `session_id` of a project
/// at `cwd` in: `<sessions_root>/<project-key>/<encoded id>`.
#[must_use]
pub fn session_dir(sessions_root: &Path, cwd: Option<&str>, session_id: &str) -> PathBuf {
    sessions_root
        .join(project_key(cwd))
        .join(encode_segment(session_id))
}

/// Serialize `lines` as newline-terminated JSON Lines.
///
/// # Errors
///
/// Returns the serialization error, which indicates a bug because a log line
/// is plain data.
pub fn encode_jsonl<L: Serialize>(lines: &[L]) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    for line in lines {
        serde_json::to_writer(&mut out, line).map_err(io::Error::other)?;
        out.push(b'\n');
    }
    Ok(out)
}

/// Parse newline-separated JSON documents. Blank lines are skipped.
///
/// # Errors
///
/// Returns the first line that does not parse as `L`.
pub fn decode_jsonl<L: DeserializeOwned>(bytes: &[u8]) -> io::Result<Vec<L>> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .map(|line| serde_json::from_slice(line).map_err(io::Error::other))
        .collect()
}

/// Compression level of a frame. Logs are text and write once, so the default
/// level is the right trade.
#[cfg(feature = "zstd")]
const ZSTD_LEVEL: i32 = zstd::DEFAULT_COMPRESSION_LEVEL;

/// Compress `input` into one complete, checksummed zstd frame — the unit the
/// harness's container is built from.
///
/// The harness compresses every frame with `ZSTD_c_checksumFlag` set, and its
/// frame scanner reads the flag out of each frame header to find the next
/// boundary. Matching it keeps a written frame byte-comparable with a
/// harness-written one.
///
/// # Errors
///
/// Returns the encoder's errors.
#[cfg(feature = "zstd")]
pub fn zstd_frame(input: &[u8]) -> io::Result<Vec<u8>> {
    use std::io::Write as _;

    let mut encoder = zstd::stream::raw::Encoder::new(ZSTD_LEVEL)?;
    encoder.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))?;

    let mut writer = zstd::stream::write::Encoder::with_encoder(Vec::new(), encoder);
    writer.write_all(input)?;
    writer.finish()
}

/// One `session.jsonl.zstd` artifact: a header frame followed by append
/// batches.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionArtifact {
    path: PathBuf,
}

#[cfg(feature = "zstd")]
impl SessionArtifact {
    /// The artifact inside `dir`.
    #[must_use]
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            path: dir.join(SESSION_FILE_NAME),
        }
    }

    /// The artifact of session `session_id` for a project at `cwd`, under
    /// `sessions_root` (`$DSH_HOME/sessions`, `~/.dsh/sessions` by default).
    #[must_use]
    pub fn locate(sessions_root: &Path, cwd: Option<&str>, session_id: &str) -> Self {
        Self::in_dir(&session_dir(sessions_root, cwd, session_id))
    }

    /// Path of the artifact.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the artifact exists on disk.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.path.is_file()
    }

    /// Create the artifact with `header` as its first and only frame. An
    /// existing artifact is replaced.
    ///
    /// # Errors
    ///
    /// Returns the directory-creation, serialization and write errors.
    pub fn create<L: Serialize>(&self, header: &L) -> io::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let frame = zstd_frame(&encode_jsonl(std::slice::from_ref(header))?)?;
        std::fs::write(&self.path, frame)
    }

    /// Append `lines` as one frame. An empty batch writes nothing.
    ///
    /// # Errors
    ///
    /// Returns the serialization and write errors.
    pub fn append<L: Serialize>(&self, lines: &[L]) -> io::Result<()> {
        use std::io::Write as _;

        if lines.is_empty() {
            return Ok(());
        }
        let frame = zstd_frame(&encode_jsonl(lines)?)?;
        let mut file = std::fs::OpenOptions::new().append(true).open(&self.path)?;
        file.write_all(&frame)?;
        file.flush()
    }

    /// Every record of the artifact, header first.
    ///
    /// # Errors
    ///
    /// Returns the read, decompression and parse errors.
    pub fn read<L: DeserializeOwned>(&self) -> io::Result<Vec<L>> {
        let bytes = std::fs::read(&self.path)?;
        decode_jsonl(&zstd::decode_all(bytes.as_slice())?)
    }

    /// The records of the first frame alone — what the harness's session
    /// listing reads, so this must be exactly the header.
    ///
    /// # Errors
    ///
    /// Returns the read, decompression and parse errors.
    pub fn read_first_frame<L: DeserializeOwned>(&self) -> io::Result<Vec<L>> {
        use std::io::Read as _;

        let bytes = std::fs::read(&self.path)?;
        let mut first = Vec::new();
        zstd::stream::read::Decoder::new(bytes.as_slice())?
            .single_frame()
            .read_to_end(&mut first)?;
        decode_jsonl(&first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            session_dir(Path::new("/r"), Some("/p"), "s/1"),
            PathBuf::from("/r/--p--/s~002F1")
        );
    }

    #[test]
    fn jsonl_round_trips_and_skips_blank_lines() {
        let lines = vec![serde_json::json!({"a": 1}), serde_json::json!({"b": "two"})];
        let bytes = encode_jsonl(&lines).expect("plain data serializes");
        assert_eq!(bytes, b"{\"a\":1}\n{\"b\":\"two\"}\n");
        let mut with_blank = bytes.clone();
        with_blank.extend_from_slice(b"\n  \n");
        let back: Vec<serde_json::Value> = decode_jsonl(&with_blank).expect("it parses back");
        assert_eq!(back, lines);
    }

    /// The artifact is a frame container, and the harness's session listing
    /// decodes only the **first** frame of every session it finds, demanding
    /// exactly one header line from it.
    #[cfg(feature = "zstd")]
    #[test]
    fn the_first_frame_holds_only_the_header_and_batches_append() {
        let root = std::env::temp_dir().join(format!("dsh-log-container-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        let artifact = SessionArtifact::locate(&root, Some("/tmp/project"), "s1");
        assert!(!artifact.exists());
        let header = serde_json::json!({"type": "session", "id": "s1"});
        artifact.create(&header).expect("the header frame writes");
        artifact
            .append(&[serde_json::json!({"seq": 0}), serde_json::json!({"seq": 1})])
            .expect("the first batch writes");
        artifact
            .append::<serde_json::Value>(&[])
            .expect("an empty batch is a no-op");
        artifact
            .append(&[serde_json::json!({"seq": 2})])
            .expect("the second batch writes");
        assert!(artifact.exists());
        assert_eq!(
            artifact.path(),
            root.join("--tmp-project--")
                .join("s1")
                .join(SESSION_FILE_NAME)
        );

        let first: Vec<serde_json::Value> = artifact.read_first_frame().expect("first frame");
        assert_eq!(first, vec![header.clone()]);

        let all: Vec<serde_json::Value> = artifact.read().expect("the container decodes");
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], header);
        assert_eq!(all[3]["seq"], 2);

        std::fs::remove_dir_all(&root).expect("the test cleans up after itself");
    }
}
