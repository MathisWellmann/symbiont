//! The module contains code to document the dependencies of the dylib
//! and provide a doc string to the LLM in the system prompt.

use std::{
    collections::{
        BTreeSet,
        HashMap,
        HashSet,
    },
    fmt::Write,
    path::{
        Path,
        PathBuf,
    },
};

use rustdoc_types::{
    Crate as RustdocCrate,
    Id,
    Item,
    ItemEnum,
    Span,
    Use,
    Visibility,
};
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
    let crate_data = rustdoc_json(crate_name, &[crate_name]).await?;
    let rendered = RustApiSynopsis::new(&crate_data).render(ApiSynopsisSubject::Host);
    trace!("api synopsis: {}", rendered.api);

    s.push_str(&rendered.api);

    for reexported_crate in rendered.external_crates {
        let package_candidates = package_candidates_for_crate(&reexported_crate);
        let package_candidates = Vec::from_iter(package_candidates.iter().map(String::as_str));
        match rustdoc_json(&reexported_crate, &package_candidates).await {
            Ok(crate_data) => {
                let rendered =
                    RustApiSynopsis::new(&crate_data).render(ApiSynopsisSubject::ReexportedCrate);
                trace!("re-exported crate api synopsis: {}", rendered.api);
                s.push('\n');
                s.push_str(&rendered.api);
            }
            Err(err) => {
                trace!("could not document re-exported crate {reexported_crate}: {err}");
                let _ = writeln!(
                    s,
                    "\nCould not generate rustdoc JSON for re-exported crate `{reexported_crate}`."
                );
            }
        }
    }

    Ok(())
}

async fn rustdoc_json(crate_name: &str, package_candidates: &[&str]) -> Result<RustdocCrate> {
    for package in package_candidates {
        let args = [
            "rustdoc",
            "-p",
            package,
            "--",
            "--output-format=json",
            "-Z",
            "unstable-options",
        ];
        let output = Command::new("cargo").args(args).output().await?;
        trace!("cargo rustdoc output for package {package}: {output:?}");
        if output.status.success() {
            return read_rustdoc_json(crate_name).await;
        }
    }

    Err(Error::CargoDoc)
}

async fn read_rustdoc_json(crate_name: &str) -> Result<RustdocCrate> {
    let filename = format!("target/doc/{}.json", crate_name.replace('-', "_"));
    let mut json_file = File::open(&filename).await?;
    let mut json_str = String::with_capacity(10_000);
    json_file.read_to_string(&mut json_str).await?;
    serde_json::from_str(&json_str).map_err(|_| Error::MdDoc)
}

fn package_candidates_for_crate(crate_name: &str) -> Vec<String> {
    let mut candidates = vec![crate_name.to_string()];
    let dashed = crate_name.replace('_', "-");
    if dashed != crate_name {
        candidates.push(dashed);
    }
    candidates
}

#[derive(Clone, Copy)]
enum ApiSynopsisSubject {
    Host,
    ReexportedCrate,
}

struct RenderedApiSynopsis {
    api: String,
    external_crates: Vec<String>,
}

/// Small rustdoc-JSON adapter that emits source-like public API snippets.
///
/// We intentionally do not reimplement rustdoc's type/signature renderer. The
/// JSON tells us which public items exist and where they came from; the synopsis
/// is built from those original Rust source spans, with function bodies and
/// constant initializers elided.
struct RustApiSynopsis<'a> {
    crate_data: &'a RustdocCrate,
    source_cache: HashMap<PathBuf, String>,
    rendered_items: HashSet<Id>,
    queued_reexports: HashSet<Id>,
    pending_reexports: Vec<Id>,
    external_crates_to_document: BTreeSet<String>,
    local_crate_alias: String,
}

impl<'a> RustApiSynopsis<'a> {
    fn new(crate_data: &'a RustdocCrate) -> Self {
        Self {
            crate_data,
            source_cache: HashMap::new(),
            rendered_items: HashSet::new(),
            queued_reexports: HashSet::new(),
            pending_reexports: Vec::new(),
            external_crates_to_document: BTreeSet::new(),
            local_crate_alias: "host".to_string(),
        }
    }

    fn render(mut self, subject: ApiSynopsisSubject) -> RenderedApiSynopsis {
        let mut out = String::new();
        let root = self.crate_data.index.get(&self.crate_data.root);
        let crate_name = root.and_then(|item| item.name.as_deref()).unwrap_or("host");

        match subject {
            ApiSynopsisSubject::Host => {
                self.local_crate_alias = "host".to_string();
                out.push_str("This is a Rust-style synopsis of the public host API available to generated code.\n");
                out.push_str("It is for reference only; do not copy the whole block. Call these APIs from the evolved function.\n");
                out.push_str("The generated dylib depends on this crate as `host` and injects `use host::prelude::*;`; do not emit that import yourself.\n");
                out.push_str("Public dependency crates re-exported by `host::prelude` are documented below for API reference, but are not direct dylib dependencies. Use their re-exported items unqualified, not through dependency crate paths.\n\n");
                out.push_str("```rust\n");
                let _ = writeln!(out, "// API synopsis for host crate `{crate_name}`.");
            }
            ApiSynopsisSubject::ReexportedCrate => {
                self.local_crate_alias = crate_name.to_string();
                let _ = writeln!(
                    out,
                    "This is a Rust-style synopsis of the public API for dependency crate `{crate_name}`, re-exported by the host prelude."
                );
                out.push_str("This is API reference for items re-exported into scope by `host::prelude::*`; the dependency crate path itself is not available. Use documented re-exported items unqualified.\n\n");
                out.push_str("```rust\n");
                let _ = writeln!(
                    out,
                    "// API synopsis for re-exported dependency crate `{crate_name}`."
                );
            }
        }
        out.push_str("// Bodies and large constant initializers are omitted.\n\n");

        if let Some(root) = root {
            self.write_module_items(&mut out, root, 0);
        }
        self.write_pending_reexports(&mut out);

        out.push_str("```\n");
        RenderedApiSynopsis {
            api: out,
            external_crates: self.external_crates_to_document.into_iter().collect(),
        }
    }

    fn write_module_items(&mut self, out: &mut String, module_item: &Item, indent: usize) {
        let ItemEnum::Module(module) = &module_item.inner else {
            return;
        };

        for item_id in &module.items {
            let Some(item) = self.crate_data.index.get(item_id) else {
                continue;
            };
            if !is_public(item) {
                continue;
            }
            self.write_item(out, item, indent);
        }
    }

    fn write_item(&mut self, out: &mut String, item: &Item, indent: usize) {
        let already_rendered = !matches!(&item.inner, ItemEnum::Use(_) | ItemEnum::Module(_))
            && !self.rendered_items.insert(item.id);
        if already_rendered {
            return;
        }

        match &item.inner {
            ItemEnum::Use(use_item) => {
                self.write_snippet(
                    out,
                    &synthesize_use(use_item, &self.local_crate_alias),
                    indent,
                );
                self.queue_use_targets(use_item);
            }
            ItemEnum::Module(_) => {
                let Some(name) = &item.name else {
                    return;
                };
                write_doc_comment(out, item.docs.as_deref(), indent);
                write_indent(out, indent);
                let _ = writeln!(out, "pub mod {name} {{");
                self.write_module_items(out, item, indent + 1);
                write_indent(out, indent);
                out.push_str("}\n\n");
            }
            ItemEnum::Struct(strukt) => {
                if let Some(snippet) = self.source_snippet(item) {
                    self.write_snippet(out, &snippet, indent);
                    self.write_inherent_impls(out, item.name.as_deref(), &strukt.impls, indent);
                }
            }
            ItemEnum::Enum(enm) => {
                if let Some(snippet) = self.source_snippet(item) {
                    self.write_snippet(out, &snippet, indent);
                    self.write_inherent_impls(out, item.name.as_deref(), &enm.impls, indent);
                }
            }
            ItemEnum::Union(union) => {
                if let Some(snippet) = self.source_snippet(item) {
                    self.write_snippet(out, &snippet, indent);
                    self.write_inherent_impls(out, item.name.as_deref(), &union.impls, indent);
                }
            }
            ItemEnum::Function(_) => {
                if let Some(snippet) = self.source_snippet(item).and_then(|s| elide_body(&s)) {
                    self.write_snippet(out, &snippet, indent);
                }
            }
            ItemEnum::Constant { .. } | ItemEnum::Static(_) => {
                if let Some(snippet) = self.source_snippet(item).map(|s| elide_initializer(&s)) {
                    self.write_snippet(out, &snippet, indent);
                }
            }
            ItemEnum::TypeAlias(_) => {
                if let Some(snippet) = self.source_snippet(item) {
                    self.write_snippet(out, &snippet, indent);
                }
            }
            _ => {}
        }
    }

    fn write_inherent_impls(
        &mut self,
        out: &mut String,
        type_name: Option<&str>,
        impl_ids: &[Id],
        indent: usize,
    ) {
        let Some(type_name) = type_name else {
            return;
        };

        let methods = impl_ids
            .iter()
            .filter_map(|impl_id| self.crate_data.index.get(impl_id))
            .filter_map(|item| match &item.inner {
                ItemEnum::Impl(impl_block) if impl_block.trait_.is_none() => Some(impl_block),
                _ => None,
            })
            .flat_map(|impl_block| impl_block.items.iter())
            .filter_map(|method_id| self.crate_data.index.get(method_id))
            .filter(|method| is_public(method))
            .filter_map(|method| self.source_snippet(method).and_then(|s| elide_body(&s)))
            .collect::<Vec<_>>();

        if methods.is_empty() {
            return;
        }

        write_indent(out, indent);
        let _ = writeln!(out, "impl {type_name} {{");
        for method in methods {
            let method = normalize_local_paths(&method, &self.local_crate_alias);
            write_snippet_lines(out, &method, indent + 1);
        }
        write_indent(out, indent);
        out.push_str("}\n\n");
    }

    fn write_pending_reexports(&mut self, out: &mut String) {
        while let Some(id) = self.pending_reexports.pop() {
            let Some(item) = self.crate_data.index.get(&id) else {
                continue;
            };
            if !is_public(item) || self.rendered_items.contains(&id) {
                continue;
            }
            self.write_item(out, item, 0);
        }
    }

    fn queue_reexport(&mut self, id: Id) {
        if self.rendered_items.contains(&id) || !self.queued_reexports.insert(id) {
            return;
        }
        self.pending_reexports.push(id);
    }

    fn queue_use_targets(&mut self, use_item: &Use) {
        if let Some(id) = use_item.id {
            self.queue_reexport(id);
            self.queue_external_crate_for_id(id);
            return;
        }

        if !use_item.is_glob {
            return;
        }

        let Some(first_segment) = use_item.source.split("::").next() else {
            return;
        };
        if matches!(first_segment, "crate" | "self" | "super" | "host") {
            return;
        }
        if self
            .crate_data
            .external_crates
            .values()
            .any(|external_crate| {
                external_crate.name == first_segment
                    && is_documentable_external_crate(external_crate)
            })
        {
            self.external_crates_to_document
                .insert(first_segment.to_string());
        }
    }

    fn queue_external_crate_for_id(&mut self, id: Id) {
        let Some(summary) = self.crate_data.paths.get(&id) else {
            return;
        };
        let Some(external_crate) = self.crate_data.external_crates.get(&summary.crate_id) else {
            return;
        };
        if is_documentable_external_crate(external_crate) {
            self.external_crates_to_document
                .insert(external_crate.name.clone());
        }
    }

    fn source_snippet(&mut self, item: &Item) -> Option<String> {
        let span = item.span.as_ref()?;
        self.snippet_for_span(span)
    }

    fn snippet_for_span(&mut self, span: &Span) -> Option<String> {
        let source = self.read_source(&span.filename)?;
        slice_span(source, span)
    }

    fn read_source(&mut self, path: &Path) -> Option<&str> {
        if !self.source_cache.contains_key(path) {
            let source = std::fs::read_to_string(path).ok()?;
            self.source_cache.insert(path.to_owned(), source);
        }
        self.source_cache.get(path).map(String::as_str)
    }

    fn write_snippet(&self, out: &mut String, snippet: &str, indent: usize) {
        let snippet = normalize_local_paths(snippet.trim(), &self.local_crate_alias);
        write_snippet_lines(out, &snippet, indent);
        out.push('\n');
    }
}

fn synthesize_use(use_item: &Use, local_crate_alias: &str) -> String {
    let source = normalize_local_paths(&use_item.source, local_crate_alias);
    if use_item.is_glob {
        format!("pub use {source}::*;")
    } else if source.ends_with(&format!("::{}", use_item.name)) {
        format!("pub use {source};")
    } else {
        format!("pub use {source} as {};", use_item.name)
    }
}

fn is_public(item: &Item) -> bool {
    matches!(item.visibility, Visibility::Public)
}

fn is_documentable_external_crate(external_crate: &rustdoc_types::ExternalCrate) -> bool {
    if matches!(
        external_crate.name.as_str(),
        "alloc" | "core" | "proc_macro" | "std" | "test"
    ) {
        return false;
    }

    !external_crate
        .html_root_url
        .as_deref()
        .is_some_and(|url| url.starts_with("https://doc.rust-lang.org/"))
}

fn write_doc_comment(out: &mut String, docs: Option<&str>, indent: usize) {
    let Some(docs) = docs else {
        return;
    };
    let docs = docs.trim();
    if docs.is_empty() {
        return;
    }

    for line in docs.lines() {
        write_indent(out, indent);
        let line = line.trim_end();
        if line.is_empty() {
            out.push_str("///\n");
        } else {
            let _ = writeln!(out, "/// {line}");
        }
    }
}

fn write_snippet_lines(out: &mut String, snippet: &str, indent: usize) {
    for line in snippet.lines() {
        if !line.trim().is_empty() {
            write_indent(out, indent);
            out.push_str(line);
        }
        out.push('\n');
    }
}

fn write_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("    ");
    }
}

fn normalize_local_paths(snippet: &str, local_crate_alias: &str) -> String {
    snippet
        .replace("crate::", &format!("{local_crate_alias}::"))
        .replace("use crate;", &format!("use {local_crate_alias};"))
}

fn slice_span(source: &str, span: &Span) -> Option<String> {
    let lines = source.lines().collect::<Vec<_>>();
    let (start_line, start_col) = span.begin;
    let (end_line, end_col) = span.end;
    let mut start_idx = start_line.checked_sub(1)?;
    let end_idx = end_line.checked_sub(1)?;
    let mut start_col = start_col;

    if start_idx >= lines.len() || end_idx >= lines.len() || start_idx > end_idx {
        return None;
    }

    while start_idx > 0 && is_item_prefix_line(lines[start_idx - 1]) {
        start_idx -= 1;
        start_col = 1;
    }

    if start_idx == end_idx {
        return Some(slice_columns(lines[start_idx], start_col, end_col).to_string());
    }

    let mut snippet = String::new();
    snippet.push_str(slice_columns_to_end(lines[start_idx], start_col));
    snippet.push('\n');
    for line in &lines[start_idx + 1..end_idx] {
        snippet.push_str(line);
        snippet.push('\n');
    }
    snippet.push_str(slice_columns(lines[end_idx], 1, end_col));
    Some(snippet)
}

fn is_item_prefix_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("///") || trimmed.starts_with("#[")
}

fn slice_columns(line: &str, start_col: usize, end_col: usize) -> &str {
    let start = column_to_byte(line, start_col);
    let end = column_to_byte(line, end_col).max(start).min(line.len());
    &line[start..end]
}

fn slice_columns_to_end(line: &str, start_col: usize) -> &str {
    let start = column_to_byte(line, start_col);
    &line[start..]
}

fn column_to_byte(line: &str, one_indexed_col: usize) -> usize {
    let zero_indexed_col = one_indexed_col.saturating_sub(1);
    line.char_indices()
        .nth(zero_indexed_col)
        .map(|(idx, _)| idx)
        .unwrap_or(line.len())
}

fn elide_body(snippet: &str) -> Option<String> {
    let (open, close) = outer_brace_pair(snippet)?;
    if !snippet[close + 1..].trim().is_empty() {
        return None;
    }
    let mut out = snippet[..open].trim_end().to_string();
    out.push(';');
    Some(out)
}

fn elide_initializer(snippet: &str) -> String {
    let Some((equals, semicolon)) = top_level_initializer_bounds(snippet) else {
        return snippet.to_string();
    };
    let mut out = snippet[..equals].trim_end().to_string();
    out.push_str(&snippet[semicolon..]);
    out
}

fn outer_brace_pair(src: &str) -> Option<(usize, usize)> {
    let events = scan_top_level_events(src);
    events
        .brace_pairs
        .into_iter()
        .find(|(_, close)| src[close + 1..].trim().is_empty())
}

fn top_level_initializer_bounds(src: &str) -> Option<(usize, usize)> {
    let events = scan_top_level_events(src);
    let equals = events.equals?;
    let semicolon = events.semicolon?;
    (equals < semicolon).then_some((equals, semicolon))
}

#[derive(Default)]
struct TopLevelEvents {
    brace_pairs: Vec<(usize, usize)>,
    equals: Option<usize>,
    semicolon: Option<usize>,
}

fn scan_top_level_events(src: &str) -> TopLevelEvents {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut depth = 0usize;
    let mut stack = Vec::new();
    let mut events = TopLevelEvents::default();

    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => i = skip_line_comment(bytes, i + 2),
            b'/' if bytes.get(i + 1) == Some(&b'*') => i = skip_block_comment(bytes, i + 2),
            b'"' => i = skip_string(bytes, i + 1),
            b'\'' => i = skip_char(bytes, i + 1),
            b'r' if raw_string_hashes(bytes, i).is_some() => i = skip_raw_string(bytes, i),
            b'(' | b'[' => {
                depth += 1;
                i += 1;
            }
            b')' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b'{' => {
                if depth == 0 {
                    stack.push(i);
                }
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if let Some(open) = (depth == 0).then(|| stack.pop()).flatten() {
                    events.brace_pairs.push((open, i));
                }
                i += 1;
            }
            b'=' if depth == 0 && events.equals.is_none() => {
                events.equals = Some(i);
                i += 1;
            }
            b';' if depth == 0 && events.semicolon.is_none() => {
                events.semicolon = Some(i);
                i += 1;
            }
            _ => i += 1,
        }
    }

    events
}

fn skip_line_comment(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i] != b'\n' {
        i += 1;
    }
    i
}

fn skip_block_comment(bytes: &[u8], mut i: usize) -> usize {
    let mut depth = 1usize;
    while i + 1 < bytes.len() && depth > 0 {
        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
            depth += 1;
            i += 2;
        } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
            depth -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    i
}

fn skip_string(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    i
}

fn skip_char(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'\'' => return i + 1,
            _ => i += 1,
        }
    }
    i
}

fn raw_string_hashes(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    let mut hashes = 0usize;
    while bytes.get(i) == Some(&b'#') {
        hashes += 1;
        i += 1;
    }
    (bytes.get(i) == Some(&b'"')).then_some(hashes)
}

fn skip_raw_string(bytes: &[u8], start: usize) -> usize {
    let Some(hashes) = raw_string_hashes(bytes, start) else {
        return start + 1;
    };
    let mut i = start + 2 + hashes;
    while i < bytes.len() {
        if bytes[i] == b'"' && bytes.get(i + 1..i + 1 + hashes) == Some(&vec![b'#'; hashes][..]) {
            return i + 1 + hashes;
        }
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: u32, name: Option<&str>, inner: ItemEnum) -> Item {
        Item {
            id: Id(id),
            crate_id: 0,
            name: name.map(str::to_string),
            span: None,
            visibility: Visibility::Public,
            docs: None,
            links: HashMap::new(),
            attrs: Vec::new(),
            deprecation: None,
            inner,
        }
    }

    #[test]
    fn doc_string_host_render_queues_external_glob_reexport_crate() {
        let crate_data = RustdocCrate {
            root: Id(0),
            crate_version: None,
            includes_private: false,
            index: HashMap::from([
                (
                    Id(0),
                    item(
                        0,
                        Some("feat_trade_flow_egui"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: true,
                            items: vec![Id(1)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(1),
                    item(
                        1,
                        Some("prelude"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: false,
                            items: vec![Id(2)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(2),
                    item(
                        2,
                        None,
                        ItemEnum::Use(Use {
                            source: "sliding_features".to_string(),
                            name: "sliding_features".to_string(),
                            id: Some(Id(99)),
                            is_glob: true,
                        }),
                    ),
                ),
            ]),
            paths: HashMap::from([(
                Id(99),
                rustdoc_types::ItemSummary {
                    crate_id: 1,
                    path: vec!["sliding_features".to_string()],
                    kind: rustdoc_types::ItemKind::Module,
                },
            )]),
            external_crates: HashMap::from([(
                1,
                rustdoc_types::ExternalCrate {
                    name: "sliding_features".to_string(),
                    html_root_url: None,
                    path: PathBuf::from("target/debug/deps/libsliding_features.rmeta"),
                },
            )]),
            target: rustdoc_types::Target {
                triple: "x86_64-unknown-linux-gnu".to_string(),
                target_features: Vec::new(),
            },
            format_version: rustdoc_types::FORMAT_VERSION,
        };

        let rendered = RustApiSynopsis::new(&crate_data).render(ApiSynopsisSubject::Host);

        assert!(rendered.api.contains("pub mod prelude"));
        assert!(rendered.api.contains("pub use sliding_features::*;"));
        assert_eq!(rendered.external_crates, vec!["sliding_features"]);
    }

    #[test]
    fn doc_string_host_render_queues_external_named_reexport_crate() {
        let crate_data = RustdocCrate {
            root: Id(0),
            crate_version: None,
            includes_private: false,
            index: HashMap::from([
                (
                    Id(0),
                    item(
                        0,
                        Some("feat_trade_flow_egui"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: true,
                            items: vec![Id(1)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(1),
                    item(
                        1,
                        Some("prelude"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: false,
                            items: vec![Id(2)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(2),
                    item(
                        2,
                        None,
                        ItemEnum::Use(Use {
                            source: "sliding_features::RollingSum".to_string(),
                            name: "RollingSum".to_string(),
                            id: Some(Id(99)),
                            is_glob: false,
                        }),
                    ),
                ),
            ]),
            paths: HashMap::from([(
                Id(99),
                rustdoc_types::ItemSummary {
                    crate_id: 1,
                    path: vec!["sliding_features".to_string(), "RollingSum".to_string()],
                    kind: rustdoc_types::ItemKind::Struct,
                },
            )]),
            external_crates: HashMap::from([(
                1,
                rustdoc_types::ExternalCrate {
                    name: "sliding_features".to_string(),
                    html_root_url: None,
                    path: PathBuf::from("target/debug/deps/libsliding_features.rmeta"),
                },
            )]),
            target: rustdoc_types::Target {
                triple: "x86_64-unknown-linux-gnu".to_string(),
                target_features: Vec::new(),
            },
            format_version: rustdoc_types::FORMAT_VERSION,
        };

        let rendered = RustApiSynopsis::new(&crate_data).render(ApiSynopsisSubject::Host);

        assert!(
            rendered
                .api
                .contains("pub use sliding_features::RollingSum;")
        );
        assert_eq!(rendered.external_crates, vec!["sliding_features"]);
    }

    #[test]
    fn doc_string_host_render_ignores_std_reexport_crate() {
        let crate_data = RustdocCrate {
            root: Id(0),
            crate_version: None,
            includes_private: false,
            index: HashMap::from([
                (
                    Id(0),
                    item(
                        0,
                        Some("feat_trade_flow_egui"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: true,
                            items: vec![Id(1)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(1),
                    item(
                        1,
                        Some("prelude"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: false,
                            items: vec![Id(2)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(2),
                    item(
                        2,
                        None,
                        ItemEnum::Use(Use {
                            source: "std::fmt::Debug".to_string(),
                            name: "Debug".to_string(),
                            id: Some(Id(99)),
                            is_glob: false,
                        }),
                    ),
                ),
            ]),
            paths: HashMap::from([(
                Id(99),
                rustdoc_types::ItemSummary {
                    crate_id: 1,
                    path: vec!["std".to_string(), "fmt".to_string(), "Debug".to_string()],
                    kind: rustdoc_types::ItemKind::Trait,
                },
            )]),
            external_crates: HashMap::from([(
                1,
                rustdoc_types::ExternalCrate {
                    name: "std".to_string(),
                    html_root_url: Some("https://doc.rust-lang.org/nightly/".to_string()),
                    path: PathBuf::from("libstd.rmeta"),
                },
            )]),
            target: rustdoc_types::Target {
                triple: "x86_64-unknown-linux-gnu".to_string(),
                target_features: Vec::new(),
            },
            format_version: rustdoc_types::FORMAT_VERSION,
        };

        let rendered = RustApiSynopsis::new(&crate_data).render(ApiSynopsisSubject::Host);

        assert!(rendered.api.contains("pub use std::fmt::Debug;"));
        assert!(rendered.external_crates.is_empty());
    }

    #[test]
    fn doc_string_package_candidates_try_crate_and_dashed_package_names() {
        assert_eq!(
            package_candidates_for_crate("sliding_features"),
            vec!["sliding_features", "sliding-features"]
        );
        assert_eq!(
            package_candidates_for_crate("sliding-features"),
            vec!["sliding-features"]
        );
    }

    #[test]
    fn doc_string_normalizes_crate_paths_to_the_active_synopsis_alias() {
        assert_eq!(
            normalize_local_paths("crate::Feature", "host"),
            "host::Feature"
        );
        assert_eq!(
            normalize_local_paths("crate::Feature", "sliding_features"),
            "sliding_features::Feature"
        );
        assert_eq!(
            normalize_local_paths("use crate;", "sliding_features"),
            "use sliding_features;"
        );
    }

    #[test]
    fn doc_string_reexported_crate_render_uses_dependency_alias() {
        let crate_data = RustdocCrate {
            root: Id(0),
            crate_version: None,
            includes_private: false,
            index: HashMap::from([
                (
                    Id(0),
                    item(
                        0,
                        Some("sliding_features"),
                        ItemEnum::Module(rustdoc_types::Module {
                            is_crate: true,
                            items: vec![Id(1)],
                            is_stripped: false,
                        }),
                    ),
                ),
                (
                    Id(1),
                    item(
                        1,
                        None,
                        ItemEnum::Use(Use {
                            source: "crate::RollingSum".to_string(),
                            name: "RollingSum".to_string(),
                            id: None,
                            is_glob: false,
                        }),
                    ),
                ),
            ]),
            paths: HashMap::new(),
            external_crates: HashMap::new(),
            target: rustdoc_types::Target {
                triple: "x86_64-unknown-linux-gnu".to_string(),
                target_features: Vec::new(),
            },
            format_version: rustdoc_types::FORMAT_VERSION,
        };

        let rendered =
            RustApiSynopsis::new(&crate_data).render(ApiSynopsisSubject::ReexportedCrate);

        assert!(rendered.api.contains("dependency crate `sliding_features`"));
        assert!(
            rendered
                .api
                .contains("pub use sliding_features::RollingSum;")
        );
        assert!(!rendered.api.contains("host::RollingSum"));
    }

    #[test]
    fn doc_string_elides_function_bodies_without_losing_signature_docs() {
        let snippet = r#"
/// Add a sample to the rolling window.
pub fn push(&mut self, sample: f64) -> f64 {
    self.sum += sample;
    self.sum
}
"#;

        let elided = elide_body(snippet).expect("function body should be elided");

        assert!(elided.contains("/// Add a sample to the rolling window."));
        assert!(elided.contains("pub fn push(&mut self, sample: f64) -> f64;"));
        assert!(!elided.contains("self.sum += sample"));
    }

    #[test]
    fn doc_string_elides_large_constant_initializers_without_losing_type() {
        let snippet = r#"
/// Coefficients available to generated code.
pub const WEIGHTS: [f64; 3] = [1.0, 2.0, 3.0];
"#;

        let elided = elide_initializer(snippet);

        assert!(elided.contains("/// Coefficients available to generated code."));
        assert!(elided.contains("pub const WEIGHTS: [f64; 3];"));
        assert!(!elided.contains("[1.0, 2.0, 3.0]"));
    }
}
