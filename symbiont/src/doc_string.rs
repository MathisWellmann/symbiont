//! The module contains code to document the dependencies of the dylib
//! and provide a doc string to the LLM in the system prompt.

use std::{
    collections::{
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

    let filename = format!("target/doc/{}.json", crate_name.replace('-', "_"));
    let mut json_file = File::open(&filename).await?;
    let mut json_str = String::with_capacity(10_000);
    json_file.read_to_string(&mut json_str).await?;
    let crate_data: RustdocCrate = serde_json::from_str(&json_str).map_err(|_| Error::MdDoc)?;
    let api = RustApiSynopsis::new(&crate_data).render();
    trace!("api synopsis: {api}");

    s.push_str(&api);

    Ok(())
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
}

impl<'a> RustApiSynopsis<'a> {
    fn new(crate_data: &'a RustdocCrate) -> Self {
        Self {
            crate_data,
            source_cache: HashMap::new(),
            rendered_items: HashSet::new(),
            queued_reexports: HashSet::new(),
            pending_reexports: Vec::new(),
        }
    }

    fn render(mut self) -> String {
        let mut out = String::new();
        let root = self.crate_data.index.get(&self.crate_data.root);
        let crate_name = root.and_then(|item| item.name.as_deref()).unwrap_or("host");

        out.push_str(
            "This is a Rust-style synopsis of the public host API available to generated code.\n",
        );
        out.push_str("It is for reference only; do not copy the whole block. Call these APIs from the evolved function.\n");
        out.push_str("The host crate is available to the generated dylib as `host`; configured preludes may already import common items.\n\n");
        out.push_str("```rust\n");
        let _ = writeln!(out, "// API synopsis for host crate `{crate_name}`.");
        out.push_str("// Bodies and large constant initializers are omitted.\n\n");

        if let Some(root) = root {
            self.write_module_items(&mut out, root, 0);
        }
        self.write_pending_reexports(&mut out);

        out.push_str("```\n");
        out
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
                write_snippet(out, &synthesize_use(use_item), indent);
                if let Some(id) = use_item.id {
                    self.queue_reexport(id);
                }
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
                    write_snippet(out, &snippet, indent);
                    self.write_inherent_impls(out, item.name.as_deref(), &strukt.impls, indent);
                }
            }
            ItemEnum::Enum(enm) => {
                if let Some(snippet) = self.source_snippet(item) {
                    write_snippet(out, &snippet, indent);
                    self.write_inherent_impls(out, item.name.as_deref(), &enm.impls, indent);
                }
            }
            ItemEnum::Union(union) => {
                if let Some(snippet) = self.source_snippet(item) {
                    write_snippet(out, &snippet, indent);
                    self.write_inherent_impls(out, item.name.as_deref(), &union.impls, indent);
                }
            }
            ItemEnum::Function(_) => {
                if let Some(snippet) = self.source_snippet(item).and_then(|s| elide_body(&s)) {
                    write_snippet(out, &snippet, indent);
                }
            }
            ItemEnum::Constant { .. } | ItemEnum::Static(_) => {
                if let Some(snippet) = self.source_snippet(item).map(|s| elide_initializer(&s)) {
                    write_snippet(out, &snippet, indent);
                }
            }
            ItemEnum::TypeAlias(_) => {
                if let Some(snippet) = self.source_snippet(item) {
                    write_snippet(out, &snippet, indent);
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
}

fn synthesize_use(use_item: &Use) -> String {
    let source = normalize_host_paths(&use_item.source);
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

fn write_snippet(out: &mut String, snippet: &str, indent: usize) {
    let snippet = normalize_host_paths(snippet.trim());
    write_snippet_lines(out, &snippet, indent);
    out.push('\n');
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

fn normalize_host_paths(snippet: &str) -> String {
    snippet
        .replace("crate::", "host::")
        .replace("use crate;", "use host;")
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
