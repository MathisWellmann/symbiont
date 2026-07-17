//! The module contains code to document the dependencies of the dylib
//! and provide a doc string to the LLM in the system prompt.

use std::{
    collections::{
        BTreeMap,
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

/// Document the exact API imported by `use host::prelude::*;`.
pub(crate) async fn write_prelude_doc_string(s: &mut String, crate_name: &str) -> Result<()> {
    let crate_data = rustdoc_json(crate_name, &[crate_name]).await?;
    let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();
    trace!("host facade synopsis: {}", rendered.api);
    s.push_str(&rendered.api);

    let mut pending = rendered.external_facades;
    let mut rendered_facades = BTreeSet::new();
    let mut crate_cache: HashMap<String, Option<RustdocCrate>> = HashMap::new();
    let mut rendered_ids: HashMap<String, HashSet<Id>> = HashMap::new();
    while let Some(crate_name) = pending.keys().next().cloned() {
        let requests = pending.remove(&crate_name).unwrap_or_default();
        let requests = requests
            .into_iter()
            .filter(|request| rendered_facades.insert((crate_name.clone(), request.clone())))
            .collect::<BTreeSet<_>>();
        if requests.is_empty() {
            continue;
        }

        if !crate_cache.contains_key(&crate_name) {
            let package_candidates = package_candidates_for_crate(&crate_name);
            let package_candidates = Vec::from_iter(package_candidates.iter().map(String::as_str));
            let crate_data = match rustdoc_json(&crate_name, &package_candidates).await {
                Ok(crate_data) => Some(crate_data),
                Err(err) => {
                    trace!("could not document reachable crate {crate_name}: {err}");
                    let _ = writeln!(
                        s,
                        "\nCould not generate rustdoc JSON for reachable crate `{crate_name}`."
                    );
                    None
                }
            };
            crate_cache.insert(crate_name.clone(), crate_data);
        }
        let Some(crate_data) = crate_cache.get(&crate_name).and_then(Option::as_ref) else {
            continue;
        };

        let rendered = RustApiSynopsis::new(crate_data)
            .with_rendered_items(rendered_ids.remove(&crate_name).unwrap_or_default())
            .render_external_facade(&crate_name, &requests);
        trace!("reachable {crate_name} facade synopsis: {}", rendered.api);
        if !rendered.api.is_empty() {
            s.push('\n');
            s.push_str(&rendered.api);
        }
        rendered_ids.insert(crate_name.clone(), rendered.rendered_items);
        merge_facades(&mut pending, rendered.external_facades);
    }

    Ok(())
}

fn merge_facades(
    target: &mut BTreeMap<String, BTreeSet<FacadeRequest>>,
    source: BTreeMap<String, BTreeSet<FacadeRequest>>,
) {
    for (crate_name, requests) in source {
        target.entry(crate_name).or_default().extend(requests);
    }
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FacadeRequest {
    Module(Vec<String>),
    Item(Vec<String>),
}

struct RenderedApiSynopsis {
    api: String,
    external_facades: BTreeMap<String, BTreeSet<FacadeRequest>>,
    /// Ids rendered by this synopsis pass, merged with any seeded ids. Used to
    /// avoid re-rendering items when the same crate is documented in multiple
    /// facade batches.
    rendered_items: HashSet<Id>,
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
    queued_reexports: HashSet<(Id, bool)>,
    pending_reexports: Vec<(Id, bool)>,
    external_facades: BTreeMap<String, BTreeSet<FacadeRequest>>,
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
            external_facades: BTreeMap::new(),
            local_crate_alias: "host".to_string(),
        }
    }

    /// Seeds the set of already rendered item ids so repeated renders of the
    /// same crate do not duplicate items.
    fn with_rendered_items(mut self, rendered_items: HashSet<Id>) -> Self {
        self.rendered_items = rendered_items;
        self
    }

    fn render_host_facade(mut self) -> RenderedApiSynopsis {
        self.local_crate_alias = "host".to_string();
        let Some(prelude) = self.find_module(&["prelude".to_string()]) else {
            trace!("host crate exposes no `prelude` module; nothing to document");
            return RenderedApiSynopsis {
                api: "The host crate does not expose a `prelude` module, so `use host::prelude::*;` imports nothing. No host API is available beyond explicit `host::...` paths.\n".to_string(),
                external_facades: self.external_facades,
                rendered_items: self.rendered_items,
            };
        };

        let mut out = "The harness injects `use host::prelude::*;`. The following is the complete API imported by that statement. Items not shown are not available. Use these names unqualified and do not emit imports.\n\n```rust\n// Reachable host facade; bodies and large constant initializers are omitted.\n\n".to_string();
        self.write_module_items(&mut out, &prelude, 0);
        self.write_pending_reexports(&mut out);

        out.push_str("```\n");
        RenderedApiSynopsis {
            api: out,
            external_facades: self.external_facades,
            rendered_items: self.rendered_items,
        }
    }

    fn render_external_facade(
        mut self,
        crate_name: &str,
        requests: &BTreeSet<FacadeRequest>,
    ) -> RenderedApiSynopsis {
        self.local_crate_alias = crate_name.to_string();
        let mut body = String::new();
        for request in requests {
            match request {
                FacadeRequest::Module(path) => {
                    if let Some(module) = self.find_module(path) {
                        self.write_module_items(&mut body, &module, 0);
                    }
                }
                FacadeRequest::Item(path) => {
                    if let Some(item) = self.find_item(path) {
                        self.write_item(&mut body, &item, 0);
                    }
                }
            }
        }
        self.write_pending_reexports(&mut body);

        let api = if body.is_empty() {
            String::new()
        } else {
            format!(
                "Reachable items re-exported from `{crate_name}` (the crate path itself is not available):\n\n```rust\n// Use these items unqualified.\n\n{body}```\n"
            )
        };
        RenderedApiSynopsis {
            api,
            external_facades: self.external_facades,
            rendered_items: self.rendered_items,
        }
    }

    fn find_module(&self, path: &[String]) -> Option<Item> {
        if path.is_empty() {
            return self.crate_data.index.get(&self.crate_data.root).cloned();
        }
        self.find_item(path)
            .filter(|item| matches!(item.inner, ItemEnum::Module(_)))
    }

    fn find_item(&self, path: &[String]) -> Option<Item> {
        if let Some(item) = self.crate_data.paths.iter().find_map(|(id, summary)| {
            (summary.path.get(1..) == Some(path))
                .then(|| self.crate_data.index.get(id).cloned())
                .flatten()
        }) {
            return Some(item);
        }

        let mut item = self.crate_data.index.get(&self.crate_data.root)?.clone();
        for segment in path {
            let ItemEnum::Module(module) = &item.inner else {
                return None;
            };
            item = module.items.iter().find_map(|id| {
                self.crate_data
                    .index
                    .get(id)
                    .filter(|child| child.name.as_ref() == Some(segment))
                    .cloned()
            })?;
        }
        Some(item)
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
            ItemEnum::Use(use_item) => self.queue_use_targets(use_item),
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
            ItemEnum::Trait(trait_def) => {
                if let Some(snippet) = self.source_snippet(item) {
                    let rendered = self.elide_trait_items(&snippet, &trait_def.items);
                    self.write_snippet(out, &rendered, indent);
                }
            }
            ItemEnum::Macro(macro_source) => {
                write_doc_comment(out, item.docs.as_deref(), indent);
                self.write_snippet(out, macro_source, indent);
            }
            ItemEnum::TypeAlias(_) | ItemEnum::TraitAlias(_) => {
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

    /// Renders a trait declaration with associated item signatures, eliding
    /// default method bodies and associated constant initializers.
    fn elide_trait_items(&mut self, snippet: &str, item_ids: &[Id]) -> String {
        let header = match outer_brace_pair(snippet) {
            Some((open, _)) => snippet[..open].trim_end(),
            None => snippet.trim_end().trim_end_matches(';').trim_end(),
        }
        .to_string();

        let assoc_items = item_ids
            .iter()
            .filter_map(|id| self.crate_data.index.get(id))
            .filter_map(|assoc| {
                let assoc_snippet = self.source_snippet(assoc)?;
                match &assoc.inner {
                    ItemEnum::Function(_) => {
                        Some(elide_body(&assoc_snippet).unwrap_or(assoc_snippet))
                    }
                    _ => Some(elide_initializer(&assoc_snippet)),
                }
            })
            .collect::<Vec<_>>();

        let mut out = header;
        if assoc_items.is_empty() {
            out.push_str(" {}");
            return out;
        }
        out.push_str(" {\n");
        for assoc_item in assoc_items {
            for line in assoc_item.trim().lines() {
                out.push_str("    ");
                out.push_str(line.trim_start());
                out.push('\n');
            }
        }
        out.push('}');
        out
    }

    fn write_pending_reexports(&mut self, out: &mut String) {
        while let Some((id, flatten_module)) = self.pending_reexports.pop() {
            let Some(item) = self.crate_data.index.get(&id).cloned() else {
                continue;
            };
            if !is_public(&item) || (!flatten_module && self.rendered_items.contains(&id)) {
                continue;
            }
            if flatten_module {
                self.write_module_items(out, &item, 0);
            } else {
                self.write_item(out, &item, 0);
            }
        }
    }

    fn queue_reexport(&mut self, id: Id, flatten_module: bool) {
        if self.rendered_items.contains(&id) || !self.queued_reexports.insert((id, flatten_module))
        {
            return;
        }
        self.pending_reexports.push((id, flatten_module));
    }

    fn queue_use_targets(&mut self, use_item: &Use) {
        let Some(id) = use_item.id else {
            self.queue_unresolved_use(use_item);
            return;
        };
        let Some(summary) = self.crate_data.paths.get(&id) else {
            // The paths table has gaps (e.g. items reachable only through
            // re-exports); fall back to rendering the target from the local
            // index if present there.
            self.queue_reexport(id, use_item.is_glob);
            return;
        };
        let Some(external_crate) = self.crate_data.external_crates.get(&summary.crate_id) else {
            self.queue_reexport(id, use_item.is_glob);
            return;
        };
        if !is_documentable_external_crate(external_crate) {
            return;
        }

        let path = summary.path.iter().skip(1).cloned().collect::<Vec<_>>();
        let request = if use_item.is_glob {
            FacadeRequest::Module(path)
        } else {
            FacadeRequest::Item(path)
        };
        self.external_facades
            .entry(external_crate.name.clone())
            .or_default()
            .insert(request);
    }

    /// Fallback for `use` items rustdoc could not resolve to an id: derive the
    /// facade request from the source path of a glob import.
    fn queue_unresolved_use(&mut self, use_item: &Use) {
        if !use_item.is_glob {
            return;
        }
        let mut segments = use_item.source.split("::");
        let Some(crate_name) = segments.next() else {
            return;
        };
        if matches!(crate_name, "crate" | "self" | "super") || crate_name == self.local_crate_alias
        {
            return;
        }
        let is_documentable = self
            .crate_data
            .external_crates
            .values()
            .any(|external_crate| {
                external_crate.name == crate_name && is_documentable_external_crate(external_crate)
            });
        if !is_documentable {
            return;
        }
        let path = segments.map(str::to_string).collect::<Vec<_>>();
        self.external_facades
            .entry(crate_name.to_string())
            .or_default()
            .insert(FacadeRequest::Module(path));
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

    fn spanned_item(id: u32, name: Option<&str>, inner: ItemEnum, span: Span) -> Item {
        Item {
            span: Some(span),
            ..item(id, name, inner)
        }
    }

    fn span(filename: &Path, begin: (usize, usize), end: (usize, usize)) -> Span {
        Span {
            filename: filename.to_path_buf(),
            begin,
            end,
        }
    }

    fn temp_source_file(tag: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "symbiont_doc_string_{tag}_{}.rs",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("test source file is writable");
        path
    }

    fn function_decl() -> ItemEnum {
        ItemEnum::Function(rustdoc_types::Function {
            sig: rustdoc_types::FunctionSignature {
                inputs: Vec::new(),
                output: None,
                is_c_variadic: false,
            },
            generics: empty_generics(),
            header: rustdoc_types::FunctionHeader {
                is_const: false,
                is_unsafe: false,
                is_async: false,
                abi: rustdoc_types::Abi::Rust,
            },
            has_body: false,
        })
    }

    fn trait_decl(items: Vec<Id>) -> ItemEnum {
        ItemEnum::Trait(rustdoc_types::Trait {
            is_auto: false,
            is_unsafe: false,
            is_dyn_compatible: true,
            items,
            generics: empty_generics(),
            bounds: Vec::new(),
            implementations: Vec::new(),
        })
    }

    fn empty_generics() -> rustdoc_types::Generics {
        rustdoc_types::Generics {
            params: Vec::new(),
            where_predicates: Vec::new(),
        }
    }

    fn local_path_entry(
        path: &[&str],
        kind: rustdoc_types::ItemKind,
    ) -> rustdoc_types::ItemSummary {
        rustdoc_types::ItemSummary {
            crate_id: 0,
            path: path.iter().map(|s| s.to_string()).collect(),
            kind,
        }
    }

    fn crate_data(
        index: HashMap<Id, Item>,
        paths: HashMap<Id, rustdoc_types::ItemSummary>,
        external_crates: HashMap<u32, rustdoc_types::ExternalCrate>,
    ) -> RustdocCrate {
        RustdocCrate {
            root: Id(0),
            crate_version: None,
            includes_private: false,
            index,
            paths,
            external_crates,
            target: rustdoc_types::Target {
                triple: "x86_64-unknown-linux-gnu".to_string(),
                target_features: Vec::new(),
            },
            format_version: rustdoc_types::FORMAT_VERSION,
        }
    }

    fn prelude_crate_with(use_item: Use, extra_items: Vec<Item>) -> HashMap<Id, Item> {
        let mut index = HashMap::from([
            (
                Id(0),
                item(
                    0,
                    Some("host_crate"),
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
            (Id(2), item(2, None, ItemEnum::Use(use_item))),
        ]);
        for extra in extra_items {
            index.insert(extra.id, extra);
        }
        index
    }

    #[test]
    fn doc_string_host_render_documents_reexported_trait_with_elided_bodies() {
        let source = temp_source_file(
            "trait_view",
            "/// Observe a sliding window.\npub trait View {\n    /// Push a new value.\n    fn update(&mut self, val: f64);\n    fn last(&self) -> Option<f64> {\n        None\n    }\n}\n",
        );

        let index = prelude_crate_with(
            Use {
                source: "crate::traits::View".to_string(),
                name: "View".to_string(),
                id: Some(Id(3)),
                is_glob: false,
            },
            vec![
                spanned_item(
                    3,
                    Some("View"),
                    trait_decl(vec![Id(4), Id(5)]),
                    span(&source, (2, 1), (8, 2)),
                ),
                spanned_item(
                    4,
                    Some("update"),
                    function_decl(),
                    span(&source, (4, 5), (4, 200)),
                ),
                spanned_item(
                    5,
                    Some("last"),
                    function_decl(),
                    span(&source, (5, 5), (7, 6)),
                ),
            ],
        );
        let paths = HashMap::from([(
            Id(3),
            local_path_entry(
                &["host_crate", "traits", "View"],
                rustdoc_types::ItemKind::Trait,
            ),
        )]);
        let crate_data = crate_data(index, paths, HashMap::new());

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert!(rendered.api.contains("/// Observe a sliding window."));
        assert!(rendered.api.contains("pub trait View {"));
        assert!(rendered.api.contains("/// Push a new value."));
        assert!(rendered.api.contains("fn update(&mut self, val: f64);"));
        assert!(rendered.api.contains("fn last(&self) -> Option<f64>;"));
        assert!(!rendered.api.contains("None"));
    }

    #[test]
    fn doc_string_host_render_documents_reexported_macro_source() {
        let index = prelude_crate_with(
            Use {
                source: "crate::macros::ema".to_string(),
                name: "ema".to_string(),
                id: Some(Id(3)),
                is_glob: false,
            },
            vec![Item {
                docs: Some("Exponential moving average.".to_string()),
                ..item(
                    3,
                    Some("ema"),
                    ItemEnum::Macro("macro_rules! ema {\n    ($x:expr) => { ... };\n}".to_string()),
                )
            }],
        );
        let paths = HashMap::from([(
            Id(3),
            local_path_entry(
                &["host_crate", "macros", "ema"],
                rustdoc_types::ItemKind::Macro,
            ),
        )]);
        let crate_data = crate_data(index, paths, HashMap::new());

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert!(rendered.api.contains("/// Exponential moving average."));
        assert!(rendered.api.contains("macro_rules! ema {"));
    }

    #[test]
    fn doc_string_host_render_falls_back_to_source_path_for_unresolved_glob() {
        let index = prelude_crate_with(
            Use {
                source: "sliding_features::pure_functions".to_string(),
                name: "pure_functions".to_string(),
                id: None,
                is_glob: true,
            },
            Vec::new(),
        );
        let external_crates = HashMap::from([(
            1,
            rustdoc_types::ExternalCrate {
                name: "sliding_features".to_string(),
                html_root_url: None,
                path: PathBuf::from("target/debug/deps/libsliding_features.rmeta"),
            },
        )]);
        let crate_data = crate_data(index, HashMap::new(), external_crates);

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert_eq!(
            rendered.external_facades["sliding_features"],
            BTreeSet::from([FacadeRequest::Module(vec!["pure_functions".to_string()])])
        );
    }

    #[test]
    fn doc_string_host_render_documents_local_reexport_missing_from_paths() {
        let index = prelude_crate_with(
            Use {
                source: "crate::indicators".to_string(),
                name: "indicators".to_string(),
                id: Some(Id(3)),
                is_glob: false,
            },
            vec![item(
                3,
                Some("indicators"),
                ItemEnum::Module(rustdoc_types::Module {
                    is_crate: false,
                    items: Vec::new(),
                    is_stripped: false,
                }),
            )],
        );
        let crate_data = crate_data(index, HashMap::new(), HashMap::new());

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert!(rendered.api.contains("pub mod indicators"));
    }

    #[test]
    fn doc_string_host_render_flattens_local_glob_reexport_missing_from_paths() {
        let index = prelude_crate_with(
            Use {
                source: "crate::indicators".to_string(),
                name: "indicators".to_string(),
                id: Some(Id(3)),
                is_glob: true,
            },
            vec![
                item(
                    3,
                    Some("indicators"),
                    ItemEnum::Module(rustdoc_types::Module {
                        is_crate: false,
                        items: vec![Id(4)],
                        is_stripped: false,
                    }),
                ),
                item(
                    4,
                    Some("rsi"),
                    ItemEnum::Module(rustdoc_types::Module {
                        is_crate: false,
                        items: Vec::new(),
                        is_stripped: false,
                    }),
                ),
            ],
        );
        let crate_data = crate_data(index, HashMap::new(), HashMap::new());

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert!(rendered.api.contains("pub mod rsi"));
        assert!(!rendered.api.contains("pub mod indicators"));
    }

    #[test]
    fn doc_string_seeded_rendered_items_prevent_duplicate_facade_output() {
        let source = temp_source_file("dedup_struct", "pub struct RollingSum;\n");
        let index = HashMap::from([
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
                spanned_item(
                    1,
                    Some("RollingSum"),
                    ItemEnum::Struct(rustdoc_types::Struct {
                        kind: rustdoc_types::StructKind::Unit,
                        generics: empty_generics(),
                        impls: Vec::new(),
                    }),
                    span(&source, (1, 1), (1, 23)),
                ),
            ),
        ]);
        let paths = HashMap::from([(
            Id(1),
            local_path_entry(
                &["sliding_features", "RollingSum"],
                rustdoc_types::ItemKind::Struct,
            ),
        )]);
        let crate_data = crate_data(index, paths, HashMap::new());

        let first = RustApiSynopsis::new(&crate_data).render_external_facade(
            "sliding_features",
            &BTreeSet::from([FacadeRequest::Item(vec!["RollingSum".to_string()])]),
        );
        assert!(first.api.contains("pub struct RollingSum;"));
        assert!(first.rendered_items.contains(&Id(1)));

        let second = RustApiSynopsis::new(&crate_data)
            .with_rendered_items(first.rendered_items)
            .render_external_facade(
                "sliding_features",
                &BTreeSet::from([FacadeRequest::Module(Vec::new())]),
            );
        assert!(second.api.is_empty());
    }

    #[test]
    fn doc_string_host_render_reports_missing_prelude_module() {
        let index = HashMap::from([(
            Id(0),
            item(
                0,
                Some("host_crate"),
                ItemEnum::Module(rustdoc_types::Module {
                    is_crate: true,
                    items: Vec::new(),
                    is_stripped: false,
                }),
            ),
        )]);
        let crate_data = crate_data(index, HashMap::new(), HashMap::new());

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert!(
            rendered
                .api
                .contains("does not expose a `prelude` module")
        );
        assert!(!rendered.api.contains("complete API"));
        assert!(!rendered.api.contains("```rust"));
        assert!(rendered.external_facades.is_empty());
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

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert_eq!(
            rendered.external_facades["sliding_features"],
            BTreeSet::from([FacadeRequest::Module(Vec::new())])
        );
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

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert_eq!(
            rendered.external_facades["sliding_features"],
            BTreeSet::from([FacadeRequest::Item(vec!["RollingSum".to_string()])])
        );
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

        let rendered = RustApiSynopsis::new(&crate_data).render_host_facade();

        assert!(rendered.external_facades.is_empty());
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

        let rendered = RustApiSynopsis::new(&crate_data).render_external_facade(
            "sliding_features",
            &BTreeSet::from([FacadeRequest::Module(Vec::new())]),
        );

        assert!(rendered.external_facades.is_empty());
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
