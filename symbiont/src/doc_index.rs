// SPDX-License-Identifier: MPL-2.0
//! The [`DocIndex`]: a queryable cache of the rustdoc JSON of the host crate
//! and the crates that the host facade re-exports.

use std::{
    collections::{
        BTreeSet,
        HashMap,
    },
    sync::{
        Arc,
        Mutex,
        OnceLock,
        RwLock,
    },
};

use rustdoc_types::{
    Crate as RustdocCrate,
    Item,
    ItemEnum,
    ItemKind,
    Use,
};

use crate::{
    Error,
    Result,
    doc_string::{
        FacadeRequest,
        RustApiSynopsis,
        is_public,
        load_crate_data,
        render_external_facades,
        rustdoc_json,
    },
};

/// The error of a [`DocIndex`] query.
#[derive(Debug, thiserror::Error)]
pub enum DocIndexError {
    /// The host crate exposes no `prelude` module.
    #[error(
        "the host crate exposes no `prelude` module, so `use host::prelude::*;` imports nothing"
    )]
    NoPrelude,
    /// No public module exists at the requested path.
    #[error(
        "no public module `{0}` exists in the host API; call `api_index` without arguments to list the prelude"
    )]
    ModuleNotFound(String),
    /// No public item exists at the requested path.
    #[error(
        "no public item `{0}` exists in the host API; call `api_index` to list the available names"
    )]
    ItemNotFound(String),
    /// A thread panicked while it held the cache lock of the index.
    #[error("the doc index cache lock is poisoned")]
    PoisonedLock,
}

/// A queryable cache of the rustdoc JSON of the host crate and the crates
/// that the host facade re-exports.
///
/// [`DocIndex::host`] builds the index one time per process and crate name.
/// The rustdoc build is slow, so the result is cached. External crates load
/// lazily on the first query that names them. A crate that fails to load is
/// not loaded again.
pub struct DocIndex {
    host_crate: String,
    crates: Arc<RwLock<HashMap<String, Option<Arc<RustdocCrate>>>>>,
}

impl DocIndex {
    /// Get the shared index for `crate_name`.
    ///
    /// The first call for a crate name runs `cargo rustdoc` and caches the
    /// result. Later calls return the cached index.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot build or parse the rustdoc JSON
    /// of the host crate.
    pub async fn host(crate_name: &str) -> Result<Arc<Self>> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<DocIndex>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(index) = cache
            .lock()
            .map_err(|_| Error::MutexPoison)?
            .get(crate_name)
        {
            return Ok(Arc::clone(index));
        }

        let crate_data = rustdoc_json(crate_name, &[crate_name]).await?;
        let index = Arc::new(Self::new(
            crate_name,
            HashMap::from([(crate_name.to_string(), Some(Arc::new(crate_data)))]),
        ));
        cache
            .lock()
            .map_err(|_| Error::MutexPoison)?
            .insert(crate_name.to_string(), Arc::clone(&index));
        Ok(index)
    }

    fn new(host_crate: &str, crates: HashMap<String, Option<Arc<RustdocCrate>>>) -> Self {
        Self {
            host_crate: host_crate.to_string(),
            crates: Arc::new(RwLock::new(crates)),
        }
    }

    /// Build an index from prepared rustdoc data, for tests.
    #[cfg(test)]
    fn from_crates(
        host_crate: &str,
        host_data: RustdocCrate,
        externals: HashMap<String, RustdocCrate>,
    ) -> Self {
        let mut crates = HashMap::from([(host_crate.to_string(), Some(Arc::new(host_data)))]);
        for (name, data) in externals {
            crates.insert(name, Some(Arc::new(data)));
        }
        Self::new(host_crate, crates)
    }

    /// Render the compact index of a host module: one `kind name` line per
    /// public item.
    ///
    /// With `None`, this function lists the `prelude` module. A re-export
    /// line shows the kind and the name of the target and the crate it comes
    /// from.
    ///
    /// # Errors
    ///
    /// Returns an error if the module does not exist or is not public.
    pub fn render_index(
        &self,
        module_path: Option<&str>,
    ) -> std::result::Result<String, DocIndexError> {
        let host = self.host_data()?;
        let path = match module_path {
            None => vec!["prelude".to_string()],
            Some(path) => normalize_path(path, &self.host_crate),
        };
        let module =
            RustApiSynopsis::new(&host)
                .find_module(&path)
                .ok_or_else(|| match module_path {
                    None => DocIndexError::NoPrelude,
                    Some(path) => DocIndexError::ModuleNotFound(path.to_string()),
                })?;
        Ok(render_module_index(&host, &module))
    }

    /// Render the full synopsis of the host item or module at `path`: the
    /// definition, the inherent methods, and the operator impls.
    ///
    /// The path is `::`-separated and relative to the host crate root. A
    /// leading `host` segment is removed. A single segment resolves against
    /// the names that `use host::prelude::*;` imports. A re-export renders
    /// the target item and loads the rustdoc JSON of its crate lazily.
    ///
    /// # Errors
    ///
    /// Returns an error if no public item or module exists at `path`.
    pub async fn render_doc(&self, path: &str) -> std::result::Result<String, DocIndexError> {
        let segments = normalize_path(path, &self.host_crate);
        let host = self.host_data()?;
        let request = find_request(&host, &segments)
            .ok_or_else(|| DocIndexError::ItemNotFound(path.to_string()))?;

        let mut out = String::new();
        let rendered = RustApiSynopsis::new(&host).render_requests(&BTreeSet::from([request]));
        out.push_str(&rendered.api);
        let crates = Arc::clone(&self.crates);
        render_external_facades(
            &mut out,
            rendered.external_facades,
            move |crate_name: String| {
                let crates = Arc::clone(&crates);
                async move { load_crate_data(&crates, &crate_name).await }
            },
        )
        .await;

        if out.trim().is_empty() {
            return Err(DocIndexError::ItemNotFound(path.to_string()));
        }
        Ok(out.trim_end().to_string())
    }

    /// Get the cached rustdoc data of the host crate.
    fn host_data(&self) -> std::result::Result<Arc<RustdocCrate>, DocIndexError> {
        self.crates
            .read()
            .map_err(|_| DocIndexError::PoisonedLock)?
            .get(&self.host_crate)
            .and_then(Option::as_ref)
            .map(Arc::clone)
            .ok_or(DocIndexError::PoisonedLock)
    }
}

/// Split a `::`-separated path into segments. Remove a leading `host`
/// segment or a leading segment with the host crate name.
fn normalize_path(path: &str, host_crate: &str) -> Vec<String> {
    let mut segments: Vec<String> = path
        .split("::")
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect();
    if segments
        .first()
        .is_some_and(|first| first == "host" || first == host_crate)
    {
        segments.remove(0);
    }
    segments
}

/// Find the facade request for `segments` in the host crate. A single
/// segment resolves against the children of the `prelude` module.
fn find_request(host: &RustdocCrate, segments: &[String]) -> Option<FacadeRequest> {
    if segments.is_empty() {
        return None;
    }
    let synopsis = RustApiSynopsis::new(host);
    if let Some(item) = synopsis.find_item(segments) {
        return Some(request_for(host, &item, segments.to_vec()));
    }
    if segments.len() == 1 {
        let prelude = synopsis.find_module(&["prelude".to_string()])?;
        let ItemEnum::Module(prelude) = &prelude.inner else {
            return None;
        };
        let item = prelude
            .items
            .iter()
            .filter_map(|id| host.index.get(id))
            .find(|item| item.name.as_deref() == Some(segments[0].as_str()))?;
        return Some(request_for(
            host,
            item,
            vec!["prelude".to_string(), segments[0].clone()],
        ));
    }
    None
}

/// Build the request that renders `item`. `fallback_path` is the path of
/// `item` relative to the crate root, for items missing from the paths
/// table.
fn request_for(host: &RustdocCrate, item: &Item, fallback_path: Vec<String>) -> FacadeRequest {
    let path = host
        .paths
        .get(&item.id)
        .map(|summary| summary.path.iter().skip(1).cloned().collect())
        .unwrap_or(fallback_path);
    if matches!(item.inner, ItemEnum::Module(_)) {
        FacadeRequest::Module(path)
    } else {
        FacadeRequest::Item(path)
    }
}

/// Render the public children of `module` as one `kind name` line per item.
fn render_module_index(crate_data: &RustdocCrate, module: &Item) -> String {
    let ItemEnum::Module(module) = &module.inner else {
        return String::new();
    };
    let mut out = String::new();
    for item_id in &module.items {
        let Some(item) = crate_data.index.get(item_id) else {
            continue;
        };
        if !is_public(item) {
            continue;
        }
        if let Some(line) = index_line(crate_data, item) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out
}

/// Render one line of the compact index for `item`.
fn index_line(crate_data: &RustdocCrate, item: &Item) -> Option<String> {
    let name = item.name.as_deref()?;
    Some(match &item.inner {
        ItemEnum::Use(use_item) => reexport_line(crate_data, use_item, name),
        inner => format!("{} {name}", kind_str(inner.item_kind())),
    })
}

/// Render the index line of a `use` item. Resolve the target to show its
/// kind and the crate it comes from.
fn reexport_line(crate_data: &RustdocCrate, use_item: &Use, name: &str) -> String {
    let Some(summary) = use_item.id.and_then(|id| crate_data.paths.get(&id)) else {
        return format!("use {}", use_item.source);
    };
    let kind = kind_str(summary.kind);
    let origin = crate_data.external_crates.get(&summary.crate_id);
    if use_item.is_glob {
        return match origin {
            Some(external) => format!("mod {name} (glob re-export from `{}`)", external.name),
            None => format!("mod {name} (glob re-export)"),
        };
    }
    match origin {
        Some(external) => format!("{kind} {name} (re-exported from `{}`)", external.name),
        None => format!("{kind} {name}"),
    }
}

/// Map an item kind to the keyword of its declaration.
fn kind_str(kind: ItemKind) -> &'static str {
    match kind {
        ItemKind::Module => "mod",
        ItemKind::Struct => "struct",
        ItemKind::Union => "union",
        ItemKind::Enum => "enum",
        ItemKind::Function => "fn",
        ItemKind::Trait | ItemKind::TraitAlias => "trait",
        ItemKind::TypeAlias => "type",
        ItemKind::Constant | ItemKind::AssocConst => "const",
        ItemKind::Static => "static",
        ItemKind::Macro | ItemKind::ProcAttribute | ItemKind::ProcDerive => "macro",
        _ => "item",
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use rustdoc_types::{
        Abi,
        ExternalCrate,
        Function,
        FunctionHeader,
        FunctionSignature,
        Generics,
        Id,
        ItemSummary,
        Module,
        Visibility,
    };

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
            stability: None,
            const_stability: None,
            inner,
        }
    }

    fn module(id: u32, name: &str, items: Vec<Id>) -> Item {
        item(
            id,
            Some(name),
            ItemEnum::Module(Module {
                is_crate: id == 0,
                items,
                is_stripped: false,
            }),
        )
    }

    fn function(id: u32, name: &str) -> Item {
        item(
            id,
            Some(name),
            ItemEnum::Function(Function {
                sig: FunctionSignature {
                    inputs: Vec::new(),
                    output: None,
                    is_c_variadic: false,
                },
                generics: Generics {
                    params: Vec::new(),
                    where_predicates: Vec::new(),
                },
                header: FunctionHeader {
                    is_const: false,
                    is_unsafe: false,
                    is_async: false,
                    abi: Abi::Rust,
                },
                has_body: true,
                default_unstable: None,
            }),
        )
    }

    fn crate_data(
        index: HashMap<Id, Item>,
        paths: HashMap<Id, ItemSummary>,
        external_crates: HashMap<u32, ExternalCrate>,
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

    /// The host crate: a `prelude` module with one external re-export, one
    /// local function, and one nested module with a function.
    fn host_fixture() -> RustdocCrate {
        let index = HashMap::from([
            (Id(0), module(0, "host_crate", vec![Id(1)])),
            (Id(1), module(1, "prelude", vec![Id(2), Id(3), Id(4)])),
            (
                Id(2),
                item(
                    2,
                    Some("decimal"),
                    ItemEnum::Use(Use {
                        source: "dep_crate::decimal".to_string(),
                        name: "decimal".to_string(),
                        id: Some(Id(100)),
                        is_glob: false,
                    }),
                ),
            ),
            (Id(3), function(3, "submit_order")),
            (Id(4), module(4, "indicators", vec![Id(5)])),
            (Id(5), function(5, "zscore")),
        ]);
        let paths = HashMap::from([(
            Id(100),
            ItemSummary {
                crate_id: 1,
                path: vec!["dep_crate".to_string(), "decimal".to_string()],
                kind: ItemKind::Macro,
            },
        )]);
        let external_crates = HashMap::from([(
            1,
            ExternalCrate {
                name: "dep_crate".to_string(),
                html_root_url: None,
                path: std::path::PathBuf::new(),
            },
        )]);
        crate_data(index, paths, external_crates)
    }

    /// The external crate: one macro at the path that the host re-exports.
    fn dep_fixture() -> RustdocCrate {
        let index = HashMap::from([
            module(0, "dep_crate", vec![Id(100)]).into_entry(),
            item(
                100,
                Some("decimal"),
                ItemEnum::Macro("macro_rules! decimal { () => {} }".to_string()),
            )
            .into_entry(),
        ]);
        let paths = HashMap::from([(
            Id(100),
            ItemSummary {
                crate_id: 0,
                path: vec!["dep_crate".to_string(), "decimal".to_string()],
                kind: ItemKind::Macro,
            },
        )]);
        crate_data(index, paths, HashMap::new())
    }

    pub(crate) fn fixture_index() -> DocIndex {
        DocIndex::from_crates(
            "host_crate",
            host_fixture(),
            HashMap::from([("dep_crate".to_string(), dep_fixture())]),
        )
    }

    trait IntoEntry {
        fn into_entry(self) -> (Id, Item);
    }

    impl IntoEntry for Item {
        fn into_entry(self) -> (Id, Item) {
            (self.id, self)
        }
    }

    #[test]
    fn render_index_lists_prelude_children_with_kinds_and_origins() {
        let index = fixture_index();
        let listing = index.render_index(None).expect("prelude exists");
        assert!(listing.contains("macro decimal (re-exported from `dep_crate`)"));
        assert!(listing.contains("fn submit_order"));
        assert!(listing.contains("mod indicators"));
    }

    #[test]
    fn render_index_expands_nested_module() {
        let index = fixture_index();
        let listing = index
            .render_index(Some("prelude::indicators"))
            .expect("module exists");
        assert_eq!(listing, "fn zscore\n");
        // A leading `host` segment or the host crate name is removed.
        let listing = index
            .render_index(Some("host::prelude::indicators"))
            .expect("module exists");
        assert_eq!(listing, "fn zscore\n");
    }

    #[test]
    fn render_index_unknown_module_errors() {
        let index = fixture_index();
        let err = index.render_index(Some("nope")).expect_err("the path does not resolve");
        assert!(matches!(err, DocIndexError::ModuleNotFound(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn render_doc_renders_local_function_by_prelude_name() {
        let index = fixture_index();
        let doc = index.render_doc("submit_order").await.expect("item exists");
        assert!(doc.contains("pub fn submit_order()"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn render_doc_follows_reexport_into_external_crate() {
        let index = fixture_index();
        let doc = index.render_doc("decimal").await.expect("item exists");
        assert!(doc.contains("macro_rules! decimal"));
        assert!(doc.contains("dep_crate"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn render_doc_unknown_path_errors() {
        let index = fixture_index();
        let err = index.render_doc("nope").await.expect_err("the path does not resolve");
        assert!(matches!(err, DocIndexError::ItemNotFound(_)));
    }
}
