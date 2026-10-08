// SPDX-License-Identifier: MPL-2.0
//! The [`DocIndex`]: a queryable cache of the rustdoc JSON of the host crate
//! and the crates that the host facade re-exports.
//!
//! The cache is complete when [`DocIndex::host`] returns. A query reads it
//! and does no I/O, because a tool call must not run `cargo rustdoc` while
//! the runtime compiles a dylib with the same cargo lock.

use std::{
    collections::{
        BTreeSet,
        HashMap,
        HashSet,
    },
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
        OnceLock,
    },
};

use rustdoc_types::{
    Crate as RustdocCrate,
    Id,
    Item,
    ItemEnum,
    ItemKind,
    Use,
};
use tokio::sync::OnceCell;

use crate::{
    Error,
    Result,
    doc_string::{
        FacadeRequest,
        RustApiSynopsis,
        cargo_workspace_root,
        exported_name as item_name,
        is_public,
        load_external_facades,
        render_cached_external_facades,
        rustdoc_json,
    },
};

/// The cache slot of one crate name: empty until the first build finishes.
type IndexCell = OnceCell<Arc<DocIndex>>;

/// The error of a [`DocIndex`] query.
#[derive(Debug, thiserror::Error)]
pub enum DocIndexError {
    /// The host crate exposes no `prelude` module.
    #[error(
        "the host crate exposes no `prelude` module, so `use host::prelude::*;` imports nothing"
    )]
    NoPrelude,
    /// No public module exists at the requested path. The second field
    /// holds the closest names the host API has.
    #[error(
        "no public module `{0}` exists in the host API{1}; call `api_index` without arguments to list the prelude"
    )]
    ModuleNotFound(String, Closest),
    /// No public item exists at the requested path. The second field holds
    /// the closest names the host API has.
    #[error(
        "no public item `{0}` exists in the host API{1}; call `api_index` to list the available names"
    )]
    ItemNotFound(String, Closest),
    /// The item exists, but the renderer produced nothing for it: the source
    /// file behind its rustdoc span could not be read.
    #[error(
        "public item `{0}` exists in the host API, but its definition could not be rendered: its source file could not be read; the name is valid, so use it without the full definition"
    )]
    RenderFailed(String),
}

/// The names of the host API closest to a path that does not resolve, as
/// the suffix of a not-found message. Empty when nothing is close.
///
/// A model that misspells a name, or guesses a related one, otherwise needs
/// an `api_index` call and a second `api_doc` call - two turns - to find the
/// name it meant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Closest(Vec<String>);

impl Closest {
    /// The suggested names, closest first.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.0
    }
}

impl std::fmt::Display for Closest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some((first, rest)) = self.0.split_first() else {
            return Ok(());
        };
        write!(f, "; the closest names are `{first}`")?;
        for name in rest {
            write!(f, ", `{name}`")?;
        }
        Ok(())
    }
}

/// The most names a not-found message suggests.
const MAX_SUGGESTIONS: usize = 5;

/// A queryable cache of the rustdoc JSON of the host crate and the crates
/// that the host facade re-exports.
///
/// [`DocIndex::host`] builds the index one time per process and crate name.
/// It runs `cargo rustdoc` for the host crate and for every crate that the
/// host facade can reach, then keeps the result. A query reads this cache
/// and does no I/O, so a tool call costs no `cargo rustdoc` run and does not
/// compete for the cargo lock with a dylib build.
pub struct DocIndex {
    host_crate: String,
    /// The rustdoc JSON of the host crate and of every reachable crate that
    /// loaded. This map never changes after [`DocIndex::host`] returns.
    crates: HashMap<String, Arc<RustdocCrate>>,
    /// The cargo workspace root that the rustdoc spans of the host crate are
    /// recorded relative to. `None` for indexes that build their rustdoc data
    /// by hand (tests), where spans are already absolute or absent.
    workspace_root: Option<PathBuf>,
}

impl DocIndex {
    /// Get the shared index for `crate_name`.
    ///
    /// The first call for a crate name builds the rustdoc JSON of the host
    /// crate and of every crate that the host facade re-exports, which is
    /// slow. Later calls return the cached index. Concurrent first calls for
    /// one crate name await the same build; only a failed build is retried.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot build or parse the rustdoc JSON
    /// of the host crate.
    pub async fn host(crate_name: &str) -> Result<Arc<Self>> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<IndexCell>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        Self::cached(cache, crate_name, Self::build_shared(crate_name)).await
    }

    /// Get the entry of `crate_name` from `cache`, running `build` if the
    /// cache has none.
    ///
    /// The map lock only hands out the cell of one crate name. The build runs
    /// under that cell, so concurrent first callers await one build instead
    /// of each running `cargo rustdoc`, and a build for another crate name is
    /// not blocked. `build` is dropped unpolled if another caller wins.
    async fn cached(
        cache: &Mutex<HashMap<String, Arc<IndexCell>>>,
        crate_name: &str,
        build: impl Future<Output = Result<Arc<Self>>>,
    ) -> Result<Arc<Self>> {
        let cell = {
            let mut crates = cache.lock().map_err(|_| Error::MutexPoison)?;
            Arc::clone(crates.entry(crate_name.to_string()).or_default())
        };

        // A failed build leaves the cell empty, so a later call retries it.
        let index = cell.get_or_try_init(move || build).await?;
        Ok(Arc::clone(index))
    }

    /// [`Self::build`] as the shared handle that the cache stores.
    async fn build_shared(crate_name: &str) -> Result<Arc<Self>> {
        Self::build(crate_name).await.map(Arc::new)
    }

    /// Build the index and warm the cache for every reachable crate.
    ///
    /// The host facade render names the external crates that a query can
    /// reach. This function loads all of them, and the crates that their own
    /// re-exports reach, so that no query needs I/O. The rendered text is not
    /// kept: the system prompt of a tool mode does not embed it.
    async fn build(crate_name: &str) -> Result<Self> {
        let host_data = Arc::new(rustdoc_json(crate_name, &[crate_name]).await?);
        let workspace_root = cargo_workspace_root().await?;
        let rendered = RustApiSynopsis::new(&host_data)
            .with_workspace_root(workspace_root.clone())
            .render_host_facade();
        let mut discarded = String::new();
        let mut crates = load_external_facades(&mut discarded, rendered.external_facades).await;
        crates.insert(crate_name.to_string(), host_data);
        Ok(Self {
            host_crate: crate_name.to_string(),
            crates,
            workspace_root: Some(workspace_root),
        })
    }

    /// Build an index from prepared rustdoc data, for tests.
    #[cfg(test)]
    fn from_crates(
        host_crate: &str,
        host_data: RustdocCrate,
        externals: HashMap<String, RustdocCrate>,
    ) -> Self {
        let mut crates = HashMap::from([(host_crate.to_string(), Arc::new(host_data))]);
        for (name, data) in externals {
            crates.insert(name, Arc::new(data));
        }
        Self {
            host_crate: host_crate.to_string(),
            crates,
            workspace_root: None,
        }
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
        let module = match module_path {
            None => self.prelude().ok_or(DocIndexError::NoPrelude)?,
            Some(path) => {
                let segments = normalize_path(&clean_query(path), &self.host_crate);
                match self.resolve_module(&segments) {
                    Some(module) => module,
                    None => return self.render_crate_index(path, &segments),
                }
            }
        };
        Ok(self.render_module_index(&module))
    }

    /// The index of a module path that names a dependency crate instead of
    /// a host module: the prelude items that come from that crate.
    ///
    /// The prelude lists its re-exports with their origin crate, so models
    /// ask for the crate (`api_index lfest`), which no host path names.
    fn render_crate_index(
        &self,
        path: &str,
        segments: &[String],
    ) -> std::result::Result<String, DocIndexError> {
        let not_found = || {
            DocIndexError::ModuleNotFound(
                path.to_string(),
                self.closest(segments, NameKind::Module),
            )
        };
        let [crate_name] = segments else {
            return Err(not_found());
        };
        let prelude = self.prelude().ok_or_else(not_found)?;
        let marker = format!("(re-exported from `{crate_name}`)");
        let listing = String::from_iter(
            self.render_module_index(&prelude)
                .lines()
                .filter(|line| line.ends_with(&marker))
                .map(|line| format!("{line}\n")),
        );
        if listing.is_empty() {
            return Err(not_found());
        }
        Ok(format!(
            "`{crate_name}` is not a module of the host API. These prelude items come from it; \
             use them unqualified:\n{listing}"
        ))
    }

    /// Render the full synopsis of the host item or module at `path`: the
    /// definition, the inherent methods, and the operator impls.
    ///
    /// The path is `::`-separated and relative to the host crate root. A
    /// leading `host` segment is removed. A single segment resolves against
    /// the names that `use host::prelude::*;` imports. A re-export renders
    /// the target item from the cached rustdoc JSON of its crate.
    ///
    /// This function reads the cache and does no I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if no public item or module exists at `path`, or if
    /// the item exists but the renderer produced nothing for it.
    pub fn render_doc(&self, path: &str) -> std::result::Result<String, DocIndexError> {
        let not_found = || DocIndexError::ItemNotFound(path.to_string(), Closest::default());
        let segments = normalize_path(&clean_query(path), &self.host_crate);
        let Some((located, note)) = self.resolve_lenient(&segments) else {
            return Err(DocIndexError::ItemNotFound(
                path.to_string(),
                self.closest(&segments, NameKind::Any),
            ));
        };
        let crate_data = self
            .crate_data(&located.crate_name)
            .ok_or_else(not_found)?
            .clone();
        let request = request_for(&crate_data, &located).ok_or_else(not_found)?;
        let requests = BTreeSet::from([request]);

        let mut synopsis = RustApiSynopsis::new(&crate_data);
        if let Some(workspace_root) = &self.workspace_root {
            synopsis = synopsis.with_workspace_root(workspace_root.clone());
        }
        let rendered = if located.crate_name == self.host_crate {
            synopsis.render_requests(&requests)
        } else {
            synopsis.render_external_facade(&located.crate_name, &requests)
        };

        let mut out = rendered.api;
        render_cached_external_facades(&mut out, rendered.external_facades, &self.crates);

        if out.trim().is_empty() {
            // The path resolved: the item exists. An empty render means its
            // source could not be read, and the model must not be told the
            // item is missing, because `api_index` lists it.
            return Err(DocIndexError::RenderFailed(path.to_string()));
        }
        match note {
            Some(note) => Ok(format!("Note: {note}\n\n{}", out.trim())),
            None => Ok(out.trim_end().to_string()),
        }
    }

    /// Resolve a path the way a model writes it, with the reason when it
    /// had to be read as another path.
    ///
    /// In order: the path as given; the path with leading segments dropped
    /// (`lfest::Account` names the prelude's `Account`: models qualify names
    /// with the crate the prelude listing says they come from, and in 0.41.0
    /// of agent-symbiont 829 of 19,267 lookups failed on that); for
    /// `Type::member`, the type, whose definition lists its methods and
    /// fields; and a name that differs only in case, if exactly one does.
    fn resolve_lenient(&self, segments: &[String]) -> Option<(Located, Option<String>)> {
        if let Some(located) = self.resolve(segments) {
            return Some((located, None));
        }
        let asked = segments.join("::");
        for start in 1..segments.len() {
            let rest = &segments[start..];
            if let Some(located) = self.resolve(rest) {
                let shown = rest.join("::");
                return Some((
                    located,
                    Some(format!(
                        "`{asked}` is not a path of the host API; this is `{shown}`, which is in \
                         scope as written here. Use `{shown}`, not `{asked}`."
                    )),
                ));
            }
        }
        if let Some((member, owner)) = segments.split_last()
            && !owner.is_empty()
            && let Some((located, _)) = self.resolve_lenient(owner)
            && let Some(target) = self.follow_reexports(located.clone())
            && is_type_like(&target.item.inner)
        {
            let owner = owner.join("::");
            return Some((
                located,
                Some(format!(
                    "`{asked}` is no item of its own; `{member}` would be a member of \
                     `{owner}`, whose definition with all its methods follows."
                )),
            ));
        }
        let [name] = segments else {
            return None;
        };
        let mut same_case = self
            .known_names(NameKind::Any)
            .into_iter()
            .filter(|known| known.eq_ignore_ascii_case(name));
        let known = same_case.next()?;
        if same_case.next().is_some() {
            return None;
        }
        let located = self.resolve(std::slice::from_ref(&known))?;
        Some((
            located,
            Some(format!("`{name}` is spelled `{known}` in the host API.")),
        ))
    }

    /// The names of the host API closest to the last segment of `segments`,
    /// best first: equal but for case, then containing it or contained in
    /// it, then by edit distance, which must stay within a third of the
    /// name's length.
    fn closest(&self, segments: &[String], kind: NameKind) -> Closest {
        let Some(wanted) = segments.last() else {
            return Closest::default();
        };
        let wanted = wanted.to_ascii_lowercase();
        let max_distance = (wanted.chars().count() / 3).max(1);
        let mut scored: Vec<(usize, String)> = self
            .known_names(kind)
            .into_iter()
            .filter_map(|known| {
                let last = known
                    .rsplit("::")
                    .next()
                    .unwrap_or(&known)
                    .to_ascii_lowercase();
                let score = if last == wanted {
                    0
                } else if last.contains(&wanted) || wanted.contains(&last) {
                    1
                } else {
                    let distance = edit_distance(&last, &wanted);
                    if distance > max_distance {
                        return None;
                    }
                    1 + distance
                };
                Some((score, known))
            })
            .collect();
        scored.sort();
        scored.dedup_by(|a, b| a.1 == b.1);
        Closest(
            scored
                .into_iter()
                .take(MAX_SUGGESTIONS)
                .map(|(_, name)| name)
                .collect(),
        )
    }

    /// Every name a lookup can resolve: the prelude's names, the root's
    /// names, and one level into the modules of both, as paths.
    fn known_names(&self, kind: NameKind) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        let tops = self
            .prelude()
            .into_iter()
            .chain(std::iter::once(self.host_root()));
        for top in tops {
            for child in self.children(&top) {
                // The prelude is listed as a top of its own, unqualified.
                let Some(name) = item_name(&child.item)
                    .filter(|name| *name != "prelude")
                    .map(str::to_string)
                else {
                    continue;
                };
                let module = self
                    .follow_reexports(child)
                    .filter(|target| matches!(target.item.inner, ItemEnum::Module(_)));
                match module {
                    Some(module) => {
                        for grandchild in self.children(&module) {
                            let inner_is_module =
                                matches!(grandchild.item.inner, ItemEnum::Module(_));
                            if let Some(inner) = item_name(&grandchild.item)
                                && kind.admits(inner_is_module)
                            {
                                names.insert(format!("{name}::{inner}"));
                            }
                        }
                        names.insert(name);
                    }
                    None if kind.admits(false) => {
                        names.insert(name);
                    }
                    None => {}
                }
            }
        }
        names
    }

    /// Get the cached rustdoc data of the host crate.
    fn host_data(&self) -> &RustdocCrate {
        self.crates
            .get(&self.host_crate)
            .expect("the constructors always insert the host crate")
    }

    /// Get the cached rustdoc data of a crate.
    fn crate_data(&self, crate_name: &str) -> Option<&Arc<RustdocCrate>> {
        self.crates.get(crate_name)
    }

    /// The `prelude` module of the host crate.
    ///
    /// This walks the host root directly instead of going through
    /// [`Self::resolve`], which asks for the prelude itself.
    fn prelude(&self) -> Option<Located> {
        let located = self.walk(self.host_root(), &["prelude".to_string()])?;
        let located = self.follow_reexports(located)?;
        matches!(located.item.inner, ItemEnum::Module(_)).then_some(located)
    }

    /// The root module of the host crate.
    fn host_root(&self) -> Located {
        let host = self.host_data();
        Located {
            crate_name: self.host_crate.clone(),
            path: Vec::new(),
            item: host
                .index
                .get(&host.root)
                .cloned()
                .unwrap_or_else(|| empty_root(host.root)),
        }
    }

    /// Resolve a normalized path to the item that it names.
    ///
    /// A one-segment path resolves against the `prelude` module first,
    /// because the generated code only has `use host::prelude::*;` in scope.
    /// A root item of the same spelling is a different item to the compiler,
    /// so the prelude name must win, as it does in Rust.
    ///
    /// Every other path walks the host crate root, then the `prelude` module,
    /// so that a name imported by `use host::prelude::*;` resolves without
    /// its module path. The last step is the paths table of the host crate,
    /// which holds items that no walk reaches.
    ///
    /// The result is the item under the name, which can be a `use` item. Call
    /// [`Self::follow_reexports`] to get the declaration behind it.
    fn resolve(&self, segments: &[String]) -> Option<Located> {
        if segments.is_empty() {
            return None;
        }
        let from_prelude = || {
            self.prelude()
                .and_then(|prelude| self.walk(prelude, segments))
        };
        if segments.len() == 1
            && let Some(located) = from_prelude()
        {
            return Some(located);
        }
        if let Some(located) = self.walk(self.host_root(), segments) {
            return Some(located);
        }
        if let Some(located) = from_prelude() {
            return Some(located);
        }
        let item = RustApiSynopsis::new(self.host_data()).find_item(segments)?;
        Some(Located {
            crate_name: self.host_crate.clone(),
            path: segments.to_vec(),
            item,
        })
    }

    /// Resolve a normalized path to the module that it names.
    fn resolve_module(&self, segments: &[String]) -> Option<Located> {
        self.resolve(segments)
            .and_then(|located| self.follow_reexports(located))
            .filter(|located| matches!(located.item.inner, ItemEnum::Module(_)))
    }

    /// Walk `segments` down from `start`, one name per segment.
    ///
    /// Every segment except the last must name a module. A segment that names
    /// a re-export of a module walks through it.
    fn walk(&self, start: Located, segments: &[String]) -> Option<Located> {
        let mut current = start;
        for (position, segment) in segments.iter().enumerate() {
            let child = self
                .children(&current)
                .into_iter()
                .find(|child| item_name(&child.item) == Some(segment.as_str()))?;
            if position + 1 == segments.len() {
                return Some(child);
            }
            current = self.follow_reexports(child)?;
        }
        Some(current)
    }

    /// The public children of a module, with the glob re-exports expanded.
    ///
    /// `pub use dep::prelude::*;` puts the names of that module in scope. It
    /// does not put the name `prelude` in scope. The listing must therefore
    /// show the names behind the glob, not the glob itself.
    ///
    /// A name that the module declares hides a name of the same spelling
    /// from a glob, as it does in Rust.
    fn children(&self, module: &Located) -> Vec<Located> {
        let mut children = Vec::new();
        self.collect_children(module, &mut children, &mut HashSet::new());
        let mut seen = HashSet::new();
        children.retain(|child| match item_name(&child.item) {
            Some(name) => seen.insert(name.to_string()),
            None => false,
        });
        children
    }

    /// Collect the public children of a module and of every module that it
    /// re-exports with a glob. `visited` stops a cycle of glob re-exports.
    fn collect_children(
        &self,
        module: &Located,
        out: &mut Vec<Located>,
        visited: &mut HashSet<(String, Id)>,
    ) {
        if !visited.insert((module.crate_name.clone(), module.item.id)) {
            return;
        }
        let Some(crate_data) = self.crate_data(&module.crate_name) else {
            return;
        };
        let ItemEnum::Module(inner) = &module.item.inner else {
            return;
        };

        let mut globs = Vec::new();
        for item in inner
            .items
            .iter()
            .filter_map(|id| crate_data.index.get(id))
            .filter(|item| is_public(item))
        {
            match &item.inner {
                ItemEnum::Use(use_item) if use_item.is_glob => globs.push((item, use_item)),
                _ if item_name(item).is_some() => out.push(module.child(item.clone())),
                _ => {}
            }
        }

        for (item, use_item) in globs {
            match self.follow_use(module, use_item) {
                Some(target) => self.collect_children(&target, out, visited),
                // The target crate has no cached rustdoc JSON. Keep the glob
                // itself so that the listing reports the gap.
                None => out.push(module.child(item.clone())),
            }
        }
    }

    /// Follow a chain of re-exports to the item that declares the name.
    ///
    /// A `use` item carries no declaration. The index needs the kind of the
    /// target, which can live in another crate.
    fn follow_reexports(&self, located: Located) -> Option<Located> {
        let mut current = located;
        for _ in 0..MAX_REEXPORT_DEPTH {
            let ItemEnum::Use(use_item) = &current.item.inner else {
                return Some(current);
            };
            current = self.follow_use(&current, use_item)?;
        }
        None
    }

    /// The target of one `use` item, in the crate that declares it.
    fn follow_use(&self, owner: &Located, use_item: &Use) -> Option<Located> {
        let owner_data = self.crate_data(&owner.crate_name)?;
        let id = use_item.id?;
        if let Some(summary) = owner_data.paths.get(&id)
            && let Some(external) = owner_data.external_crates.get(&summary.crate_id)
        {
            let target_data = self.crate_data(&external.name)?;
            let path = Vec::from_iter(summary.path.iter().skip(1).cloned());
            let item = RustApiSynopsis::new(target_data).find_item(&path)?;
            return Some(Located {
                crate_name: external.name.clone(),
                path,
                item,
            });
        }
        let item = owner_data.index.get(&id)?.clone();
        let path = match owner_data.paths.get(&id) {
            Some(summary) => Vec::from_iter(summary.path.iter().skip(1).cloned()),
            None => owner.path.clone(),
        };
        Some(Located {
            crate_name: owner.crate_name.clone(),
            path,
            item,
        })
    }

    /// Render the public children of a module as one `kind name` line each.
    fn render_module_index(&self, module: &Located) -> String {
        let mut out = String::new();
        for child in self.children(module) {
            if let Some(line) = self.index_line(&child) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        out
    }

    /// Render one line of the compact index.
    ///
    /// The name is the one that the module exports. A re-export shows the
    /// kind of its target and the crate that declares it.
    fn index_line(&self, child: &Located) -> Option<String> {
        let name = item_name(&child.item)?;
        let ItemEnum::Use(use_item) = &child.item.inner else {
            let kind = kind_str(child.item.inner.item_kind());
            // A child from a glob re-export already is the declaration. Name
            // the crate it comes from, as a named re-export does.
            let origin = (child.crate_name != self.host_crate).then_some(child.crate_name.as_str());
            return Some(reexport_line(kind, name, origin));
        };
        if use_item.is_glob {
            return Some(self.glob_index_line(child, use_item, name));
        }
        match self.follow_reexports(child.clone()) {
            Some(target) => Some(reexport_line(
                kind_str(target.item.inner.item_kind()),
                name,
                (target.crate_name != child.crate_name).then_some(target.crate_name.as_str()),
            )),
            // The target crate has no cached rustdoc JSON. The paths table of
            // the owner still gives the kind and the origin.
            None => Some(self.summary_index_line(child, use_item, name)),
        }
    }

    /// Render the index line of a glob re-export that does not resolve.
    ///
    /// A glob that resolves never reaches this function: its names replace it
    /// in the listing. This line reports that the target has no cached
    /// rustdoc JSON, so the names behind it are not known.
    fn glob_index_line(&self, child: &Located, use_item: &Use, name: &str) -> String {
        match self.reexport_origin(child, use_item) {
            Some(origin) => format!("mod {name} (glob re-export from `{origin}`)"),
            None => format!("mod {name} (glob re-export)"),
        }
    }

    /// The crate that a `use` item points at, from the paths table alone.
    fn reexport_origin(&self, child: &Located, use_item: &Use) -> Option<String> {
        let crate_data = self.crate_data(&child.crate_name)?;
        let summary = crate_data.paths.get(&use_item.id?)?;
        let external = crate_data.external_crates.get(&summary.crate_id)?;
        Some(external.name.clone())
    }

    /// Render the index line of a re-export from the paths table alone.
    fn summary_index_line(&self, child: &Located, use_item: &Use, name: &str) -> String {
        let summary = self
            .crate_data(&child.crate_name)
            .and_then(|crate_data| crate_data.paths.get(&use_item.id?));
        let Some(summary) = summary else {
            return format!("use {}", use_item.source);
        };
        reexport_line(
            kind_str(summary.kind),
            name,
            self.reexport_origin(child, use_item).as_deref(),
        )
    }
}

/// Which names a suggestion may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NameKind {
    /// Modules only, for `api_index`.
    Module,
    /// Every item, for `api_doc`.
    Any,
}

impl NameKind {
    fn admits(self, is_module: bool) -> bool {
        match self {
            Self::Module => is_module,
            Self::Any => true,
        }
    }
}

/// Whether an item has members a `Type::member` path can name.
fn is_type_like(inner: &ItemEnum) -> bool {
    matches!(
        inner,
        ItemEnum::Struct(_)
            | ItemEnum::Enum(_)
            | ItemEnum::Union(_)
            | ItemEnum::Trait(_)
            | ItemEnum::TypeAlias(_)
    )
}

/// The Levenshtein distance between two strings, by characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(ca != *cb);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// A path as a model writes it, without what is no part of a Rust path:
/// backticks, a leading item keyword (`fn conformant_price`), generic
/// arguments (`QuoteCurrency<i64, D>`), and a trailing `()` or `!`.
fn clean_query(path: &str) -> String {
    let mut text = path.trim().trim_matches('`').trim();
    for keyword in [
        "fn ", "struct ", "enum ", "trait ", "type ", "const ", "mod ", "macro ", "static ",
        "union ",
    ] {
        if let Some(rest) = text.strip_prefix(keyword) {
            text = rest.trim_start();
            break;
        }
    }
    let mut cleaned = String::with_capacity(text.len());
    let mut depth = 0_usize;
    for c in text.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => cleaned.push(c),
            _ => {}
        }
    }
    let cleaned = cleaned.trim();
    let cleaned = cleaned.strip_suffix("()").unwrap_or(cleaned);
    let cleaned = cleaned.strip_suffix('!').unwrap_or(cleaned);
    cleaned.trim().to_string()
}

/// The maximum number of re-exports that one name can chain through.
const MAX_REEXPORT_DEPTH: usize = 8;

/// An item, the crate that holds it, and its path inside that crate.
///
/// The path starts under the crate root and is what a facade request needs.
#[derive(Clone)]
struct Located {
    crate_name: String,
    path: Vec<String>,
    item: Item,
}

impl Located {
    /// A child of this module, one path segment deeper.
    fn child(&self, item: Item) -> Self {
        let mut path = self.path.clone();
        path.extend(item_name(&item).map(str::to_string));
        Self {
            crate_name: self.crate_name.clone(),
            path,
            item,
        }
    }
}

/// Render one index line for a re-export.
fn reexport_line(kind: &str, name: &str, origin: Option<&str>) -> String {
    match origin {
        Some(origin) => format!("{kind} {name} (re-exported from `{origin}`)"),
        None => format!("{kind} {name}"),
    }
}

/// A stand-in for a crate root that the index does not hold.
fn empty_root(id: Id) -> Item {
    Item {
        id,
        crate_id: 0,
        name: None,
        span: None,
        visibility: rustdoc_types::Visibility::Public,
        docs: None,
        links: HashMap::new(),
        attrs: Vec::new(),
        deprecation: None,
        stability: None,
        const_stability: None,
        inner: ItemEnum::Module(rustdoc_types::Module {
            is_crate: true,
            items: Vec::new(),
            is_stripped: false,
        }),
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

/// Build the request that renders `located` from its own crate.
///
/// The path must be valid inside that crate. The paths table gives it. The
/// walked path is the fallback for the items that the table omits, for
/// example a `use` item.
fn request_for(crate_data: &RustdocCrate, located: &Located) -> Option<FacadeRequest> {
    let path: Vec<String> = match crate_data.paths.get(&located.item.id) {
        Some(summary) => summary.path.iter().skip(1).cloned().collect(),
        None if located.path.is_empty() => return None,
        None => located.path.clone(),
    };
    Some(if matches!(located.item.inner, ItemEnum::Module(_)) {
        FacadeRequest::Module(path)
    } else {
        FacadeRequest::Item(path)
    })
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
    use std::sync::atomic::{
        AtomicUsize,
        Ordering,
    };

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

    /// A `use` item that re-exports one name.
    ///
    /// Rustdoc leaves `Item::name` empty on a `use` item and puts the
    /// exported name in `Use::name`. The fixtures must match that, or the
    /// index looks correct here and stays empty against a real crate.
    fn reexport(id: u32, name: &str, source: &str, target: u32) -> Item {
        item(
            id,
            None,
            ItemEnum::Use(Use {
                source: source.to_string(),
                name: name.to_string(),
                id: Some(Id(target)),
                is_glob: false,
            }),
        )
    }

    /// A `use` item that re-exports a whole module.
    fn glob_reexport(id: u32, name: &str, source: &str, target: u32) -> Item {
        item(
            id,
            None,
            ItemEnum::Use(Use {
                source: source.to_string(),
                name: name.to_string(),
                id: Some(Id(target)),
                is_glob: true,
            }),
        )
    }

    /// A summary for the paths table of a crate.
    fn summary(crate_id: u32, path: &[&str], kind: ItemKind) -> ItemSummary {
        ItemSummary {
            crate_id,
            path: path.iter().map(|segment| segment.to_string()).collect(),
            kind,
        }
    }

    /// The host crate. The `prelude` module holds one external re-export, one
    /// local function, one nested module, one re-export of a module from
    /// another module, and one renamed re-export.
    fn host_fixture() -> RustdocCrate {
        let index = HashMap::from([
            (Id(0), module(0, "host_crate", vec![Id(1), Id(10)])),
            (
                Id(1),
                module(1, "prelude", vec![Id(2), Id(3), Id(4), Id(6), Id(7)]),
            ),
            (Id(2), reexport(2, "decimal", "dep_crate::decimal", 100)),
            (Id(3), function(3, "submit_order")),
            (Id(4), module(4, "indicators", vec![Id(5)])),
            (Id(5), function(5, "zscore")),
            (Id(6), reexport(6, "rolling", "crate::deep::rolling", 11)),
            (Id(7), reexport(7, "Bar", "crate::deep::Foo", 13)),
            (Id(10), module(10, "deep", vec![Id(11), Id(13)])),
            (Id(11), module(11, "rolling", vec![Id(12)])),
            (Id(12), function(12, "mean")),
            (Id(13), function(13, "Foo")),
        ]);
        let paths = HashMap::from([
            (
                Id(100),
                summary(1, &["dep_crate", "decimal"], ItemKind::Macro),
            ),
            (
                Id(10),
                summary(0, &["host_crate", "deep"], ItemKind::Module),
            ),
            (
                Id(11),
                summary(0, &["host_crate", "deep", "rolling"], ItemKind::Module),
            ),
            (
                Id(13),
                summary(0, &["host_crate", "deep", "Foo"], ItemKind::Function),
            ),
        ]);
        let external_crates = HashMap::from([(
            1,
            ExternalCrate {
                name: "dep_crate".to_string(),
                html_root_url: None,
                path: PathBuf::new(),
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
            summary(0, &["dep_crate", "decimal"], ItemKind::Macro),
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

    /// A host crate whose prelude re-exports a module of `dep_crate` with a
    /// glob, next to one name that it declares itself.
    fn glob_host_fixture() -> RustdocCrate {
        let index = HashMap::from([
            (Id(0), module(0, "host_crate", vec![Id(1)])),
            (Id(1), module(1, "prelude", vec![Id(2), Id(3)])),
            (
                Id(2),
                glob_reexport(2, "prelude", "dep_crate::prelude::*", 200),
            ),
            (Id(3), function(3, "submit_order")),
        ]);
        let paths = HashMap::from([(
            Id(200),
            summary(1, &["dep_crate", "prelude"], ItemKind::Module),
        )]);
        let external_crates = HashMap::from([(
            1,
            ExternalCrate {
                name: "dep_crate".to_string(),
                html_root_url: None,
                path: PathBuf::new(),
            },
        )]);
        crate_data(index, paths, external_crates)
    }

    /// The crate behind the glob: a `prelude` module with two names.
    fn glob_dep_fixture() -> RustdocCrate {
        let index = HashMap::from([
            module(0, "dep_crate", vec![Id(200)]).into_entry(),
            module(200, "prelude", vec![Id(201), Id(202)]).into_entry(),
            function(201, "quote").into_entry(),
            function(202, "submit_order").into_entry(),
        ]);
        let paths = HashMap::from([
            (
                Id(200),
                summary(0, &["dep_crate", "prelude"], ItemKind::Module),
            ),
            (
                Id(201),
                summary(0, &["dep_crate", "prelude", "quote"], ItemKind::Function),
            ),
        ]);
        crate_data(index, paths, HashMap::new())
    }

    fn glob_fixture_index() -> DocIndex {
        DocIndex::from_crates(
            "host_crate",
            glob_host_fixture(),
            HashMap::from([("dep_crate".to_string(), glob_dep_fixture())]),
        )
    }

    /// A host crate whose root declares a name that the prelude also exports,
    /// with a different declaration behind it.
    fn shadow_host_fixture() -> RustdocCrate {
        let index = HashMap::from([
            (Id(0), module(0, "host_crate", vec![Id(1), Id(3)])),
            (Id(1), module(1, "prelude", vec![Id(2)])),
            (Id(2), reexport(2, "decimal", "dep_crate::decimal", 100)),
            (Id(3), function(3, "decimal")),
        ]);
        let paths = HashMap::from([
            (
                Id(100),
                summary(1, &["dep_crate", "decimal"], ItemKind::Macro),
            ),
            (
                Id(3),
                summary(0, &["host_crate", "decimal"], ItemKind::Function),
            ),
        ]);
        let external_crates = HashMap::from([(
            1,
            ExternalCrate {
                name: "dep_crate".to_string(),
                html_root_url: None,
                path: PathBuf::new(),
            },
        )]);
        crate_data(index, paths, external_crates)
    }

    trait IntoEntry {
        fn into_entry(self) -> (Id, Item);
    }

    impl IntoEntry for Item {
        fn into_entry(self) -> (Id, Item) {
            (self.id, self)
        }
    }

    /// A build that counts its runs. It yields once, so a second caller that
    /// starts while it runs has to wait for it.
    async fn counted_build(builds: &AtomicUsize) -> Result<Arc<DocIndex>> {
        builds.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        Ok(Arc::new(fixture_index()))
    }

    /// Two concurrent first callers for one crate name must share one build.
    /// Each build runs `cargo rustdoc` for the host crate and every reachable
    /// crate, so a duplicate is slow and fights the same cargo lock as the
    /// dylib builds.
    #[tokio::test]
    async fn host_cache_builds_one_index_per_crate_name() {
        let cache = Mutex::new(HashMap::new());
        let builds = AtomicUsize::new(0);

        let (first, second) = tokio::join!(
            DocIndex::cached(&cache, "host_crate", counted_build(&builds)),
            DocIndex::cached(&cache, "host_crate", counted_build(&builds)),
        );
        let first = first.expect("the build succeeds");
        let second = second.expect("the build succeeds");

        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&first, &second));
    }

    /// A failed build must not poison the cache: the next call retries it.
    #[tokio::test]
    async fn host_cache_retries_after_a_failed_build() {
        let cache = Mutex::new(HashMap::new());
        let builds = AtomicUsize::new(0);

        let failed = DocIndex::cached(&cache, "host_crate", std::future::ready(Err(Error::MdDoc)))
            .await
            .err();
        assert!(failed.is_some(), "the build must report its error");

        DocIndex::cached(&cache, "host_crate", counted_build(&builds))
            .await
            .expect("the retry succeeds");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
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
        let err = index
            .render_index(Some("nope"))
            .expect_err("the path does not resolve");
        assert!(matches!(err, DocIndexError::ModuleNotFound(..)));
    }

    #[test]
    fn render_doc_renders_local_function_by_prelude_name() {
        let index = fixture_index();
        let doc = index.render_doc("submit_order").expect("item exists");
        assert!(doc.contains("pub fn submit_order()"));
    }

    #[test]
    fn render_doc_follows_reexport_into_external_crate() {
        let index = fixture_index();
        let doc = index.render_doc("decimal").expect("item exists");
        assert!(doc.contains("macro_rules! decimal"));
        assert!(doc.contains("dep_crate"));
    }

    /// A query must never load a crate. When the cache has no data for a
    /// re-export target, the render says so instead of running `cargo`.
    #[test]
    fn render_doc_notes_a_dependency_that_is_not_cached() {
        let index = DocIndex::from_crates("host_crate", host_fixture(), HashMap::new());
        let doc = index.render_doc("decimal").expect("the re-export resolves");
        assert!(doc.contains("Could not generate rustdoc JSON"), "{doc}");
        assert!(doc.contains("dep_crate"), "{doc}");
    }

    /// A module that the prelude re-exports must be listable under the name
    /// that the index shows, without the internal path of the host crate.
    #[test]
    fn render_index_follows_a_reexported_module() {
        let index = fixture_index();
        let listing = index.render_index(None).expect("prelude exists");
        assert!(listing.contains("mod rolling"), "{listing}");

        for path in ["rolling", "prelude::rolling", "deep::rolling"] {
            let listing = index
                .render_index(Some(path))
                .unwrap_or_else(|err| panic!("`{path}` must resolve: {err}"));
            assert_eq!(listing, "fn mean\n", "{path}");
        }
    }

    /// A renamed re-export keeps the name of the prelude in the index.
    #[test]
    fn render_index_keeps_the_exported_name_of_a_renamed_reexport() {
        let index = fixture_index();
        let listing = index.render_index(None).expect("prelude exists");
        assert!(listing.contains("fn Bar"), "{listing}");
        assert!(!listing.contains("fn Foo"), "{listing}");
    }

    #[test]
    fn render_doc_follows_a_reexported_module() {
        let index = fixture_index();
        let doc = index.render_doc("rolling").expect("the module resolves");
        assert!(doc.contains("pub fn mean()"), "{doc}");
    }

    /// `use host::prelude::*;` puts the names behind a glob in scope. It does
    /// not put the name of the glob target module in scope. The listing must
    /// show the names, not the module.
    #[test]
    fn render_index_expands_a_glob_reexport() {
        let index = glob_fixture_index();
        let listing = index.render_index(None).expect("prelude exists");
        assert!(
            listing.contains("fn quote (re-exported from `dep_crate`)"),
            "{listing}"
        );
        assert!(!listing.contains("glob re-export"), "{listing}");
        assert!(!listing.contains("mod prelude"), "{listing}");
    }

    /// A name that the prelude declares hides the name of the same spelling
    /// behind a glob, as it does in Rust.
    #[test]
    fn render_index_prefers_a_declared_name_over_a_glob_name() {
        let index = glob_fixture_index();
        let listing = index.render_index(None).expect("prelude exists");
        assert_eq!(listing.matches("submit_order").count(), 1, "{listing}");
        assert!(listing.contains("fn submit_order\n"), "{listing}");
    }

    /// The agent reads a name from the listing and asks for its definition.
    /// That name must resolve even though the host crate never declares it.
    #[test]
    fn render_doc_resolves_a_name_from_a_glob_reexport() {
        let index = glob_fixture_index();
        let doc = index.render_doc("quote").expect("the glob name resolves");
        assert!(doc.contains("pub fn quote()"), "{doc}");
        assert!(doc.contains("dep_crate"), "{doc}");
    }

    /// Without cached data for the target crate, the listing keeps the glob
    /// line so that the gap stays visible.
    #[test]
    fn render_index_keeps_a_glob_that_does_not_resolve() {
        let index = DocIndex::from_crates("host_crate", glob_host_fixture(), HashMap::new());
        let listing = index.render_index(None).expect("prelude exists");
        assert!(
            listing.contains("mod prelude (glob re-export from `dep_crate`)"),
            "{listing}"
        );
    }

    /// The generated code only has `use host::prelude::*;` in scope, so a
    /// one-segment name means the prelude name, not a root item of the same
    /// spelling. The definition must match the name the agent compiles
    /// against, and it must match what the listing shows.
    #[test]
    fn render_doc_prefers_the_prelude_name_over_a_root_item() {
        let index = DocIndex::from_crates(
            "host_crate",
            shadow_host_fixture(),
            HashMap::from([("dep_crate".to_string(), dep_fixture())]),
        );

        let listing = index.render_index(None).expect("prelude exists");
        assert!(
            listing.contains("macro decimal (re-exported from `dep_crate`)"),
            "{listing}"
        );

        let doc = index
            .render_doc("decimal")
            .expect("the prelude name exists");
        assert!(doc.contains("macro_rules! decimal"), "{doc}");
        assert!(!doc.contains("pub fn decimal"), "{doc}");
    }

    #[test]
    fn render_doc_unknown_path_errors() {
        let index = fixture_index();
        let err = index
            .render_doc("nope")
            .expect_err("the path does not resolve");
        assert!(matches!(err, DocIndexError::ItemNotFound(..)));
    }

    /// The path resolves but the renderer produces nothing: the item carries
    /// no readable source. The error must say that the item exists, not that
    /// it is missing, because `api_index` lists it.
    #[test]
    fn render_doc_reports_a_resolved_item_that_cannot_render() {
        let index =
            DocIndex::from_crates("host_crate", unrenderable_host_fixture(), HashMap::new());
        let err = index
            .render_doc("Widget")
            .expect_err("the item resolves but renders nothing");
        assert!(
            matches!(err, DocIndexError::RenderFailed(ref name) if name == "Widget"),
            "{err:?}"
        );
    }

    /// A host fixture whose prelude names a struct that rustdoc JSON cannot
    /// render: the entry carries no source span.
    fn unrenderable_host_fixture() -> RustdocCrate {
        let index = HashMap::from([
            (Id(0), module(0, "host_crate", vec![Id(1)])),
            (Id(1), module(1, "prelude", vec![Id(2)])),
            (
                Id(2),
                item(
                    2,
                    Some("Widget"),
                    ItemEnum::Struct(rustdoc_types::Struct {
                        kind: rustdoc_types::StructKind::Unit,
                        generics: Generics {
                            params: Vec::new(),
                            where_predicates: Vec::new(),
                        },
                        impls: Vec::new(),
                    }),
                ),
            ),
        ]);
        let paths = HashMap::from([(
            Id(2),
            ItemSummary {
                crate_id: 0,
                path: vec![
                    "host_crate".to_string(),
                    "prelude".to_string(),
                    "Widget".to_string(),
                ],
                kind: ItemKind::Struct,
            },
        )]);
        crate_data(index, paths, HashMap::new())
    }

    /// A struct `Order` in the prelude whose source span points at a real
    /// file, so it renders like a struct of a real crate.
    fn struct_fixture_index(source: &std::path::Path) -> DocIndex {
        std::fs::write(source, "pub struct Order;\n").expect("write the fixture source");
        let mut order = item(
            2,
            Some("Order"),
            ItemEnum::Struct(rustdoc_types::Struct {
                kind: rustdoc_types::StructKind::Unit,
                generics: Generics {
                    params: Vec::new(),
                    where_predicates: Vec::new(),
                },
                impls: Vec::new(),
            }),
        );
        order.span = Some(rustdoc_types::Span {
            filename: source.to_path_buf(),
            begin: (1, 1),
            end: (1, 18),
        });
        let index = HashMap::from([
            (Id(0), module(0, "host_crate", vec![Id(1)])),
            (Id(1), module(1, "prelude", vec![Id(2)])),
            (Id(2), order),
        ]);
        let paths = HashMap::from([(
            Id(2),
            summary(0, &["host_crate", "prelude", "Order"], ItemKind::Struct),
        )]);
        DocIndex::from_crates(
            "host_crate",
            crate_data(index, paths, HashMap::new()),
            HashMap::new(),
        )
    }

    /// A name qualified with the crate the prelude re-exports it from
    /// resolves to the prelude name, and the answer says which path to use.
    #[test]
    fn render_doc_reads_a_crate_qualified_name_as_the_prelude_name() {
        let doc = fixture_index()
            .render_doc("dep_crate::decimal")
            .expect("the prelude re-exports `decimal` from `dep_crate`");
        assert!(doc.contains("macro_rules! decimal"), "{doc}");
        assert!(
            doc.starts_with(
                "Note: `dep_crate::decimal` is not a path of the host API; this is `decimal`"
            ),
            "{doc}"
        );
        assert!(
            doc.contains("Use `decimal`, not `dep_crate::decimal`."),
            "{doc}"
        );
        // A path that resolves as written gets no note.
        let doc = fixture_index().render_doc("decimal").expect("resolves");
        assert!(!doc.contains("Note:"), "{doc}");
    }

    /// Item keywords, backticks, generic arguments and call parentheses are
    /// no part of a path.
    #[test]
    fn render_doc_ignores_the_decorations_of_a_name() {
        for query in [
            "`fn submit_order()`",
            "fn submit_order",
            "submit_order()",
            " submit_order ",
            "prelude::submit_order",
        ] {
            let doc = fixture_index().render_doc(query).expect(query);
            assert!(doc.contains("pub fn submit_order()"), "{query}: {doc}");
            assert!(!doc.contains("Note:"), "{query}: {doc}");
        }
        let doc = fixture_index().render_doc("decimal!").expect("macro call");
        assert!(doc.contains("macro_rules! decimal"), "{doc}");
    }

    #[test]
    fn render_doc_resolves_a_unique_name_that_differs_only_in_case() {
        let doc = fixture_index()
            .render_doc("SUBMIT_ORDER")
            .expect("one match");
        assert!(doc.contains("pub fn submit_order()"), "{doc}");
        assert!(
            doc.contains("`SUBMIT_ORDER` is spelled `submit_order` in the host API."),
            "{doc}"
        );
    }

    /// A misspelled name is answered with the closest names, so the next call
    /// can use one directly; a name close to nothing gets no suggestion.
    #[test]
    fn render_doc_suggests_the_closest_names_for_an_unknown_path() {
        let err = fixture_index()
            .render_doc("submit_ordr")
            .expect_err("misspelled");
        let DocIndexError::ItemNotFound(path, closest) = &err else {
            panic!("not found: {err}");
        };
        assert_eq!(path, "submit_ordr");
        assert_eq!(
            closest.names().first().map(String::as_str),
            Some("submit_order")
        );
        assert!(
            err.to_string().contains(
                "no public item `submit_ordr` exists in the host API; the closest names are \
                 `submit_order`"
            ),
            "{err}"
        );
        // A name contained in a known one is close, at any length.
        let err = fixture_index().render_doc("order").expect_err("partial");
        assert!(err.to_string().contains("`submit_order`"), "{err}");

        let err = fixture_index()
            .render_doc("xyzzy_plugh")
            .expect_err("close to nothing");
        let DocIndexError::ItemNotFound(_, closest) = &err else {
            panic!("not found: {err}");
        };
        assert!(closest.names().is_empty(), "{closest:?}");
        assert_eq!(
            err.to_string(),
            "no public item `xyzzy_plugh` exists in the host API; call `api_index` to list the \
             available names"
        );
    }

    /// `Type::member` shows the type, whose definition lists the members;
    /// a member path under a function or a module is not read that way.
    #[test]
    fn render_doc_shows_the_type_of_a_member_path() {
        let dir = std::env::temp_dir().join(format!("doc_index_member_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let index = struct_fixture_index(&dir.join("order.rs"));
        let doc = index.render_doc("Order::price").expect("the type renders");
        assert!(doc.contains("pub struct Order;"), "{doc}");
        assert!(
            doc.contains("`price` would be a member of `Order`, whose definition"),
            "{doc}"
        );
        let doc = index
            .render_doc("host_crate::Order::price")
            .expect("both rules combine");
        assert!(doc.contains("pub struct Order;"), "{doc}");
        std::fs::remove_dir_all(&dir).expect("clean up");

        assert!(matches!(
            fixture_index().render_doc("submit_order::price"),
            Err(DocIndexError::ItemNotFound(..))
        ));
        assert!(matches!(
            fixture_index().render_doc("indicators::nope"),
            Err(DocIndexError::ItemNotFound(..))
        ));
    }

    /// `api_index` on a crate the prelude re-exports from lists the prelude
    /// items that come from it; an unknown module gets suggestions.
    #[test]
    fn render_index_of_a_dependency_crate_lists_its_prelude_items() {
        let listing = fixture_index()
            .render_index(Some("dep_crate"))
            .expect("the prelude re-exports from `dep_crate`");
        assert!(
            listing.starts_with("`dep_crate` is not a module of the host API."),
            "{listing}"
        );
        assert!(
            listing.contains("macro decimal (re-exported from `dep_crate`)\n"),
            "{listing}"
        );
        assert!(!listing.contains("submit_order"), "{listing}");

        let err = fixture_index()
            .render_index(Some("indicatrs"))
            .expect_err("misspelled module");
        let DocIndexError::ModuleNotFound(_, closest) = &err else {
            panic!("not found: {err}");
        };
        assert_eq!(closest.names(), ["indicators".to_string()]);
    }

    #[test]
    fn the_edit_distance_counts_insertions_deletions_and_substitutions() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("account", "acount"), 1);
        assert_eq!(edit_distance("ab", "ba"), 2);
    }

    #[test]
    fn a_query_is_cleaned_to_its_path() {
        assert_eq!(clean_query("`fn conformant_price`"), "conformant_price");
        assert_eq!(clean_query("QuoteCurrency<i64, D>"), "QuoteCurrency");
        assert_eq!(
            clean_query("Account<i64, DECIMALS, Cur, UserOrderId>::new()"),
            "Account::new"
        );
        assert_eq!(clean_query("struct  Position"), "Position");
        assert_eq!(clean_query("decimal!"), "decimal");
        assert_eq!(clean_query("lfest::Account"), "lfest::Account");
        assert_eq!(clean_query("fnord"), "fnord");
        assert_eq!(clean_query(""), "");
    }
}
