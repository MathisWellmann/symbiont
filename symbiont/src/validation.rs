use std::collections::HashMap;

use prettyplease::unparse;
// SPDX-License-Identifier: MPL-2.0
use quote::ToTokens;
use syn::{
    FnArg,
    Signature,
    Type,
    Visibility,
    visit::{
        self,
        Visit,
    },
};
use tracing::debug;

use crate::{
    Error,
    Result,
    utils::{
        is_no_mangle,
        is_pub,
    },
};

/// Validate that a parsed AST enforces typed generation:
/// - No `unsafe` code and no forbidden constructs (see
///   [`enforce_code_policy`])
/// - Every **declared** function is `pub` and carries `#[unsafe(no_mangle)]`
/// - Every other (helper) function stays unexported: `#[unsafe(no_mangle)]`
///   is stripped if the model added one anyway
/// - All function signatures match the expected signatures from lib.rs.
///
/// Signatures are compared by function name, argument **types**, and return
/// type. Argument names are ignored: they carry no meaning at the dylib ABI
/// boundary (symbol lookup is by function name only), so renaming an unused
/// parameter (e.g. to `_market_state`) is not a mismatch.
///
/// # Why helpers must not be exported
///
/// `#[unsafe(no_mangle)]` in a `dylib` publishes the symbol with default ELF
/// visibility, which makes it *preemptible*: the compiler routes even the
/// dylib's own calls to it through the GOT, and the dynamic loader resolves
/// those against the global scope (the host executable and everything it
/// pulled in, libc included) **before** the dylib itself. A generated helper
/// named like a libc function — `qsort`, `random`, `div`, `remove`, … —
/// therefore silently hijacks the call to libc's version with completely
/// different arguments, which segfaults the host. Unexported helpers keep
/// their mangled, crate-hashed names, cannot collide, and can be inlined.
///
/// Returns `Err` with a descriptive message if any check fails.
pub(crate) fn validate_generated_ast(
    file: &mut syn::File,
    expected_sigs: &[String],
    denied_paths: &[String],
) -> Result<()> {
    enforce_code_policy(file, denied_paths)?;

    if expected_sigs.is_empty() {
        return Ok(());
    }

    let declared: Vec<&str> = expected_sigs
        .iter()
        .filter_map(|sig| expected_fn_name(sig))
        .collect();

    let mut found_sigs: Vec<(String, String)> = Vec::with_capacity(4);

    for item in &mut file.items {
        if let syn::Item::Fn(item_fn) = item {
            let name = item_fn.sig.ident.to_string();

            if declared.contains(&name.as_str()) {
                // Add `pub` visibility if missing
                if !is_pub(item_fn) {
                    debug!("Function `{name}` missing `pub` visibility, adding it");
                    item_fn.vis = Visibility::Public(syn::token::Pub::default());
                }
                // Add #[unsafe(no_mangle)] if missing
                if !is_no_mangle(item_fn) {
                    debug!("Function `{name}` missing #[unsafe(no_mangle)], adding it");
                    let attr: syn::Attribute = syn::parse_quote!(#[unsafe(no_mangle)]);
                    item_fn.attrs.insert(0, attr);
                }
            } else if is_no_mangle(item_fn) {
                // Helper the model exported on its own: unexport it, or its
                // name may be preempted by a libc symbol at load time.
                debug!("Helper `{name}` carries #[unsafe(no_mangle)], removing it");
                item_fn.attrs.retain(|attr| {
                    !(attr.path().is_ident("unsafe")
                        && matches!(&attr.meta, syn::Meta::List(list)
                            if list.tokens.to_string() == "no_mangle"))
                });
            }

            let sig = format_signature(&item_fn.sig)
                .unwrap_or_else(|| normalize_tokens(item_fn.sig.to_token_stream().to_string()));
            found_sigs.push((name, sig));
        }
    }

    // Compare each expected signature against the generated function of the
    // same name, so the error pinpoints the offending function instead of the
    // first expected signature that had no exact match. Expected signatures
    // retain argument names for prompts; only this compatibility comparison
    // canonicalizes them to argument types.
    for expected in expected_sigs {
        let Some(fn_name) = expected_fn_name(expected) else {
            continue;
        };
        let expected_canonical = syn::parse_str::<Signature>(expected)
            .ok()
            .and_then(|sig| format_signature(&sig))
            .unwrap_or_else(|| expected.clone());
        let found = found_sigs.iter().find(|(name, _)| name == fn_name);
        match found {
            Some((_, sig)) if sig == &expected_canonical => {}
            Some((_, sig)) => {
                return Err(Error::SignatureMismatch {
                    code: unparse(file),
                    expected: expected.clone(),
                    got: sig.clone(),
                });
            }
            None => {
                return Err(Error::SignatureMismatch {
                    code: unparse(file),
                    expected: expected.clone(),
                    got: format!("function `{fn_name}` not found"),
                });
            }
        }
    }

    Ok(())
}

/// Reject stub bodies outright, and candidates that implement nothing.
///
/// An empty body or a lone placeholder macro (`todo!()`, `unimplemented!()`,
/// `unreachable!()`) always rejects, even when other declared functions
/// genuinely evolved: such a body panics at run time or fails to compile. A
/// verbatim echo of the declared default body only rejects when *no*
/// declared function changed. A candidate that genuinely evolves one
/// function while leaving another at its default is a partial evolution and
/// is accepted; the untouched functions keep their default behaviour.
///
/// Runs after [`validate_generated_ast`] succeeds, so argument types already
/// match the declarations and bodies of declared functions can be compared
/// positionally. Argument identifiers are rewritten to positional
/// placeholders before comparison because the signature check deliberately
/// ignores argument names: an echo that renamed an unused parameter must
/// still count as an echo.
///
/// `default_bodies` maps every declared function name to the normalized
/// token string of its default body (see [`default_body_tokens`]).
pub(crate) fn check_implementation_bodies(
    file: &syn::File,
    default_bodies: &HashMap<String, String>,
) -> Result<()> {
    // First declared function still sitting at its default body, and whether
    // any declared function genuinely changed.
    let mut unchanged: Option<String> = None;
    let mut changed = false;
    for item in &file.items {
        let syn::Item::Fn(item_fn) = item else {
            continue;
        };
        let name = item_fn.sig.ident.to_string();
        let Some(default_body) = default_bodies.get(&name) else {
            // Helper function, not an evolution target.
            continue;
        };
        let block = &item_fn.block;
        if block.stmts.is_empty() {
            return Err(Error::UnimplementedFunction {
                code: unparse(file),
                fn_name: name,
                reason: "the body is empty".to_string(),
            });
        }
        if let Some(placeholder) = placeholder_macro_of(block) {
            return Err(Error::UnimplementedFunction {
                code: unparse(file),
                fn_name: name,
                reason: format!("the body is only `{placeholder}!()`"),
            });
        }
        let generated = body_tokens(block, &arg_names(&item_fn.sig));
        if generated == *default_body {
            unchanged.get_or_insert(name);
        } else {
            changed = true;
        }
    }
    // Every declared function still sits at its default body: the candidate
    // implements nothing.
    if !changed && let Some(name) = unchanged {
        return Err(Error::UnimplementedFunction {
            code: unparse(file),
            fn_name: name,
            reason: "the body is identical to the default implementation".to_string(),
        });
    }
    Ok(())
}

/// The name of the placeholder macro if the whole body is a single
/// `todo!()`, `unimplemented!()` or `unreachable!()` call, else `None`.
fn placeholder_macro_of(block: &syn::Block) -> Option<&'static str> {
    let [stmt] = block.stmts.as_slice() else {
        return None;
    };
    // A macro call parses to `Stmt::Macro` in statement position
    // (`todo!();`) and to `Expr::Macro` as the body's final expression
    // (`todo!()`).
    let mac: &syn::Macro = match stmt {
        syn::Stmt::Macro(sm) => &sm.mac,
        syn::Stmt::Expr(syn::Expr::Macro(m), _) => &m.mac,
        _ => return None,
    };
    ["todo", "unimplemented", "unreachable"]
        .into_iter()
        .find(|name| mac.path.is_ident(name))
}

/// Normalized body tokens of a declared function, the echo reference used
/// by [`check_implementation_bodies`].
pub(crate) fn default_body_tokens(item_fn: &syn::ItemFn) -> String {
    body_tokens(&item_fn.block, &arg_names(&item_fn.sig))
}

/// The body's token stream with argument identifiers rewritten to
/// positional placeholders, rendered as a string. Comments do not survive
/// into token streams, so the result is insensitive to both formatting and
/// comments.
fn body_tokens(block: &syn::Block, arg_names: &[String]) -> String {
    normalize_arg_idents(block.to_token_stream(), arg_names).to_string()
}

/// The argument identifiers of `sig`, positionally. Wildcard parameters
/// (`_`) carry no identifier and are skipped; an echo that named such a
/// parameter is then missed rather than misreported.
fn arg_names(sig: &Signature) -> Vec<String> {
    sig.inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pat) => match &*pat.pat {
                syn::Pat::Ident(p) => Some(p.ident.to_string()),
                _ => None,
            },
            FnArg::Receiver(_) => None,
        })
        .collect()
}

/// Rewrite every identifier matching one of `arg_names` to
/// `__symbiont_arg_{i}`, positionally, so body comparisons ignore argument
/// names. Recurses into delimiter groups.
fn normalize_arg_idents(
    tokens: proc_macro2::TokenStream,
    arg_names: &[String],
) -> proc_macro2::TokenStream {
    tokens
        .into_iter()
        .map(|tt| match tt {
            proc_macro2::TokenTree::Ident(ident) => {
                let name = ident.to_string();
                match arg_names.iter().position(|arg| arg == &name) {
                    Some(i) => proc_macro2::TokenTree::Ident(proc_macro2::Ident::new(
                        &format!("__symbiont_arg_{i}"),
                        ident.span(),
                    )),
                    None => proc_macro2::TokenTree::Ident(ident),
                }
            }
            proc_macro2::TokenTree::Group(group) => {
                let stream = normalize_arg_idents(group.stream(), arg_names);
                proc_macro2::TokenTree::Group(proc_macro2::Group::new(group.delimiter(), stream))
            }
            _ => tt,
        })
        .collect()
}

/// Path prefixes that are always denied, independent of
/// [`crate::DylibConfig`]: replacing the panic hook would break the
/// harness's panic-reporting protocol — the injected preamble owns the
/// hook inside the dylib.
const ALWAYS_DENIED_PATHS: &[&str] = &[
    "std::panic::set_hook",
    "std::panic::take_hook",
    "std::panic::update_hook",
];

/// Attributes that hijack the dylib's runtime and break its contract with
/// the host (shared System allocator, unwinding panics, program entry).
const DENIED_ATTRIBUTES: &[&str] = &[
    "global_allocator",
    "panic_handler",
    "alloc_error_handler",
    "no_main",
];

/// Reject `unsafe` code and forbidden constructs in LLM-generated code.
///
/// Enforced on the parsed AST *before* compilation: the rejection is
/// cheap (no cargo round-trip), pinpoints the offending construct for the
/// backpressure prompt, and cannot be evaded with `#[allow(unsafe_code)]`
/// the way a compiler lint could. A crate-level `#![forbid(unsafe_code)]`
/// is not an option anyway: the injected panic preamble is legitimately
/// unsafe, and the `#[unsafe(no_mangle)]` export attribute on every
/// evolvable function trips the `unsafe_code` lint in edition 2024.
///
/// Rejected as **unsafe** ([`Error::UnsafeCode`]):
/// - `unsafe { .. }` blocks
/// - `unsafe fn` (free, impl, trait, and foreign)
/// - `unsafe impl` and `unsafe trait`
/// - `extern` blocks
/// - unsafe attributes such as `#[unsafe(export_name = ..)]` — except the
///   exact `#[unsafe(no_mangle)]` export attribute the harness itself
///   manages
/// - an `unsafe` token anywhere inside a macro definition or invocation,
///   which would otherwise smuggle unsafe code past the AST scan
///
/// Rejected as **forbidden** ([`Error::ForbiddenConstruct`]):
/// - `static` items and `thread_local!` — dylib-local state silently
///   resets on every evolution and every retained revision has its own
///   copy; state must be host-owned (see CAVEATS.md)
/// - `macro_rules!` definitions — macro bodies would otherwise be a
///   blind spot for every other rule
/// - [`DENIED_ATTRIBUTES`] — allocator/panic/entry overrides
/// - references to [`ALWAYS_DENIED_PATHS`] and the host-configurable
///   `denied_paths` prefixes ([`crate::DylibConfig::default_denied_paths`]),
///   matched after resolving the file's `use` aliases; glob imports of a
///   denied module are rejected outright
///
/// This bounds what evolvable code can *name*. It is guidance for the
/// evolution loop, not a security sandbox: safe Rust reached through a
/// host-provided API can still do I/O on the host's behalf.
pub(crate) fn enforce_code_policy(file: &syn::File, denied_paths: &[String]) -> Result<()> {
    let denied: Vec<Vec<&str>> = denied_paths
        .iter()
        .map(String::as_str)
        .chain(ALWAYS_DENIED_PATHS.iter().copied())
        .map(|path| path.split("::").collect())
        .collect();

    let mut collector = AliasCollector::default();
    collector.visit_file(file);

    let mut scan = PolicyScan {
        denied: &denied,
        aliases: collector.aliases,
        finding: None,
    };
    scan.visit_file(file);
    match scan.finding {
        Some(Finding::Unsafe(construct)) => Err(Error::UnsafeCode {
            code: unparse(file),
            construct,
        }),
        Some(Finding::Forbidden { construct, reason }) => Err(Error::ForbiddenConstruct {
            code: unparse(file),
            construct,
            reason,
        }),
        None => Ok(()),
    }
}

/// The first policy violation found in the AST.
enum Finding {
    /// An `unsafe` construct; reported as [`Error::UnsafeCode`].
    Unsafe(String),
    /// A forbidden (but safe) construct; reported as
    /// [`Error::ForbiddenConstruct`].
    Forbidden { construct: String, reason: String },
}

/// Render `tokens` as a short snippet for feedback messages.
fn snippet(tokens: &dyn ToTokens) -> String {
    let mut snippet = tokens.to_token_stream().to_string();
    if snippet.len() > 120 {
        let cut = (0..=120)
            .rev()
            .find(|i| snippet.is_char_boundary(*i))
            .unwrap_or(0);
        snippet.truncate(cut);
        snippet.push('…');
    }
    snippet
}

/// Resolves `use` imports to absolute paths so denied-path matching sees
/// through aliases (`use std::process::exit as quit;`, `use std as s;`,
/// `use std::fs::File;` + `File::open(..)`). The generated code is a
/// single file without external macro expansion, so this resolution is
/// complete for non-glob imports.
#[derive(Default)]
struct AliasCollector {
    /// Local name -> absolute path segments.
    aliases: HashMap<String, Vec<String>>,
}

impl<'ast> Visit<'ast> for AliasCollector {
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        collect_use_tree(&node.tree, &mut Vec::new(), &mut self.aliases);
    }
}

/// Walk a use tree, recording every imported leaf under its local name.
fn collect_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    aliases: &mut HashMap<String, Vec<String>>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_tree(&path.tree, prefix, aliases);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = prefix.clone();
            // `use std::process::{self};` imports the module itself.
            if name.ident != "self" {
                full.push(name.ident.to_string());
            }
            let local = full.last().cloned().unwrap_or_default();
            aliases.insert(local, full);
        }
        syn::UseTree::Rename(rename) => {
            let mut full = prefix.clone();
            if rename.ident != "self" {
                full.push(rename.ident.to_string());
            }
            aliases.insert(rename.rename.to_string(), full);
        }
        syn::UseTree::Glob(_) => {}
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_tree(item, prefix, aliases);
            }
        }
    }
}

/// AST visitor recording the first policy violation.
struct PolicyScan<'a> {
    /// Denied path prefixes, pre-split into segments.
    denied: &'a [Vec<&'a str>],
    /// `use` aliases of this file, local name -> absolute segments.
    aliases: HashMap<String, Vec<String>>,
    finding: Option<Finding>,
}

impl PolicyScan<'_> {
    fn record_unsafe(&mut self, what: &str, tokens: &dyn ToTokens) {
        if self.finding.is_none() {
            self.finding = Some(Finding::Unsafe(format!("{what}: `{}`", snippet(tokens))));
        }
    }

    fn record_forbidden(&mut self, what: &str, tokens: &dyn ToTokens, reason: String) {
        if self.finding.is_none() {
            self.finding = Some(Finding::Forbidden {
                construct: format!("{what}: `{}`", snippet(tokens)),
                reason,
            });
        }
    }

    /// Match `segments` (after alias expansion) against the denied
    /// prefixes; returns the matched denied prefix.
    fn denied_prefix_of(&self, segments: &[String]) -> Option<String> {
        let expanded: Vec<&str> = match segments.split_first() {
            Some((first, rest)) => match self.aliases.get(first.as_str()) {
                Some(full) => full
                    .iter()
                    .map(String::as_str)
                    .chain(rest.iter().map(String::as_str))
                    .collect(),
                None => segments.iter().map(String::as_str).collect(),
            },
            None => return None,
        };
        self.denied
            .iter()
            .find(|denied| expanded.len() >= denied.len() && expanded[..denied.len()] == denied[..])
            .map(|denied| denied.join("::"))
    }

    fn check_path(&mut self, segments: Vec<String>, tokens: &dyn ToTokens) {
        if let Some(denied) = self.denied_prefix_of(&segments) {
            self.record_forbidden(
                &format!("a use of `{denied}`"),
                tokens,
                format!(
                    "access to `{denied}` is denied for evolvable code (hosts control this via `DylibConfig`)"
                ),
            );
        }
    }
}

/// `true` if any token (recursively) is the `unsafe` keyword.
fn tokens_contain_unsafe(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|tt| match tt {
        proc_macro2::TokenTree::Ident(ident) => ident == "unsafe",
        proc_macro2::TokenTree::Group(group) => tokens_contain_unsafe(group.stream()),
        _ => false,
    })
}

impl<'ast> Visit<'ast> for PolicyScan<'_> {
    fn visit_expr_unsafe(&mut self, node: &'ast syn::ExprUnsafe) {
        self.record_unsafe("an `unsafe` block", node);
    }

    // Covers free functions, impl methods, trait methods, and foreign fns.
    fn visit_signature(&mut self, node: &'ast Signature) {
        if matches!(node.safety, syn::Safety::Unsafe(_)) {
            self.record_unsafe("an `unsafe fn`", node);
        }
        visit::visit_signature(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if node.unsafety.is_some() {
            self.record_unsafe("an `unsafe impl`", node);
        }
        visit::visit_item_impl(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        if node.unsafety.is_some() {
            self.record_unsafe("an `unsafe trait`", node);
        }
        visit::visit_item_trait(self, node);
    }

    fn visit_item_foreign_mod(&mut self, node: &'ast syn::ItemForeignMod) {
        self.record_unsafe("an `extern` block", node);
    }

    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        self.record_forbidden(
            "a `static` item",
            node,
            "static state silently resets on every evolution and every retained revision has \
             its own copy; keep state host-owned and pass it in via arguments, or use `const`"
                .to_string(),
        );
        visit::visit_item_static(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        // `macro_rules! name { .. }` carries the definition's name.
        if node.ident.is_some() {
            self.record_forbidden(
                "a `macro_rules!` definition",
                node,
                "defining macros is forbidden in evolvable code; write the logic directly"
                    .to_string(),
            );
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        // The exact export attribute the harness manages is the only
        // permitted unsafe attribute; see `crate::utils::is_no_mangle`.
        let is_no_mangle_export = node.path().is_ident("unsafe")
            && matches!(&node.meta, syn::Meta::List(list) if list.tokens.to_string() == "no_mangle");
        if node.path().is_ident("unsafe") && !is_no_mangle_export {
            self.record_unsafe("an unsafe attribute", node);
        }
        if let Some(denied) = DENIED_ATTRIBUTES
            .iter()
            .find(|attr| node.path().is_ident(attr))
        {
            self.record_forbidden(
                &format!("a `#[{denied}]` attribute"),
                node,
                "overriding the allocator, panic handling, or program entry breaks the \
                 contract between host and dylib"
                    .to_string(),
            );
        }
        visit::visit_attribute(self, node);
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        self.check_use_tree(&node.tree, &mut Vec::new());
        visit::visit_item_use(self, node);
    }

    fn visit_path(&mut self, node: &'ast syn::Path) {
        let segments: Vec<String> = node
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        self.check_path(segments, node);
        visit::visit_path(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if node
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "thread_local")
        {
            self.record_forbidden(
                "a `thread_local!` declaration",
                node,
                "thread-local state silently resets on every evolution; keep state host-owned \
                 and pass it in via arguments"
                    .to_string(),
            );
        }
        if tokens_contain_unsafe(node.tokens.clone()) {
            self.record_unsafe("an `unsafe` token inside a macro", node);
        }
        // Denied paths inside macro tokens: `to_string` renders paths with
        // canonical `a :: b` spacing, so a plain substring match suffices.
        let text = node.tokens.to_string();
        if let Some(denied) = self
            .denied
            .iter()
            .find(|denied| text.contains(&denied.join(" :: ")))
        {
            let denied = denied.join("::");
            self.record_forbidden(
                &format!("a use of `{denied}` inside a macro"),
                node,
                format!(
                    "access to `{denied}` is denied for evolvable code (hosts control this via `DylibConfig`)"
                ),
            );
        }
        visit::visit_macro(self, node);
    }
}

impl PolicyScan<'_> {
    /// Check a `use` tree: leaves are matched against the denied prefixes,
    /// and glob imports overlapping a denied module are rejected outright
    /// (they would make denied items nameable without their path).
    fn check_use_tree(&mut self, tree: &syn::UseTree, prefix: &mut Vec<String>) {
        match tree {
            syn::UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.check_use_tree(&path.tree, prefix);
                prefix.pop();
            }
            syn::UseTree::Name(name) => {
                let mut full = prefix.clone();
                if name.ident != "self" {
                    full.push(name.ident.to_string());
                }
                self.check_path(full, name);
            }
            syn::UseTree::Rename(rename) => {
                let mut full = prefix.clone();
                if rename.ident != "self" {
                    full.push(rename.ident.to_string());
                }
                self.check_path(full, rename);
            }
            syn::UseTree::Glob(glob) => {
                let overlaps = self.denied.iter().find(|denied| {
                    let shorter = prefix.len().min(denied.len());
                    prefix[..shorter]
                        .iter()
                        .map(String::as_str)
                        .eq(denied[..shorter].iter().copied())
                });
                if let Some(denied) = overlaps {
                    let denied = denied.join("::");
                    self.record_forbidden(
                        &format!("a glob import of `{}`", prefix.join("::")),
                        glob,
                        format!(
                            "glob imports overlapping the denied module `{denied}` are rejected; import items explicitly"
                        ),
                    );
                }
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.check_use_tree(item, prefix);
                }
            }
        }
    }
}

/// Extract the function name from a signature rendered by [`format_signature`],
/// e.g. `fn step(&mut usize)` -> `step`.
fn expected_fn_name(sig: &str) -> Option<&str> {
    sig.strip_prefix("fn ")?.split('(').next()
}

/// Format a `syn::Signature` into a canonical string for comparison.
///
/// Renders `fn name(ty0, ty1, ...) -> ret` **without argument names**: two
/// signatures that differ only in argument names are the same function at the
/// dylib boundary and must compare equal.
fn format_signature(sig: &Signature) -> Option<String> {
    let mut out = String::from("fn ");
    out.push_str(&sig.ident.to_string());

    if sig.asyncness.is_some()
        || matches!(sig.safety, syn::Safety::Unsafe(_))
        || sig.abi.is_some()
        || sig.variadic.is_some()
        || !sig.generics.params.is_empty()
    {
        return None;
    }

    out.push('(');
    let inputs: Vec<_> = sig.inputs.iter().map(format_fn_arg).collect();
    out.push_str(&inputs.join(", "));
    out.push(')');

    if let syn::ReturnType::Type(_, ty) = &sig.output {
        out.push_str(" -> ");
        out.push_str(&format_return_type(ty));
    }

    Some(out)
}

fn format_fn_arg(arg: &FnArg) -> String {
    use FnArg::*;
    match arg {
        Receiver(recv) => {
            if recv.mutability.is_some() {
                "&mut self".into()
            } else {
                "&self".into()
            }
        }
        Typed(pat) => normalize_tokens(pat.ty.to_token_stream().to_string()),
    }
}

fn normalize_tokens(mut s: String) -> String {
    s = s.replace("& mut ", "&mut ");
    s = s.replace("& ref ", "&ref ");
    s = s.replace("* mut ", "*mut ");
    s = s.replace("* const ", "*const ");
    s = s.replace("mut & ", "mut &");
    s
}

fn format_return_type(ty: &Type) -> String {
    normalize_tokens(ty.to_token_stream().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_rust_code;

    /// The default denied-path configuration used by the tests.
    fn denied() -> Vec<String> {
        crate::DylibConfig::default_denied_paths()
    }

    #[test]
    fn test_validate_valid_code() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected, &denied()).expect("validation passed");
    }

    #[test]
    fn test_validate_missing_no_mangle_gets_added() {
        let input = "```rust
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected, &denied())
            .expect("should succeed by adding #[unsafe(no_mangle)]");

        // Verify the attribute was actually added
        let item_fn = match &file.items[0] {
            syn::Item::Fn(f) => f,
            _ => panic!("expected fn item"),
        };
        assert!(
            is_no_mangle(item_fn),
            "expected #[unsafe(no_mangle)] to be present after validation"
        );
    }

    #[test]
    fn test_validate_non_public_gets_pub_added() {
        let input = "```rust
fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected, &denied())
            .expect("should succeed by adding `pub` and #[unsafe(no_mangle)]");

        let item_fn = match &file.items[0] {
            syn::Item::Fn(f) => f,
            _ => panic!("expected fn item"),
        };
        assert!(
            is_pub(item_fn),
            "expected `pub` visibility after validation"
        );
        assert!(
            is_no_mangle(item_fn),
            "expected #[unsafe(no_mangle)] after validation"
        );
    }

    #[test]
    fn helpers_are_not_exported() {
        // A helper named like a libc function must not become an exported,
        // preemptible symbol: the dylib's own call to it would otherwise be
        // resolved to libc's `qsort` and crash the host.
        let input = "```rust
fn step(counter: &mut usize) {
    qsort(counter);
}

#[unsafe(no_mangle)]
pub fn qsort(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected, &denied()).expect("validation passed");

        let fns: Vec<&syn::ItemFn> = file
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Fn(f) => Some(f),
                _ => None,
            })
            .collect();
        let declared = fns
            .iter()
            .find(|f| f.sig.ident == "step")
            .expect("declared fn is present");
        let helper = fns
            .iter()
            .find(|f| f.sig.ident == "qsort")
            .expect("helper fn is present");
        assert!(is_no_mangle(declared), "declared fn must be exported");
        assert!(is_pub(declared), "declared fn must be pub");
        assert!(!is_no_mangle(helper), "helper must not be exported");
    }

    #[test]
    fn test_validate_signature_mismatch() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        let err =
            validate_generated_ast(&mut file, &expected, &denied()).expect_err("should error");
        dbg!(&err);
        match err {
            Error::SignatureMismatch {
                code,
                expected,
                got,
            } => {
                // `add` is not a declared function, so its export attribute
                // was stripped before the mismatch was reported.
                assert_eq!(&code, "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n");
                assert_eq!(expected, "fn step(counter: &mut usize)");
                assert_eq!(got, "function `step` not found");
            }
            _ => panic!("Invalid error"),
        }
    }

    #[test]
    fn argument_names_are_ignored() {
        // Renaming an argument (e.g. marking it unused) is ABI-compatible:
        // dylib dispatch is by function name only.
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(_counter: &mut usize) {
    *_counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected, &denied())
            .expect("renamed argument must validate");
    }

    #[test]
    fn abi_incompatible_modifiers_are_reported() {
        // `unsafe` modifiers are intercepted by the unsafe scan first.
        for input in [
            "pub unsafe fn step(counter: &mut usize) {}",
            "pub unsafe extern \"C\" fn step(counter: &mut usize, ...) {}",
        ] {
            let mut file = syn::parse_str(input).expect("can parse");
            let expected = vec!["fn step(counter: &mut usize)".to_string()];
            let err = validate_generated_ast(&mut file, &expected, &denied())
                .expect_err("unsafe signature must be rejected");
            assert!(
                matches!(
                    err,
                    Error::UnsafeCode { ref construct, .. } if construct.contains("an `unsafe fn`")
                ),
                "feedback must identify the unsafe fn in {input}: {err}"
            );
        }

        // Safe but ABI-incompatible modifiers surface as signature mismatches.
        for (input, modifier) in [
            ("pub async fn step(counter: &mut usize) {}", "async"),
            (
                "pub extern \"C\" fn step(counter: &mut usize) {}",
                "extern \"C\"",
            ),
        ] {
            let mut file = syn::parse_str(input).expect("can parse");
            let expected = vec!["fn step(counter: &mut usize)".to_string()];
            let err = validate_generated_ast(&mut file, &expected, &denied())
                .expect_err("incompatible signature must be rejected");
            assert!(
                matches!(
                    err,
                    Error::SignatureMismatch { ref got, .. } if got.contains(modifier)
                ),
                "feedback must identify `{modifier}` in {input}: {err}"
            );
        }
    }

    #[test]
    fn mismatch_names_the_offending_function() {
        // `action` matches but `on_order_update` does not; the error must
        // point at `on_order_update`, not at the first expected signature.
        let input = "```rust
#[unsafe(no_mangle)]
pub fn action(tick: &Tick) {
    let _ = tick;
}
#[unsafe(no_mangle)]
pub fn on_order_update(update: &Update, extra: bool) {
    let _ = (update, extra);
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec![
            "fn action(tick: & Tick)".to_string(),
            "fn on_order_update(update: & Update)".to_string(),
        ];
        let err =
            validate_generated_ast(&mut file, &expected, &denied()).expect_err("should error");
        match err {
            Error::SignatureMismatch { expected, got, .. } => {
                assert_eq!(expected, "fn on_order_update(update: & Update)");
                assert_eq!(got, "fn on_order_update(& Update, bool)");
            }
            _ => panic!("Invalid error"),
        }
    }

    #[test]
    fn test_validate_unsafe_no_mangle() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize)".to_string()];
        validate_generated_ast(&mut file, &expected, &denied())
            .expect("#[unsafe(no_mangle)] should be valid");
    }

    #[test]
    fn multi_fn_with_renamed_unused_argument_validates() {
        // Regression test for a stuck self-healing loop: the generated
        // `on_order_update` renamed the unused `market_state` parameter to
        // `_market_state`, which must not be a signature mismatch.
        let input = "```rust
#[unsafe(no_mangle)]
pub fn action(
    step_data: &TickData,
    account: &Account<i64, DECIMALS, Cur, UserOrderId>,
    market_state: &MarketState<i64, DECIMALS>,
    account_tracker: &FullAccountTracker<DECIMALS, Cur>,
    commands: &mut CommandBuffer<DECIMALS, Cur>,
) {
}
#[unsafe(no_mangle)]
pub fn on_order_update(
    order_update: OrderUpdate<DECIMALS, Cur>,
    account: &Account<i64, DECIMALS, Cur, UserOrderId>,
    _market_state: &MarketState<i64, DECIMALS>,
    commands: &mut CommandBuffer<DECIMALS, Cur>,
) {
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec![
            "fn action(step_data: & TickData, account: & Account < i64, DECIMALS, Cur, UserOrderId >, market_state: & MarketState < i64, DECIMALS >, account_tracker: & FullAccountTracker < DECIMALS, Cur >, commands: &mut CommandBuffer < DECIMALS, Cur >)".to_string(),
            "fn on_order_update(order_update: OrderUpdate < DECIMALS, Cur >, account: & Account < i64, DECIMALS, Cur, UserOrderId >, market_state: & MarketState < i64, DECIMALS >, commands: &mut CommandBuffer < DECIMALS, Cur >)".to_string(),
        ];
        validate_generated_ast(&mut file, &expected, &denied())
            .expect("renamed `_market_state` argument must validate");
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_validate_with_return_type() {
        let input = "```rust
#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) -> usize {
    *counter
}
```";
        let mut file = parse_rust_code(input).expect("can parse");
        let expected = vec!["fn step(counter: &mut usize) -> usize".to_string()];
        validate_generated_ast(&mut file, &expected, &denied())
            .expect("validation with return type passed");
    }

    /// A `default_bodies` map holding the one function declared by
    /// `default_src`.
    fn bodies_for(default_src: &str) -> HashMap<String, String> {
        let item: syn::ItemFn = syn::parse_str(default_src).expect("default body parses");
        let mut bodies = HashMap::new();
        bodies.insert(item.sig.ident.to_string(), default_body_tokens(&item));
        bodies
    }
    /// A `default_bodies` map for the two declared functions `step_src` and
    /// `bump_src`.
    fn two_bodies(step_src: &str, bump_src: &str) -> HashMap<String, String> {
        let mut bodies = bodies_for(step_src);
        let bump: syn::ItemFn = syn::parse_str(bump_src).expect("default body parses");
        bodies.insert(bump.sig.ident.to_string(), default_body_tokens(&bump));
        bodies
    }
    /// Assert `check_implementation_bodies` rejects `code` with
    /// `fn_name` / `reason`.
    fn assert_rejects_unimplemented(code: &str, bodies: &HashMap<String, String>, reason: &str) {
        let file: syn::File = syn::parse_str(code).expect("can parse");
        let err = check_implementation_bodies(&file, bodies)
            .expect_err("unimplemented body must be rejected");
        match err {
            Error::UnimplementedFunction {
                fn_name,
                reason: found,
                ..
            } => {
                assert_eq!(fn_name, "step", "for `{code}`");
                assert_eq!(found, reason, "for `{code}`");
            }
            other => panic!("expected UnimplementedFunction in `{code}`, got {other:?}"),
        }
    }

    #[test]
    fn empty_body_is_rejected() {
        let bodies = bodies_for("pub fn step(counter: &mut usize) { *counter += 1; }");
        assert_rejects_unimplemented(
            "pub fn step(counter: &mut usize) {}",
            &bodies,
            "the body is empty",
        );
        // A body of only comments is empty as far as the token stream is
        // concerned.
        assert_rejects_unimplemented(
            "pub fn step(counter: &mut usize) {\n    // will be implemented\n}",
            &bodies,
            "the body is empty",
        );
    }

    #[test]
    fn placeholder_macro_bodies_are_rejected() {
        let bodies = bodies_for("pub fn step(counter: &mut usize) { *counter += 1; }");
        for (macro_name, reason) in [
            ("todo", "the body is only `todo!()`"),
            ("unimplemented", "the body is only `unimplemented!()`"),
            ("unreachable", "the body is only `unreachable!()`"),
        ] {
            for code in [
                &format!("pub fn step(counter: &mut usize) {{ {macro_name}!(); }}"),
                &format!("pub fn step(counter: &mut usize) {{ {macro_name}!() }}"),
            ] {
                assert_rejects_unimplemented(code, &bodies, reason);
            }
        }
        // A placeholder next to real work is an implementation, not a stub.
        let file: syn::File =
            syn::parse_str("pub fn step(counter: &mut usize) { *counter += 1; todo!(); }")
                .expect("can parse");
        check_implementation_bodies(&file, &bodies).expect("placeholder with real work must pass");
    }

    #[test]
    fn verbatim_default_echo_is_rejected() {
        let bodies = bodies_for("pub fn step(counter: &mut usize) { *counter += 1; }");
        // Formatted differently and carrying a comment: token streams are
        // insensitive to both, so it still counts as an echo.
        let code =
            "pub fn step(counter: &mut usize) {\n    // keeping the default\n    *counter+= 1;\n}";
        assert_rejects_unimplemented(
            code,
            &bodies,
            "the body is identical to the default implementation",
        );
    }

    #[test]
    fn echo_with_renamed_argument_is_rejected() {
        // The signature check ignores argument names, so the echo check must
        // not be evadable by renaming an unused parameter.
        let bodies = bodies_for("pub fn step(counter: &mut usize) { *counter += 1; }");
        assert_rejects_unimplemented(
            "pub fn step(_counter: &mut usize) { *_counter += 1; }",
            &bodies,
            "the body is identical to the default implementation",
        );
    }

    #[test]
    fn changed_body_is_accepted() {
        let bodies = bodies_for("pub fn step(counter: &mut usize) { *counter += 1; }");
        let file: syn::File = syn::parse_str("pub fn step(counter: &mut usize) { *counter += 7; }")
            .expect("can parse");
        check_implementation_bodies(&file, &bodies).expect("a genuine evolution must pass");
    }

    #[test]
    fn helper_functions_are_not_checked() {
        // Only declared (evolvable) functions are checked; a helper with an
        // empty body is the model's business.
        let bodies = bodies_for("pub fn step(counter: &mut usize) { *counter += 1; }");
        let file: syn::File =
            syn::parse_str("fn helper() {}\npub fn step(counter: &mut usize) { *counter += 7; }")
                .expect("can parse");
        check_implementation_bodies(&file, &bodies).expect("helpers must not be checked");
    }

    #[test]
    fn partial_evolution_with_one_default_body_is_accepted() {
        let bodies = two_bodies(
            "pub fn step(counter: &mut usize) { *counter += 1; }",
            "pub fn bump(count: usize) -> usize { count + 1 }",
        );
        // `step` evolves, `bump` keeps its default body: a partial evolution
        // must land, or a multi-function host could never improve one
        // function at a time.
        let file: syn::File = syn::parse_str(
            "pub fn step(counter: &mut usize) { *counter += 7; }\n\
             pub fn bump(count: usize) -> usize { count + 1 }",
        )
        .expect("can parse");
        check_implementation_bodies(&file, &bodies).expect("a partial evolution must pass");
    }

    #[test]
    fn all_default_bodies_are_rejected() {
        let bodies = two_bodies(
            "pub fn step(counter: &mut usize) { *counter += 1; }",
            "pub fn bump(count: usize) -> usize { count + 1 }",
        );
        let file: syn::File = syn::parse_str(
            "pub fn step(counter: &mut usize) { *counter += 1; }\n\
             pub fn bump(count: usize) -> usize { count + 1 }",
        )
        .expect("can parse");
        let err = check_implementation_bodies(&file, &bodies)
            .expect_err("a candidate with no changed body must be rejected");
        match err {
            Error::UnimplementedFunction {
                fn_name,
                reason: found,
                ..
            } => {
                assert_eq!(fn_name, "step", "the first unchanged function is named");
                assert_eq!(found, "the body is identical to the default implementation");
            }
            other => panic!("expected UnimplementedFunction, got {other:?}"),
        }
    }

    #[test]
    fn stub_is_rejected_even_beside_a_genuine_implementation() {
        let bodies = two_bodies(
            "pub fn step(counter: &mut usize) { *counter += 1; }",
            "pub fn bump(count: usize) -> usize { count + 1 }",
        );
        // `step` genuinely evolves, but `bump` is a placeholder that would
        // panic at run time: stubs reject regardless of the other functions.
        let file: syn::File = syn::parse_str(
            "pub fn step(counter: &mut usize) { *counter += 7; }\n\
             pub fn bump(count: usize) -> usize { todo!() }",
        )
        .expect("can parse");
        let err = check_implementation_bodies(&file, &bodies)
            .expect_err("a stub must be rejected even in a partial evolution");
        match err {
            Error::UnimplementedFunction {
                fn_name,
                reason: found,
                ..
            } => {
                assert_eq!(fn_name, "bump");
                assert_eq!(found, "the body is only `todo!()`");
            }
            other => panic!("expected UnimplementedFunction, got {other:?}"),
        }
    }

    /// Assert `enforce_code_policy` rejects `code` with an unsafe finding
    /// naming `construct`.
    fn assert_rejects_unsafe(code: &str, construct: &str) {
        let file: syn::File = syn::parse_str(code).expect("can parse");
        let err =
            enforce_code_policy(&file, &denied()).expect_err("unsafe construct must be rejected");
        match err {
            Error::UnsafeCode {
                construct: found, ..
            } => {
                assert!(
                    found.contains(construct),
                    "expected construct `{construct}`, got `{found}`"
                );
            }
            other => panic!("expected Error::UnsafeCode, got {other}"),
        }
    }

    #[test]
    fn unsafe_block_is_rejected() {
        assert_rejects_unsafe(
            "pub fn step(counter: &mut usize) { unsafe { *counter += 1; } }",
            "an `unsafe` block",
        );
    }

    #[test]
    fn unsafe_fn_is_rejected() {
        assert_rejects_unsafe("pub unsafe fn helper(ptr: *mut usize) {}", "an `unsafe fn`");
        assert_rejects_unsafe(
            "struct S; impl S { unsafe fn helper(&self) {} }",
            "an `unsafe fn`",
        );
    }

    #[test]
    fn unsafe_impl_and_trait_are_rejected() {
        assert_rejects_unsafe("struct S; unsafe impl Send for S {}", "an `unsafe impl`");
        assert_rejects_unsafe("unsafe trait Scary {}", "an `unsafe trait`");
    }

    #[test]
    fn extern_block_is_rejected() {
        assert_rejects_unsafe(
            "unsafe extern \"C\" { fn malloc(size: usize) -> *mut u8; }",
            "an `extern` block",
        );
    }

    #[test]
    fn unsafe_attribute_is_rejected_but_no_mangle_is_allowed() {
        assert_rejects_unsafe(
            "#[unsafe(export_name = \"evil\")] pub fn step(counter: &mut usize) {}",
            "an unsafe attribute",
        );

        let file: syn::File = syn::parse_str(
            "#[unsafe(no_mangle)] pub fn step(counter: &mut usize) { *counter += 1; }",
        )
        .expect("can parse");
        enforce_code_policy(&file, &denied())
            .expect("#[unsafe(no_mangle)] is harness-managed and allowed");
    }

    #[test]
    fn unsafe_smuggled_through_macro_is_rejected() {
        // Defining a smuggling macro is already rejected as a macro
        // definition; invocation-side smuggling is caught by the token scan.
        assert_rejects_unsafe(
            "pub fn step(counter: &mut usize) { let _ = stringify!(unsafe); }",
            "an `unsafe` token inside a macro",
        );
    }

    #[test]
    fn safe_code_passes_unsafe_scan() {
        let file: syn::File = syn::parse_str(
            "pub fn step(counter: &mut usize) {\n\
                 let values = vec![1usize, 2, 3];\n\
                 *counter += values.iter().sum::<usize>();\n\
                 println!(\"counter is {counter}\");\n\
             }",
        )
        .expect("can parse");
        enforce_code_policy(&file, &denied()).expect("safe code must pass");
    }

    /// Assert `enforce_code_policy` rejects `code` with a forbidden finding
    /// naming `construct`.
    fn assert_rejects_forbidden(code: &str, construct: &str) {
        let file: syn::File = syn::parse_str(code).expect("can parse");
        let err = enforce_code_policy(&file, &denied())
            .expect_err("forbidden construct must be rejected");
        match err {
            Error::ForbiddenConstruct {
                construct: found, ..
            } => {
                assert!(
                    found.contains(construct),
                    "expected construct `{construct}`, got `{found}`"
                );
            }
            other => panic!("expected Error::ForbiddenConstruct, got {other}"),
        }
    }

    #[test]
    fn static_items_are_rejected() {
        assert_rejects_forbidden(
            "static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);\n\
             pub fn step(counter: &mut usize) { *counter += 1; }",
            "a `static` item",
        );
    }

    #[test]
    fn thread_local_is_rejected() {
        assert_rejects_forbidden(
            "thread_local! { static CACHE: std::cell::Cell<usize> = std::cell::Cell::new(0); }\n\
             pub fn step(counter: &mut usize) {}",
            "a `thread_local!` declaration",
        );
    }

    #[test]
    fn macro_definitions_are_rejected() {
        assert_rejects_forbidden(
            "macro_rules! helper { () => { 1 }; }\npub fn step(counter: &mut usize) { *counter += helper!(); }",
            "a `macro_rules!` definition",
        );
    }

    #[test]
    fn runtime_hijacking_attributes_are_rejected() {
        assert_rejects_forbidden(
            "#[panic_handler] fn handle(info: &PanicInfo) -> ! { loop {} }",
            "a `#[panic_handler]` attribute",
        );
        // `#[global_allocator]` requires a static item, which the static
        // rule already rejects first.
        assert_rejects_forbidden(
            "#[global_allocator] static A: std::alloc::System = std::alloc::System;",
            "a `static` item",
        );
    }

    #[test]
    fn panic_hook_tampering_is_rejected() {
        assert_rejects_forbidden(
            "pub fn step(counter: &mut usize) { std::panic::set_hook(Box::new(|_| {})); }",
            "a use of `std::panic::set_hook`",
        );
    }

    #[test]
    fn denied_std_paths_are_rejected() {
        assert_rejects_forbidden(
            "pub fn step(counter: &mut usize) { std::process::exit(0); }",
            "a use of `std::process`",
        );
        assert_rejects_forbidden(
            "use std::fs::File;\npub fn step(counter: &mut usize) {}",
            "a use of `std::fs`",
        );
        assert_rejects_forbidden(
            "pub fn step(counter: &mut usize) { std::thread::sleep(std::time::Duration::from_secs(1)); }",
            "a use of `std::thread`",
        );
    }

    #[test]
    fn denied_paths_behind_aliases_are_rejected() {
        // Renamed leaf import.
        assert_rejects_forbidden(
            "use std::process::exit as quit;\npub fn step(counter: &mut usize) { quit(0); }",
            "a use of `std::process`",
        );
        // Renamed crate root.
        assert_rejects_forbidden(
            "use std as s;\npub fn step(counter: &mut usize) { s::process::abort(); }",
            "a use of `std::process`",
        );
        // Usage through an imported parent module.
        assert_rejects_forbidden(
            "use std::io;\npub fn step(counter: &mut usize) { let mut s = String::new(); let _ = io::stdin().read_line(&mut s); }",
            "a use of `std::io::stdin`",
        );
    }

    #[test]
    fn glob_import_of_denied_module_is_rejected() {
        assert_rejects_forbidden(
            "use std::process::*;\npub fn step(counter: &mut usize) {}",
            "a glob import of `std::process`",
        );
    }

    #[test]
    fn denied_path_inside_macro_is_rejected() {
        assert_rejects_forbidden(
            "pub fn step(counter: &mut usize) { let _ = stringify!(std::process::exit(0)); }",
            "a use of `std::process` inside a macro",
        );
    }

    #[test]
    fn allowed_path_configuration_is_respected() {
        let code = "pub fn step(counter: &mut usize) { let _ = std::fs::read(\"data.bin\"); }";
        let file: syn::File = syn::parse_str(code).expect("can parse");

        // Denied by default.
        enforce_code_policy(&file, &denied()).expect_err("std::fs is denied by default");

        // Allowed once the host opts out.
        let relaxed = crate::DylibConfig::standalone(crate::Profile::Debug)
            .with_allowed_path("std::fs")
            .denied_paths()
            .clone();
        enforce_code_policy(&file, &relaxed).expect("std::fs was explicitly allowed");
    }

    #[test]
    fn custom_denied_path_is_enforced() {
        let code = "pub fn step(counter: &mut usize) { host::dangerous::wipe(); }";
        let file: syn::File = syn::parse_str(code).expect("can parse");

        enforce_code_policy(&file, &denied()).expect("host paths are allowed by default");

        let strict = crate::DylibConfig::standalone(crate::Profile::Debug)
            .with_denied_path("host::dangerous")
            .denied_paths()
            .clone();
        enforce_code_policy(&file, &strict).expect_err("custom denied path must be enforced");
    }

    #[test]
    fn panic_hook_paths_stay_denied_with_empty_config() {
        let file: syn::File =
            syn::parse_str("pub fn step(counter: &mut usize) { let _ = std::panic::take_hook(); }")
                .expect("can parse");
        enforce_code_policy(&file, &[]).expect_err("hook tampering is always denied");
    }
}
