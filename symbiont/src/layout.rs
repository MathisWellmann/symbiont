// SPDX-License-Identifier: MPL-2.0
//! The layout of the generated dylib's `src/lib.rs`.
//!
//! ```text
//! <candidate>                     the agent's code, byte for byte, from offset 0
//!
//! // ---- symbiont: harness glue ----
//! <prelude>                       configured imports and inline `evolvable!` items
//! mod __symbiont_panic { .. }     the panic-buffer protocol (`unwind.rs`)
//! mod __symbiont_state { .. }     the state protocol (`state.rs`), only with `type State`
//! mod __symbiont_exports { .. }   one exported `catch_unwind` wrapper per declared fn,
//!                                 plus the state constructor, destructor and tag
//! ```
//!
//! The candidate comes first, unmodified, so that every compiler location
//! inside it is a location in the agent's own text: line 6 of a diagnostic
//! is line 6 of the code block the agent wrote, and a byte offset indexes
//! the candidate string directly. Everything the harness adds follows. Item
//! order is irrelevant to Rust, so the glue can live at the end.
//!
//! # Exports
//!
//! The agent's functions are never exported themselves. For every declared
//! function the glue emits a wrapper that carries the export and forwards to
//! the agent's implementation:
//!
//! ```ignore
//! mod __symbiont_exports {
//!     use super::*;
//!     #[unsafe(export_name = "sort")]
//!     pub fn __symbiont_export_sort(__symbiont_arg_0: &mut [f64], __symbiont_arg_1: usize) {
//!         super::__symbiont_panic::__symbiont_install_panic_hook();
//!         match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
//!             super::sort(__symbiont_arg_0, __symbiont_arg_1)
//!         })) {
//!             Ok(v) => v,
//!             Err(e) => { /* store the payload as a fallback */ Default::default() }
//!         }
//!     }
//! }
//! ```
//!
//! This keeps the agent's text untouched (no attribute is injected, no body
//! is rewritten) and it makes the export set exactly the declared functions:
//! a helper the agent invents keeps its mangled name. That matters because
//! an exported symbol in an ELF dylib is preemptible, and a helper named like
//! a libc function (`qsort`, `random`, ...) would otherwise hijack the
//! dylib's own call to it. Validation rejects export attributes in the
//! candidate for the same reason.
//!
//! # Only the entry points catch panics
//!
//! The wrapper is the only place a panic can escape into the host, so it is
//! the only place with a `catch_unwind` frame. Wrapping the agent's helpers
//! too would be actively harmful:
//!
//! - Every call, including each level of a recursive helper, would set up a
//!   `catch_unwind` frame and re-install the panic hook. That is call
//!   overhead and stack footprint in exactly the hot, deeply recursive code
//!   this harness exists to optimize; enlarged frames have overflowed the
//!   shared stack before.
//! - A panicking helper would return `Default::default()` to its *caller*,
//!   which would then keep computing with a placeholder value. Letting the
//!   unwind travel to the exported entry point aborts the whole call
//!   instead, which is what the host's `take_panic` contract describes.
//! - Helpers would have to return `Default` types. The `evolvable!` macro
//!   enforces that for declared signatures, but a helper returning, say,
//!   `Ordering` or `&f64` is perfectly reasonable.
//!
//! # State arguments
//!
//! With a declared `type State`, an argument `&mut State` or `&State` of a
//! declared function is the host's opaque `symbiont::State` handle. The
//! wrapper takes it as `&mut Handle` or `&Handle`, a reference to the
//! handle's `repr(C)` prefix with the same borrow, checks that
//! this dylib created it, and passes the agent's own `State` on (see
//! [`crate::state`]). The check is outside `catch_unwind`: a state of another
//! revision is a host bug, and it aborts.
//!
//! The `Err` arm substitutes `Default::default()` as a placeholder return
//! value, which is safe for every type (a zeroed value would be undefined
//! behaviour for `String` or `&T`). The panic *message with its source
//! location* is recorded by the hook installed via
//! `__symbiont_install_panic_hook`: the hook runs at panic time, the only
//! point where `std::panic::Location` is available. The `Err` arm only stores
//! the location-less payload as a fallback for panics that bypassed the hook.

use std::fmt::Write as _;

use proc_macro2::TokenStream;
use quote::{
    format_ident,
    quote,
};

use crate::{
    EXPECT_WRITE,
    EvolvableDecl,
    StateDecl,
    state::{
        DROP_SYMBOL,
        NEW_SYMBOL,
        STATE_MODULE,
        STATE_PREAMBLE,
        TAG_SYMBOL,
    },
    unwind::{
        PANIC_MODULE,
        PANIC_PREAMBLE,
    },
};

/// The name of the declared state type in the agent's code.
const STATE_IDENT: &str = "State";

/// The module holding the export wrappers.
pub(crate) const EXPORTS_MODULE: &str = "__symbiont_exports";

/// The line that separates the candidate from the harness glue.
pub(crate) const GLUE_MARKER: &str =
    "// ---- symbiont: harness glue. Generated after the candidate, never edited. ----";

/// The `lib.rs` of the dylib crate: the candidate followed by `glue`.
///
/// The candidate is written verbatim from byte 0. A newline separates it
/// from the glue so the glue's first line never continues the candidate's
/// last one.
pub(crate) fn assemble_lib_rs(candidate: &str, glue: &str) -> String {
    let mut lib_rs = String::with_capacity(candidate.len() + glue.len() + 2);
    lib_rs.push_str(candidate);
    if !candidate.ends_with('\n') {
        lib_rs.push('\n');
    }
    lib_rs.push('\n');
    lib_rs.push_str(glue);
    lib_rs
}

/// Everything the harness appends after the candidate. Constant for the
/// lifetime of a runtime, so [`crate::Runtime::new`] renders it once.
pub(crate) fn harness_glue(decls: &[EvolvableDecl], prelude: &[String]) -> String {
    let mut glue = String::with_capacity(8 * 1024);
    glue.push_str(GLUE_MARKER);
    glue.push('\n');
    for part in prelude.iter().filter(|part| !part.is_empty()) {
        glue.push_str(part);
        if !part.ends_with('\n') {
            glue.push('\n');
        }
    }
    writeln!(glue, "\nmod {PANIC_MODULE} {{\n{PANIC_PREAMBLE}}}\n").expect(EXPECT_WRITE);
    if state_decl(decls).is_some() {
        writeln!(glue, "mod {STATE_MODULE} {{\n{STATE_PREAMBLE}}}\n").expect(EXPECT_WRITE);
    }
    glue.push_str(&export_wrappers(decls));
    glue
}

/// The `type State` declaration shared by `decls`, if the block has one.
///
/// # Panics
///
/// If some declarations carry one and others do not, or they carry
/// different ones: `evolvable!` gives every declaration of a block the same.
pub(crate) fn state_decl(decls: &[EvolvableDecl]) -> Option<&'static StateDecl> {
    let first = decls.first()?.state;
    for decl in decls {
        match (first, decl.state) {
            (None, None) => {}
            (Some(a), Some(b)) => assert!(
                std::ptr::eq(a, b),
                "declarations of one evolvable! block share one StateDecl"
            ),
            _ => panic!("either every declaration has the state or none has"),
        }
    }
    first
}

/// The candidate of the initial revision: every declared function with its
/// default body, as `evolvable!` rendered it.
pub(crate) fn initial_candidate(decls: &[EvolvableDecl]) -> String {
    let mut candidate = String::with_capacity(1024);
    if let Some(state) = state_decl(decls) {
        candidate.push_str(state.initial);
        if !state.initial.ends_with('\n') {
            candidate.push('\n');
        }
    }
    for (idx, decl) in decls.iter().enumerate() {
        if idx > 0 || !candidate.is_empty() {
            candidate.push('\n');
        }
        candidate.push_str(decl.full_source);
        if !decl.full_source.ends_with('\n') {
            candidate.push('\n');
        }
    }
    candidate
}

/// The `mod __symbiont_exports { .. }` item with one wrapper per declaration.
fn export_wrappers(decls: &[EvolvableDecl]) -> String {
    let module = format_ident!("{EXPORTS_MODULE}");
    let has_state = state_decl(decls).is_some();
    let wrappers = decls.iter().map(|decl| export_wrapper(decl, has_state));
    let state = state_decl(decls).map(state_exports);
    let file: syn::File = syn::parse_quote! {
        mod #module {
            use super::*;
            #state
            #(#wrappers)*
        }
    };
    prettyplease::unparse(&file)
}

/// The exported constructor, destructor and tag of the agent's `State`,
/// and the assertion that it satisfies the declared bounds.
fn state_exports(state: &StateDecl) -> TokenStream {
    let bounds: syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]> =
        syn::parse::Parser::parse_str(
            syn::punctuated::Punctuated::parse_separated_nonempty,
            state.bounds,
        )
        .expect("StateDecl::bounds is generated by evolvable! and must parse");
    let state_module = format_ident!("{STATE_MODULE}");
    let panic_module = format_ident!("{PANIC_MODULE}");
    let state_ident = format_ident!("{STATE_IDENT}");
    let (new_symbol, drop_symbol, tag_symbol) = (NEW_SYMBOL, DROP_SYMBOL, TAG_SYMBOL);
    let store_panic = store_panic_fallback(&panic_module);
    quote! {
        const _: fn() = || {
            fn __symbiont_state_satisfies_the_declared_bounds<T: #bounds>() {}
            __symbiont_state_satisfies_the_declared_bounds::<super::#state_ident>();
        };

        #[unsafe(export_name = #new_symbol)]
        pub fn __symbiont_export_state_new() -> *mut () {
            super::#panic_module::__symbiont_install_panic_hook();
            match ::std::panic::catch_unwind(super::#state_module::new::<super::#state_ident>) {
                ::core::result::Result::Ok(__symbiont_val) => __symbiont_val,
                ::core::result::Result::Err(__symbiont_err) => {
                    #store_panic
                    ::core::ptr::null_mut()
                }
            }
        }

        /// # Safety
        ///
        /// `data` must come from `__symbiont_export_state_new` of this dylib.
        #[unsafe(export_name = #drop_symbol)]
        pub unsafe fn __symbiont_export_state_drop(data: *mut ()) {
            super::#panic_module::__symbiont_install_panic_hook();
            // A panicking `Drop` leaks the rest of the value; it must not
            // unwind into the host.
            if let ::core::result::Result::Err(__symbiont_err) =
                ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| unsafe {
                    super::#state_module::drop_state::<super::#state_ident>(data)
                }))
            {
                #store_panic
            }
        }

        #[unsafe(export_name = #tag_symbol)]
        pub fn __symbiont_export_state_tag() -> usize {
            super::#state_module::tag()
        }
    }
}

/// Store the payload of a caught panic `__symbiont_err` as a fallback.
fn store_panic_fallback(panic_module: &syn::Ident) -> TokenStream {
    quote! {
        let __symbiont_msg = if let ::core::option::Option::Some(s) =
            __symbiont_err.downcast_ref::<&str>()
        {
            *s
        } else if let ::core::option::Option::Some(s) =
            __symbiont_err.downcast_ref::<::std::string::String>()
        {
            s.as_str()
        } else {
            "unknown panic"
        };
        super::#panic_module::__symbiont_store_panic_fallback(__symbiont_msg);
    }
}

/// How an argument crosses the boundary.
enum ArgKind {
    /// As written in the declaration.
    Plain,
    /// The host's state handle for a `&mut State`.
    StateMut,
    /// The host's state handle for a `&State`.
    StateRef,
}

/// The kind of an argument of type `ty`. Only a block with a declared
/// state has state arguments.
fn arg_kind(ty: &syn::Type, has_state: bool) -> ArgKind {
    let syn::Type::Reference(reference) = ty else {
        return ArgKind::Plain;
    };
    let is_state = matches!(
        reference.elem.as_ref(),
        syn::Type::Path(path) if path.qself.is_none() && path.path.is_ident(STATE_IDENT)
    );
    match (has_state && is_state, reference.mutability.is_some()) {
        (false, _) => ArgKind::Plain,
        (true, true) => ArgKind::StateMut,
        (true, false) => ArgKind::StateRef,
    }
}

/// The exported `catch_unwind` wrapper of one declared function.
fn export_wrapper(decl: &EvolvableDecl, has_state: bool) -> TokenStream {
    let item: syn::ItemFn = syn::parse_str(decl.full_source)
        .expect("full_source is generated by evolvable! and must parse");
    let target = &item.sig.ident;
    let symbol = decl.name;
    let wrapper = format_ident!("__symbiont_export_{}", decl.name);
    let panic_module = format_ident!("{PANIC_MODULE}");
    let state_module = format_ident!("{STATE_MODULE}");
    let state_ident = format_ident!("{STATE_IDENT}");
    let args: Vec<syn::Ident> = (0..item.sig.inputs.len())
        .map(|idx| format_ident!("__symbiont_arg_{idx}"))
        .collect();
    let mut params = Vec::with_capacity(args.len());
    let mut unwrap_state = Vec::new();
    for (arg, input) in args.iter().zip(&item.sig.inputs) {
        let syn::FnArg::Typed(pat) = input else {
            unreachable!("evolvable! rejects self receivers");
        };
        let ty = pat.ty.as_ref();
        let (handle, access) = match arg_kind(ty, has_state) {
            ArgKind::Plain => {
                params.push(quote! { #arg: #ty });
                continue;
            }
            ArgKind::StateMut => (quote! { &mut }, quote! { state_mut }),
            ArgKind::StateRef => (quote! { & }, quote! { state_ref }),
        };
        params.push(quote! { #arg: #handle super::#state_module::Handle });
        unwrap_state.push(quote! {
            // SAFETY: the host passes its `symbiont::State`, whose `repr(C)`
            // prefix is `Handle`, borrowed as the declaration says;
            // `state_*` aborts unless this dylib created it.
            let #arg = unsafe {
                super::#state_module::#access::<super::#state_ident>(#arg, #symbol)
            };
        });
    }
    let output = &item.sig.output;
    let store_panic = store_panic_fallback(&panic_module);
    quote! {
        #[unsafe(export_name = #symbol)]
        pub fn #wrapper(#(#params),*) #output {
            #(#unwrap_state)*
            super::#panic_module::__symbiont_install_panic_hook();
            match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                super::#target(#(#args),*)
            })) {
                ::core::result::Result::Ok(__symbiont_val) => __symbiont_val,
                ::core::result::Result::Err(__symbiont_err) => {
                    #store_panic
                    ::core::default::Default::default()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicPtr;

    use super::*;

    static FN_PTR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

    fn decl(
        name: &'static str,
        signature: &'static str,
        full_source: &'static str,
    ) -> EvolvableDecl {
        EvolvableDecl {
            name,
            signature,
            full_source,
            fn_ptr: &FN_PTR,
            state: None,
        }
    }

    #[test]
    fn candidate_is_the_byte_prefix_of_lib_rs() {
        let candidate = "pub fn step(counter: &mut usize) {\n    *counter += 5;\n}";
        let lib_rs = assemble_lib_rs(candidate, "// glue");
        assert!(lib_rs.starts_with(candidate));
        assert_eq!(&lib_rs[candidate.len()..], "\n\n// glue");
        let lines = candidate.lines().count();
        assert_eq!(lib_rs.lines().nth(lines - 1), candidate.lines().last());
    }

    #[test]
    fn candidate_with_trailing_newline_gets_no_extra_one() {
        let lib_rs = assemble_lib_rs("fn f() {}\n", "// glue");
        assert_eq!(lib_rs, "fn f() {}\n\n// glue");
    }

    #[test]
    fn export_wrapper_forwards_to_the_candidate_function() {
        let wrappers = export_wrappers(&[decl(
            "sort",
            "fn sort(data: &mut [f64], len: usize)",
            "pub fn sort(data: &mut [f64], len: usize) {\n    let _ = (data, len);\n}\n",
        )]);
        assert_eq!(
            wrappers,
            r#"mod __symbiont_exports {
    use super::*;
    #[unsafe(export_name = "sort")]
    pub fn __symbiont_export_sort(
        __symbiont_arg_0: &mut [f64],
        __symbiont_arg_1: usize,
    ) {
        super::__symbiont_panic::__symbiont_install_panic_hook();
        match ::std::panic::catch_unwind(
            ::std::panic::AssertUnwindSafe(|| {
                super::sort(__symbiont_arg_0, __symbiont_arg_1)
            }),
        ) {
            ::core::result::Result::Ok(__symbiont_val) => __symbiont_val,
            ::core::result::Result::Err(__symbiont_err) => {
                let __symbiont_msg = if let ::core::option::Option::Some(s) = __symbiont_err
                    .downcast_ref::<&str>()
                {
                    *s
                } else if let ::core::option::Option::Some(s) = __symbiont_err
                    .downcast_ref::<::std::string::String>()
                {
                    s.as_str()
                } else {
                    "unknown panic"
                };
                super::__symbiont_panic::__symbiont_store_panic_fallback(__symbiont_msg);
                ::core::default::Default::default()
            }
        }
    }
}
"#
        );
    }

    #[test]
    fn export_wrapper_keeps_the_return_type() {
        let wrappers = export_wrappers(&[decl(
            "pick",
            "fn pick(data: &[usize], idx: usize) -> usize",
            "pub fn pick(data: &[usize], idx: usize) -> usize {\n    data[idx]\n}\n",
        )]);
        assert!(wrappers.contains(
            "pub fn __symbiont_export_pick(\n        __symbiont_arg_0: &[usize],\n        __symbiont_arg_1: usize,\n    ) -> usize {"
        ));
        assert!(wrappers.contains("super::pick(__symbiont_arg_0, __symbiont_arg_1)"));
    }

    #[test]
    fn glue_holds_prelude_panic_module_and_exports_in_order() {
        let glue = harness_glue(
            &[decl(
                "step",
                "fn step(counter: &mut usize)",
                "pub fn step(counter: &mut usize) {\n    *counter += 1;\n}\n",
            )],
            &[String::new(), "use host::prelude::*;".to_string()],
        );
        let marker = glue.find(GLUE_MARKER).expect("marker");
        let prelude = glue.find("use host::prelude::*;\n").expect("prelude");
        let panic = glue.find("mod __symbiont_panic {\n").expect("panic module");
        let take_panic = glue
            .find("pub unsafe fn __symbiont_take_panic")
            .expect("preamble body");
        let exports = glue.find("mod __symbiont_exports {\n").expect("exports");
        assert!(marker < prelude && prelude < panic && panic < take_panic && take_panic < exports);
        // The glue parses as Rust on its own.
        syn::parse_file(&glue).expect("glue is valid Rust");
    }

    static STATE: StateDecl = StateDecl {
        bounds: "Default + Send + 'static",
        initial: "#[derive(Default)]\nstruct State {\n    calls: usize,\n}\n",
    };

    fn stateful(name: &'static str, full_source: &'static str) -> EvolvableDecl {
        EvolvableDecl {
            state: Some(&STATE),
            ..decl(name, "", full_source)
        }
    }

    #[test]
    fn state_arguments_cross_as_handles() {
        let wrappers = export_wrappers(&[
            stateful(
                "step",
                "pub fn step(state: &mut State, x: f64) -> f64 {\n    x\n}\n",
            ),
            stateful("peek", "pub fn peek(state: &State) -> usize {\n    0\n}\n"),
        ]);
        syn::parse_file(&wrappers).expect("wrappers are valid Rust");
        let compact: String = wrappers.split_whitespace().collect();
        assert!(
            compact.contains(
                "__symbiont_arg_0:&mutsuper::__symbiont_state::Handle,__symbiont_arg_1:f64"
            )
        );
        assert!(compact.contains("__symbiont_arg_0:&super::__symbiont_state::Handle"));
        assert!(compact.contains("state_mut::<super::State>(__symbiont_arg_0,\"step\")"));
        assert!(compact.contains("state_ref::<super::State>(__symbiont_arg_0,\"peek\")"));
        for symbol in [NEW_SYMBOL, DROP_SYMBOL, TAG_SYMBOL] {
            assert!(wrappers.contains(&format!("export_name = \"{symbol}\"")));
        }
        assert!(
            compact
                .contains("__symbiont_state_satisfies_the_declared_bounds<T:Default+Send+'static>")
        );
    }

    #[test]
    fn a_block_without_state_has_no_state_glue() {
        let glue = harness_glue(
            &[decl(
                "f",
                "fn f(state: &mut State)",
                "pub fn f(state: &mut State) {}\n",
            )],
            &[],
        );
        assert!(!glue.contains(STATE_MODULE));
        assert!(!glue.contains(NEW_SYMBOL));
        // A host type named `State` passes through unchanged.
        assert!(glue.contains("__symbiont_arg_0: &mut State"));
    }

    #[test]
    fn the_state_module_is_part_of_the_glue() {
        let glue = harness_glue(&[stateful("f", "pub fn f(state: &mut State) {}\n")], &[]);
        let panic = glue.find("mod __symbiont_panic {").expect("panic module");
        let state = glue.find("mod __symbiont_state {").expect("state module");
        let exports = glue.find("mod __symbiont_exports {").expect("exports");
        assert!(panic < state);
        assert!(state < exports);
        syn::parse_file(&glue).expect("glue is valid Rust");
    }

    #[test]
    fn the_initial_candidate_starts_with_the_initial_state() {
        let candidate = initial_candidate(&[stateful("f", "pub fn f(state: &mut State) {}\n")]);
        assert_eq!(
            candidate,
            "#[derive(Default)]\nstruct State {\n    calls: usize,\n}\n\npub fn f(state: &mut State) {}\n"
        );
    }

    #[test]
    #[should_panic(expected = "either every declaration has the state or none has")]
    fn mixed_state_declarations_are_a_bug() {
        let _ = state_decl(&[
            stateful("a", "pub fn a() {}\n"),
            decl("b", "fn b()", "pub fn b() {}\n"),
        ]);
    }

    #[test]
    fn initial_candidate_joins_the_declared_sources() {
        let candidate = initial_candidate(&[
            decl("a", "fn a()", "pub fn a() {}\n"),
            decl("b", "fn b()", "pub fn b() {}"),
        ]);
        assert_eq!(candidate, "pub fn a() {}\n\npub fn b() {}\n");
    }
}
