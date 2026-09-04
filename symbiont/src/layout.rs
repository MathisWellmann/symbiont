// SPDX-License-Identifier: MPL-2.0
//! The layout of the generated dylib's `src/lib.rs`.
//!
//! ```text
//! <candidate>                     the agent's code, byte for byte, from offset 0
//!
//! // ---- symbiont: harness glue ----
//! <prelude>                       configured imports and inline `evolvable!` items
//! mod __symbiont_panic { .. }     the panic-buffer protocol (`unwind.rs`)
//! mod __symbiont_exports { .. }   one exported `catch_unwind` wrapper per declared fn
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
    unwind::{
        PANIC_MODULE,
        PANIC_PREAMBLE,
    },
};

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
    glue.push_str(&export_wrappers(decls));
    glue
}

/// The candidate of the initial revision: every declared function with its
/// default body, as `evolvable!` rendered it.
pub(crate) fn initial_candidate(decls: &[EvolvableDecl]) -> String {
    let mut candidate = String::with_capacity(1024);
    for (idx, decl) in decls.iter().enumerate() {
        if idx > 0 {
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
    let wrappers = decls.iter().map(export_wrapper);
    let file: syn::File = syn::parse_quote! {
        mod #module {
            use super::*;
            #(#wrappers)*
        }
    };
    prettyplease::unparse(&file)
}

/// The exported `catch_unwind` wrapper of one declared function.
fn export_wrapper(decl: &EvolvableDecl) -> TokenStream {
    let item: syn::ItemFn = syn::parse_str(decl.full_source)
        .expect("full_source is generated by evolvable! and must parse");
    let target = &item.sig.ident;
    let symbol = decl.name;
    let wrapper = format_ident!("__symbiont_export_{}", decl.name);
    let panic_module = format_ident!("{PANIC_MODULE}");
    let args: Vec<syn::Ident> = (0..item.sig.inputs.len())
        .map(|idx| format_ident!("__symbiont_arg_{idx}"))
        .collect();
    let types = item.sig.inputs.iter().map(|arg| match arg {
        syn::FnArg::Typed(pat) => pat.ty.as_ref(),
        syn::FnArg::Receiver(_) => unreachable!("evolvable! rejects self receivers"),
    });
    let output = &item.sig.output;
    quote! {
        #[unsafe(export_name = #symbol)]
        pub fn #wrapper(#(#args: #types),*) #output {
            super::#panic_module::__symbiont_install_panic_hook();
            match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                super::#target(#(#args),*)
            })) {
                ::core::result::Result::Ok(__symbiont_val) => __symbiont_val,
                ::core::result::Result::Err(__symbiont_err) => {
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

    #[test]
    fn initial_candidate_joins_the_declared_sources() {
        let candidate = initial_candidate(&[
            decl("a", "fn a()", "pub fn a() {}\n"),
            decl("b", "fn b()", "pub fn b() {}"),
        ]);
        assert_eq!(candidate, "pub fn a() {}\n\npub fn b() {}\n");
    }
}
