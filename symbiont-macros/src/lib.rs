// SPDX-License-Identifier: MPL-2.0
//! Proc macros for the `symbiont` crate.
//!
//! Provides the [`evolvable!`] function-like macro that declares
//! hot-reloadable functions and generates dispatch wrappers.

mod evolvable;
mod full_source;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{
    ToTokens,
    quote,
};
use syn::{
    FnArg,
    ReturnType,
    Signature,
};

use crate::{
    evolvable::EvolvableBlock,
    full_source::build_full_source,
};

/// Format a `syn::Signature` into a human-readable string like `"fn step(counter: &mut usize)"`.
///
/// This mirrors the format used by `symbiont`'s validation module.
fn format_signature(sig: &Signature) -> String {
    let mut out = String::from("fn ");
    out.push_str(&sig.ident.to_string());

    out.push('(');
    let inputs: Vec<String> = sig.inputs.iter().map(format_fn_arg).collect();
    out.push_str(&inputs.join(", "));
    out.push(')');

    if let ReturnType::Type(_, ty) = &sig.output {
        out.push_str(" -> ");
        out.push_str(&normalize_tokens(ty.to_token_stream().to_string()));
    }

    out
}

fn format_fn_arg(arg: &FnArg) -> String {
    match arg {
        FnArg::Receiver(recv) => {
            if recv.mutability.is_some() {
                "&mut self".into()
            } else {
                "&self".into()
            }
        }
        FnArg::Typed(pat) => {
            let name = pat.pat.to_token_stream().to_string();
            let ty = normalize_tokens(pat.ty.to_token_stream().to_string());
            format!("{name}: {ty}")
        }
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

/// Declare hot-reloadable functions that are compiled into a temporary dylib and loaded at runtime.
///
/// # Examples
///
/// ```rust,ignore
/// symbiont::evolvable! {
///     fn step(counter: &mut usize) {
///         *counter += 1;  // default implementation
///     }
///
///     fn compute(x: f64) -> f64;  // bodyless, defaults to todo!()
/// }
/// ```
///
/// This generates:
/// - A `SYMBIONT_DECLS` constant with metadata for each function
/// - Wrapper functions that dispatch calls through the loaded dylib
#[proc_macro]
pub fn evolvable(input: TokenStream) -> TokenStream {
    let block = syn::parse_macro_input!(input as EvolvableBlock);

    if block.functions.is_empty() {
        return syn::Error::new(
            Span::call_site(),
            "evolvable! block must contain at least one function",
        )
        .to_compile_error()
        .into();
    }

    let mut decl_entries = Vec::new();
    let mut wrapper_fns = Vec::new();

    for func in &block.functions {
        let sig = func.sig();
        let vis = func.vis();
        let ident = &sig.ident;
        let fn_name_str = ident.to_string();
        let signature_str = format_signature(sig);
        let full_source = build_full_source(func);

        // Generate a unique static name for the cached function pointer
        let static_name = syn::Ident::new(
            &format!("__SYMBIONT_FN_{}", fn_name_str.to_uppercase()),
            ident.span(),
        );

        // Build the argument types and names for the extern fn type and call
        let mut arg_types = Vec::new();
        let mut arg_names = Vec::new();
        for arg in &sig.inputs {
            match arg {
                FnArg::Typed(pat_type) => {
                    arg_types.push(pat_type.ty.as_ref().clone());
                    arg_names.push(pat_type.pat.as_ref().clone());
                }
                FnArg::Receiver(_) => {
                    return syn::Error::new_spanned(
                        arg,
                        "self receivers are not supported in evolvable functions",
                    )
                    .to_compile_error()
                    .into();
                }
            }
        }

        let ret_ty = match &sig.output {
            ReturnType::Default => quote! { () },
            ReturnType::Type(_, ty) => quote! { #ty },
        };

        // Build the EvolvableDecl entry (with reference to the AtomicPtr static)
        decl_entries.push(quote! {
            ::symbiont::EvolvableDecl {
                name: #fn_name_str,
                signature: #signature_str,
                full_source: #full_source,
                fn_ptr: &#static_name,
            }
        });

        // Build the per-function AtomicPtr static and lock-free dispatch wrapper
        let fn_inputs = &sig.inputs;
        let fn_output = &sig.output;

        wrapper_fns.push(quote! {
            #[doc(hidden)]
            static #static_name: ::std::sync::atomic::AtomicPtr<()> =
                ::std::sync::atomic::AtomicPtr::new(::std::ptr::null_mut());

            #vis fn #ident(#fn_inputs) #fn_output {
                // In debug builds, track this call so evolve() can assert
                // no functions are in flight. Compiled away in release.
                #[cfg(debug_assertions)]
                let _call_guard = ::symbiont::__internal::enter_call();

                let ptr = #static_name.load(::std::sync::atomic::Ordering::Acquire);
                debug_assert!(
                    !ptr.is_null(),
                    concat!("symbiont: function '", #fn_name_str, "' not initialized; call Runtime::init() first")
                );
                let f: fn(#(#arg_types),*) -> #ret_ty = unsafe { ::std::mem::transmute(ptr) };
                f(#(#arg_names),*)
            }
        });
    }

    // Render the prelude into the per-call constant slice. The prelude has
    // two sources:
    //
    //   1. Inline items declared inside `evolvable! { ... }` — re-emitted
    //      verbatim in the host crate and converted to a string literal
    //      that the runtime will splice into the dylib's `lib.rs`.
    //   2. `shared Foo, Bar;` references — each resolves to a
    //      `__SYMBIONT_SHARED_<Ident>` const produced by the
    //      `#[symbiont::shared]` attribute macro applied to types defined
    //      outside the macro.
    let prelude_items = &block.prelude_items;
    let inline_prelude_source = if prelude_items.is_empty() {
        String::new()
    } else {
        let file = syn::File {
            shebang: None,
            attrs: Vec::new(),
            items: prelude_items.clone(),
        };
        prettyplease::unparse(&file)
    };

    let mut prelude_parts: Vec<proc_macro2::TokenStream> = Vec::new();
    if !inline_prelude_source.is_empty() {
        prelude_parts.push(quote! { #inline_prelude_source });
    }
    for ident in &block.shared_refs {
        let const_ident = syn::Ident::new(&format!("__SYMBIONT_SHARED_{ident}"), ident.span());
        prelude_parts.push(quote! { #const_ident });
    }

    let expanded = quote! {
        #(#prelude_items)*

        #[doc(hidden)]
        const SYMBIONT_PRELUDE: &[&::core::primitive::str] = &[
            #(#prelude_parts),*
        ];

        const SYMBIONT_DECLS: &[::symbiont::EvolvableDecl] = &[
            #(#decl_entries),*
        ];

        #(#wrapper_fns)*
    };

    expanded.into()
}

/// Attribute macro that marks a type (struct, enum, or type alias) as
/// shared between the host crate and any `evolvable!`-generated dylib.
///
/// Expands the annotated item unchanged, and additionally emits a
/// `pub const __SYMBIONT_SHARED_<Ident>: &str = "<source>"` constant
/// holding the item's source code. The `evolvable!` macro picks this up
/// via `shared <Ident>;` declarations.
///
/// # Example
///
/// ```rust,ignore
/// #[symbiont::shared]
/// #[derive(Debug, Clone)]
/// struct GameState { x: usize, y: usize }
///
/// symbiont::evolvable! {
///     shared GameState;
///     fn step(state: &mut GameState) { state.x += 1; }
/// }
/// ```
#[proc_macro_attribute]
pub fn shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_ts: proc_macro2::TokenStream = item.clone().into();
    let parsed = match syn::parse::<syn::Item>(item) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    use syn::Item::*;
    let ident = match &parsed {
        Struct(s) => &s.ident,
        Enum(e) => &e.ident,
        Type(t) => &t.ident,
        Union(u) => &u.ident,
        _ => {
            return syn::Error::new_spanned(
                &parsed,
                "#[symbiont::shared] only supports `struct`, `enum`, `union`, and `type` items",
            )
            .to_compile_error()
            .into();
        }
    };

    // Render the annotated item back to source so the runtime can splice
    // the exact same definition into the dylib's `lib.rs`.
    let file = syn::File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![parsed.clone()],
    };
    let source = prettyplease::unparse(&file);

    let const_ident = syn::Ident::new(&format!("__SYMBIONT_SHARED_{ident}"), ident.span());

    quote! {
        #item_ts

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub const #const_ident: &::core::primitive::str = #source;
    }
    .into()
}
