// SPDX-License-Identifier: MPL-2.0
//! Proc macros for the `symbiont` crate.
//!
//! Provides the [`evolvable!`] function-like macro that declares
//! hot-reloadable functions and generates dispatch wrappers.

mod evolvable;
mod full_source;
mod utils;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{
    quote,
    quote_spanned,
};
use syn::{
    FnArg,
    ReturnType,
    spanned::Spanned,
};

use crate::{
    evolvable::EvolvableBlock,
    full_source::build_full_source,
    utils::format_signature,
};

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
/// # Return types
///
/// Every return type must implement [`Default`]: when an evolved
/// implementation panics, the in-dylib `catch_unwind` wrapper substitutes
/// `Default::default()` as a safe placeholder return value (retrieve the
/// panic message with `Runtime::take_panic`). The bound is enforced with a
/// compile error at the declaration site, so generated dylibs always
/// compile.
///
/// This generates:
/// - A `SYMBIONT_DECLS` constant with metadata for each function
/// - Wrapper functions that dispatch calls through the loaded dylib
/// - Per-function `<name>_fn(revision)` accessors returning typed
///   `RevisionFn` handles to any retained revision
#[proc_macro]
#[expect(
    clippy::too_many_lines,
    reason = "One big macro, better be left undisturbed."
)]
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

        // On panic inside the dylib, the `catch_unwind` wrapper substitutes
        // `Default::default()` as the return value, so every evolvable
        // return type must implement `Default`. Enforce this at declaration
        // time so generated dylibs always compile.
        let ret_span = match &sig.output {
            ReturnType::Type(_, ty) => ty.span(),
            ReturnType::Default => ident.span(),
        };
        let assert_ident = syn::Ident::new(
            &format!("__symbiont_return_type_of_{fn_name_str}_must_implement_default"),
            ident.span(),
        );
        wrapper_fns.push(quote_spanned! {ret_span=>
            const _: fn() = || {
                fn #assert_ident<T: ::core::default::Default>() {}
                #assert_ident::<#ret_ty>();
            };
        });

        // Build the EvolvableDecl entry (with reference to the AtomicPtr static)
        decl_entries.push(quote! {
            ::symbiont::EvolvableDecl {
                name: #fn_name_str,
                signature: #signature_str,
                full_source: #full_source,
                fn_ptr: &#static_name,
            }
        });

        // The per-function AtomicPtr static and lock-free dispatch wrapper.
        wrapper_fns.push(dispatch_wrapper(
            vis,
            sig,
            &static_name,
            &arg_types,
            &arg_names,
            &ret_ty,
        ));

        // Per-revision typed handle accessor, e.g. `step_fn(revision)`.
        wrapper_fns.push(revision_accessor(
            vis,
            ident,
            &fn_name_str,
            &static_name,
            &arg_types,
            &ret_ty,
        ));
    }

    // Render inline items declared inside `evolvable! { ... }` into the
    // per-call constant slice. These items are re-emitted verbatim in the
    // host crate and converted to a string literal that the runtime will
    // splice into the dylib's `lib.rs`.
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

/// Generate the per-function `AtomicPtr` static and the lock-free dispatch
/// wrapper that calls into the currently active revision.
fn dispatch_wrapper(
    vis: &syn::Visibility,
    sig: &syn::Signature,
    static_name: &syn::Ident,
    arg_types: &[syn::Type],
    arg_names: &[syn::Pat],
    ret_ty: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let ident = &sig.ident;
    let fn_name_str = ident.to_string();
    let fn_inputs = &sig.inputs;
    let fn_output = &sig.output;
    quote! {
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
    }
}

/// Generate the `<name>_fn(revision)` accessor that returns a typed
/// `RevisionFn` handle to `<name>`'s implementation in any retained revision.
fn revision_accessor(
    vis: &syn::Visibility,
    ident: &syn::Ident,
    fn_name_str: &str,
    static_name: &syn::Ident,
    arg_types: &[syn::Type],
    ret_ty: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let accessor_ident = syn::Ident::new(&format!("{fn_name_str}_fn"), ident.span());
    let accessor_doc = format!(
        "Typed handle to the `{fn_name_str}` implementation of a specific retained \
         `Revision`.\n\n\
         Returns `None` if the runtime is not initialized or `revision` is not \
         registered. The handle pins the revision's dylib; calls through it never \
         touch the swappable dispatch pointers, so they may run concurrently with \
         `Runtime::evolve` / `Runtime::activate_revision` and alongside handles of \
         other revisions. Panics of handle calls are stored per revision — read \
         them with `RevisionFn::take_panic`, not `Runtime::take_panic`."
    );
    quote! {
        #[doc = #accessor_doc]
        #vis fn #accessor_ident(
            revision: ::symbiont::Revision,
        ) -> ::core::option::Option<::symbiont::RevisionFn<fn(#(#arg_types),*) -> #ret_ty>> {
            let untyped = ::symbiont::__internal::revision_fn_lookup(&#static_name, revision)?;
            // SAFETY: the runtime validated this revision's code against exactly
            // this signature before compiling its dylib, and the handle keeps the
            // library loaded, so the symbol pointer is valid for this fn type.
            ::core::option::Option::Some(unsafe {
                untyped.cast::<fn(#(#arg_types),*) -> #ret_ty>()
            })
        }
    }
}
