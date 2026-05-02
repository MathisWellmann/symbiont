use syn::{
    ForeignItemFn,
    ItemFn,
    Signature,
    Visibility,
    parse::{
        Parse,
        ParseStream,
    },
};

/// A single function declaration inside `evolvable! { ... }`.
///
/// Supports two forms:
/// - With body: `fn step(counter: &mut usize) { *counter += 1; }`
/// - Without body: `fn step(counter: &mut usize);`
pub(crate) enum EvolvableFn {
    WithBody(ItemFn),
    WithoutBody(ForeignItemFn),
}

impl EvolvableFn {
    pub(crate) fn sig(&self) -> &Signature {
        match self {
            EvolvableFn::WithBody(f) => &f.sig,
            EvolvableFn::WithoutBody(f) => &f.sig,
        }
    }

    pub(crate) fn vis(&self) -> &Visibility {
        match self {
            EvolvableFn::WithBody(f) => &f.vis,
            EvolvableFn::WithoutBody(f) => &f.vis,
        }
    }

    pub(crate) fn attrs(&self) -> &[syn::Attribute] {
        match self {
            EvolvableFn::WithBody(f) => &f.attrs,
            EvolvableFn::WithoutBody(f) => &f.attrs,
        }
    }
}

/// The contents of an `evolvable! { ... }` block: zero or more function declarations.
#[expect(clippy::field_scoped_visibility_modifiers, reason = "Good here")]
pub(crate) struct EvolvableBlock {
    pub(crate) functions: Vec<EvolvableFn>,
}

impl Parse for EvolvableBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut functions = Vec::new();
        while !input.is_empty() {
            // Try parsing as a full function (with body) first
            let fork = input.fork();
            if fork.parse::<ItemFn>().is_ok() {
                functions.push(EvolvableFn::WithBody(input.parse::<ItemFn>()?));
            } else {
                // Fall back to bodyless (foreign-style) declaration
                functions.push(EvolvableFn::WithoutBody(input.parse::<ForeignItemFn>()?));
            }
        }
        Ok(EvolvableBlock { functions })
    }
}
