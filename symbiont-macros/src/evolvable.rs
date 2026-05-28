use syn::{
    ForeignItemFn,
    Item,
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

/// The contents of an `evolvable! { ... }` block.
///
/// Holds zero or more function declarations plus a prelude of supporting
/// inline items. Inline structs, enums, type aliases, `use` statements, etc.
/// declared directly inside the macro body are re-emitted in both the host
/// crate and the generated dylib.
#[expect(clippy::field_scoped_visibility_modifiers, reason = "Good here")]
pub(crate) struct EvolvableBlock {
    pub(crate) functions: Vec<EvolvableFn>,
    /// Non-fn items declared inline inside the macro.
    pub(crate) prelude_items: Vec<Item>,
}

impl Parse for EvolvableBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut functions = Vec::new();
        let mut prelude_items = Vec::new();

        while !input.is_empty() {
            // Try parsing as a full function (with body) first.
            let fork = input.fork();
            if fork.parse::<ItemFn>().is_ok() {
                functions.push(EvolvableFn::WithBody(input.parse::<ItemFn>()?));
                continue;
            }

            // Try parsing as a bodyless function declaration.
            let fork = input.fork();
            if fork.parse::<ForeignItemFn>().is_ok() {
                let parsed: ForeignItemFn = input.parse()?;
                functions.push(EvolvableFn::WithoutBody(parsed));
                continue;
            }

            // Fall back to any other top-level item (struct/enum/type/use/...).
            let item: Item = input.parse()?;
            match item {
                Item::Fn(f) => functions.push(EvolvableFn::WithBody(f)),
                other => prelude_items.push(other),
            }
        }
        Ok(EvolvableBlock {
            functions,
            prelude_items,
        })
    }
}
