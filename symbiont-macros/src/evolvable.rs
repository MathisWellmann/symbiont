use quote::ToTokens;
use syn::{
    Attribute,
    ForeignItemFn,
    Item,
    ItemFn,
    Signature,
    Token,
    TraitItemType,
    Visibility,
    parse::{
        Parse,
        ParseStream,
    },
};

/// The name of the state type an `evolvable!` block may declare.
pub(crate) const STATE_IDENT: &str = "State";

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

    pub(crate) fn attrs(&self) -> &[Attribute] {
        match self {
            EvolvableFn::WithBody(f) => &f.attrs,
            EvolvableFn::WithoutBody(f) => &f.attrs,
        }
    }
}

/// The `type State: Default + Send + 'static;` declaration of a block.
#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "Plain parse output without invariants, like EvolvableBlock"
)]
pub(crate) struct StateTypeDecl {
    pub(crate) attrs: Vec<Attribute>,
    pub(crate) vis: Visibility,
    pub(crate) decl: TraitItemType,
}

/// The contents of an `evolvable! { ... }` block.
///
/// Holds zero or more function declarations plus a prelude of supporting
/// inline items. Inline structs, enums, type aliases, `use` statements, etc.
/// declared directly inside the macro body are re-emitted in both the host
/// crate and the generated dylib.
///
/// Two items are special: a `type State: ..;` declaration, and the
/// `struct State` or `enum State` that is its initial definition. Neither is
/// part of the prelude: in the host `State` is the opaque
/// `symbiont::State` handle, and in the dylib the agent defines it.
#[expect(clippy::field_scoped_visibility_modifiers, reason = "Good here")]
pub(crate) struct EvolvableBlock {
    pub(crate) functions: Vec<EvolvableFn>,
    /// Non-fn items declared inline inside the macro.
    pub(crate) prelude_items: Vec<Item>,
    /// The declared state type, if any.
    pub(crate) state: Option<StateTypeDecl>,
    /// The initial definition of the state type, if the block has one.
    pub(crate) state_initial: Option<Item>,
}

impl Parse for EvolvableBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut functions = Vec::new();
        let mut prelude_items = Vec::new();
        let mut state: Option<StateTypeDecl> = None;
        let mut state_initial: Option<Item> = None;

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

            // A bodyless `type X: Bounds;`, which is not a valid free item.
            if is_bodyless_type(input) {
                let parsed = StateTypeDecl {
                    attrs: input.call(Attribute::parse_outer)?,
                    vis: input.parse()?,
                    decl: input.parse()?,
                };
                if parsed.decl.ident != STATE_IDENT {
                    return Err(syn::Error::new_spanned(
                        &parsed.decl.ident,
                        "the only bodyless type an evolvable! block may declare is `type State`",
                    ));
                }
                if state.is_some() {
                    return Err(syn::Error::new_spanned(
                        &parsed.decl,
                        "`type State` is declared more than once",
                    ));
                }
                state = Some(parsed);
                continue;
            }

            // Fall back to any other top-level item (struct/enum/type/use/...).
            let item: Item = input.parse()?;
            match item {
                Item::Fn(f) => functions.push(EvolvableFn::WithBody(f)),
                other if defines_state(&other) => {
                    if state_initial.is_some() {
                        return Err(syn::Error::new_spanned(
                            &other,
                            "the initial `State` is defined more than once",
                        ));
                    }
                    state_initial = Some(other);
                }
                other => prelude_items.push(other),
            }
        }
        if let (None, Some(initial)) = (&state, &state_initial) {
            // Without the declaration `State` is an ordinary prelude type:
            // the host may well have one of its own.
            prelude_items.push(initial.clone());
            state_initial = None;
        }
        Ok(EvolvableBlock {
            functions,
            prelude_items,
            state,
            state_initial,
        })
    }
}

/// Whether the input starts with `[attrs] [vis] type Ident [: Bounds];`.
fn is_bodyless_type(input: ParseStream) -> bool {
    let fork = input.fork();
    if fork.call(Attribute::parse_outer).is_err() || fork.parse::<Visibility>().is_err() {
        return false;
    }
    fork.peek(Token![type])
        && fork
            .parse::<TraitItemType>()
            .is_ok_and(|decl| decl.default.is_none())
}

/// Whether `item` is a `struct State` or `enum State`.
fn defines_state(item: &Item) -> bool {
    match item {
        Item::Struct(s) => s.ident == STATE_IDENT,
        Item::Enum(e) => e.ident == STATE_IDENT,
        _ => false,
    }
}

/// Whether `tokens` contain the identifier `State` anywhere.
pub(crate) fn mentions_state(tokens: &impl ToTokens) -> bool {
    fn walk(stream: proc_macro2::TokenStream) -> bool {
        stream.into_iter().any(|tt| match tt {
            proc_macro2::TokenTree::Ident(ident) => ident == STATE_IDENT,
            proc_macro2::TokenTree::Group(group) => walk(group.stream()),
            _ => false,
        })
    }
    walk(tokens.to_token_stream())
}

/// Whether `ty` is exactly `&State` or `&mut State`, the only forms in which
/// a declared state may cross the boundary.
pub(crate) fn is_state_reference(ty: &syn::Type) -> bool {
    let syn::Type::Reference(reference) = ty else {
        return false;
    };
    if reference.lifetime.is_some() {
        return false;
    }
    let syn::Type::Path(path) = reference.elem.as_ref() else {
        return false;
    };
    path.qself.is_none() && path.path.is_ident(STATE_IDENT)
}
