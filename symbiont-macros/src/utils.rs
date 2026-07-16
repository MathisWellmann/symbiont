use quote::ToTokens;
use syn::{
    FnArg,
    ReturnType,
    Signature,
};

/// Format a `syn::Signature` into a human-readable string like `"fn step(counter: &mut usize)"`.
///
/// This mirrors the source-like format used in prompts. Validation separately
/// canonicalizes signatures to ignore argument names for ABI compatibility.
pub(crate) fn format_signature(sig: &Signature) -> String {
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

pub(crate) fn format_fn_arg(arg: &FnArg) -> String {
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
