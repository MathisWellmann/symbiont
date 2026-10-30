// SPDX-License-Identifier: MPL-2.0
//! Integration test: a host's source transform decides what is compiled,
//! while the revision keeps the agent's text, and compile errors come back
//! located in the agent's text.
//!
//! One test per binary: [`symbiont::Runtime`] is a process-wide singleton.
#![expect(
    unused_crate_dependencies,
    reason = "Integration tests don't use them all"
)]

use std::sync::Arc;

use symbiont::{
    Error,
    Profile,
    Runtime,
};

#[tokio::test]
#[cfg_attr(
    miri,
    ignore = "compiles and dlopens dylibs, which Miri does not support"
)]
async fn the_transform_shapes_the_dylib_but_not_the_revision() {
    symbiont::evolvable! {
        fn st_value() -> u32 {
            0
        }
    };
    let rt = Runtime::new(SYMBIONT_DECLS, SYMBIONT_PRELUDE, Profile::Debug)
        .await
        .expect("Can init runtime");
    rt.set_source_transform(Arc::new(|source: &str| {
        if source.contains("REFUSE") {
            Err("refused: write MAGIC instead".to_string())
        } else {
            Ok(source.replace("MAGIC", "7"))
        }
    }))
    .unwrap_or_else(|_| panic!("the first transform is accepted"));
    assert!(
        rt.set_source_transform(Arc::new(|s: &str| Ok(s.to_string())))
            .is_err(),
        "the transform is set once"
    );

    // `MAGIC` only compiles after the transform.
    let source = "pub fn st_value() -> u32 {\n    MAGIC\n}\n".to_string();
    let revision = rt
        .build_and_register(source.clone(), &mut None)
        .await
        .expect("the transformed source compiles");
    assert_eq!(rt.revision_code(revision).as_deref(), Some(source.as_str()));
    let value = st_value_fn(revision)
        .expect("the revision has the fn")
        .get();
    assert_eq!(value(), 7);
    let again = rt
        .build_and_register(source.clone(), &mut None)
        .await
        .expect("a byte-identical candidate is found");
    assert_eq!(again, revision, "deduplicated by the agent's text");

    let refused = "pub fn st_value() -> u32 {\n    REFUSE\n}\n".to_string();
    let Err(Error::CompilationFailed { code, err, .. }) =
        rt.build_and_register(refused.clone(), &mut None).await
    else {
        panic!("a refused transform fails the build")
    };
    assert_eq!(code, refused);
    assert_eq!(err, "refused: write MAGIC instead");

    // A type error on a line the transform left alone comes back with a
    // span in the agent's text; the error on the rewritten line keeps its
    // message but loses its span.
    let broken = "pub fn st_value() -> u32 {\n    let x: u32 = \"no\";\n    MAGIC + x\n}\n";
    let Err(Error::CompilationFailed {
        code, diagnostics, ..
    }) = rt.build_and_register(broken.to_string(), &mut None).await
    else {
        panic!("a type error fails the build")
    };
    assert_eq!(code, broken);
    let span = diagnostics
        .iter()
        .flat_map(|d| &d.spans)
        .find(|span| span.line_start == 2)
        .expect("the error on line 2 keeps its span");
    assert_eq!(&broken[span.bytes.clone()], "\"no\"");
}
