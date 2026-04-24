#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Syn(#[from] syn::Error),

    #[error(transparent)]
    RigPrompt(#[from] rig::completion::PromptError),

    #[error(transparent)]
    RigHttp(#[from] rig::http_client::Error),

    #[error("The text does not contain any rust code.")]
    NoRustCode,

    #[error("Could not parse Rust code.")]
    CouldNotParseRust,

    #[error("Failed to write lib.rs: {0}")]
    WriteLib(String),

    #[error("Validation failed: function '{0}' is not `pub`")]
    NonPublicFunction(String),

    #[error("Validation failed: function '{0}' is missing `#[no_mangle]`")]
    MissingNoMangle(String),

    #[error("Validation failed: signature mismatch for '{name}'. Expected: {expected}. Got: {got}")]
    SignatureMismatch {
        name: String,
        expected: String,
        got: String,
    },

    #[error("Compilation failed:\n{0}")]
    CompilationFailed(String),
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;
