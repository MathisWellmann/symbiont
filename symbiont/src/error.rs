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

    #[error("Validation failed: function '{0}' is not `pub`")]
    NonPublicFunction(String),

    #[error("Validation failed: function '{0}' is missing `#[no_mangle]`")]
    MissingNoMangle(String),

    #[error("Validation failed: signature mismatch for '{0}'. Expected: {1}. Got: {2}")]
    SignatureMismatch(String, String, String),
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;
