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
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;
