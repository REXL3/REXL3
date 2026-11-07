//! The model runtime's error type.
use gb_cuda::CudaError;
use std::fmt;

#[derive(Debug)]
pub enum ModelError {
    Pack(crate::pack::PackError),
    Cuda(CudaError),
    Config(String),
    /// A request the runtime cannot serve (too many rows for one step, a slot that was never prefilled, ...).
    Request(String),
    /// The pack is of a family this release recognises and plans for but cannot run yet.
    NotBuilt(String),
    /// The request is fine but the memory set aside for it (an image's features, say) is all in use right now: retry shortly. The server answers 503
    /// with a `Retry-After`; nothing is wrong with the request and nothing is lost.
    Busy(String),
    /// Device state the engine depends on could not be brought up to date (accepted rows not replayed into the recurrent state):
    /// every later token of every sequence would be wrong, so this is fatal whatever call it surfaced in.
    StateLost(String),
}

impl ModelError {
    /// The process cannot go on serving after this error: a fatal CUDA error (sticky, or a code this build does not know) or lost
    /// device state. Everything else only spoils the request it came from.
    pub fn is_fatal(&self) -> bool {
        match self {
            ModelError::Cuda(e) => e.is_fatal(),
            ModelError::StateLost(_) => true,
            _ => false,
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Pack(e) => write!(f, "{e}"),
            ModelError::Cuda(e) => write!(f, "{e}"),
            ModelError::Config(m) => write!(f, "configuration: {m}"),
            ModelError::Request(m) => write!(f, "{m}"),
            ModelError::NotBuilt(m) => write!(f, "{m}"),
            ModelError::Busy(m) => write!(f, "{m}"),
            ModelError::StateLost(m) => write!(f, "device state lost: {m}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<crate::pack::PackError> for ModelError {
    fn from(e: crate::pack::PackError) -> Self {
        ModelError::Pack(e)
    }
}

impl From<CudaError> for ModelError {
    fn from(e: CudaError) -> Self {
        ModelError::Cuda(e)
    }
}
