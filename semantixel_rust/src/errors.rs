use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemantiXelError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Invalid extension: {0}")]
    InvalidExtension(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SemantiXelError>;
