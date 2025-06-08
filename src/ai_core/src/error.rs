//! Error types for the UNIA AI Core.

use std::path::PathBuf;
use thiserror::Error;

/// A specialized Result type for AI Core operations.
pub type Result<T> = std::result::Result<T, AIError>;

/// Errors that can occur in the AI Core.
#[derive(Error, Debug)]
pub enum AIError {
    /// Error occurred during model loading.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// Error occurred during model unloading.
    #[error("Failed to unload model: {0}")]
    ModelUnloadError(String),

    /// Error occurred during inference.
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Error occurred during model initialization.
    #[error("Model initialization error: {0}")]
    ModelInitError(String),

    /// Error occurred during model validation.
    #[error("Model validation error: {0}")]
    ModelValidationError(String),

    /// Error occurred during configuration loading.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Error occurred during I/O operations.
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),

    /// Error occurred during JSON parsing.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error occurred during model serialization/deserialization.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Error occurred during distributed operations.
    #[error("Distributed processing error: {0}")]
    DistributedError(String),

    /// Error occurred during tensor operations.
    #[error("Tensor error: {0}")]
    TensorError(String),

    /// Error occurred during ONNX operations.
    #[error("ONNX error: {0}")]
    OnnxError(String),

    /// Error occurred during PyTorch operations.
    #[error("PyTorch error: {0}")]
    TorchError(String),

    /// Error occurred during model registry operations.
    #[error("Model registry error: {0}")]
    RegistryError(String),

    /// Error occurred during storage operations.
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Error occurred during network operations.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Error occurred during GPU operations.
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Error occurred during resource allocation.
    #[error("Resource allocation error: {0}")]
    ResourceError(String),

    /// Error occurred due to unsupported operation.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Error occurred due to invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Error occurred due to model not found.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Error occurred due to model already loaded.
    #[error("Model already loaded: {0}")]
    ModelAlreadyLoaded(String),

    /// Error occurred due to model version mismatch.
    #[error("Model version mismatch: expected {expected}, found {found}")]
    ModelVersionMismatch {
        /// Expected version
        expected: String,
        /// Found version
        found: String,
    },

    /// Error occurred due to missing file.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Error occurred due to timeout.
    #[error("Operation timed out after {0} seconds")]
    Timeout(u64),

    /// Error occurred due to insufficient resources.
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    /// Error occurred due to hardware limitation.
    #[error("Hardware limitation: {0}")]
    HardwareLimitation(String),

    /// Error occurred due to system error.
    #[error("System error: {0}")]
    SystemError(String),

    /// Error occurred due to internal error.
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Extension trait for converting various error types to AIError.
pub trait IntoAIError<T> {
    /// Convert the error to an AIError.
    fn into_ai_error(self, context: &str) -> Result<T>;
}

impl<T, E: std::fmt::Display> IntoAIError<T> for std::result::Result<T, E> {
    fn into_ai_error(self, context: &str) -> Result<T> {
        self.map_err(|e| AIError::InternalError(format!("{}: {}", context, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_conversion() {
        let io_result: std::result::Result<(), std::io::Error> = 
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "test error"));
        
        let ai_result: Result<()> = io_result.into_ai_error("IO operation failed");
        
        assert!(ai_result.is_err());
        if let Err(e) = ai_result {
            assert!(matches!(e, AIError::InternalError(_)));
        }
    }
}
