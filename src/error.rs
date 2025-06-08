//! Error types for UNIA Operating System.

use std::fmt;
use std::error::Error as StdError;

/// Result type for UNIA operations.
pub type Result<T> = std::result::Result<T, AIError>;

/// Error types for AI operations.
#[derive(Debug)]
pub enum AIError {
    /// I/O error
    IoError(String),
    
    /// Network error
    NetworkError(String),
    
    /// Permission denied
    PermissionDenied(String),
    
    /// Resource not found
    NotFound(String),
    
    /// Invalid input
    InvalidInput(String),
    
    /// Resource exhausted
    ResourceExhausted(String),
    
    /// Serialization error
    SerializationError(String),
    
    /// AI model error
    ModelError(String),
    
    /// External service error
    ExternalServiceError(String),
    
    /// Internal error
    InternalError(String),
}

impl fmt::Display for AIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AIError::IoError(msg) => write!(f, "I/O error: {}", msg),
            AIError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            AIError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            AIError::NotFound(msg) => write!(f, "Not found: {}", msg),
            AIError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AIError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            AIError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            AIError::ModelError(msg) => write!(f, "AI model error: {}", msg),
            AIError::ExternalServiceError(msg) => write!(f, "External service error: {}", msg),
            AIError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl StdError for AIError {}

impl From<std::io::Error> for AIError {
    fn from(err: std::io::Error) -> Self {
        AIError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for AIError {
    fn from(err: serde_json::Error) -> Self {
        AIError::SerializationError(err.to_string())
    }
}

impl From<webrtc::Error> for AIError {
    fn from(err: webrtc::Error) -> Self {
        AIError::NetworkError(err.to_string())
    }
}
