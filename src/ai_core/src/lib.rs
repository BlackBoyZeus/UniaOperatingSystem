//! # UNIA AI Core
//!
//! The core AI framework for the UNIA Operating System, providing high-performance
//! inference, model management, and specialized gaming AI capabilities.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

use std::sync::Arc;

pub mod error;
pub mod inference;
pub mod model;
pub mod game_ai;
pub mod distributed;
pub mod utils;
pub mod config;
pub mod telemetry;
pub mod storage;

use crate::config::AIConfig;
use crate::error::Result;
use crate::inference::InferenceEngine;
use crate::model::ModelManager;

/// The main entry point for the UNIA AI Core framework.
///
/// This struct provides access to all AI capabilities of the UNIA system.
/// It manages the lifecycle of AI components and coordinates their interactions.
#[derive(Clone)]
pub struct AICore {
    /// Configuration for the AI Core
    config: Arc<AIConfig>,
    
    /// The inference engine for executing AI models
    inference_engine: Arc<InferenceEngine>,
    
    /// The model manager for handling AI model lifecycle
    model_manager: Arc<ModelManager>,
}

impl AICore {
    /// Create a new instance of the AI Core with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the AI Core
    ///
    /// # Returns
    ///
    /// A Result containing the initialized AI Core or an error
    pub async fn new(config: AIConfig) -> Result<Self> {
        let config = Arc::new(config);
        
        // Initialize the model manager
        let model_manager = Arc::new(ModelManager::new(config.clone()).await?);
        
        // Initialize the inference engine
        let inference_engine = Arc::new(InferenceEngine::new(config.clone(), model_manager.clone()).await?);
        
        Ok(Self {
            config,
            inference_engine,
            model_manager,
        })
    }
    
    /// Get a reference to the inference engine.
    pub fn inference_engine(&self) -> Arc<InferenceEngine> {
        self.inference_engine.clone()
    }
    
    /// Get a reference to the model manager.
    pub fn model_manager(&self) -> Arc<ModelManager> {
        self.model_manager.clone()
    }
    
    /// Initialize the AI Core with default models and prepare it for use.
    ///
    /// This method loads the default models specified in the configuration
    /// and performs any necessary initialization steps.
    pub async fn initialize(&self) -> Result<()> {
        // Load default models
        self.model_manager.load_default_models().await?;
        
        // Initialize the inference engine
        self.inference_engine.initialize().await?;
        
        tracing::info!("AI Core initialized successfully");
        Ok(())
    }
    
    /// Shutdown the AI Core and release resources.
    pub async fn shutdown(&self) -> Result<()> {
        // Shutdown the inference engine
        self.inference_engine.shutdown().await?;
        
        // Unload models
        self.model_manager.unload_all_models().await?;
        
        tracing::info!("AI Core shut down successfully");
        Ok(())
    }
}

/// Initialize the global tracing subscriber for logging.
///
/// This function should be called at the start of the application to set up
/// the logging infrastructure.
pub fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
    
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ai_core_lifecycle() {
        // Create a test configuration
        let config = AIConfig::for_testing();
        
        // Create the AI Core
        let ai_core = AICore::new(config).await.expect("Failed to create AI Core");
        
        // Initialize the AI Core
        ai_core.initialize().await.expect("Failed to initialize AI Core");
        
        // Shutdown the AI Core
        ai_core.shutdown().await.expect("Failed to shutdown AI Core");
    }
}
