//! Inference engine for executing AI models.

mod backend;
mod tensor;
mod cache;
mod scheduler;
mod profiler;

use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::sync::Semaphore;

use crate::config::{AIConfig, DeviceType, PrecisionType};
use crate::error::{AIError, Result};
use crate::model::{ModelManager, ModelId, ModelInfo};

use self::backend::{Backend, BackendFactory, CpuBackend, CudaBackend};
use self::tensor::{Tensor, TensorType};
use self::cache::InferenceCache;
use self::scheduler::InferenceScheduler;
use self::profiler::InferenceProfiler;

/// Options for inference execution.
#[derive(Debug, Clone)]
pub struct InferenceOptions {
    /// Device to use for inference
    pub device: DeviceType,
    
    /// Precision to use for inference
    pub precision: PrecisionType,
    
    /// Whether to use caching
    pub use_cache: bool,
    
    /// Priority of the inference task
    pub priority: InferencePriority,
    
    /// Timeout for inference in milliseconds
    pub timeout_ms: Option<u64>,
    
    /// Additional backend-specific options
    pub backend_options: HashMap<String, String>,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            device: DeviceType::CPU,
            precision: PrecisionType::FP32,
            use_cache: true,
            priority: InferencePriority::Normal,
            timeout_ms: None,
            backend_options: HashMap::new(),
        }
    }
}

/// Priority levels for inference tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InferencePriority {
    /// Low priority, can be delayed
    Low,
    
    /// Normal priority
    Normal,
    
    /// High priority, should be executed promptly
    High,
    
    /// Critical priority, should be executed immediately
    Critical,
}

/// Result of an inference operation.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Output tensors
    pub outputs: HashMap<String, Tensor>,
    
    /// Metadata about the inference
    pub metadata: InferenceMetadata,
}

/// Metadata about an inference operation.
#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    /// Model ID used for inference
    pub model_id: ModelId,
    
    /// Device used for inference
    pub device: DeviceType,
    
    /// Precision used for inference
    pub precision: PrecisionType,
    
    /// Time taken for inference in microseconds
    pub duration_us: u64,
    
    /// Whether the result was from cache
    pub from_cache: bool,
    
    /// Backend used for inference
    pub backend: String,
    
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

/// The main inference engine for executing AI models.
pub struct InferenceEngine {
    /// Configuration for the inference engine
    config: Arc<AIConfig>,
    
    /// Model manager for accessing models
    model_manager: Arc<ModelManager>,
    
    /// Available backends for inference
    backends: RwLock<HashMap<DeviceType, Arc<dyn Backend>>>,
    
    /// Cache for inference results
    cache: InferenceCache,
    
    /// Scheduler for inference tasks
    scheduler: InferenceScheduler,
    
    /// Profiler for inference performance
    profiler: InferenceProfiler,
    
    /// Semaphore for limiting concurrent inferences
    concurrency_limiter: Semaphore,
}

impl InferenceEngine {
    /// Create a new inference engine.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the inference engine
    /// * `model_manager` - Model manager for accessing models
    ///
    /// # Returns
    ///
    /// A Result containing the initialized inference engine or an error
    pub async fn new(config: Arc<AIConfig>, model_manager: Arc<ModelManager>) -> Result<Self> {
        let cache = InferenceCache::new(config.inference.max_memory_mb * 1024 * 1024);
        let scheduler = InferenceScheduler::new();
        let profiler = InferenceProfiler::new();
        
        let concurrency_limiter = Semaphore::new(config.general.worker_threads);
        
        let mut engine = Self {
            config,
            model_manager,
            backends: RwLock::new(HashMap::new()),
            cache,
            scheduler,
            profiler,
            concurrency_limiter,
        };
        
        // Initialize backends
        engine.initialize_backends()?;
        
        Ok(engine)
    }
    
    /// Initialize the inference engine.
    ///
    /// This method prepares the inference engine for use.
    pub async fn initialize(&self) -> Result<()> {
        // Nothing to do here for now
        Ok(())
    }
    
    /// Shutdown the inference engine.
    ///
    /// This method releases resources used by the inference engine.
    pub async fn shutdown(&self) -> Result<()> {
        // Clear the cache
        self.cache.clear();
        
        // Shutdown backends
        let backends = self.backends.read();
        for (_, backend) in backends.iter() {
            backend.shutdown()?;
        }
        
        Ok(())
    }
    
    /// Initialize available backends based on configuration.
    fn initialize_backends(&mut self) -> Result<()> {
        let mut backends = self.backends.write();
        
        // Always add CPU backend
        backends.insert(DeviceType::CPU, Arc::new(CpuBackend::new()) as Arc<dyn Backend>);
        
        // Add CUDA backend if enabled and available
        if self.config.is_cuda_enabled() {
            match CudaBackend::new() {
                Ok(backend) => {
                    backends.insert(DeviceType::CUDA, Arc::new(backend) as Arc<dyn Backend>);
                    tracing::info!("CUDA backend initialized");
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize CUDA backend: {}", e);
                }
            }
        }
        
        // Add TensorRT backend if enabled and available
        if self.config.is_tensorrt_enabled() {
            match BackendFactory::create_tensorrt_backend() {
                Ok(backend) => {
                    backends.insert(DeviceType::TensorRT, backend);
                    tracing::info!("TensorRT backend initialized");
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize TensorRT backend: {}", e);
                }
            }
        }
        
        tracing::info!("Initialized {} inference backends", backends.len());
        Ok(())
    }
    
    /// Run inference using the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - ID of the model to use
    /// * `inputs` - Input tensors for inference
    /// * `options` - Options for inference execution
    ///
    /// # Returns
    ///
    /// A Result containing the inference result or an error
    pub async fn run_inference(
        &self,
        model_id: &ModelId,
        inputs: HashMap<String, Tensor>,
        options: Option<InferenceOptions>,
    ) -> Result<InferenceResult> {
        let options = options.unwrap_or_default();
        
        // Check if the result is in cache
        if options.use_cache && self.config.inference.enable_caching {
            if let Some(result) = self.cache.get(model_id, &inputs, &options) {
                return Ok(result);
            }
        }
        
        // Get the model
        let model_info = self.model_manager.get_model(model_id).await?;
        
        // Select the backend
        let backend = self.select_backend(&options, &model_info)?;
        
        // Acquire concurrency permit
        let _permit = self.concurrency_limiter.acquire().await.map_err(|e| {
            AIError::InternalError(format!("Failed to acquire concurrency permit: {}", e))
        })?;
        
        // Start profiling
        let profiling_id = self.profiler.start_inference(model_id, &options);
        
        // Run inference
        let start_time = std::time::Instant::now();
        let outputs = backend.run_inference(&model_info, inputs.clone(), &options)?;
        let duration_us = start_time.elapsed().as_micros() as u64;
        
        // Stop profiling
        self.profiler.end_inference(profiling_id, duration_us);
        
        // Create result
        let result = InferenceResult {
            outputs,
            metadata: InferenceMetadata {
                model_id: model_id.clone(),
                device: options.device,
                precision: options.precision,
                duration_us,
                from_cache: false,
                backend: backend.name().to_string(),
                extra: HashMap::new(),
            },
        };
        
        // Cache the result if caching is enabled
        if options.use_cache && self.config.inference.enable_caching {
            self.cache.put(model_id, &inputs, &options, result.clone());
        }
        
        Ok(result)
    }
    
    /// Select the appropriate backend for inference.
    fn select_backend(
        &self,
        options: &InferenceOptions,
        model_info: &ModelInfo,
    ) -> Result<Arc<dyn Backend>> {
        let backends = self.backends.read();
        
        // Check if the requested device is available
        if let Some(backend) = backends.get(&options.device) {
            // Check if the backend supports the model
            if backend.supports_model(model_info)? {
                return Ok(backend.clone());
            }
        }
        
        // Fall back to CPU if the requested device is not available or doesn't support the model
        if options.device != DeviceType::CPU {
            tracing::warn!(
                "Requested device {:?} not available or doesn't support model, falling back to CPU",
                options.device
            );
            
            if let Some(cpu_backend) = backends.get(&DeviceType::CPU) {
                return Ok(cpu_backend.clone());
            }
        }
        
        Err(AIError::InferenceError(format!(
            "No suitable backend found for device {:?} and model {}",
            options.device, model_info.id
        )))
    }
    
    /// Get performance statistics for the inference engine.
    pub fn get_performance_stats(&self) -> HashMap<String, f64> {
        self.profiler.get_statistics()
    }
    
    /// Clear the inference cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelMetadata;
    
    #[tokio::test]
    async fn test_inference_engine_creation() {
        let config = Arc::new(AIConfig::for_testing());
        let model_manager = Arc::new(ModelManager::new(config.clone()).await.unwrap());
        
        let engine = InferenceEngine::new(config, model_manager).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_backend_selection() {
        let config = Arc::new(AIConfig::for_testing());
        let model_manager = Arc::new(ModelManager::new(config.clone()).await.unwrap());
        
        let engine = InferenceEngine::new(config, model_manager).await.unwrap();
        
        let model_info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            description: "A test model".to_string(),
            metadata: ModelMetadata::default(),
            input_shapes: HashMap::new(),
            output_shapes: HashMap::new(),
        };
        
        let options = InferenceOptions {
            device: DeviceType::CPU,
            ..Default::default()
        };
        
        let backend = engine.select_backend(&options, &model_info);
        assert!(backend.is_ok());
        assert_eq!(backend.unwrap().name(), "CPU");
    }
}
