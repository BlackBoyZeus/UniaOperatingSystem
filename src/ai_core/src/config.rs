//! Configuration for the UNIA AI Core.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Duration;

use crate::error::{AIError, Result};

/// Configuration for the AI Core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIConfig {
    /// General configuration
    pub general: GeneralConfig,
    
    /// Inference engine configuration
    pub inference: InferenceConfig,
    
    /// Model management configuration
    pub model: ModelConfig,
    
    /// Distributed processing configuration
    pub distributed: DistributedConfig,
    
    /// Telemetry configuration
    pub telemetry: TelemetryConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
}

/// General configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Application name
    pub app_name: String,
    
    /// Log level
    pub log_level: String,
    
    /// Enable debug mode
    pub debug_mode: bool,
    
    /// Number of worker threads
    pub worker_threads: usize,
}

/// Inference engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Default device to use for inference
    pub default_device: DeviceType,
    
    /// Enable CUDA for inference
    pub enable_cuda: bool,
    
    /// Enable TensorRT for inference
    pub enable_tensorrt: bool,
    
    /// Default precision for inference
    pub default_precision: PrecisionType,
    
    /// Maximum batch size for inference
    pub max_batch_size: usize,
    
    /// Inference timeout in seconds
    pub timeout_seconds: u64,
    
    /// Enable model caching
    pub enable_caching: bool,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
}

/// Model management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Directory for model storage
    pub model_dir: PathBuf,
    
    /// Default models to load at startup
    pub default_models: Vec<String>,
    
    /// Enable model versioning
    pub enable_versioning: bool,
    
    /// Enable automatic model updates
    pub auto_update: bool,
    
    /// Model registry URL
    pub registry_url: Option<String>,
    
    /// Authentication token for model registry
    pub registry_token: Option<String>,
}

/// Distributed processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Enable distributed processing
    pub enabled: bool,
    
    /// Cluster nodes
    pub nodes: Vec<String>,
    
    /// Node ID
    pub node_id: String,
    
    /// Coordination service URL
    pub coordination_url: Option<String>,
    
    /// Communication timeout in seconds
    pub timeout_seconds: u64,
}

/// Telemetry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry
    pub enabled: bool,
    
    /// Telemetry endpoint URL
    pub endpoint: Option<String>,
    
    /// Sampling rate (0.0 - 1.0)
    pub sampling_rate: f32,
    
    /// Include performance metrics
    pub include_performance: bool,
    
    /// Include usage metrics
    pub include_usage: bool,
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage directory
    pub storage_dir: PathBuf,
    
    /// Enable persistence
    pub enable_persistence: bool,
    
    /// Compression level (0-9)
    pub compression_level: u8,
    
    /// Cache size in MB
    pub cache_size_mb: usize,
}

/// Device types for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU device
    CPU,
    /// CUDA device
    CUDA,
    /// TensorRT device
    TensorRT,
    /// OpenCL device
    OpenCL,
    /// Vulkan device
    Vulkan,
    /// Metal device (for macOS)
    Metal,
}

/// Precision types for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionType {
    /// FP32 precision
    FP32,
    /// FP16 precision
    FP16,
    /// INT8 precision
    INT8,
    /// Mixed precision
    Mixed,
}

impl AIConfig {
    /// Load configuration from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// A Result containing the loaded configuration or an error
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_str = fs::read_to_string(path.as_ref())
            .map_err(|e| AIError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        serde_json::from_str(&config_str)
            .map_err(|e| AIError::ConfigError(format!("Failed to parse config file: {}", e)))
    }
    
    /// Save configuration to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let config_str = serde_json::to_string_pretty(self)
            .map_err(|e| AIError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(path.as_ref(), config_str)
            .map_err(|e| AIError::ConfigError(format!("Failed to write config file: {}", e)))
    }
    
    /// Create a default configuration.
    pub fn default() -> Self {
        Self {
            general: GeneralConfig {
                app_name: "UNIA AI Core".to_string(),
                log_level: "info".to_string(),
                debug_mode: false,
                worker_threads: num_cpus::get(),
            },
            inference: InferenceConfig {
                default_device: DeviceType::CPU,
                enable_cuda: true,
                enable_tensorrt: false,
                default_precision: PrecisionType::FP32,
                max_batch_size: 16,
                timeout_seconds: 30,
                enable_caching: true,
                max_memory_mb: 4096,
            },
            model: ModelConfig {
                model_dir: PathBuf::from("/var/lib/unia/models"),
                default_models: vec![
                    "npc-behavior-basic".to_string(),
                    "object-detection".to_string(),
                ],
                enable_versioning: true,
                auto_update: false,
                registry_url: None,
                registry_token: None,
            },
            distributed: DistributedConfig {
                enabled: false,
                nodes: Vec::new(),
                node_id: "node-0".to_string(),
                coordination_url: None,
                timeout_seconds: 10,
            },
            telemetry: TelemetryConfig {
                enabled: false,
                endpoint: None,
                sampling_rate: 0.1,
                include_performance: true,
                include_usage: true,
            },
            storage: StorageConfig {
                storage_dir: PathBuf::from("/var/lib/unia/storage"),
                enable_persistence: true,
                compression_level: 6,
                cache_size_mb: 1024,
            },
        }
    }
    
    /// Create a configuration for testing.
    #[cfg(test)]
    pub fn for_testing() -> Self {
        use std::env::temp_dir;
        
        let mut temp_dir = temp_dir();
        temp_dir.push("unia_test");
        
        let mut model_dir = temp_dir.clone();
        model_dir.push("models");
        
        let mut storage_dir = temp_dir.clone();
        storage_dir.push("storage");
        
        Self {
            general: GeneralConfig {
                app_name: "UNIA AI Core Test".to_string(),
                log_level: "debug".to_string(),
                debug_mode: true,
                worker_threads: 2,
            },
            inference: InferenceConfig {
                default_device: DeviceType::CPU,
                enable_cuda: false,
                enable_tensorrt: false,
                default_precision: PrecisionType::FP32,
                max_batch_size: 4,
                timeout_seconds: 5,
                enable_caching: true,
                max_memory_mb: 1024,
            },
            model: ModelConfig {
                model_dir,
                default_models: vec![
                    "test-model".to_string(),
                ],
                enable_versioning: true,
                auto_update: false,
                registry_url: None,
                registry_token: None,
            },
            distributed: DistributedConfig {
                enabled: false,
                nodes: Vec::new(),
                node_id: "test-node".to_string(),
                coordination_url: None,
                timeout_seconds: 5,
            },
            telemetry: TelemetryConfig {
                enabled: false,
                endpoint: None,
                sampling_rate: 0.0,
                include_performance: false,
                include_usage: false,
            },
            storage: StorageConfig {
                storage_dir,
                enable_persistence: true,
                compression_level: 0,
                cache_size_mb: 128,
            },
        }
    }
    
    /// Get the inference timeout as a Duration.
    pub fn inference_timeout(&self) -> Duration {
        Duration::from_secs(self.inference.timeout_seconds)
    }
    
    /// Get the distributed communication timeout as a Duration.
    pub fn distributed_timeout(&self) -> Duration {
        Duration::from_secs(self.distributed.timeout_seconds)
    }
    
    /// Check if CUDA is enabled and available.
    pub fn is_cuda_enabled(&self) -> bool {
        self.inference.enable_cuda && is_cuda_available()
    }
    
    /// Check if TensorRT is enabled and available.
    pub fn is_tensorrt_enabled(&self) -> bool {
        self.inference.enable_tensorrt && is_tensorrt_available()
    }
}

/// Check if CUDA is available on the system.
fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Use tch to check CUDA availability
        tch::Cuda::is_available()
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Check if TensorRT is available on the system.
fn is_tensorrt_available() -> bool {
    #[cfg(feature = "tensorrt")]
    {
        // This would need to be implemented with proper TensorRT detection
        // For now, just return false
        false
    }
    
    #[cfg(not(feature = "tensorrt"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_config_serialization() {
        let config = AIConfig::default();
        
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        
        // Save config to file
        config.save_to_file(&config_path).unwrap();
        
        // Load config from file
        let loaded_config = AIConfig::from_file(&config_path).unwrap();
        
        // Check that the loaded config matches the original
        assert_eq!(
            serde_json::to_string(&config).unwrap(),
            serde_json::to_string(&loaded_config).unwrap()
        );
    }
    
    #[test]
    fn test_config_for_testing() {
        let config = AIConfig::for_testing();
        
        // Check that test config has expected values
        assert!(config.general.debug_mode);
        assert_eq!(config.inference.default_device, DeviceType::CPU);
        assert!(!config.inference.enable_cuda);
        assert!(!config.telemetry.enabled);
    }
}
