//! Hardware acceleration for AI operations.
//!
//! This module provides hardware acceleration capabilities for AI operations,
//! supporting various acceleration technologies like CUDA, OpenCL, and custom NPUs.

pub mod cuda;
pub mod opencl;
pub mod vulkan;
pub mod npu;
pub mod cpu;

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use crate::error::Result;
use crate::inference::Tensor;

/// Hardware acceleration device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorType {
    /// CPU
    CPU,
    /// CUDA (NVIDIA GPUs)
    CUDA,
    /// OpenCL (cross-platform)
    OpenCL,
    /// Vulkan Compute (cross-platform)
    Vulkan,
    /// Neural Processing Unit
    NPU,
    /// Custom accelerator
    Custom,
}

/// Hardware acceleration capabilities.
#[derive(Debug, Clone)]
pub struct AcceleratorCapabilities {
    /// Accelerator type
    pub accelerator_type: AcceleratorType,
    
    /// Device name
    pub device_name: String,
    
    /// Device vendor
    pub device_vendor: String,
    
    /// Device version
    pub device_version: String,
    
    /// Available memory in bytes
    pub available_memory: u64,
    
    /// Supported operations
    pub supported_operations: Vec<String>,
    
    /// Supported precisions
    pub supported_precisions: Vec<String>,
    
    /// Compute units
    pub compute_units: u32,
    
    /// Maximum workgroup size
    pub max_workgroup_size: u32,
    
    /// Maximum dimensions
    pub max_dimensions: u32,
    
    /// Device extensions
    pub extensions: Vec<String>,
}

/// Hardware acceleration operation.
#[derive(Debug, Clone)]
pub enum AcceleratorOperation {
    /// Matrix multiplication
    MatrixMultiply {
        /// Input A
        a: Tensor,
        /// Input B
        b: Tensor,
    },
    
    /// Convolution
    Convolution {
        /// Input
        input: Tensor,
        /// Kernel
        kernel: Tensor,
        /// Stride
        stride: [u32; 2],
        /// Padding
        padding: [u32; 2],
    },
    
    /// Element-wise operation
    ElementWise {
        /// Input A
        a: Tensor,
        /// Input B
        b: Tensor,
        /// Operation type
        op_type: ElementWiseOpType,
    },
    
    /// Reduction operation
    Reduction {
        /// Input
        input: Tensor,
        /// Axes
        axes: Vec<u32>,
        /// Operation type
        op_type: ReductionOpType,
    },
    
    /// Activation function
    Activation {
        /// Input
        input: Tensor,
        /// Activation type
        activation_type: ActivationType,
    },
    
    /// Pooling operation
    Pooling {
        /// Input
        input: Tensor,
        /// Kernel size
        kernel_size: [u32; 2],
        /// Stride
        stride: [u32; 2],
        /// Padding
        padding: [u32; 2],
        /// Pooling type
        pooling_type: PoolingType,
    },
    
    /// Softmax operation
    Softmax {
        /// Input
        input: Tensor,
        /// Axis
        axis: u32,
    },
    
    /// Custom operation
    Custom {
        /// Operation name
        name: String,
        /// Inputs
        inputs: Vec<Tensor>,
        /// Parameters
        parameters: std::collections::HashMap<String, String>,
    },
}

/// Element-wise operation type.
#[derive(Debug, Clone, Copy)]
pub enum ElementWiseOpType {
    /// Addition
    Add,
    /// Subtraction
    Subtract,
    /// Multiplication
    Multiply,
    /// Division
    Divide,
    /// Maximum
    Maximum,
    /// Minimum
    Minimum,
    /// Power
    Power,
}

/// Reduction operation type.
#[derive(Debug, Clone, Copy)]
pub enum ReductionOpType {
    /// Sum
    Sum,
    /// Product
    Product,
    /// Maximum
    Maximum,
    /// Minimum
    Minimum,
    /// Mean
    Mean,
}

/// Activation function type.
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    /// ReLU
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// Leaky ReLU
    LeakyReLU,
    /// ELU
    ELU,
    /// GELU
    GELU,
}

/// Pooling operation type.
#[derive(Debug, Clone, Copy)]
pub enum PoolingType {
    /// Max pooling
    Max,
    /// Average pooling
    Average,
}

/// Hardware accelerator trait.
#[async_trait]
pub trait Accelerator: Send + Sync + Debug {
    /// Get the accelerator type.
    fn accelerator_type(&self) -> AcceleratorType;
    
    /// Get the accelerator capabilities.
    fn capabilities(&self) -> &AcceleratorCapabilities;
    
    /// Check if the accelerator supports an operation.
    fn supports_operation(&self, operation: &AcceleratorOperation) -> bool;
    
    /// Execute an operation.
    async fn execute(&self, operation: AcceleratorOperation) -> Result<Tensor>;
    
    /// Execute multiple operations in a batch.
    async fn execute_batch(&self, operations: Vec<AcceleratorOperation>) -> Result<Vec<Tensor>> {
        let mut results = Vec::with_capacity(operations.len());
        
        for operation in operations {
            let result = self.execute(operation).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Initialize the accelerator.
    async fn initialize(&self) -> Result<()>;
    
    /// Shutdown the accelerator.
    async fn shutdown(&self) -> Result<()>;
}

/// Hardware acceleration manager.
pub struct AccelerationManager {
    /// Available accelerators
    accelerators: RwLock<Vec<Arc<dyn Accelerator>>>,
    
    /// Default accelerator
    default_accelerator: RwLock<Option<Arc<dyn Accelerator>>>,
}

impl AccelerationManager {
    /// Create a new acceleration manager.
    pub fn new() -> Self {
        Self {
            accelerators: RwLock::new(Vec::new()),
            default_accelerator: RwLock::new(None),
        }
    }
    
    /// Initialize the acceleration manager.
    pub async fn initialize(&self) -> Result<()> {
        // Discover available accelerators
        self.discover_accelerators().await?;
        
        // Initialize accelerators
        for accelerator in self.accelerators.read().iter() {
            accelerator.initialize().await?;
        }
        
        // Set default accelerator
        if self.default_accelerator.read().is_none() {
            let accelerators = self.accelerators.read();
            if !accelerators.is_empty() {
                *self.default_accelerator.write() = Some(accelerators[0].clone());
            }
        }
        
        Ok(())
    }
    
    /// Discover available accelerators.
    async fn discover_accelerators(&self) -> Result<()> {
        let mut accelerators = self.accelerators.write();
        
        // Clear existing accelerators
        accelerators.clear();
        
        // Discover CUDA accelerators
        if let Ok(cuda_accelerators) = cuda::discover_accelerators().await {
            for accelerator in cuda_accelerators {
                accelerators.push(Arc::new(accelerator));
            }
        }
        
        // Discover OpenCL accelerators
        if let Ok(opencl_accelerators) = opencl::discover_accelerators().await {
            for accelerator in opencl_accelerators {
                accelerators.push(Arc::new(accelerator));
            }
        }
        
        // Discover Vulkan accelerators
        if let Ok(vulkan_accelerators) = vulkan::discover_accelerators().await {
            for accelerator in vulkan_accelerators {
                accelerators.push(Arc::new(accelerator));
            }
        }
        
        // Discover NPU accelerators
        if let Ok(npu_accelerators) = npu::discover_accelerators().await {
            for accelerator in npu_accelerators {
                accelerators.push(Arc::new(accelerator));
            }
        }
        
        // Always add CPU accelerator as fallback
        let cpu_accelerator = cpu::CPUAccelerator::new();
        accelerators.push(Arc::new(cpu_accelerator));
        
        tracing::info!("Discovered {} accelerators", accelerators.len());
        
        Ok(())
    }
    
    /// Get all available accelerators.
    pub fn get_accelerators(&self) -> Vec<Arc<dyn Accelerator>> {
        self.accelerators.read().clone()
    }
    
    /// Get accelerators of a specific type.
    pub fn get_accelerators_by_type(&self, accelerator_type: AcceleratorType) -> Vec<Arc<dyn Accelerator>> {
        self.accelerators.read().iter()
            .filter(|a| a.accelerator_type() == accelerator_type)
            .cloned()
            .collect()
    }
    
    /// Get the default accelerator.
    pub fn get_default_accelerator(&self) -> Option<Arc<dyn Accelerator>> {
        self.default_accelerator.read().clone()
    }
    
    /// Set the default accelerator.
    pub fn set_default_accelerator(&self, accelerator: Arc<dyn Accelerator>) {
        *self.default_accelerator.write() = Some(accelerator);
    }
    
    /// Execute an operation using the best available accelerator.
    pub async fn execute(&self, operation: AcceleratorOperation) -> Result<Tensor> {
        // Find the best accelerator for this operation
        let accelerator = self.find_best_accelerator(&operation)?;
        
        // Execute the operation
        accelerator.execute(operation).await
    }
    
    /// Execute multiple operations in a batch using the best available accelerator.
    pub async fn execute_batch(&self, operations: Vec<AcceleratorOperation>) -> Result<Vec<Tensor>> {
        // Group operations by best accelerator
        let mut operation_groups: std::collections::HashMap<AcceleratorType, Vec<AcceleratorOperation>> = std::collections::HashMap::new();
        
        for operation in operations {
            let accelerator = self.find_best_accelerator(&operation)?;
            operation_groups.entry(accelerator.accelerator_type()).or_default().push(operation);
        }
        
        // Execute each group
        let mut results = Vec::new();
        
        for (accelerator_type, ops) in operation_groups {
            let accelerator = self.get_accelerators_by_type(accelerator_type).first()
                .ok_or_else(|| crate::error::AIError::InternalError("Accelerator not found".to_string()))?
                .clone();
            
            let mut batch_results = accelerator.execute_batch(ops).await?;
            results.append(&mut batch_results);
        }
        
        Ok(results)
    }
    
    /// Find the best accelerator for an operation.
    fn find_best_accelerator(&self, operation: &AcceleratorOperation) -> Result<Arc<dyn Accelerator>> {
        // Try to find an accelerator that supports this operation
        for accelerator in self.accelerators.read().iter() {
            if accelerator.supports_operation(operation) {
                return Ok(accelerator.clone());
            }
        }
        
        // Fall back to CPU
        for accelerator in self.accelerators.read().iter() {
            if accelerator.accelerator_type() == AcceleratorType::CPU {
                return Ok(accelerator.clone());
            }
        }
        
        Err(crate::error::AIError::InternalError("No suitable accelerator found".to_string()))
    }
    
    /// Shutdown the acceleration manager.
    pub async fn shutdown(&self) -> Result<()> {
        // Shutdown all accelerators
        for accelerator in self.accelerators.read().iter() {
            accelerator.shutdown().await?;
        }
        
        // Clear accelerators
        self.accelerators.write().clear();
        *self.default_accelerator.write() = None;
        
        Ok(())
    }
}

impl Default for AccelerationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for AccelerationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccelerationManager")
            .field("accelerators", &format!("{} accelerators", self.accelerators.read().len()))
            .field("default_accelerator", &self.default_accelerator.read())
            .finish()
    }
}
