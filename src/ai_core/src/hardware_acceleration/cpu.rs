//! CPU-based accelerator implementation.

use std::sync::Arc;

use async_trait::async_trait;
use ndarray::{Array, ArrayD, Axis, Ix2};

use crate::error::{AIError, Result};
use crate::inference::Tensor;

use super::{
    Accelerator, AcceleratorCapabilities, AcceleratorOperation, AcceleratorType,
    ActivationType, ElementWiseOpType, PoolingType, ReductionOpType,
};

/// CPU-based accelerator.
#[derive(Debug)]
pub struct CPUAccelerator {
    /// Accelerator capabilities
    capabilities: AcceleratorCapabilities,
}

impl CPUAccelerator {
    /// Create a new CPU accelerator.
    pub fn new() -> Self {
        let capabilities = AcceleratorCapabilities {
            accelerator_type: AcceleratorType::CPU,
            device_name: "CPU".to_string(),
            device_vendor: "Generic".to_string(),
            device_version: "1.0".to_string(),
            available_memory: 0, // Will be detected at runtime
            supported_operations: vec![
                "MatrixMultiply".to_string(),
                "Convolution".to_string(),
                "ElementWise".to_string(),
                "Reduction".to_string(),
                "Activation".to_string(),
                "Pooling".to_string(),
                "Softmax".to_string(),
            ],
            supported_precisions: vec![
                "float32".to_string(),
                "float64".to_string(),
                "int32".to_string(),
                "int64".to_string(),
            ],
            compute_units: num_cpus::get() as u32,
            max_workgroup_size: 1024,
            max_dimensions: 3,
            extensions: Vec::new(),
        };
        
        Self { capabilities }
    }
    
    /// Convert a tensor to an ndarray.
    fn tensor_to_ndarray(&self, tensor: &Tensor) -> Result<ArrayD<f32>> {
        // This is a simplified implementation for demonstration purposes
        // In a real implementation, this would handle different tensor types
        
        let shape = tensor.shape.dims.clone();
        let data = match &tensor.data {
            crate::inference::TensorData::Float32(data) => data.clone(),
            _ => return Err(AIError::InvalidInput("Only Float32 tensors are supported".to_string())),
        };
        
        let array = Array::from_shape_vec(shape, (*data).clone())
            .map_err(|e| AIError::InvalidInput(format!("Failed to convert tensor to ndarray: {}", e)))?;
        
        Ok(array)
    }
    
    /// Convert an ndarray to a tensor.
    fn ndarray_to_tensor(&self, array: ArrayD<f32>) -> Result<Tensor> {
        let shape = crate::inference::TensorShape::new(array.shape().to_vec());
        let data = Arc::new(array.into_raw_vec());
        
        Ok(Tensor::new(shape, crate::inference::TensorData::Float32(data)))
    }
    
    /// Execute matrix multiplication.
    fn execute_matrix_multiply(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_array = self.tensor_to_ndarray(a)?;
        let b_array = self.tensor_to_ndarray(b)?;
        
        // Reshape to 2D for matrix multiplication
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();
        
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(AIError::InvalidInput("Tensors must have at least 2 dimensions".to_string()));
        }
        
        let a_rows = a_shape[a_shape.len() - 2];
        let a_cols = a_shape[a_shape.len() - 1];
        let b_rows = b_shape[b_shape.len() - 2];
        let b_cols = b_shape[b_shape.len() - 1];
        
        if a_cols != b_rows {
            return Err(AIError::InvalidInput(format!(
                "Incompatible dimensions for matrix multiplication: {}x{} and {}x{}",
                a_rows, a_cols, b_rows, b_cols
            )));
        }
        
        // Reshape to 2D
        let a_flat = a_array.into_shape((a_rows, a_cols)).unwrap();
        let b_flat = b_array.into_shape((b_rows, b_cols)).unwrap();
        
        // Perform matrix multiplication
        let result = a_flat.dot(&b_flat);
        
        // Convert back to tensor
        let result_array = result.into_dyn();
        self.ndarray_to_tensor(result_array)
    }
    
    /// Execute element-wise operation.
    fn execute_element_wise(&self, a: &Tensor, b: &Tensor, op_type: ElementWiseOpType) -> Result<Tensor> {
        let a_array = self.tensor_to_ndarray(a)?;
        let b_array = self.tensor_to_ndarray(b)?;
        
        // Check shapes
        if a_array.shape() != b_array.shape() {
            return Err(AIError::InvalidInput("Tensors must have the same shape for element-wise operations".to_string()));
        }
        
        // Perform element-wise operation
        let result = match op_type {
            ElementWiseOpType::Add => &a_array + &b_array,
            ElementWiseOpType::Subtract => &a_array - &b_array,
            ElementWiseOpType::Multiply => &a_array * &b_array,
            ElementWiseOpType::Divide => &a_array / &b_array,
            ElementWiseOpType::Maximum => a_array.mapv(|a| a.max(b_array[a_array.ndim()])),
            ElementWiseOpType::Minimum => a_array.mapv(|a| a.min(b_array[a_array.ndim()])),
            ElementWiseOpType::Power => a_array.mapv(|a| a.powf(b_array[a_array.ndim()])),
        };
        
        // Convert back to tensor
        self.ndarray_to_tensor(result)
    }
    
    /// Execute reduction operation.
    fn execute_reduction(&self, input: &Tensor, axes: &[u32], op_type: ReductionOpType) -> Result<Tensor> {
        let input_array = self.tensor_to_ndarray(input)?;
        
        // Convert axes to usize
        let axes: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
        
        // Perform reduction
        let result = match op_type {
            ReductionOpType::Sum => {
                let mut result = input_array.clone();
                for &axis in axes.iter().rev() {
                    result = result.sum_axis(Axis(axis));
                }
                result
            }
            ReductionOpType::Product => {
                let mut result = input_array.clone();
                for &axis in axes.iter().rev() {
                    result = result.fold(1.0, Axis(axis), |&a, &b| a * b);
                }
                result
            }
            ReductionOpType::Maximum => {
                let mut result = input_array.clone();
                for &axis in axes.iter().rev() {
                    result = result.fold(f32::NEG_INFINITY, Axis(axis), |&a, &b| a.max(b));
                }
                result
            }
            ReductionOpType::Minimum => {
                let mut result = input_array.clone();
                for &axis in axes.iter().rev() {
                    result = result.fold(f32::INFINITY, Axis(axis), |&a, &b| a.min(b));
                }
                result
            }
            ReductionOpType::Mean => {
                let mut result = input_array.clone();
                for &axis in axes.iter().rev() {
                    let len = result.len_of(Axis(axis)) as f32;
                    result = result.sum_axis(Axis(axis)) / len;
                }
                result
            }
        };
        
        // Convert back to tensor
        self.ndarray_to_tensor(result)
    }
    
    /// Execute activation function.
    fn execute_activation(&self, input: &Tensor, activation_type: ActivationType) -> Result<Tensor> {
        let input_array = self.tensor_to_ndarray(input)?;
        
        // Apply activation function
        let result = match activation_type {
            ActivationType::ReLU => input_array.mapv(|x| x.max(0.0)),
            ActivationType::Sigmoid => input_array.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::Tanh => input_array.mapv(|x| x.tanh()),
            ActivationType::LeakyReLU => input_array.mapv(|x| if x > 0.0 { x } else { 0.01 * x }),
            ActivationType::ELU => input_array.mapv(|x| if x > 0.0 { x } else { (x.exp() - 1.0) }),
            ActivationType::GELU => input_array.mapv(|x| {
                let sqrt_2_over_pi = 0.7978845608028654;
                let coef = sqrt_2_over_pi * (0.044715 * x.powi(3) + x);
                0.5 * x * (1.0 + coef.tanh())
            }),
        };
        
        // Convert back to tensor
        self.ndarray_to_tensor(result)
    }
    
    /// Execute softmax operation.
    fn execute_softmax(&self, input: &Tensor, axis: u32) -> Result<Tensor> {
        let input_array = self.tensor_to_ndarray(input)?;
        
        // Apply softmax along the specified axis
        let axis = axis as usize;
        
        // Find max along axis for numerical stability
        let max_along_axis = input_array.fold(f32::NEG_INFINITY, Axis(axis), |&a, &b| a.max(b));
        
        // Subtract max and compute exp
        let exp_input = input_array.clone() - max_along_axis.insert_axis(Axis(axis));
        let exp_input = exp_input.mapv(|x| x.exp());
        
        // Sum along axis
        let sum_along_axis = exp_input.sum_axis(Axis(axis));
        
        // Divide by sum
        let result = exp_input / sum_along_axis.insert_axis(Axis(axis));
        
        // Convert back to tensor
        self.ndarray_to_tensor(result)
    }
}

#[async_trait]
impl Accelerator for CPUAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::CPU
    }
    
    fn capabilities(&self) -> &AcceleratorCapabilities {
        &self.capabilities
    }
    
    fn supports_operation(&self, operation: &AcceleratorOperation) -> bool {
        match operation {
            AcceleratorOperation::MatrixMultiply { .. } => true,
            AcceleratorOperation::ElementWise { .. } => true,
            AcceleratorOperation::Reduction { .. } => true,
            AcceleratorOperation::Activation { .. } => true,
            AcceleratorOperation::Softmax { .. } => true,
            AcceleratorOperation::Convolution { .. } => false, // Not implemented yet
            AcceleratorOperation::Pooling { .. } => false, // Not implemented yet
            AcceleratorOperation::Custom { .. } => false,
        }
    }
    
    async fn execute(&self, operation: AcceleratorOperation) -> Result<Tensor> {
        match operation {
            AcceleratorOperation::MatrixMultiply { a, b } => {
                self.execute_matrix_multiply(&a, &b)
            }
            AcceleratorOperation::ElementWise { a, b, op_type } => {
                self.execute_element_wise(&a, &b, op_type)
            }
            AcceleratorOperation::Reduction { input, axes, op_type } => {
                self.execute_reduction(&input, &axes, op_type)
            }
            AcceleratorOperation::Activation { input, activation_type } => {
                self.execute_activation(&input, activation_type)
            }
            AcceleratorOperation::Softmax { input, axis } => {
                self.execute_softmax(&input, axis)
            }
            _ => Err(AIError::UnsupportedOperation(format!(
                "Operation not supported by CPU accelerator: {:?}",
                operation
            ))),
        }
    }
    
    async fn initialize(&self) -> Result<()> {
        // Nothing to initialize for CPU
        Ok(())
    }
    
    async fn shutdown(&self) -> Result<()> {
        // Nothing to shut down for CPU
        Ok(())
    }
}
