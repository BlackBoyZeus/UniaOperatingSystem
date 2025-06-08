//! Tensor implementation for the inference engine.

use std::fmt;
use std::sync::Arc;
use std::hash::{Hash, Hasher};

use serde::{Serialize, Deserialize};

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape {
    /// Dimensions of the tensor
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new tensor shape.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
    
    /// Get the number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Get the number of dimensions in the tensor.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }
    
    /// Check if the tensor is a scalar (0 dimensions).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
    
    /// Check if the tensor is a vector (1 dimension).
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }
    
    /// Check if the tensor is a matrix (2 dimensions).
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

/// Data type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorType {
    /// 32-bit floating point
    Float32,
    
    /// 16-bit floating point
    Float16,
    
    /// 8-bit integer
    Int8,
    
    /// 32-bit integer
    Int32,
    
    /// 64-bit integer
    Int64,
    
    /// Boolean
    Bool,
}

impl TensorType {
    /// Get the size of this data type in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            TensorType::Float32 => 4,
            TensorType::Float16 => 2,
            TensorType::Int8 => 1,
            TensorType::Int32 => 4,
            TensorType::Int64 => 8,
            TensorType::Bool => 1,
        }
    }
    
    /// Get the name of this data type.
    pub fn name(&self) -> &'static str {
        match self {
            TensorType::Float32 => "float32",
            TensorType::Float16 => "float16",
            TensorType::Int8 => "int8",
            TensorType::Int32 => "int32",
            TensorType::Int64 => "int64",
            TensorType::Bool => "bool",
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Storage for tensor data.
#[derive(Clone)]
pub enum TensorData {
    /// 32-bit floating point data
    Float32(Arc<Vec<f32>>),
    
    /// 16-bit floating point data (stored as u16 for compatibility)
    Float16(Arc<Vec<u16>>),
    
    /// 8-bit integer data
    Int8(Arc<Vec<i8>>),
    
    /// 32-bit integer data
    Int32(Arc<Vec<i32>>),
    
    /// 64-bit integer data
    Int64(Arc<Vec<i64>>),
    
    /// Boolean data
    Bool(Arc<Vec<bool>>),
}

impl TensorData {
    /// Get the data type of this tensor data.
    pub fn data_type(&self) -> TensorType {
        match self {
            TensorData::Float32(_) => TensorType::Float32,
            TensorData::Float16(_) => TensorType::Float16,
            TensorData::Int8(_) => TensorType::Int8,
            TensorData::Int32(_) => TensorType::Int32,
            TensorData::Int64(_) => TensorType::Int64,
            TensorData::Bool(_) => TensorType::Bool,
        }
    }
    
    /// Get the number of elements in this tensor data.
    pub fn len(&self) -> usize {
        match self {
            TensorData::Float32(data) => data.len(),
            TensorData::Float16(data) => data.len(),
            TensorData::Int8(data) => data.len(),
            TensorData::Int32(data) => data.len(),
            TensorData::Int64(data) => data.len(),
            TensorData::Bool(data) => data.len(),
        }
    }
    
    /// Check if this tensor data is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the size of this tensor data in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            TensorData::Float32(data) => data.len() * 4,
            TensorData::Float16(data) => data.len() * 2,
            TensorData::Int8(data) => data.len(),
            TensorData::Int32(data) => data.len() * 4,
            TensorData::Int64(data) => data.len() * 8,
            TensorData::Bool(data) => data.len(),
        }
    }
}

impl fmt::Debug for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorData::Float32(data) => {
                write!(f, "Float32({} elements)", data.len())
            }
            TensorData::Float16(data) => {
                write!(f, "Float16({} elements)", data.len())
            }
            TensorData::Int8(data) => {
                write!(f, "Int8({} elements)", data.len())
            }
            TensorData::Int32(data) => {
                write!(f, "Int32({} elements)", data.len())
            }
            TensorData::Int64(data) => {
                write!(f, "Int64({} elements)", data.len())
            }
            TensorData::Bool(data) => {
                write!(f, "Bool({} elements)", data.len())
            }
        }
    }
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorData::Float32(a), TensorData::Float32(b)) => a == b,
            (TensorData::Float16(a), TensorData::Float16(b)) => a == b,
            (TensorData::Int8(a), TensorData::Int8(b)) => a == b,
            (TensorData::Int32(a), TensorData::Int32(b)) => a == b,
            (TensorData::Int64(a), TensorData::Int64(b)) => a == b,
            (TensorData::Bool(a), TensorData::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for TensorData {}

impl Hash for TensorData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            TensorData::Float32(data) => {
                for &x in data.iter() {
                    state.write_u32(x.to_bits());
                }
            }
            TensorData::Float16(data) => {
                for &x in data.iter() {
                    state.write_u16(x);
                }
            }
            TensorData::Int8(data) => {
                for &x in data.iter() {
                    state.write_i8(x);
                }
            }
            TensorData::Int32(data) => {
                for &x in data.iter() {
                    state.write_i32(x);
                }
            }
            TensorData::Int64(data) => {
                for &x in data.iter() {
                    state.write_i64(x);
                }
            }
            TensorData::Bool(data) => {
                for &x in data.iter() {
                    state.write_u8(x as u8);
                }
            }
        }
    }
}

/// A tensor for AI operations.
#[derive(Clone)]
pub struct Tensor {
    /// Shape of the tensor
    pub shape: TensorShape,
    
    /// Data of the tensor
    pub data: TensorData,
}

impl Tensor {
    /// Create a new tensor with the given shape and data.
    pub fn new(shape: TensorShape, data: TensorData) -> Self {
        // Validate that the data size matches the shape
        let expected_elements = shape.num_elements();
        let actual_elements = data.len();
        
        if expected_elements != actual_elements {
            panic!(
                "Tensor data size mismatch: expected {} elements for shape {}, got {}",
                expected_elements, shape, actual_elements
            );
        }
        
        Self { shape, data }
    }
    
    /// Create a new f32 tensor with the given shape.
    pub fn new_f32(shape: TensorShape) -> Self {
        let num_elements = shape.num_elements();
        let data = Arc::new(vec![0.0f32; num_elements]);
        Self::new(shape, TensorData::Float32(data))
    }
    
    /// Create a new f16 tensor with the given shape.
    pub fn new_f16(shape: TensorShape) -> Self {
        let num_elements = shape.num_elements();
        let data = Arc::new(vec![0u16; num_elements]);
        Self::new(shape, TensorData::Float16(data))
    }
    
    /// Create a new i8 tensor with the given shape.
    pub fn new_i8(shape: TensorShape) -> Self {
        let num_elements = shape.num_elements();
        let data = Arc::new(vec![0i8; num_elements]);
        Self::new(shape, TensorData::Int8(data))
    }
    
    /// Create a new i32 tensor with the given shape.
    pub fn new_i32(shape: TensorShape) -> Self {
        let num_elements = shape.num_elements();
        let data = Arc::new(vec![0i32; num_elements]);
        Self::new(shape, TensorData::Int32(data))
    }
    
    /// Create a new i64 tensor with the given shape.
    pub fn new_i64(shape: TensorShape) -> Self {
        let num_elements = shape.num_elements();
        let data = Arc::new(vec![0i64; num_elements]);
        Self::new(shape, TensorData::Int64(data))
    }
    
    /// Create a new bool tensor with the given shape.
    pub fn new_bool(shape: TensorShape) -> Self {
        let num_elements = shape.num_elements();
        let data = Arc::new(vec![false; num_elements]);
        Self::new(shape, TensorData::Bool(data))
    }
    
    /// Get the data type of this tensor.
    pub fn data_type(&self) -> TensorType {
        self.data.data_type()
    }
    
    /// Get the number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }
    
    /// Get the size of this tensor in bytes.
    pub fn size_in_bytes(&self) -> usize {
        self.data.size_in_bytes()
    }
    
    /// Check if this tensor is a scalar.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }
    
    /// Check if this tensor is a vector.
    pub fn is_vector(&self) -> bool {
        self.shape.is_vector()
    }
    
    /// Check if this tensor is a matrix.
    pub fn is_matrix(&self) -> bool {
        self.shape.is_matrix()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={}, type={})",
            self.shape,
            self.data_type()
        )
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl Eq for Tensor {}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.data.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        
        assert_eq!(shape.num_elements(), 24);
        assert_eq!(shape.num_dims(), 3);
        assert!(!shape.is_scalar());
        assert!(!shape.is_vector());
        assert!(!shape.is_matrix());
        
        let vector_shape = TensorShape::new(vec![5]);
        assert!(vector_shape.is_vector());
        
        let matrix_shape = TensorShape::new(vec![2, 3]);
        assert!(matrix_shape.is_matrix());
        
        let scalar_shape = TensorShape::new(vec![]);
        assert!(scalar_shape.is_scalar());
    }
    
    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::new(vec![2, 3]);
        
        let f32_tensor = Tensor::new_f32(shape.clone());
        assert_eq!(f32_tensor.data_type(), TensorType::Float32);
        assert_eq!(f32_tensor.num_elements(), 6);
        
        let f16_tensor = Tensor::new_f16(shape.clone());
        assert_eq!(f16_tensor.data_type(), TensorType::Float16);
        
        let i8_tensor = Tensor::new_i8(shape.clone());
        assert_eq!(i8_tensor.data_type(), TensorType::Int8);
        
        let i32_tensor = Tensor::new_i32(shape.clone());
        assert_eq!(i32_tensor.data_type(), TensorType::Int32);
        
        let i64_tensor = Tensor::new_i64(shape.clone());
        assert_eq!(i64_tensor.data_type(), TensorType::Int64);
        
        let bool_tensor = Tensor::new_bool(shape);
        assert_eq!(bool_tensor.data_type(), TensorType::Bool);
    }
    
    #[test]
    #[should_panic(expected = "Tensor data size mismatch")]
    fn test_tensor_size_mismatch() {
        let shape = TensorShape::new(vec![2, 3]);
        let data = Arc::new(vec![0.0f32; 5]); // Should be 6 elements
        
        Tensor::new(shape, TensorData::Float32(data));
    }
}
