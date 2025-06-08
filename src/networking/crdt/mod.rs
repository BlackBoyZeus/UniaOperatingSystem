//! CRDT (Conflict-free Replicated Data Types) implementation for UNIA.
//!
//! This module provides a robust CRDT implementation for distributed state
//! synchronization in multiplayer gaming scenarios.

pub mod counter;
pub mod register;
pub mod set;
pub mod map;
pub mod sequence;
pub mod clock;

use std::fmt::Debug;
use std::hash::Hash;
use serde::{Serialize, Deserialize};

use self::clock::VectorClock;

/// CRDT trait for conflict-free replicated data types.
pub trait CRDT: Clone + Debug + Send + Sync {
    /// The type of value this CRDT represents.
    type Value;
    
    /// The type of operation that can be applied to this CRDT.
    type Operation: Clone + Debug + Serialize + for<'de> Deserialize<'de> + Send + Sync;
    
    /// Get the current value of the CRDT.
    fn value(&self) -> Self::Value;
    
    /// Apply an operation to the CRDT.
    fn apply(&mut self, operation: Self::Operation);
    
    /// Merge another CRDT of the same type into this one.
    fn merge(&mut self, other: &Self);
}

/// Node ID for CRDT operations.
pub type NodeId = String;

/// Timestamp for CRDT operations.
pub type Timestamp = u64;

/// Operation metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationMetadata {
    /// Node that created the operation
    pub node_id: NodeId,
    
    /// Timestamp when the operation was created
    pub timestamp: Timestamp,
    
    /// Vector clock at the time of the operation
    pub vector_clock: VectorClock,
}

impl OperationMetadata {
    /// Create new operation metadata.
    pub fn new(node_id: NodeId, timestamp: Timestamp, vector_clock: VectorClock) -> Self {
        Self {
            node_id,
            timestamp,
            vector_clock,
        }
    }
    
    /// Check if this operation happened before another operation.
    pub fn happened_before(&self, other: &Self) -> bool {
        self.vector_clock.happened_before(&other.vector_clock)
    }
    
    /// Check if this operation is concurrent with another operation.
    pub fn concurrent_with(&self, other: &Self) -> bool {
        self.vector_clock.concurrent_with(&other.vector_clock)
    }
}

/// CRDT operation with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation<T> {
    /// Operation data
    pub data: T,
    
    /// Operation metadata
    pub metadata: OperationMetadata,
}

impl<T> Operation<T> {
    /// Create a new operation.
    pub fn new(data: T, metadata: OperationMetadata) -> Self {
        Self {
            data,
            metadata,
        }
    }
}

/// CRDT registry for managing multiple CRDTs.
#[derive(Debug)]
pub struct CRDTRegistry<T: CRDT> {
    /// CRDTs by ID
    crdts: std::collections::HashMap<String, T>,
    
    /// Node ID
    node_id: NodeId,
    
    /// Vector clock
    vector_clock: VectorClock,
}

impl<T: CRDT> CRDTRegistry<T> {
    /// Create a new CRDT registry.
    pub fn new(node_id: NodeId) -> Self {
        let mut vector_clock = VectorClock::new();
        vector_clock.update(&node_id, 0);
        
        Self {
            crdts: std::collections::HashMap::new(),
            node_id,
            vector_clock,
        }
    }
    
    /// Get a CRDT by ID.
    pub fn get(&self, id: &str) -> Option<&T> {
        self.crdts.get(id)
    }
    
    /// Get a mutable CRDT by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut T> {
        self.crdts.get_mut(id)
    }
    
    /// Insert a CRDT.
    pub fn insert(&mut self, id: String, crdt: T) {
        self.crdts.insert(id, crdt);
    }
    
    /// Remove a CRDT.
    pub fn remove(&mut self, id: &str) -> Option<T> {
        self.crdts.remove(id)
    }
    
    /// Apply an operation to a CRDT.
    pub fn apply(&mut self, id: &str, operation: T::Operation) -> Result<(), String> {
        let crdt = self.crdts.get_mut(id).ok_or_else(|| format!("CRDT not found: {}", id))?;
        crdt.apply(operation);
        Ok(())
    }
    
    /// Merge another CRDT registry into this one.
    pub fn merge(&mut self, other: &Self) {
        for (id, other_crdt) in &other.crdts {
            if let Some(crdt) = self.crdts.get_mut(id) {
                crdt.merge(other_crdt);
            } else {
                self.crdts.insert(id.clone(), other_crdt.clone());
            }
        }
        
        // Merge vector clocks
        self.vector_clock.merge(&other.vector_clock);
    }
    
    /// Create operation metadata for a new operation.
    pub fn create_metadata(&mut self) -> OperationMetadata {
        // Increment the vector clock for this node
        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        self.vector_clock.increment(&self.node_id);
        
        OperationMetadata::new(
            self.node_id.clone(),
            timestamp,
            self.vector_clock.clone(),
        )
    }
    
    /// Get the node ID.
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
    
    /// Get the vector clock.
    pub fn vector_clock(&self) -> &VectorClock {
        &self.vector_clock
    }
    
    /// Get all CRDT IDs.
    pub fn ids(&self) -> Vec<String> {
        self.crdts.keys().cloned().collect()
    }
    
    /// Get the number of CRDTs.
    pub fn len(&self) -> usize {
        self.crdts.len()
    }
    
    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.crdts.is_empty()
    }
}
