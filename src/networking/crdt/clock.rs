//! Vector clock implementation for CRDT causality tracking.

use std::collections::HashMap;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};

use super::NodeId;

/// Vector clock for tracking causality between events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorClock {
    /// Clock values by node ID
    clocks: HashMap<NodeId, u64>,
}

impl VectorClock {
    /// Create a new empty vector clock.
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }
    
    /// Get the clock value for a node.
    pub fn get(&self, node_id: &NodeId) -> u64 {
        *self.clocks.get(node_id).unwrap_or(&0)
    }
    
    /// Update the clock value for a node.
    pub fn update(&mut self, node_id: &NodeId, value: u64) {
        self.clocks.insert(node_id.clone(), value);
    }
    
    /// Increment the clock value for a node.
    pub fn increment(&mut self, node_id: &NodeId) {
        let value = self.get(node_id) + 1;
        self.update(node_id, value);
    }
    
    /// Merge another vector clock into this one.
    pub fn merge(&mut self, other: &Self) {
        for (node_id, &value) in &other.clocks {
            let current = self.get(node_id);
            if value > current {
                self.update(node_id, value);
            }
        }
    }
    
    /// Check if this vector clock happened before another vector clock.
    pub fn happened_before(&self, other: &Self) -> bool {
        // This happened before other if:
        // 1. For all nodes in self, self[node] <= other[node]
        // 2. There exists at least one node where self[node] < other[node]
        
        let mut less_than_exists = false;
        
        for (node_id, &value) in &self.clocks {
            let other_value = other.get(node_id);
            
            if value > other_value {
                return false;
            }
            
            if value < other_value {
                less_than_exists = true;
            }
        }
        
        // Check for nodes in other but not in self
        for node_id in other.clocks.keys() {
            if !self.clocks.contains_key(node_id) && other.get(node_id) > 0 {
                less_than_exists = true;
            }
        }
        
        less_than_exists
    }
    
    /// Check if this vector clock is concurrent with another vector clock.
    pub fn concurrent_with(&self, other: &Self) -> bool {
        !self.happened_before(other) && !other.happened_before(self)
    }
    
    /// Compare two vector clocks.
    pub fn compare(&self, other: &Self) -> VectorClockOrdering {
        if self == other {
            VectorClockOrdering::Equal
        } else if self.happened_before(other) {
            VectorClockOrdering::Less
        } else if other.happened_before(self) {
            VectorClockOrdering::Greater
        } else {
            VectorClockOrdering::Concurrent
        }
    }
    
    /// Get all node IDs in this vector clock.
    pub fn nodes(&self) -> Vec<NodeId> {
        self.clocks.keys().cloned().collect()
    }
    
    /// Get the number of nodes in this vector clock.
    pub fn len(&self) -> usize {
        self.clocks.len()
    }
    
    /// Check if this vector clock is empty.
    pub fn is_empty(&self) -> bool {
        self.clocks.is_empty()
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Vector clock ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorClockOrdering {
    /// Less than (happened before)
    Less,
    /// Equal
    Equal,
    /// Greater than (happened after)
    Greater,
    /// Concurrent (incomparable)
    Concurrent,
}

impl From<VectorClockOrdering> for Option<Ordering> {
    fn from(ordering: VectorClockOrdering) -> Self {
        match ordering {
            VectorClockOrdering::Less => Some(Ordering::Less),
            VectorClockOrdering::Equal => Some(Ordering::Equal),
            VectorClockOrdering::Greater => Some(Ordering::Greater),
            VectorClockOrdering::Concurrent => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_clock_basics() {
        let mut clock = VectorClock::new();
        
        // Initial state
        assert_eq!(clock.get(&"node1".to_string()), 0);
        
        // Update and get
        clock.update(&"node1".to_string(), 5);
        assert_eq!(clock.get(&"node1".to_string()), 5);
        
        // Increment
        clock.increment(&"node1".to_string());
        assert_eq!(clock.get(&"node1".to_string()), 6);
        
        // Multiple nodes
        clock.update(&"node2".to_string(), 3);
        assert_eq!(clock.get(&"node2".to_string()), 3);
    }
    
    #[test]
    fn test_vector_clock_merge() {
        let mut clock1 = VectorClock::new();
        clock1.update(&"node1".to_string(), 5);
        clock1.update(&"node2".to_string(), 3);
        
        let mut clock2 = VectorClock::new();
        clock2.update(&"node1".to_string(), 4);
        clock2.update(&"node2".to_string(), 7);
        clock2.update(&"node3".to_string(), 2);
        
        clock1.merge(&clock2);
        
        assert_eq!(clock1.get(&"node1".to_string()), 5);
        assert_eq!(clock1.get(&"node2".to_string()), 7);
        assert_eq!(clock1.get(&"node3".to_string()), 2);
    }
    
    #[test]
    fn test_vector_clock_happened_before() {
        let mut clock1 = VectorClock::new();
        clock1.update(&"node1".to_string(), 5);
        clock1.update(&"node2".to_string(), 3);
        
        let mut clock2 = VectorClock::new();
        clock2.update(&"node1".to_string(), 6);
        clock2.update(&"node2".to_string(), 3);
        
        assert!(clock1.happened_before(&clock2));
        assert!(!clock2.happened_before(&clock1));
        
        let mut clock3 = VectorClock::new();
        clock3.update(&"node1".to_string(), 5);
        clock3.update(&"node2".to_string(), 4);
        
        assert!(clock1.happened_before(&clock3));
        assert!(!clock3.happened_before(&clock1));
        
        let mut clock4 = VectorClock::new();
        clock4.update(&"node1".to_string(), 6);
        clock4.update(&"node2".to_string(), 2);
        
        assert!(!clock1.happened_before(&clock4));
        assert!(!clock4.happened_before(&clock1));
        assert!(clock1.concurrent_with(&clock4));
    }
    
    #[test]
    fn test_vector_clock_compare() {
        let mut clock1 = VectorClock::new();
        clock1.update(&"node1".to_string(), 5);
        clock1.update(&"node2".to_string(), 3);
        
        let mut clock2 = VectorClock::new();
        clock2.update(&"node1".to_string(), 5);
        clock2.update(&"node2".to_string(), 3);
        
        let mut clock3 = VectorClock::new();
        clock3.update(&"node1".to_string(), 6);
        clock3.update(&"node2".to_string(), 3);
        
        let mut clock4 = VectorClock::new();
        clock4.update(&"node1".to_string(), 6);
        clock4.update(&"node2".to_string(), 2);
        
        assert_eq!(clock1.compare(&clock2), VectorClockOrdering::Equal);
        assert_eq!(clock1.compare(&clock3), VectorClockOrdering::Less);
        assert_eq!(clock3.compare(&clock1), VectorClockOrdering::Greater);
        assert_eq!(clock1.compare(&clock4), VectorClockOrdering::Concurrent);
    }
}
