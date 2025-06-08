//! Advanced behavior tree implementation for NPCs.
//!
//! This module provides a sophisticated behavior tree system for
//! creating complex, realistic NPC behaviors in games.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Serialize, Deserialize};
use parking_lot::RwLock;
use uuid::Uuid;

use crate::error::Result;
use crate::game_ai::npc::{NPCState, NPCDecision};

/// Behavior tree node result.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeResult {
    /// Node succeeded
    Success,
    
    /// Node failed
    Failure,
    
    /// Node is still running
    Running,
}

/// Behavior tree blackboard for sharing data between nodes.
#[derive(Debug, Clone, Default)]
pub struct Blackboard {
    /// Data storage
    data: HashMap<String, serde_json::Value>,
    
    /// Timestamps for timed data
    timestamps: HashMap<String, Instant>,
}

impl Blackboard {
    /// Create a new blackboard.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            timestamps: HashMap::new(),
        }
    }
    
    /// Set a value in the blackboard.
    pub fn set<T: Serialize>(&mut self, key: &str, value: T) -> Result<()> {
        let json_value = serde_json::to_value(value)?;
        self.data.insert(key.to_string(), json_value);
        self.timestamps.insert(key.to_string(), Instant::now());
        Ok(())
    }
    
    /// Get a value from the blackboard.
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        if let Some(value) = self.data.get(key) {
            let typed_value = serde_json::from_value(value.clone())?;
            Ok(Some(typed_value))
        } else {
            Ok(None)
        }
    }
    
    /// Check if a key exists in the blackboard.
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
    
    /// Remove a value from the blackboard.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.timestamps.remove(key);
        self.data.remove(key)
    }
    
    /// Get the time since a value was last set.
    pub fn time_since(&self, key: &str) -> Option<Duration> {
        self.timestamps.get(key).map(|time| time.elapsed())
    }
    
    /// Clear all values from the blackboard.
    pub fn clear(&mut self) {
        self.data.clear();
        self.timestamps.clear();
    }
}

/// Behavior tree node context.
#[derive(Debug, Clone)]
pub struct NodeContext<'a> {
    /// NPC state
    pub npc: &'a NPCState,
    
    /// Game context
    pub context: &'a HashMap<String, serde_json::Value>,
    
    /// Blackboard for sharing data between nodes
    pub blackboard: &'a mut Blackboard,
    
    /// Delta time since last update
    pub delta_time: f32,
}

/// Behavior tree node.
pub trait BehaviorNode: Send + Sync {
    /// Execute the node.
    fn execute(&self, context: &mut NodeContext) -> NodeResult;
    
    /// Get the node name.
    fn name(&self) -> &str;
    
    /// Reset the node state.
    fn reset(&self);
}

/// Composite node that executes child nodes in sequence until one fails.
pub struct SequenceNode {
    /// Node name
    name: String,
    
    /// Child nodes
    children: Vec<Arc<dyn BehaviorNode>>,
    
    /// Current running child index
    current_child: RwLock<usize>,
}

impl SequenceNode {
    /// Create a new sequence node.
    pub fn new(name: &str, children: Vec<Arc<dyn BehaviorNode>>) -> Self {
        Self {
            name: name.to_string(),
            children,
            current_child: RwLock::new(0),
        }
    }
}

impl BehaviorNode for SequenceNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        let mut current_child = self.current_child.write();
        
        // If we have no children, succeed immediately
        if self.children.is_empty() {
            return NodeResult::Success;
        }
        
        // Execute children in sequence
        while *current_child < self.children.len() {
            let child = &self.children[*current_child];
            
            match child.execute(context) {
                NodeResult::Success => {
                    // Child succeeded, move to the next one
                    *current_child += 1;
                }
                NodeResult::Running => {
                    // Child is still running, return Running
                    return NodeResult::Running;
                }
                NodeResult::Failure => {
                    // Child failed, reset and return Failure
                    *current_child = 0;
                    return NodeResult::Failure;
                }
            }
        }
        
        // All children succeeded, reset and return Success
        *current_child = 0;
        NodeResult::Success
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        *self.current_child.write() = 0;
        
        // Reset all children
        for child in &self.children {
            child.reset();
        }
    }
}

/// Composite node that executes child nodes in sequence until one succeeds.
pub struct SelectorNode {
    /// Node name
    name: String,
    
    /// Child nodes
    children: Vec<Arc<dyn BehaviorNode>>,
    
    /// Current running child index
    current_child: RwLock<usize>,
}

impl SelectorNode {
    /// Create a new selector node.
    pub fn new(name: &str, children: Vec<Arc<dyn BehaviorNode>>) -> Self {
        Self {
            name: name.to_string(),
            children,
            current_child: RwLock::new(0),
        }
    }
}

impl BehaviorNode for SelectorNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        let mut current_child = self.current_child.write();
        
        // If we have no children, fail immediately
        if self.children.is_empty() {
            return NodeResult::Failure;
        }
        
        // Execute children in sequence
        while *current_child < self.children.len() {
            let child = &self.children[*current_child];
            
            match child.execute(context) {
                NodeResult::Success => {
                    // Child succeeded, reset and return Success
                    *current_child = 0;
                    return NodeResult::Success;
                }
                NodeResult::Running => {
                    // Child is still running, return Running
                    return NodeResult::Running;
                }
                NodeResult::Failure => {
                    // Child failed, move to the next one
                    *current_child += 1;
                }
            }
        }
        
        // All children failed, reset and return Failure
        *current_child = 0;
        NodeResult::Failure
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        *self.current_child.write() = 0;
        
        // Reset all children
        for child in &self.children {
            child.reset();
        }
    }
}

/// Decorator node that inverts the result of its child.
pub struct InverterNode {
    /// Node name
    name: String,
    
    /// Child node
    child: Arc<dyn BehaviorNode>,
}

impl InverterNode {
    /// Create a new inverter node.
    pub fn new(name: &str, child: Arc<dyn BehaviorNode>) -> Self {
        Self {
            name: name.to_string(),
            child,
        }
    }
}

impl BehaviorNode for InverterNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        match self.child.execute(context) {
            NodeResult::Success => NodeResult::Failure,
            NodeResult::Failure => NodeResult::Success,
            NodeResult::Running => NodeResult::Running,
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        self.child.reset();
    }
}

/// Decorator node that succeeds after a certain number of ticks.
pub struct SucceederNode {
    /// Node name
    name: String,
    
    /// Child node
    child: Arc<dyn BehaviorNode>,
}

impl SucceederNode {
    /// Create a new succeeder node.
    pub fn new(name: &str, child: Arc<dyn BehaviorNode>) -> Self {
        Self {
            name: name.to_string(),
            child,
        }
    }
}

impl BehaviorNode for SucceederNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        // Execute the child
        let result = self.child.execute(context);
        
        // If the child is still running, return Running
        if result == NodeResult::Running {
            return NodeResult::Running;
        }
        
        // Otherwise, always return Success
        NodeResult::Success
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        self.child.reset();
    }
}

/// Decorator node that repeats its child a certain number of times.
pub struct RepeatNode {
    /// Node name
    name: String,
    
    /// Child node
    child: Arc<dyn BehaviorNode>,
    
    /// Number of repetitions (None for infinite)
    repetitions: Option<usize>,
    
    /// Current repetition count
    current_repetition: RwLock<usize>,
}

impl RepeatNode {
    /// Create a new repeat node.
    pub fn new(name: &str, child: Arc<dyn BehaviorNode>, repetitions: Option<usize>) -> Self {
        Self {
            name: name.to_string(),
            child,
            repetitions,
            current_repetition: RwLock::new(0),
        }
    }
}

impl BehaviorNode for RepeatNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        let mut current_repetition = self.current_repetition.write();
        
        // Check if we've reached the maximum number of repetitions
        if let Some(repetitions) = self.repetitions {
            if *current_repetition >= repetitions {
                *current_repetition = 0;
                return NodeResult::Success;
            }
        }
        
        // Execute the child
        match self.child.execute(context) {
            NodeResult::Success => {
                // Child succeeded, increment repetition count
                *current_repetition += 1;
                
                // Check if we've reached the maximum number of repetitions
                if let Some(repetitions) = self.repetitions {
                    if *current_repetition >= repetitions {
                        *current_repetition = 0;
                        return NodeResult::Success;
                    }
                }
                
                // Reset the child and return Running
                self.child.reset();
                NodeResult::Running
            }
            NodeResult::Failure => {
                // Child failed, reset and return Failure
                *current_repetition = 0;
                NodeResult::Failure
            }
            NodeResult::Running => {
                // Child is still running, return Running
                NodeResult::Running
            }
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        *self.current_repetition.write() = 0;
        self.child.reset();
    }
}

/// Decorator node that runs its child for a certain amount of time.
pub struct TimeLimitNode {
    /// Node name
    name: String,
    
    /// Child node
    child: Arc<dyn BehaviorNode>,
    
    /// Time limit in seconds
    time_limit: f32,
    
    /// Start time
    start_time: RwLock<Option<Instant>>,
}

impl TimeLimitNode {
    /// Create a new time limit node.
    pub fn new(name: &str, child: Arc<dyn BehaviorNode>, time_limit: f32) -> Self {
        Self {
            name: name.to_string(),
            child,
            time_limit,
            start_time: RwLock::new(None),
        }
    }
}

impl BehaviorNode for TimeLimitNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        let mut start_time = self.start_time.write();
        
        // Initialize start time if not set
        if start_time.is_none() {
            *start_time = Some(Instant::now());
        }
        
        // Check if time limit has been reached
        if let Some(time) = *start_time {
            if time.elapsed().as_secs_f32() >= self.time_limit {
                // Time limit reached, reset and return Failure
                *start_time = None;
                self.child.reset();
                return NodeResult::Failure;
            }
        }
        
        // Execute the child
        match self.child.execute(context) {
            NodeResult::Success => {
                // Child succeeded, reset and return Success
                *start_time = None;
                NodeResult::Success
            }
            NodeResult::Failure => {
                // Child failed, reset and return Failure
                *start_time = None;
                NodeResult::Failure
            }
            NodeResult::Running => {
                // Child is still running, return Running
                NodeResult::Running
            }
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        *self.start_time.write() = None;
        self.child.reset();
    }
}

/// Condition node that checks a condition.
pub struct ConditionNode {
    /// Node name
    name: String,
    
    /// Condition function
    condition: Box<dyn Fn(&NodeContext) -> bool + Send + Sync>,
}

impl ConditionNode {
    /// Create a new condition node.
    pub fn new(name: &str, condition: Box<dyn Fn(&NodeContext) -> bool + Send + Sync>) -> Self {
        Self {
            name: name.to_string(),
            condition,
        }
    }
}

impl BehaviorNode for ConditionNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        if (self.condition)(context) {
            NodeResult::Success
        } else {
            NodeResult::Failure
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        // Nothing to reset
    }
}

/// Action node that performs an action.
pub struct ActionNode {
    /// Node name
    name: String,
    
    /// Action function
    action: Box<dyn Fn(&mut NodeContext) -> NodeResult + Send + Sync>,
}

impl ActionNode {
    /// Create a new action node.
    pub fn new(name: &str, action: Box<dyn Fn(&mut NodeContext) -> NodeResult + Send + Sync>) -> Self {
        Self {
            name: name.to_string(),
            action,
        }
    }
}

impl BehaviorNode for ActionNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        (self.action)(context)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        // Nothing to reset
    }
}

/// Decision node that creates an NPC decision.
pub struct DecisionNode {
    /// Node name
    name: String,
    
    /// Decision function
    decision: Box<dyn Fn(&NodeContext) -> NPCDecision + Send + Sync>,
}

impl DecisionNode {
    /// Create a new decision node.
    pub fn new(name: &str, decision: Box<dyn Fn(&NodeContext) -> NPCDecision + Send + Sync>) -> Self {
        Self {
            name: name.to_string(),
            decision,
        }
    }
}

impl BehaviorNode for DecisionNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        // Create the decision
        let decision = (self.decision)(context);
        
        // Store the decision in the blackboard
        if let Err(e) = context.blackboard.set("decision", decision) {
            tracing::error!("Failed to store decision in blackboard: {}", e);
            return NodeResult::Failure;
        }
        
        NodeResult::Success
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        // Nothing to reset
    }
}

/// Parallel node that executes all children in parallel.
pub struct ParallelNode {
    /// Node name
    name: String,
    
    /// Child nodes
    children: Vec<Arc<dyn BehaviorNode>>,
    
    /// Success policy
    success_policy: ParallelPolicy,
    
    /// Failure policy
    failure_policy: ParallelPolicy,
    
    /// Child results
    child_results: RwLock<Vec<NodeResult>>,
}

/// Parallel node policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelPolicy {
    /// Require all children to succeed/fail
    RequireAll,
    
    /// Require only one child to succeed/fail
    RequireOne,
}

impl ParallelNode {
    /// Create a new parallel node.
    pub fn new(
        name: &str,
        children: Vec<Arc<dyn BehaviorNode>>,
        success_policy: ParallelPolicy,
        failure_policy: ParallelPolicy,
    ) -> Self {
        Self {
            name: name.to_string(),
            children,
            success_policy,
            failure_policy,
            child_results: RwLock::new(vec![NodeResult::Running; children.len()]),
        }
    }
}

impl BehaviorNode for ParallelNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        let mut child_results = self.child_results.write();
        
        // If we have no children, succeed immediately
        if self.children.is_empty() {
            return NodeResult::Success;
        }
        
        // Execute all children
        let mut success_count = 0;
        let mut failure_count = 0;
        
        for (i, child) in self.children.iter().enumerate() {
            // Skip children that have already completed
            if child_results[i] == NodeResult::Success || child_results[i] == NodeResult::Failure {
                if child_results[i] == NodeResult::Success {
                    success_count += 1;
                } else {
                    failure_count += 1;
                }
                continue;
            }
            
            // Execute the child
            child_results[i] = child.execute(context);
            
            // Count successes and failures
            if child_results[i] == NodeResult::Success {
                success_count += 1;
            } else if child_results[i] == NodeResult::Failure {
                failure_count += 1;
            }
        }
        
        // Check success policy
        if (self.success_policy == ParallelPolicy::RequireOne && success_count > 0) ||
           (self.success_policy == ParallelPolicy::RequireAll && success_count == self.children.len()) {
            // Reset child results
            for result in child_results.iter_mut() {
                *result = NodeResult::Running;
            }
            return NodeResult::Success;
        }
        
        // Check failure policy
        if (self.failure_policy == ParallelPolicy::RequireOne && failure_count > 0) ||
           (self.failure_policy == ParallelPolicy::RequireAll && failure_count == self.children.len()) {
            // Reset child results
            for result in child_results.iter_mut() {
                *result = NodeResult::Running;
            }
            return NodeResult::Failure;
        }
        
        // Still running
        NodeResult::Running
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        let mut child_results = self.child_results.write();
        
        // Reset all child results
        for result in child_results.iter_mut() {
            *result = NodeResult::Running;
        }
        
        // Reset all children
        for child in &self.children {
            child.reset();
        }
    }
}

/// Monitor node that executes its child and reports the result.
pub struct MonitorNode {
    /// Node name
    name: String,
    
    /// Child node
    child: Arc<dyn BehaviorNode>,
    
    /// Monitor function
    monitor: Box<dyn Fn(&str, &NodeResult) + Send + Sync>,
}

impl MonitorNode {
    /// Create a new monitor node.
    pub fn new(
        name: &str,
        child: Arc<dyn BehaviorNode>,
        monitor: Box<dyn Fn(&str, &NodeResult) + Send + Sync>,
    ) -> Self {
        Self {
            name: name.to_string(),
            child,
            monitor,
        }
    }
}

impl BehaviorNode for MonitorNode {
    fn execute(&self, context: &mut NodeContext) -> NodeResult {
        // Execute the child
        let result = self.child.execute(context);
        
        // Call the monitor function
        (self.monitor)(self.child.name(), &result);
        
        // Return the child's result
        result
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn reset(&self) {
        self.child.reset();
    }
}

/// Behavior tree for NPC decision making.
pub struct BehaviorTree {
    /// Tree name
    name: String,
    
    /// Root node
    root: Arc<dyn BehaviorNode>,
    
    /// Blackboard for sharing data between nodes
    blackboard: Blackboard,
}

impl BehaviorTree {
    /// Create a new behavior tree.
    pub fn new(name: &str, root: Arc<dyn BehaviorNode>) -> Self {
        Self {
            name: name.to_string(),
            root,
            blackboard: Blackboard::new(),
        }
    }
    
    /// Execute the behavior tree.
    pub fn execute(
        &mut self,
        npc: &NPCState,
        context: &HashMap<String, serde_json::Value>,
        delta_time: f32,
    ) -> Option<NPCDecision> {
        // Create node context
        let mut node_context = NodeContext {
            npc,
            context,
            blackboard: &mut self.blackboard,
            delta_time,
        };
        
        // Execute the root node
        let result = self.root.execute(&mut node_context);
        
        // If the tree succeeded, return the decision
        if result == NodeResult::Success {
            return self.blackboard.get("decision").unwrap_or(None);
        }
        
        None
    }
    
    /// Reset the behavior tree.
    pub fn reset(&mut self) {
        self.root.reset();
        self.blackboard.clear();
    }
    
    /// Get the tree name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Behavior tree builder for creating behavior trees.
pub struct BehaviorTreeBuilder {
    /// Tree name
    name: String,
    
    /// Root node
    root: Option<Arc<dyn BehaviorNode>>,
}

impl BehaviorTreeBuilder {
    /// Create a new behavior tree builder.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            root: None,
        }
    }
    
    /// Set the root node.
    pub fn with_root(mut self, root: Arc<dyn BehaviorNode>) -> Self {
        self.root = Some(root);
        self
    }
    
    /// Build the behavior tree.
    pub fn build(self) -> Result<BehaviorTree> {
        let root = self.root.ok_or_else(|| {
            crate::error::AIError::InvalidInput("Behavior tree must have a root node".to_string())
        })?;
        
        Ok(BehaviorTree::new(&self.name, root))
    }
}
