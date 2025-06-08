//! Behavior tree implementation for game AI.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::error::Result;
use super::npc::{NPCState, NPCDecision};

/// A behavior node in a behavior tree.
pub type BehaviorNode = dyn Fn(&NPCState, &HashMap<String, serde_json::Value>) -> Option<NPCDecision> + Send + Sync;

/// A behavior tree for game AI.
pub struct BehaviorTree {
    /// Name of the behavior tree
    name: String,
    
    /// Root node of the behavior tree
    root: Box<dyn BehaviorNode>,
}

impl BehaviorTree {
    /// Create a new behavior tree.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the behavior tree
    /// * `root` - Root node of the behavior tree
    ///
    /// # Returns
    ///
    /// A new behavior tree
    pub fn new(name: &str, root: Box<dyn BehaviorNode>) -> Self {
        Self {
            name: name.to_string(),
            root,
        }
    }
    
    /// Get the name of the behavior tree.
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Execute the behavior tree.
    ///
    /// # Arguments
    ///
    /// * `state` - State of the entity
    /// * `context` - Context information for decision making
    ///
    /// # Returns
    ///
    /// A Result containing the decision or None if no decision was made
    pub fn execute(
        &self,
        state: &NPCState,
        context: &HashMap<String, serde_json::Value>,
    ) -> Result<Option<NPCDecision>> {
        // Execute the root node
        let decision = (self.root)(state, context);
        
        Ok(decision)
    }
}

impl fmt::Debug for BehaviorTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BehaviorTree")
            .field("name", &self.name)
            .finish()
    }
}

/// A sequence node that executes child nodes in sequence until one fails.
pub struct SequenceNode {
    /// Child nodes
    children: Vec<Arc<dyn BehaviorNode>>,
}

impl SequenceNode {
    /// Create a new sequence node.
    ///
    /// # Arguments
    ///
    /// * `children` - Child nodes
    ///
    /// # Returns
    ///
    /// A new sequence node
    pub fn new(children: Vec<Arc<dyn BehaviorNode>>) -> Self {
        Self { children }
    }
    
    /// Create a boxed sequence node.
    ///
    /// # Arguments
    ///
    /// * `children` - Child nodes
    ///
    /// # Returns
    ///
    /// A boxed sequence node
    pub fn boxed(children: Vec<Arc<dyn BehaviorNode>>) -> Box<dyn BehaviorNode> {
        let node = Self::new(children);
        
        Box::new(move |state, context| {
            for child in &node.children {
                if let Some(decision) = child(state, context) {
                    return Some(decision);
                }
            }
            
            None
        })
    }
}

/// A selector node that executes child nodes in sequence until one succeeds.
pub struct SelectorNode {
    /// Child nodes
    children: Vec<Arc<dyn BehaviorNode>>,
}

impl SelectorNode {
    /// Create a new selector node.
    ///
    /// # Arguments
    ///
    /// * `children` - Child nodes
    ///
    /// # Returns
    ///
    /// A new selector node
    pub fn new(children: Vec<Arc<dyn BehaviorNode>>) -> Self {
        Self { children }
    }
    
    /// Create a boxed selector node.
    ///
    /// # Arguments
    ///
    /// * `children` - Child nodes
    ///
    /// # Returns
    ///
    /// A boxed selector node
    pub fn boxed(children: Vec<Arc<dyn BehaviorNode>>) -> Box<dyn BehaviorNode> {
        let node = Self::new(children);
        
        Box::new(move |state, context| {
            for child in &node.children {
                if let Some(decision) = child(state, context) {
                    return Some(decision);
                }
            }
            
            None
        })
    }
}

/// A condition node that checks a condition.
pub struct ConditionNode {
    /// Condition function
    condition: Box<dyn Fn(&NPCState, &HashMap<String, serde_json::Value>) -> bool + Send + Sync>,
    
    /// Child node to execute if the condition is true
    child: Arc<dyn BehaviorNode>,
}

impl ConditionNode {
    /// Create a new condition node.
    ///
    /// # Arguments
    ///
    /// * `condition` - Condition function
    /// * `child` - Child node to execute if the condition is true
    ///
    /// # Returns
    ///
    /// A new condition node
    pub fn new(
        condition: Box<dyn Fn(&NPCState, &HashMap<String, serde_json::Value>) -> bool + Send + Sync>,
        child: Arc<dyn BehaviorNode>,
    ) -> Self {
        Self { condition, child }
    }
    
    /// Create a boxed condition node.
    ///
    /// # Arguments
    ///
    /// * `condition` - Condition function
    /// * `child` - Child node to execute if the condition is true
    ///
    /// # Returns
    ///
    /// A boxed condition node
    pub fn boxed(
        condition: Box<dyn Fn(&NPCState, &HashMap<String, serde_json::Value>) -> bool + Send + Sync>,
        child: Arc<dyn BehaviorNode>,
    ) -> Box<dyn BehaviorNode> {
        let node = Self::new(condition, child);
        
        Box::new(move |state, context| {
            if (node.condition)(state, context) {
                node.child(state, context)
            } else {
                None
            }
        })
    }
}

/// An action node that performs an action.
pub struct ActionNode {
    /// Action name
    action: String,
    
    /// Action parameters
    parameters: HashMap<String, serde_json::Value>,
    
    /// Action priority
    priority: f32,
    
    /// Action duration
    duration: f32,
    
    /// Whether the action can be interrupted
    interruptible: bool,
}

impl ActionNode {
    /// Create a new action node.
    ///
    /// # Arguments
    ///
    /// * `action` - Action name
    /// * `parameters` - Action parameters
    /// * `priority` - Action priority
    /// * `duration` - Action duration
    /// * `interruptible` - Whether the action can be interrupted
    ///
    /// # Returns
    ///
    /// A new action node
    pub fn new(
        action: &str,
        parameters: HashMap<String, serde_json::Value>,
        priority: f32,
        duration: f32,
        interruptible: bool,
    ) -> Self {
        Self {
            action: action.to_string(),
            parameters,
            priority,
            duration,
            interruptible,
        }
    }
    
    /// Create a boxed action node.
    ///
    /// # Arguments
    ///
    /// * `action` - Action name
    /// * `parameters` - Action parameters
    /// * `priority` - Action priority
    /// * `duration` - Action duration
    /// * `interruptible` - Whether the action can be interrupted
    ///
    /// # Returns
    ///
    /// A boxed action node
    pub fn boxed(
        action: &str,
        parameters: HashMap<String, serde_json::Value>,
        priority: f32,
        duration: f32,
        interruptible: bool,
    ) -> Box<dyn BehaviorNode> {
        let node = Self::new(action, parameters, priority, duration, interruptible);
        
        Box::new(move |_state, _context| {
            Some(NPCDecision {
                action: node.action.clone(),
                target: None,
                parameters: node.parameters.clone(),
                priority: node.priority,
                duration: node.duration,
                interruptible: node.interruptible,
            })
        })
    }
}

/// A decorator node that modifies the behavior of a child node.
pub struct DecoratorNode {
    /// Child node
    child: Arc<dyn BehaviorNode>,
    
    /// Decorator function
    decorator: Box<
        dyn Fn(
                Option<NPCDecision>,
                &NPCState,
                &HashMap<String, serde_json::Value>,
            ) -> Option<NPCDecision>
            + Send
            + Sync,
    >,
}

impl DecoratorNode {
    /// Create a new decorator node.
    ///
    /// # Arguments
    ///
    /// * `child` - Child node
    /// * `decorator` - Decorator function
    ///
    /// # Returns
    ///
    /// A new decorator node
    pub fn new(
        child: Arc<dyn BehaviorNode>,
        decorator: Box<
            dyn Fn(
                    Option<NPCDecision>,
                    &NPCState,
                    &HashMap<String, serde_json::Value>,
                ) -> Option<NPCDecision>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self { child, decorator }
    }
    
    /// Create a boxed decorator node.
    ///
    /// # Arguments
    ///
    /// * `child` - Child node
    /// * `decorator` - Decorator function
    ///
    /// # Returns
    ///
    /// A boxed decorator node
    pub fn boxed(
        child: Arc<dyn BehaviorNode>,
        decorator: Box<
            dyn Fn(
                    Option<NPCDecision>,
                    &NPCState,
                    &HashMap<String, serde_json::Value>,
                ) -> Option<NPCDecision>
                + Send
                + Sync,
        >,
    ) -> Box<dyn BehaviorNode> {
        let node = Self::new(child, decorator);
        
        Box::new(move |state, context| {
            let decision = node.child(state, context);
            (node.decorator)(decision, state, context)
        })
    }
}

/// Create an inverter decorator.
///
/// # Arguments
///
/// * `child` - Child node
///
/// # Returns
///
/// A boxed decorator node that inverts the result of the child node
pub fn inverter(child: Arc<dyn BehaviorNode>) -> Box<dyn BehaviorNode> {
    DecoratorNode::boxed(
        child,
        Box::new(|decision, _state, _context| {
            if decision.is_none() {
                Some(NPCDecision {
                    action: "idle".to_string(),
                    target: None,
                    parameters: HashMap::new(),
                    priority: 0.1,
                    duration: 1.0,
                    interruptible: true,
                })
            } else {
                None
            }
        }),
    )
}

/// Create a repeater decorator.
///
/// # Arguments
///
/// * `child` - Child node
/// * `count` - Number of times to repeat
///
/// # Returns
///
/// A boxed decorator node that repeats the child node
pub fn repeater(child: Arc<dyn BehaviorNode>, count: usize) -> Box<dyn BehaviorNode> {
    Box::new(move |state, context| {
        for _ in 0..count {
            if let Some(decision) = child(state, context) {
                return Some(decision);
            }
        }
        
        None
    })
}
