//! NPC AI system for intelligent non-player characters.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

use crate::config::AIConfig;
use crate::error::{AIError, Result};
use crate::inference::{InferenceEngine, InferenceOptions, Tensor};
use crate::model::ModelId;
use super::behavior::{BehaviorTree, BehaviorNode};

/// Type alias for NPC ID
pub type NPCId = String;

/// NPC state information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NPCState {
    /// NPC ID
    pub id: NPCId,
    
    /// NPC name
    pub name: String,
    
    /// NPC type or role
    pub npc_type: String,
    
    /// Position in 3D space
    pub position: [f32; 3],
    
    /// Rotation in 3D space (euler angles)
    pub rotation: [f32; 3],
    
    /// Current health
    pub health: f32,
    
    /// Maximum health
    pub max_health: f32,
    
    /// Current action being performed
    pub current_action: String,
    
    /// Current emotional state
    pub emotion: String,
    
    /// Relationship values with other entities
    pub relationships: HashMap<String, f32>,
    
    /// Knowledge and memories
    pub knowledge: HashMap<String, String>,
    
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

impl NPCState {
    /// Create a new NPC state.
    pub fn new(id: NPCId, name: String, npc_type: String) -> Self {
        Self {
            id,
            name,
            npc_type,
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            health: 100.0,
            max_health: 100.0,
            current_action: "idle".to_string(),
            emotion: "neutral".to_string(),
            relationships: HashMap::new(),
            knowledge: HashMap::new(),
            attributes: HashMap::new(),
        }
    }
    
    /// Convert the NPC state to a tensor for inference.
    pub fn to_tensor(&self) -> Result<HashMap<String, Tensor>> {
        // This is a simplified implementation for demonstration purposes
        // In a real implementation, this would properly convert the state to tensors
        
        // For now, just return an empty tensor map
        Ok(HashMap::new())
    }
}

/// NPC behavior decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NPCDecision {
    /// Action to perform
    pub action: String,
    
    /// Target of the action (if any)
    pub target: Option<String>,
    
    /// Parameters for the action
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Priority of the decision (0.0 - 1.0)
    pub priority: f32,
    
    /// Duration of the action in seconds
    pub duration: f32,
    
    /// Whether this action can be interrupted
    pub interruptible: bool,
}

/// NPC AI system for managing intelligent non-player characters.
pub struct NPCSystem {
    /// Configuration for the AI Core
    config: Arc<AIConfig>,
    
    /// Inference engine for AI operations
    inference_engine: Arc<InferenceEngine>,
    
    /// NPC states
    npc_states: RwLock<HashMap<NPCId, NPCState>>,
    
    /// Behavior trees for NPCs
    behavior_trees: RwLock<HashMap<String, Arc<BehaviorTree>>>,
    
    /// Default behavior model ID
    default_behavior_model: ModelId,
}

impl NPCSystem {
    /// Create a new NPC system.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the AI Core
    /// * `inference_engine` - Inference engine for AI operations
    ///
    /// # Returns
    ///
    /// A Result containing the initialized NPC system or an error
    pub async fn new(config: Arc<AIConfig>, inference_engine: Arc<InferenceEngine>) -> Result<Self> {
        Ok(Self {
            config,
            inference_engine,
            npc_states: RwLock::new(HashMap::new()),
            behavior_trees: RwLock::new(HashMap::new()),
            default_behavior_model: "npc-behavior-basic".to_string(),
        })
    }
    
    /// Initialize the NPC system.
    ///
    /// This method prepares the NPC system for use.
    pub async fn initialize(&self) -> Result<()> {
        // Load default behavior trees
        self.load_default_behaviors().await?;
        
        tracing::info!("NPC system initialized");
        Ok(())
    }
    
    /// Shutdown the NPC system.
    ///
    /// This method releases resources used by the NPC system.
    pub async fn shutdown(&self) -> Result<()> {
        // Clear NPC states
        self.npc_states.write().clear();
        
        // Clear behavior trees
        self.behavior_trees.write().clear();
        
        tracing::info!("NPC system shut down");
        Ok(())
    }
    
    /// Load default behavior trees.
    async fn load_default_behaviors(&self) -> Result<()> {
        let mut behaviors = self.behavior_trees.write();
        
        // Create some basic behavior trees
        
        // Guard behavior
        let guard_tree = BehaviorTree::new("guard", create_guard_behavior());
        behaviors.insert("guard".to_string(), Arc::new(guard_tree));
        
        // Merchant behavior
        let merchant_tree = BehaviorTree::new("merchant", create_merchant_behavior());
        behaviors.insert("merchant".to_string(), Arc::new(merchant_tree));
        
        // Villager behavior
        let villager_tree = BehaviorTree::new("villager", create_villager_behavior());
        behaviors.insert("villager".to_string(), Arc::new(villager_tree));
        
        tracing::info!("Loaded {} default behaviors", behaviors.len());
        Ok(())
    }
    
    /// Register a new NPC.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the NPC
    /// * `npc_type` - Type or role of the NPC
    ///
    /// # Returns
    ///
    /// The ID of the newly registered NPC
    pub fn register_npc(&self, name: String, npc_type: String) -> NPCId {
        let id = format!("npc-{}", Uuid::new_v4());
        let state = NPCState::new(id.clone(), name, npc_type);
        
        self.npc_states.write().insert(id.clone(), state);
        
        tracing::debug!("Registered NPC: {}", id);
        id
    }
    
    /// Unregister an NPC.
    ///
    /// # Arguments
    ///
    /// * `npc_id` - ID of the NPC to unregister
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    pub fn unregister_npc(&self, npc_id: &NPCId) -> Result<()> {
        let mut states = self.npc_states.write();
        
        if states.remove(npc_id).is_none() {
            return Err(AIError::InvalidInput(format!("NPC not found: {}", npc_id)));
        }
        
        tracing::debug!("Unregistered NPC: {}", npc_id);
        Ok(())
    }
    
    /// Get the state of an NPC.
    ///
    /// # Arguments
    ///
    /// * `npc_id` - ID of the NPC
    ///
    /// # Returns
    ///
    /// A Result containing the NPC state or an error
    pub fn get_npc_state(&self, npc_id: &NPCId) -> Result<NPCState> {
        let states = self.npc_states.read();
        
        states
            .get(npc_id)
            .cloned()
            .ok_or_else(|| AIError::InvalidInput(format!("NPC not found: {}", npc_id)))
    }
    
    /// Update the state of an NPC.
    ///
    /// # Arguments
    ///
    /// * `npc_id` - ID of the NPC
    /// * `state` - New state for the NPC
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    pub fn update_npc_state(&self, npc_id: &NPCId, state: NPCState) -> Result<()> {
        let mut states = self.npc_states.write();
        
        if !states.contains_key(npc_id) {
            return Err(AIError::InvalidInput(format!("NPC not found: {}", npc_id)));
        }
        
        states.insert(npc_id.clone(), state);
        Ok(())
    }
    
    /// Make a decision for an NPC.
    ///
    /// # Arguments
    ///
    /// * `npc_id` - ID of the NPC
    /// * `context` - Context information for decision making
    ///
    /// # Returns
    ///
    /// A Result containing the NPC decision or an error
    pub async fn make_decision(
        &self,
        npc_id: &NPCId,
        context: HashMap<String, serde_json::Value>,
    ) -> Result<NPCDecision> {
        // Get the NPC state
        let state = self.get_npc_state(npc_id)?;
        
        // Get the behavior tree for this NPC type
        let behaviors = self.behavior_trees.read();
        let behavior_tree = behaviors
            .get(&state.npc_type)
            .cloned()
            .unwrap_or_else(|| {
                // Use default behavior if no specific behavior is found
                behaviors
                    .get("villager")
                    .cloned()
                    .expect("Default behavior not found")
            });
        
        // Execute the behavior tree
        let decision = behavior_tree.execute(&state, &context)?;
        
        // If the behavior tree didn't produce a decision, use the AI model
        if decision.is_none() {
            return self.make_ai_decision(npc_id, context).await;
        }
        
        Ok(decision.unwrap())
    }
    
    /// Make a decision for an NPC using the AI model.
    ///
    /// # Arguments
    ///
    /// * `npc_id` - ID of the NPC
    /// * `context` - Context information for decision making
    ///
    /// # Returns
    ///
    /// A Result containing the NPC decision or an error
    async fn make_ai_decision(
        &self,
        npc_id: &NPCId,
        context: HashMap<String, serde_json::Value>,
    ) -> Result<NPCDecision> {
        // Get the NPC state
        let state = self.get_npc_state(npc_id)?;
        
        // Convert state and context to tensors
        let mut inputs = state.to_tensor()?;
        
        // Add context tensor
        // In a real implementation, this would properly convert the context to tensors
        
        // Run inference
        let options = InferenceOptions {
            use_cache: true,
            ..Default::default()
        };
        
        let result = self
            .inference_engine
            .run_inference(&self.default_behavior_model, inputs, Some(options))
            .await?;
        
        // Parse the result
        // In a real implementation, this would properly parse the output tensors
        
        // For now, just return a dummy decision
        Ok(NPCDecision {
            action: "idle".to_string(),
            target: None,
            parameters: HashMap::new(),
            priority: 0.5,
            duration: 1.0,
            interruptible: true,
        })
    }
    
    /// Update all NPCs.
    ///
    /// # Arguments
    ///
    /// * `delta_time` - Time elapsed since the last update in seconds
    /// * `global_context` - Global context information
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    pub async fn update_all(
        &self,
        delta_time: f32,
        global_context: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Get all NPC IDs
        let npc_ids: Vec<NPCId> = {
            let states = self.npc_states.read();
            states.keys().cloned().collect()
        };
        
        // Update each NPC
        for npc_id in npc_ids {
            // Create context for this NPC
            let mut context = global_context.clone();
            context.insert("delta_time".to_string(), serde_json::json!(delta_time));
            
            // Make a decision
            match self.make_decision(&npc_id, context).await {
                Ok(decision) => {
                    // Apply the decision
                    self.apply_decision(&npc_id, decision)?;
                }
                Err(e) => {
                    tracing::error!("Failed to make decision for NPC {}: {}", npc_id, e);
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply a decision to an NPC.
    ///
    /// # Arguments
    ///
    /// * `npc_id` - ID of the NPC
    /// * `decision` - Decision to apply
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    fn apply_decision(&self, npc_id: &NPCId, decision: NPCDecision) -> Result<()> {
        // Get the NPC state
        let mut state = self.get_npc_state(npc_id)?;
        
        // Update the state based on the decision
        state.current_action = decision.action;
        
        // Update the NPC state
        self.update_npc_state(npc_id, state)?;
        
        Ok(())
    }
}

/// Create a guard behavior tree.
fn create_guard_behavior() -> Box<dyn BehaviorNode> {
    // This is a simplified implementation for demonstration purposes
    // In a real implementation, this would create a proper behavior tree
    
    Box::new(move |state: &NPCState, context: &HashMap<String, serde_json::Value>| {
        // Simple guard behavior: patrol and attack enemies
        
        // Check if there's an enemy nearby
        if let Some(enemy) = context.get("nearest_enemy") {
            // Attack the enemy
            let enemy_id = enemy.as_str().unwrap_or("unknown");
            
            Some(NPCDecision {
                action: "attack".to_string(),
                target: Some(enemy_id.to_string()),
                parameters: HashMap::new(),
                priority: 0.8,
                duration: 2.0,
                interruptible: false,
            })
        } else {
            // Patrol
            Some(NPCDecision {
                action: "patrol".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.5,
                duration: 5.0,
                interruptible: true,
            })
        }
    })
}

/// Create a merchant behavior tree.
fn create_merchant_behavior() -> Box<dyn BehaviorNode> {
    // This is a simplified implementation for demonstration purposes
    
    Box::new(move |state: &NPCState, context: &HashMap<String, serde_json::Value>| {
        // Simple merchant behavior: attend to customers or manage inventory
        
        // Check if there's a customer nearby
        if let Some(customer) = context.get("nearest_player") {
            // Interact with the customer
            let customer_id = customer.as_str().unwrap_or("unknown");
            
            Some(NPCDecision {
                action: "talk".to_string(),
                target: Some(customer_id.to_string()),
                parameters: HashMap::new(),
                priority: 0.8,
                duration: 3.0,
                interruptible: true,
            })
        } else {
            // Manage inventory
            Some(NPCDecision {
                action: "manage_inventory".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.4,
                duration: 4.0,
                interruptible: true,
            })
        }
    })
}

/// Create a villager behavior tree.
fn create_villager_behavior() -> Box<dyn BehaviorNode> {
    // This is a simplified implementation for demonstration purposes
    
    Box::new(move |state: &NPCState, context: &HashMap<String, serde_json::Value>| {
        // Simple villager behavior: daily routines
        
        // Get the time of day
        let time_of_day = context
            .get("time_of_day")
            .and_then(|v| v.as_f64())
            .unwrap_or(12.0);
        
        // Different behaviors based on time of day
        if time_of_day < 8.0 || time_of_day > 22.0 {
            // Sleep at night
            Some(NPCDecision {
                action: "sleep".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.9,
                duration: 1.0,
                interruptible: false,
            })
        } else if time_of_day < 12.0 {
            // Work in the morning
            Some(NPCDecision {
                action: "work".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.7,
                duration: 5.0,
                interruptible: true,
            })
        } else if time_of_day < 14.0 {
            // Eat lunch
            Some(NPCDecision {
                action: "eat".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.8,
                duration: 2.0,
                interruptible: true,
            })
        } else if time_of_day < 18.0 {
            // Work in the afternoon
            Some(NPCDecision {
                action: "work".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.7,
                duration: 5.0,
                interruptible: true,
            })
        } else {
            // Socialize in the evening
            Some(NPCDecision {
                action: "socialize".to_string(),
                target: None,
                parameters: HashMap::new(),
                priority: 0.6,
                duration: 3.0,
                interruptible: true,
            })
        }
    })
}
