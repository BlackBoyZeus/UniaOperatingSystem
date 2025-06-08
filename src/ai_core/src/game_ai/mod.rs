//! Game AI systems for the UNIA AI Core.

pub mod npc;
pub mod procedural;
pub mod player;
pub mod behavior;
pub mod pathfinding;

use std::sync::Arc;

use crate::config::AIConfig;
use crate::error::Result;
use crate::inference::InferenceEngine;

/// Manager for game AI systems.
pub struct GameAIManager {
    /// Configuration for the AI Core
    config: Arc<AIConfig>,
    
    /// Inference engine for AI operations
    inference_engine: Arc<InferenceEngine>,
    
    /// NPC AI system
    npc_system: npc::NPCSystem,
    
    /// Procedural generation system
    procedural_system: procedural::ProceduralSystem,
    
    /// Player modeling system
    player_system: player::PlayerSystem,
}

impl GameAIManager {
    /// Create a new game AI manager.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the AI Core
    /// * `inference_engine` - Inference engine for AI operations
    ///
    /// # Returns
    ///
    /// A Result containing the initialized game AI manager or an error
    pub async fn new(config: Arc<AIConfig>, inference_engine: Arc<InferenceEngine>) -> Result<Self> {
        let npc_system = npc::NPCSystem::new(config.clone(), inference_engine.clone()).await?;
        let procedural_system = procedural::ProceduralSystem::new(config.clone(), inference_engine.clone()).await?;
        let player_system = player::PlayerSystem::new(config.clone(), inference_engine.clone()).await?;
        
        Ok(Self {
            config,
            inference_engine,
            npc_system,
            procedural_system,
            player_system,
        })
    }
    
    /// Get the NPC AI system.
    pub fn npc_system(&self) -> &npc::NPCSystem {
        &self.npc_system
    }
    
    /// Get the procedural generation system.
    pub fn procedural_system(&self) -> &procedural::ProceduralSystem {
        &self.procedural_system
    }
    
    /// Get the player modeling system.
    pub fn player_system(&self) -> &player::PlayerSystem {
        &self.player_system
    }
    
    /// Initialize the game AI systems.
    ///
    /// This method prepares the game AI systems for use.
    pub async fn initialize(&self) -> Result<()> {
        // Initialize subsystems
        self.npc_system.initialize().await?;
        self.procedural_system.initialize().await?;
        self.player_system.initialize().await?;
        
        tracing::info!("Game AI systems initialized");
        Ok(())
    }
    
    /// Shutdown the game AI systems.
    ///
    /// This method releases resources used by the game AI systems.
    pub async fn shutdown(&self) -> Result<()> {
        // Shutdown subsystems
        self.npc_system.shutdown().await?;
        self.procedural_system.shutdown().await?;
        self.player_system.shutdown().await?;
        
        tracing::info!("Game AI systems shut down");
        Ok(())
    }
}
