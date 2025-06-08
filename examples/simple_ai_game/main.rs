//! Simple AI Game Example
//!
//! This example demonstrates a simple game that uses the UNIA AI Core
//! for NPC behavior and procedural content generation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{App, Arg};
use tokio::sync::Mutex;
use tokio::time;

use unia_ai_core::{AICore, init_tracing};
use unia_ai_core::config::AIConfig;
use unia_ai_core::game_ai::npc::{NPCState, NPCDecision};

/// Game state.
struct GameState {
    /// Player position
    player_position: [f32; 2],
    
    /// Player health
    player_health: f32,
    
    /// NPCs
    npcs: HashMap<String, NPCState>,
    
    /// Game time
    game_time: f32,
    
    /// Day/night cycle (0.0 - 1.0)
    day_cycle: f32,
    
    /// Weather (0.0 = clear, 1.0 = stormy)
    weather: f32,
}

impl GameState {
    /// Create a new game state.
    fn new() -> Self {
        Self {
            player_position: [0.0, 0.0],
            player_health: 100.0,
            npcs: HashMap::new(),
            game_time: 0.0,
            day_cycle: 0.5,
            weather: 0.0,
        }
    }
    
    /// Update the game state.
    fn update(&mut self, delta_time: f32) {
        // Update game time
        self.game_time += delta_time;
        
        // Update day/night cycle (full cycle every 10 minutes)
        self.day_cycle = (self.game_time / 600.0) % 1.0;
        
        // Update weather (changes slowly)
        self.weather += (rand::random::<f32>() - 0.5) * 0.01;
        self.weather = self.weather.clamp(0.0, 1.0);
    }
    
    /// Convert to a context map for AI.
    fn to_context(&self) -> HashMap<String, serde_json::Value> {
        let mut context = HashMap::new();
        
        context.insert("player_position".to_string(), serde_json::json!(self.player_position));
        context.insert("player_health".to_string(), serde_json::json!(self.player_health));
        context.insert("game_time".to_string(), serde_json::json!(self.game_time));
        context.insert("day_cycle".to_string(), serde_json::json!(self.day_cycle));
        context.insert("time_of_day".to_string(), serde_json::json!(self.day_cycle * 24.0));
        context.insert("weather".to_string(), serde_json::json!(self.weather));
        
        context
    }
}

/// Game.
struct Game {
    /// Game state
    state: Mutex<GameState>,
    
    /// AI Core
    ai_core: Arc<AICore>,
    
    /// Running flag
    running: Mutex<bool>,
}

impl Game {
    /// Create a new game.
    async fn new() -> Result<Arc<Self>> {
        // Initialize AI Core
        let config = AIConfig::default();
        let ai_core = AICore::new(config).await?;
        
        // Initialize the AI Core
        ai_core.initialize().await?;
        
        // Create game
        let game = Arc::new(Self {
            state: Mutex::new(GameState::new()),
            ai_core,
            running: Mutex::new(true),
        });
        
        // Initialize NPCs
        game.initialize_npcs().await?;
        
        Ok(game)
    }
    
    /// Initialize NPCs.
    async fn initialize_npcs(&self) -> Result<()> {
        let npc_system = self.ai_core.game_ai_manager().npc_system();
        
        // Register NPCs
        let npc_types = ["guard", "merchant", "villager"];
        let npc_names = ["Alice", "Bob", "Charlie", "Diana", "Ethan"];
        
        let mut state = self.state.lock().await;
        
        for i in 0..5 {
            let npc_type = npc_types[i % npc_types.len()];
            let npc_name = npc_names[i % npc_names.len()];
            
            // Register NPC
            let npc_id = npc_system.register_npc(npc_name.to_string(), npc_type.to_string());
            
            // Get NPC state
            let mut npc_state = npc_system.get_npc_state(&npc_id)?;
            
            // Set initial position (random)
            npc_state.position = [
                (rand::random::<f32>() - 0.5) * 20.0,
                (rand::random::<f32>() - 0.5) * 20.0,
                0.0,
            ];
            
            // Update NPC state
            npc_system.update_npc_state(&npc_id, npc_state.clone())?;
            
            // Add to game state
            state.npcs.insert(npc_id, npc_state);
        }
        
        println!("Initialized {} NPCs", state.npcs.len());
        
        Ok(())
    }
    
    /// Run the game loop.
    async fn run(self: &Arc<Self>) -> Result<()> {
        println!("Starting game loop...");
        
        let mut last_time = Instant::now();
        let frame_duration = Duration::from_millis(33); // ~30 FPS
        
        // Game loop
        while *self.running.lock().await {
            // Calculate delta time
            let current_time = Instant::now();
            let delta_time = current_time.duration_since(last_time).as_secs_f32();
            last_time = current_time;
            
            // Update game state
            self.update(delta_time).await?;
            
            // Render (just print some info)
            self.render().await?;
            
            // Sleep to maintain frame rate
            time::sleep(frame_duration).await;
        }
        
        println!("Game loop ended");
        
        Ok(())
    }
    
    /// Update the game state.
    async fn update(&self, delta_time: f32) -> Result<()> {
        // Update game state
        let mut state = self.state.lock().await;
        state.update(delta_time);
        
        // Get game context
        let context = state.to_context();
        
        // Update NPCs
        let npc_system = self.ai_core.game_ai_manager().npc_system();
        
        for (npc_id, npc_state) in state.npcs.iter_mut() {
            // Make a decision for the NPC
            let decision = npc_system.make_decision(npc_id, context.clone()).await?;
            
            // Apply the decision
            self.apply_npc_decision(npc_state, &decision, delta_time);
        }
        
        Ok(())
    }
    
    /// Apply an NPC decision.
    fn apply_npc_decision(&self, npc_state: &mut NPCState, decision: &NPCDecision, delta_time: f32) {
        // Update NPC state based on the decision
        npc_state.current_action = decision.action.clone();
        
        // Update position based on action
        match decision.action.as_str() {
            "patrol" => {
                // Move in a random direction
                let angle = rand::random::<f32>() * std::f32::consts::PI * 2.0;
                let speed = 1.0;
                npc_state.position[0] += angle.cos() * speed * delta_time;
                npc_state.position[1] += angle.sin() * speed * delta_time;
            }
            "move_to_target" => {
                // Move towards target
                if let Some(target) = &decision.target {
                    // In a real game, we would look up the target's position
                    // For this example, just move towards the origin
                    let direction = [
                        -npc_state.position[0],
                        -npc_state.position[1],
                    ];
                    
                    // Normalize direction
                    let length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
                    if length > 0.001 {
                        let speed = 2.0;
                        npc_state.position[0] += direction[0] / length * speed * delta_time;
                        npc_state.position[1] += direction[1] / length * speed * delta_time;
                    }
                }
            }
            _ => {
                // Other actions don't affect position
            }
        }
    }
    
    /// Render the game.
    async fn render(&self) -> Result<()> {
        let state = self.state.lock().await;
        
        // Clear screen
        print!("\x1B[2J\x1B[1;1H");
        
        // Print game info
        println!("=== Simple AI Game ===");
        println!("Time: {:.1}s", state.game_time);
        println!("Time of Day: {:.1}h", state.day_cycle * 24.0);
        println!("Weather: {}", if state.weather < 0.3 {
            "Clear"
        } else if state.weather < 0.7 {
            "Cloudy"
        } else {
            "Stormy"
        });
        println!("Player: pos=({:.1}, {:.1}) health={:.0}", 
            state.player_position[0], state.player_position[1], state.player_health);
        
        println!("\nNPCs:");
        for (npc_id, npc_state) in &state.npcs {
            println!("- {}: {} ({}) at ({:.1}, {:.1}) - {}",
                npc_id,
                npc_state.name,
                npc_state.npc_type,
                npc_state.position[0],
                npc_state.position[1],
                npc_state.current_action);
        }
        
        println!("\nPress Ctrl+C to exit");
        
        Ok(())
    }
    
    /// Stop the game.
    async fn stop(&self) {
        *self.running.lock().await = false;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let matches = App::new("Simple AI Game")
        .version("1.0")
        .author("UNIA Team")
        .about("A simple game using UNIA AI Core")
        .arg(Arg::with_name("debug")
            .short("d")
            .long("debug")
            .help("Enable debug logging"))
        .get_matches();
    
    // Initialize tracing
    init_tracing();
    
    // Create and run the game
    let game = Game::new().await.context("Failed to create game")?;
    
    // Set up signal handler for graceful shutdown
    let game_clone = Arc::clone(&game);
    ctrlc::set_handler(move || {
        let game = game_clone.clone();
        tokio::spawn(async move {
            game.stop().await;
        });
    }).context("Failed to set Ctrl+C handler")?;
    
    // Run the game
    game.run().await.context("Game loop failed")?;
    
    // Shutdown AI Core
    game.ai_core.shutdown().await.context("Failed to shutdown AI Core")?;
    
    println!("Game exited successfully");
    
    Ok(())
}
