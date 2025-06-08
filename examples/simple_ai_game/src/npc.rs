use unia_ai::behavior_tree::BehaviorTree;
use unia_ai::world_model::WorldModel;
use unia_core::entity::Entity;

/// NPC (Non-Player Character) implementation
pub struct NPC {
    /// Entity ID
    pub entity: Entity,
    
    /// NPC type (e.g., "Villager", "Guard", "Merchant")
    pub npc_type: String,
    
    /// NPC position in the world (x, y)
    pub position: (f32, f32),
    
    /// NPC health
    pub health: i32,
    
    /// NPC behavior tree
    behavior_tree: Option<BehaviorTree>,
    
    /// Time since last behavior update
    behavior_update_timer: f32,
    
    /// Behavior update interval in seconds
    behavior_update_interval: f32,
}

impl NPC {
    /// Create a new NPC
    pub fn new(entity: Entity, npc_type: &str) -> Self {
        Self {
            entity,
            npc_type: npc_type.to_string(),
            position: (0.0, 0.0),
            health: 100,
            behavior_tree: None,
            behavior_update_timer: 0.0,
            behavior_update_interval: 1.0, // Update behavior every second
        }
    }
    
    /// Set the NPC's behavior tree
    pub fn set_behavior_tree(&mut self, behavior_tree: BehaviorTree) {
        self.behavior_tree = Some(behavior_tree);
    }
    
    /// Update the NPC
    pub fn update(&mut self, delta_time: f32, world_model: &WorldModel) {
        // Update behavior timer
        self.behavior_update_timer += delta_time;
        
        // Execute behavior tree at regular intervals
        if self.behavior_update_timer >= self.behavior_update_interval {
            self.behavior_update_timer = 0.0;
            
            // Execute behavior tree if available
            if let Some(bt) = &mut self.behavior_tree {
                bt.execute(world_model);
            }
        }
        
        // Simulate movement based on NPC type
        match self.npc_type.as_str() {
            "Villager" => self.simulate_villager_movement(delta_time, world_model),
            "Guard" => self.simulate_guard_movement(delta_time, world_model),
            "Merchant" => self.simulate_merchant_movement(delta_time, world_model),
            _ => {}
        }
    }
    
    /// Simulate villager movement
    fn simulate_villager_movement(&mut self, delta_time: f32, world_model: &WorldModel) {
        // Villagers wander during the day and stay still at night
        if world_model.get_value("time_of_day") == "day" || 
           world_model.get_value("time_of_day") == "morning" {
            // Simple random movement
            let dx = (rand::random::<f32>() - 0.5) * delta_time * 10.0;
            let dy = (rand::random::<f32>() - 0.5) * delta_time * 10.0;
            
            self.position.0 += dx;
            self.position.1 += dy;
            
            // Keep within bounds
            self.position.0 = self.position.0.max(0.0).min(512.0);
            self.position.1 = self.position.1.max(0.0).min(512.0);
        }
    }
    
    /// Simulate guard movement
    fn simulate_guard_movement(&mut self, delta_time: f32, world_model: &WorldModel) {
        // Guards patrol in a fixed pattern
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f32();
        
        // Circular patrol pattern
        let radius = 50.0;
        let center_x = 300.0;
        let center_y = 300.0;
        let speed = 0.2;
        
        self.position.0 = center_x + radius * (time * speed).cos();
        self.position.1 = center_y + radius * (time * speed).sin();
        
        // If player is nearby, move towards player
        let player_distance = world_model.get_value("player_distance").parse::<f32>().unwrap_or(f32::MAX);
        if player_distance < 50.0 {
            let player_x = world_model.get_value("player_position_x").parse::<f32>().unwrap_or(0.0);
            let player_y = world_model.get_value("player_position_y").parse::<f32>().unwrap_or(0.0);
            
            // Move towards player
            let dx = player_x - self.position.0;
            let dy = player_y - self.position.1;
            let distance = (dx * dx + dy * dy).sqrt();
            
            if distance > 0.0 {
                self.position.0 += dx / distance * delta_time * 20.0;
                self.position.1 += dy / distance * delta_time * 20.0;
            }
        }
    }
    
    /// Simulate merchant movement
    fn simulate_merchant_movement(&mut self, delta_time: f32, world_model: &WorldModel) {
        // Merchants stay near their shop during the day and go home at night
        if world_model.get_value("time_of_day") == "night" {
            // Go home (move towards a specific location)
            let home_x = 400.0;
            let home_y = 200.0;
            
            let dx = home_x - self.position.0;
            let dy = home_y - self.position.1;
            let distance = (dx * dx + dy * dy).sqrt();
            
            if distance > 5.0 {
                self.position.0 += dx / distance * delta_time * 15.0;
                self.position.1 += dy / distance * delta_time * 15.0;
            }
        } else {
            // Stay near shop with small movements
            let shop_x = 350.0;
            let shop_y = 200.0;
            
            // Random movement around shop
            let dx = (rand::random::<f32>() - 0.5) * delta_time * 5.0;
            let dy = (rand::random::<f32>() - 0.5) * delta_time * 5.0;
            
            self.position.0 += dx;
            self.position.1 += dy;
            
            // Stay within range of shop
            let dx = self.position.0 - shop_x;
            let dy = self.position.1 - shop_y;
            let distance = (dx * dx + dy * dy).sqrt();
            
            if distance > 20.0 {
                // Move back towards shop
                self.position.0 = shop_x + (dx / distance) * 20.0;
                self.position.1 = shop_y + (dy / distance) * 20.0;
            }
        }
    }
}
