use unia_core::entity::Entity;
use crate::inventory::Inventory;

/// Player character implementation
pub struct Player {
    /// Entity ID
    pub entity: Entity,
    
    /// Player position in the world (x, y)
    pub position: (f32, f32),
    
    /// Player health
    pub health: i32,
    
    /// Player stamina
    pub stamina: i32,
    
    /// Player inventory
    pub inventory: Inventory,
    
    /// Player movement speed
    pub speed: f32,
    
    /// Player experience points
    pub experience: i32,
    
    /// Player level
    pub level: i32,
}

impl Player {
    /// Create a new player
    pub fn new(entity: Entity) -> Self {
        Self {
            entity,
            position: (0.0, 0.0),
            health: 100,
            stamina: 100,
            inventory: Inventory::new(20), // 20 slots
            speed: 50.0,
            experience: 0,
            level: 1,
        }
    }
    
    /// Update the player
    pub fn update(&mut self, delta_time: f32) {
        // In a real game, this would handle player input
        // For this example, we'll just simulate some basic regeneration
        
        // Regenerate stamina
        if self.stamina < 100 {
            self.stamina += (delta_time * 5.0) as i32;
            if self.stamina > 100 {
                self.stamina = 100;
            }
        }
        
        // Slow health regeneration when not in combat
        if self.health < 100 {
            self.health += (delta_time * 1.0) as i32;
            if self.health > 100 {
                self.health = 100;
            }
        }
    }
    
    /// Move the player
    pub fn move_by(&mut self, dx: f32, dy: f32) {
        self.position.0 += dx;
        self.position.1 += dy;
        
        // Keep within world bounds
        self.position.0 = self.position.0.max(0.0).min(512.0);
        self.position.1 = self.position.1.max(0.0).min(512.0);
    }
    
    /// Take damage
    pub fn take_damage(&mut self, amount: i32) {
        self.health -= amount;
        if self.health < 0 {
            self.health = 0;
        }
    }
    
    /// Heal the player
    pub fn heal(&mut self, amount: i32) {
        self.health += amount;
        if self.health > 100 {
            self.health = 100;
        }
    }
    
    /// Use stamina
    pub fn use_stamina(&mut self, amount: i32) -> bool {
        if self.stamina >= amount {
            self.stamina -= amount;
            true
        } else {
            false
        }
    }
    
    /// Add experience points
    pub fn add_experience(&mut self, amount: i32) {
        self.experience += amount;
        
        // Check for level up
        let next_level_exp = self.level * 100; // Simple level formula
        if self.experience >= next_level_exp {
            self.level_up();
        }
    }
    
    /// Level up the player
    fn level_up(&mut self) {
        self.level += 1;
        
        // Increase stats
        self.health += 10;
        self.stamina += 5;
        self.speed += 2.0;
        
        println!("Player leveled up to level {}!", self.level);
    }
}
