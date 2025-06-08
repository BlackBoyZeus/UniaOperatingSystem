//! Core gameplay mechanics for the Advanced AI Game.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Serialize, Deserialize};
use uuid::Uuid;

use unia_ai_core::game_ai::npc::{NPCState, NPCDecision};
use unia_ai_core::game_ai::procedural::terrain_generator::{Terrain, TerrainType};

/// Player stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerStats {
    /// Health points
    pub health: f32,
    
    /// Maximum health points
    pub max_health: f32,
    
    /// Energy points
    pub energy: f32,
    
    /// Maximum energy points
    pub max_energy: f32,
    
    /// Experience points
    pub experience: u32,
    
    /// Level
    pub level: u32,
    
    /// Strength (affects damage)
    pub strength: u32,
    
    /// Agility (affects speed and dodge)
    pub agility: u32,
    
    /// Intelligence (affects skills and crafting)
    pub intelligence: u32,
    
    /// Endurance (affects health and stamina)
    pub endurance: u32,
}

impl Default for PlayerStats {
    fn default() -> Self {
        Self {
            health: 100.0,
            max_health: 100.0,
            energy: 100.0,
            max_energy: 100.0,
            experience: 0,
            level: 1,
            strength: 10,
            agility: 10,
            intelligence: 10,
            endurance: 10,
        }
    }
}

/// Player inventory item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryItem {
    /// Item ID
    pub id: String,
    
    /// Item name
    pub name: String,
    
    /// Item type
    pub item_type: ItemType,
    
    /// Item rarity
    pub rarity: ItemRarity,
    
    /// Item stats
    pub stats: HashMap<String, f32>,
    
    /// Item quantity
    pub quantity: u32,
    
    /// Item description
    pub description: String,
}

/// Item type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemType {
    /// Weapon
    Weapon,
    /// Armor
    Armor,
    /// Consumable
    Consumable,
    /// Material
    Material,
    /// Quest item
    QuestItem,
}

/// Item rarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemRarity {
    /// Common
    Common,
    /// Uncommon
    Uncommon,
    /// Rare
    Rare,
    /// Epic
    Epic,
    /// Legendary
    Legendary,
}

/// Player inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    /// Items in the inventory
    pub items: HashMap<String, InventoryItem>,
    
    /// Maximum number of items
    pub max_items: usize,
    
    /// Equipped items
    pub equipped: HashMap<String, String>,
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            items: HashMap::new(),
            max_items: 20,
            equipped: HashMap::new(),
        }
    }
}

/// Quest status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Quest objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestObjective {
    /// Objective ID
    pub id: String,
    
    /// Objective description
    pub description: String,
    
    /// Current progress
    pub progress: u32,
    
    /// Required progress to complete
    pub required: u32,
    
    /// Whether the objective is completed
    pub completed: bool,
}

/// Quest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    /// Quest ID
    pub id: String,
    
    /// Quest title
    pub title: String,
    
    /// Quest description
    pub description: String,
    
    /// Quest giver NPC ID
    pub giver: Option<String>,
    
    /// Quest objectives
    pub objectives: Vec<QuestObjective>,
    
    /// Quest status
    pub status: QuestStatus,
    
    /// Quest rewards
    pub rewards: HashMap<String, u32>,
    
    /// Quest experience reward
    pub experience: u32,
}

/// Player quest log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestLog {
    /// Active quests
    pub active: HashMap<String, Quest>,
    
    /// Completed quests
    pub completed: HashMap<String, Quest>,
    
    /// Failed quests
    pub failed: HashMap<String, Quest>,
}

impl Default for QuestLog {
    fn default() -> Self {
        Self {
            active: HashMap::new(),
            completed: HashMap::new(),
            failed: HashMap::new(),
        }
    }
}

/// Player state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerState {
    /// Player ID
    pub id: String,
    
    /// Player name
    pub name: String,
    
    /// Player position
    pub position: [f32; 3],
    
    /// Player rotation
    pub rotation: [f32; 3],
    
    /// Player stats
    pub stats: PlayerStats,
    
    /// Player inventory
    pub inventory: Inventory,
    
    /// Player quest log
    pub quest_log: QuestLog,
    
    /// Player skills
    pub skills: HashMap<String, u32>,
    
    /// Player reputation with factions
    pub reputation: HashMap<String, i32>,
    
    /// Player discovered locations
    pub discovered_locations: Vec<String>,
}

impl PlayerState {
    /// Create a new player state.
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            stats: PlayerStats::default(),
            inventory: Inventory::default(),
            quest_log: QuestLog::default(),
            skills: HashMap::new(),
            reputation: HashMap::new(),
            discovered_locations: Vec::new(),
        }
    }
    
    /// Add an item to the inventory.
    pub fn add_item(&mut self, item: InventoryItem) -> Result<(), String> {
        if self.inventory.items.len() >= self.inventory.max_items {
            return Err("Inventory is full".to_string());
        }
        
        // Check if we already have this item (for stackable items)
        if let Some(existing_item) = self.inventory.items.get_mut(&item.id) {
            existing_item.quantity += item.quantity;
        } else {
            self.inventory.items.insert(item.id.clone(), item);
        }
        
        Ok(())
    }
    
    /// Remove an item from the inventory.
    pub fn remove_item(&mut self, item_id: &str, quantity: u32) -> Result<(), String> {
        if let Some(item) = self.inventory.items.get_mut(item_id) {
            if item.quantity < quantity {
                return Err("Not enough items".to_string());
            }
            
            item.quantity -= quantity;
            
            if item.quantity == 0 {
                self.inventory.items.remove(item_id);
                
                // If the item was equipped, unequip it
                self.inventory.equipped.retain(|_, equipped_id| equipped_id != item_id);
            }
            
            Ok(())
        } else {
            Err("Item not found".to_string())
        }
    }
    
    /// Equip an item.
    pub fn equip_item(&mut self, item_id: &str, slot: &str) -> Result<(), String> {
        if !self.inventory.items.contains_key(item_id) {
            return Err("Item not found".to_string());
        }
        
        self.inventory.equipped.insert(slot.to_string(), item_id.to_string());
        Ok(())
    }
    
    /// Unequip an item.
    pub fn unequip_item(&mut self, slot: &str) -> Result<(), String> {
        if !self.inventory.equipped.contains_key(slot) {
            return Err("No item equipped in this slot".to_string());
        }
        
        self.inventory.equipped.remove(slot);
        Ok(())
    }
    
    /// Add a quest.
    pub fn add_quest(&mut self, quest: Quest) -> Result<(), String> {
        if self.quest_log.active.contains_key(&quest.id) ||
           self.quest_log.completed.contains_key(&quest.id) ||
           self.quest_log.failed.contains_key(&quest.id) {
            return Err("Quest already exists".to_string());
        }
        
        self.quest_log.active.insert(quest.id.clone(), quest);
        Ok(())
    }
    
    /// Update quest progress.
    pub fn update_quest_progress(&mut self, quest_id: &str, objective_id: &str, progress: u32) -> Result<bool, String> {
        if let Some(quest) = self.quest_log.active.get_mut(quest_id) {
            for objective in &mut quest.objectives {
                if objective.id == objective_id {
                    objective.progress += progress;
                    
                    if objective.progress >= objective.required {
                        objective.completed = true;
                        objective.progress = objective.required;
                    }
                    
                    // Check if all objectives are completed
                    let all_completed = quest.objectives.iter().all(|obj| obj.completed);
                    
                    if all_completed {
                        // Complete the quest
                        let mut quest = self.quest_log.active.remove(quest_id).unwrap();
                        quest.status = QuestStatus::Completed;
                        
                        // Add rewards
                        self.stats.experience += quest.experience;
                        
                        // Check for level up
                        self.check_level_up();
                        
                        // Add the quest to completed quests
                        self.quest_log.completed.insert(quest_id.to_string(), quest);
                        
                        return Ok(true);
                    }
                    
                    return Ok(false);
                }
            }
            
            Err("Objective not found".to_string())
        } else {
            Err("Quest not found".to_string())
        }
    }
    
    /// Check if the player should level up.
    fn check_level_up(&mut self) {
        let required_exp = self.stats.level * 100;
        
        if self.stats.experience >= required_exp {
            self.stats.level += 1;
            self.stats.experience -= required_exp;
            
            // Increase stats
            self.stats.max_health += 10.0;
            self.stats.health = self.stats.max_health;
            self.stats.max_energy += 5.0;
            self.stats.energy = self.stats.max_energy;
            
            // Check for additional level ups
            self.check_level_up();
        }
    }
    
    /// Take damage.
    pub fn take_damage(&mut self, amount: f32) -> bool {
        self.stats.health -= amount;
        
        if self.stats.health <= 0.0 {
            self.stats.health = 0.0;
            return true; // Player died
        }
        
        false
    }
    
    /// Heal.
    pub fn heal(&mut self, amount: f32) {
        self.stats.health += amount;
        
        if self.stats.health > self.stats.max_health {
            self.stats.health = self.stats.max_health;
        }
    }
    
    /// Use energy.
    pub fn use_energy(&mut self, amount: f32) -> bool {
        if self.stats.energy < amount {
            return false; // Not enough energy
        }
        
        self.stats.energy -= amount;
        true
    }
    
    /// Restore energy.
    pub fn restore_energy(&mut self, amount: f32) {
        self.stats.energy += amount;
        
        if self.stats.energy > self.stats.max_energy {
            self.stats.energy = self.stats.max_energy;
        }
    }
    
    /// Move the player.
    pub fn move_player(&mut self, direction: [f32; 3], speed: f32, delta_time: f32) {
        self.position[0] += direction[0] * speed * delta_time;
        self.position[1] += direction[1] * speed * delta_time;
        self.position[2] += direction[2] * speed * delta_time;
    }
    
    /// Rotate the player.
    pub fn rotate_player(&mut self, rotation: [f32; 3]) {
        self.rotation = rotation;
    }
    
    /// Get the player's movement speed.
    pub fn get_movement_speed(&self) -> f32 {
        let base_speed = 5.0;
        let agility_bonus = self.stats.agility as f32 * 0.1;
        
        base_speed + agility_bonus
    }
    
    /// Get the player's attack damage.
    pub fn get_attack_damage(&self) -> f32 {
        let base_damage = 5.0;
        let strength_bonus = self.stats.strength as f32 * 0.5;
        
        // Check for equipped weapon
        if let Some(weapon_id) = self.inventory.equipped.get("weapon") {
            if let Some(weapon) = self.inventory.items.get(weapon_id) {
                if let Some(&weapon_damage) = weapon.stats.get("damage") {
                    return base_damage + strength_bonus + weapon_damage;
                }
            }
        }
        
        base_damage + strength_bonus
    }
    
    /// Get the player's defense.
    pub fn get_defense(&self) -> f32 {
        let base_defense = 1.0;
        let endurance_bonus = self.stats.endurance as f32 * 0.2;
        
        // Check for equipped armor
        let mut armor_defense = 0.0;
        for (slot, item_id) in &self.inventory.equipped {
            if slot.contains("armor") {
                if let Some(armor) = self.inventory.items.get(item_id) {
                    if let Some(&defense) = armor.stats.get("defense") {
                        armor_defense += defense;
                    }
                }
            }
        }
        
        base_defense + endurance_bonus + armor_defense
    }
}

/// Combat system.
pub struct CombatSystem {
    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl CombatSystem {
    /// Create a new combat system.
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
    
    /// Calculate damage from an attack.
    pub fn calculate_damage(&mut self, attacker_damage: f32, defender_defense: f32) -> f32 {
        let base_damage = attacker_damage - defender_defense;
        let min_damage = attacker_damage * 0.1; // Always do at least 10% damage
        
        // Add some randomness
        let variance = attacker_damage * 0.2; // ±20% variance
        let random_factor = self.rng.gen_range(-variance..variance);
        
        (base_damage + random_factor).max(min_damage)
    }
    
    /// Check if an attack hits.
    pub fn check_hit(&mut self, attacker_accuracy: f32, defender_evasion: f32) -> bool {
        let hit_chance = (attacker_accuracy / (attacker_accuracy + defender_evasion)).clamp(0.1, 0.9);
        let roll = self.rng.gen_range(0.0..1.0);
        
        roll <= hit_chance
    }
    
    /// Check if an attack is a critical hit.
    pub fn check_critical(&mut self, critical_chance: f32) -> bool {
        let roll = self.rng.gen_range(0.0..1.0);
        roll <= critical_chance
    }
    
    /// Process an attack.
    pub fn process_attack(
        &mut self,
        attacker_damage: f32,
        attacker_accuracy: f32,
        attacker_critical_chance: f32,
        defender_defense: f32,
        defender_evasion: f32,
    ) -> Option<f32> {
        // Check if the attack hits
        if !self.check_hit(attacker_accuracy, defender_evasion) {
            return None; // Miss
        }
        
        // Calculate base damage
        let mut damage = self.calculate_damage(attacker_damage, defender_defense);
        
        // Check for critical hit
        if self.check_critical(attacker_critical_chance) {
            damage *= 2.0; // Double damage on critical hit
        }
        
        Some(damage)
    }
}

/// Crafting recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftingRecipe {
    /// Recipe ID
    pub id: String,
    
    /// Recipe name
    pub name: String,
    
    /// Required ingredients
    pub ingredients: HashMap<String, u32>,
    
    /// Required skills
    pub required_skills: HashMap<String, u32>,
    
    /// Result item
    pub result: InventoryItem,
    
    /// Experience gained
    pub experience: u32,
}

/// Crafting system.
pub struct CraftingSystem {
    /// Available recipes
    recipes: HashMap<String, CraftingRecipe>,
}

impl CraftingSystem {
    /// Create a new crafting system.
    pub fn new() -> Self {
        Self {
            recipes: HashMap::new(),
        }
    }
    
    /// Add a recipe.
    pub fn add_recipe(&mut self, recipe: CraftingRecipe) {
        self.recipes.insert(recipe.id.clone(), recipe);
    }
    
    /// Get a recipe.
    pub fn get_recipe(&self, recipe_id: &str) -> Option<&CraftingRecipe> {
        self.recipes.get(recipe_id)
    }
    
    /// Check if a player can craft a recipe.
    pub fn can_craft(&self, player: &PlayerState, recipe_id: &str) -> Result<(), String> {
        let recipe = self.recipes.get(recipe_id).ok_or_else(|| "Recipe not found".to_string())?;
        
        // Check skills
        for (skill, level) in &recipe.required_skills {
            let player_level = player.skills.get(skill).unwrap_or(&0);
            if player_level < level {
                return Err(format!("Requires {} level {}", skill, level));
            }
        }
        
        // Check ingredients
        for (ingredient_id, quantity) in &recipe.ingredients {
            let player_quantity = player.inventory.items.get(ingredient_id)
                .map(|item| item.quantity)
                .unwrap_or(0);
            
            if player_quantity < *quantity {
                return Err(format!("Not enough {}", ingredient_id));
            }
        }
        
        Ok(())
    }
    
    /// Craft an item.
    pub fn craft(&self, player: &mut PlayerState, recipe_id: &str) -> Result<InventoryItem, String> {
        // Check if the player can craft the recipe
        self.can_craft(player, recipe_id)?;
        
        let recipe = self.recipes.get(recipe_id).unwrap();
        
        // Remove ingredients
        for (ingredient_id, quantity) in &recipe.ingredients {
            player.remove_item(ingredient_id, *quantity)?;
        }
        
        // Add experience
        player.stats.experience += recipe.experience;
        player.check_level_up();
        
        // Return the crafted item
        Ok(recipe.result.clone())
    }
}

/// Game world.
pub struct GameWorld {
    /// Terrain
    pub terrain: Terrain,
    
    /// NPCs
    pub npcs: HashMap<String, NPCState>,
    
    /// Items in the world
    pub items: HashMap<String, (InventoryItem, [f32; 3])>,
    
    /// Time of day (0.0 - 1.0)
    pub time_of_day: f32,
    
    /// Weather (0.0 = clear, 1.0 = stormy)
    pub weather: f32,
    
    /// Game time in seconds
    pub game_time: f32,
}

impl GameWorld {
    /// Create a new game world.
    pub fn new(terrain: Terrain) -> Self {
        Self {
            terrain,
            npcs: HashMap::new(),
            items: HashMap::new(),
            time_of_day: 0.5,
            weather: 0.0,
            game_time: 0.0,
        }
    }
    
    /// Update the game world.
    pub fn update(&mut self, delta_time: f32) {
        // Update game time
        self.game_time += delta_time;
        
        // Update time of day (full cycle every 24 minutes)
        self.time_of_day = (self.game_time / (24.0 * 60.0)) % 1.0;
        
        // Update weather (changes slowly)
        self.weather += (rand::random::<f32>() - 0.5) * 0.01;
        self.weather = self.weather.clamp(0.0, 1.0);
    }
    
    /// Get the biome at a position.
    pub fn get_biome_at(&self, x: usize, y: usize) -> TerrainType {
        if x < self.terrain.width && y < self.terrain.height {
            self.terrain.biome_map[y * self.terrain.width + x]
        } else {
            TerrainType::Ocean
        }
    }
    
    /// Get the height at a position.
    pub fn get_height_at(&self, x: usize, y: usize) -> f32 {
        self.terrain.heightmap.get(x, y)
    }
    
    /// Add an NPC to the world.
    pub fn add_npc(&mut self, npc: NPCState) {
        self.npcs.insert(npc.id.clone(), npc);
    }
    
    /// Remove an NPC from the world.
    pub fn remove_npc(&mut self, npc_id: &str) -> Option<NPCState> {
        self.npcs.remove(npc_id)
    }
    
    /// Add an item to the world.
    pub fn add_item(&mut self, item: InventoryItem, position: [f32; 3]) {
        self.items.insert(item.id.clone(), (item, position));
    }
    
    /// Remove an item from the world.
    pub fn remove_item(&mut self, item_id: &str) -> Option<(InventoryItem, [f32; 3])> {
        self.items.remove(item_id)
    }
    
    /// Get NPCs near a position.
    pub fn get_npcs_near(&self, position: [f32; 3], radius: f32) -> Vec<&NPCState> {
        self.npcs.values()
            .filter(|npc| {
                let dx = npc.position[0] - position[0];
                let dy = npc.position[1] - position[1];
                let dz = npc.position[2] - position[2];
                let distance_squared = dx * dx + dy * dy + dz * dz;
                distance_squared <= radius * radius
            })
            .collect()
    }
    
    /// Get items near a position.
    pub fn get_items_near(&self, position: [f32; 3], radius: f32) -> Vec<(&InventoryItem, [f32; 3])> {
        self.items.values()
            .filter(|(_, item_pos)| {
                let dx = item_pos[0] - position[0];
                let dy = item_pos[1] - position[1];
                let dz = item_pos[2] - position[2];
                let distance_squared = dx * dx + dy * dy + dz * dz;
                distance_squared <= radius * radius
            })
            .map(|(item, pos)| (item, *pos))
            .collect()
    }
}

/// Game state.
pub struct GameState {
    /// Player state
    pub player: PlayerState,
    
    /// Game world
    pub world: GameWorld,
    
    /// Combat system
    pub combat: CombatSystem,
    
    /// Crafting system
    pub crafting: CraftingSystem,
}

impl GameState {
    /// Create a new game state.
    pub fn new(player_name: &str, terrain: Terrain) -> Self {
        Self {
            player: PlayerState::new(player_name),
            world: GameWorld::new(terrain),
            combat: CombatSystem::new(),
            crafting: CraftingSystem::new(),
        }
    }
    
    /// Update the game state.
    pub fn update(&mut self, delta_time: f32) {
        // Update the world
        self.world.update(delta_time);
        
        // Regenerate energy
        self.player.restore_energy(delta_time * 2.0);
    }
    
    /// Process player movement.
    pub fn process_movement(&mut self, direction: [f32; 3], sprint: bool) {
        let delta_time = 1.0 / 60.0; // Assume 60 FPS
        
        let mut speed = self.player.get_movement_speed();
        
        if sprint {
            if self.player.use_energy(delta_time * 10.0) {
                speed *= 1.5; // 50% speed boost when sprinting
            }
        }
        
        self.player.move_player(direction, speed, delta_time);
    }
    
    /// Process player attack.
    pub fn process_attack(&mut self, target_id: &str) -> Result<Option<f32>, String> {
        // Get the target NPC
        let npc = self.world.npcs.get(target_id).ok_or_else(|| "Target not found".to_string())?;
        
        // Calculate attack parameters
        let player_damage = self.player.get_attack_damage();
        let player_accuracy = 0.8 + (self.player.stats.agility as f32 * 0.01);
        let player_critical_chance = 0.05 + (self.player.stats.agility as f32 * 0.005);
        
        let npc_defense = 1.0; // Simple NPC defense
        let npc_evasion = 0.2; // Simple NPC evasion
        
        // Process the attack
        let damage = self.combat.process_attack(
            player_damage,
            player_accuracy,
            player_critical_chance,
            npc_defense,
            npc_evasion,
        );
        
        // If the attack hit, apply damage to the NPC
        if let Some(damage) = damage {
            // In a real implementation, we would update the NPC's health
            // For now, just return the damage dealt
            return Ok(Some(damage));
        }
        
        Ok(None)
    }
    
    /// Process player interaction with the world.
    pub fn process_interaction(&mut self) -> Result<String, String> {
        // Get nearby items
        let nearby_items = self.world.get_items_near(self.player.position, 2.0);
        
        if !nearby_items.is_empty() {
            // Pick up the closest item
            let (item, _) = nearby_items[0];
            let item_clone = item.clone();
            
            // Add to inventory
            self.player.add_item(item_clone.clone())?;
            
            // Remove from world
            self.world.remove_item(&item.id);
            
            return Ok(format!("Picked up {}", item.name));
        }
        
        // Get nearby NPCs
        let nearby_npcs = self.world.get_npcs_near(self.player.position, 2.0);
        
        if !nearby_npcs.is_empty() {
            // Interact with the closest NPC
            let npc = nearby_npcs[0];
            
            return Ok(format!("Talking to {}", npc.name));
        }
        
        Err("Nothing to interact with".to_string())
    }
    
    /// Craft an item.
    pub fn craft_item(&mut self, recipe_id: &str) -> Result<String, String> {
        let item = self.crafting.craft(&mut self.player, recipe_id)?;
        self.player.add_item(item.clone())?;
        
        Ok(format!("Crafted {}", item.name))
    }
}
