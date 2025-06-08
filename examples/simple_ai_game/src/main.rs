use std::time::{Duration, Instant};
use std::thread;

// Import UNIA modules
use unia_ai::behavior_tree::{BehaviorTree, BehaviorNode, Status};
use unia_ai::world_model::WorldModel;
use unia_graphics::renderer::Renderer;
use unia_core::game_loop::GameLoop;
use unia_core::entity::{Entity, EntityManager};
use unia_core::time::TimeSystem;

// Game-specific imports
mod npc;
mod player;
mod inventory;
mod world;

use npc::NPC;
use player::Player;
use inventory::Item;
use world::{World, TimeOfDay};

fn main() {
    println!("Starting Simple AI Game Example");
    
    // Initialize UNIA systems
    let mut renderer = Renderer::new("Simple AI Game", 1280, 720);
    let mut entity_manager = EntityManager::new();
    let mut world_model = WorldModel::new();
    let mut time_system = TimeSystem::new();
    
    // Create game world
    let mut game_world = World::new();
    game_world.generate_terrain(512, 512);
    
    // Create player
    let player_entity = entity_manager.create_entity();
    let mut player = Player::new(player_entity);
    player.position = (256.0, 256.0);
    player.health = 100;
    player.stamina = 100;
    
    // Add some items to player inventory
    player.inventory.add_item(Item::new("Sword", 10));
    player.inventory.add_item(Item::new("Health Potion", 5));
    
    // Create NPCs with behavior trees
    let mut npcs = Vec::new();
    
    // Create a villager NPC
    let villager_entity = entity_manager.create_entity();
    let mut villager = NPC::new(villager_entity, "Villager");
    villager.position = (200.0, 200.0);
    
    // Create behavior tree for villager
    let villager_bt = create_villager_behavior_tree();
    villager.set_behavior_tree(villager_bt);
    npcs.push(villager);
    
    // Create a guard NPC
    let guard_entity = entity_manager.create_entity();
    let mut guard = NPC::new(guard_entity, "Guard");
    guard.position = (300.0, 300.0);
    
    // Create behavior tree for guard
    let guard_bt = create_guard_behavior_tree();
    guard.set_behavior_tree(guard_bt);
    npcs.push(guard);
    
    // Create a merchant NPC
    let merchant_entity = entity_manager.create_entity();
    let mut merchant = NPC::new(merchant_entity, "Merchant");
    merchant.position = (350.0, 200.0);
    
    // Create behavior tree for merchant
    let merchant_bt = create_merchant_behavior_tree();
    merchant.set_behavior_tree(merchant_bt);
    npcs.push(merchant);
    
    // Main game loop
    let mut game_loop = GameLoop::new(60); // 60 FPS target
    
    game_loop.run(|delta_time| {
        // Update time system
        time_system.update(delta_time);
        
        // Update world time (day/night cycle)
        game_world.update_time(delta_time);
        
        // Update world model with current state
        update_world_model(&mut world_model, &game_world, &player, &npcs);
        
        // Update player
        player.update(delta_time);
        
        // Update NPCs
        for npc in &mut npcs {
            npc.update(delta_time, &world_model);
        }
        
        // Render game state
        render_game(&mut renderer, &game_world, &player, &npcs);
        
        // Check for game exit condition
        if player.health <= 0 {
            return false; // Exit game loop
        }
        
        true // Continue game loop
    });
    
    println!("Simple AI Game Example completed");
}

// Create behavior tree for villager NPC
fn create_villager_behavior_tree() -> BehaviorTree {
    let mut bt = BehaviorTree::new();
    
    // Create a selector node (runs child nodes until one succeeds)
    let selector = bt.add_selector("villager_main");
    
    // Add sequence for when it's daytime
    let day_sequence = bt.add_sequence("day_activities");
    day_sequence.add_condition("is_daytime", |world: &WorldModel| -> Status {
        if world.get_value("time_of_day") == "day" {
            Status::Success
        } else {
            Status::Failure
        }
    });
    
    // Daytime activities
    day_sequence.add_action("wander_village", |_| {
        println!("Villager is wandering around the village");
        Status::Success
    });
    
    day_sequence.add_action("talk_to_others", |_| {
        println!("Villager is talking to other villagers");
        Status::Success
    });
    
    day_sequence.add_action("work_field", |_| {
        println!("Villager is working in the field");
        Status::Success
    });
    
    // Add sequence for when it's nighttime
    let night_sequence = bt.add_sequence("night_activities");
    night_sequence.add_condition("is_nighttime", |world: &WorldModel| -> Status {
        if world.get_value("time_of_day") == "night" {
            Status::Success
        } else {
            Status::Failure
        }
    });
    
    // Nighttime activities
    night_sequence.add_action("go_home", |_| {
        println!("Villager is going home");
        Status::Success
    });
    
    night_sequence.add_action("sleep", |_| {
        println!("Villager is sleeping");
        Status::Success
    });
    
    // Add sequences to selector
    selector.add_child(day_sequence);
    selector.add_child(night_sequence);
    
    bt
}

// Create behavior tree for guard NPC
fn create_guard_behavior_tree() -> BehaviorTree {
    let mut bt = BehaviorTree::new();
    
    // Create a selector node
    let selector = bt.add_selector("guard_main");
    
    // Add sequence for when player is nearby
    let player_nearby_sequence = bt.add_sequence("player_nearby");
    player_nearby_sequence.add_condition("is_player_nearby", |world: &WorldModel| -> Status {
        if world.get_value("player_distance") < "50.0" {
            Status::Success
        } else {
            Status::Failure
        }
    });
    
    // Player nearby activities
    player_nearby_sequence.add_action("greet_player", |_| {
        println!("Guard greets the player");
        Status::Success
    });
    
    player_nearby_sequence.add_action("watch_player", |_| {
        println!("Guard is watching the player carefully");
        Status::Success
    });
    
    // Add sequence for patrolling
    let patrol_sequence = bt.add_sequence("patrol");
    
    // Patrol activities
    patrol_sequence.add_action("patrol_area", |_| {
        println!("Guard is patrolling the area");
        Status::Success
    });
    
    patrol_sequence.add_action("check_surroundings", |_| {
        println!("Guard is checking surroundings");
        Status::Success
    });
    
    // Add sequences to selector
    selector.add_child(player_nearby_sequence);
    selector.add_child(patrol_sequence);
    
    bt
}

// Create behavior tree for merchant NPC
fn create_merchant_behavior_tree() -> BehaviorTree {
    let mut bt = BehaviorTree::new();
    
    // Create a selector node
    let selector = bt.add_selector("merchant_main");
    
    // Add sequence for when player is nearby
    let player_nearby_sequence = bt.add_sequence("player_nearby");
    player_nearby_sequence.add_condition("is_player_nearby", |world: &WorldModel| -> Status {
        if world.get_value("player_distance") < "30.0" {
            Status::Success
        } else {
            Status::Failure
        }
    });
    
    // Player nearby activities
    player_nearby_sequence.add_action("offer_goods", |_| {
        println!("Merchant offers goods to the player");
        Status::Success
    });
    
    player_nearby_sequence.add_action("haggle", |_| {
        println!("Merchant haggles with the player");
        Status::Success
    });
    
    // Add sequence for normal activities
    let normal_sequence = bt.add_sequence("normal_activities");
    
    // Check time of day
    normal_sequence.add_condition("is_daytime", |world: &WorldModel| -> Status {
        if world.get_value("time_of_day") == "day" {
            Status::Success
        } else {
            Status::Failure
        }
    });
    
    // Normal activities
    normal_sequence.add_action("manage_inventory", |_| {
        println!("Merchant is managing inventory");
        Status::Success
    });
    
    normal_sequence.add_action("call_for_customers", |_| {
        println!("Merchant is calling for customers");
        Status::Success
    });
    
    // Add sequence for nighttime
    let night_sequence = bt.add_sequence("night_activities");
    night_sequence.add_condition("is_nighttime", |world: &WorldModel| -> Status {
        if world.get_value("time_of_day") == "night" {
            Status::Success
        } else {
            Status::Failure
        }
    });
    
    // Nighttime activities
    night_sequence.add_action("close_shop", |_| {
        println!("Merchant is closing the shop");
        Status::Success
    });
    
    night_sequence.add_action("count_earnings", |_| {
        println!("Merchant is counting the day's earnings");
        Status::Success
    });
    
    // Add sequences to selector
    selector.add_child(player_nearby_sequence);
    selector.add_child(normal_sequence);
    selector.add_child(night_sequence);
    
    bt
}

// Update world model with current game state
fn update_world_model(world_model: &mut WorldModel, game_world: &World, player: &Player, npcs: &[NPC]) {
    // Update time of day
    match game_world.time_of_day {
        TimeOfDay::Morning => world_model.set_value("time_of_day", "morning"),
        TimeOfDay::Day => world_model.set_value("time_of_day", "day"),
        TimeOfDay::Evening => world_model.set_value("time_of_day", "evening"),
        TimeOfDay::Night => world_model.set_value("time_of_day", "night"),
    }
    
    // Update player information
    world_model.set_value("player_health", &player.health.to_string());
    world_model.set_value("player_position_x", &player.position.0.to_string());
    world_model.set_value("player_position_y", &player.position.1.to_string());
    
    // Update NPC information
    for (i, npc) in npcs.iter().enumerate() {
        // Calculate distance to player
        let dx = npc.position.0 - player.position.0;
        let dy = npc.position.1 - player.position.1;
        let distance = (dx * dx + dy * dy).sqrt();
        
        world_model.set_value(&format!("npc_{}_distance", i), &distance.to_string());
        
        // If this is the closest NPC to the player, update player_distance
        if i == 0 || distance < world_model.get_value("player_distance").parse::<f32>().unwrap_or(f32::MAX) {
            world_model.set_value("player_distance", &distance.to_string());
        }
    }
}

// Render the game state
fn render_game(renderer: &mut Renderer, game_world: &World, player: &Player, npcs: &[NPC]) {
    renderer.begin_frame();
    
    // Render world
    renderer.render_terrain(&game_world.terrain);
    
    // Render player
    renderer.render_entity(player.entity, player.position.0, player.position.1, "player");
    
    // Render NPCs
    for npc in npcs {
        renderer.render_entity(npc.entity, npc.position.0, npc.position.1, &npc.npc_type);
    }
    
    // Render UI
    renderer.render_text(10.0, 10.0, &format!("Health: {}", player.health));
    renderer.render_text(10.0, 30.0, &format!("Stamina: {}", player.stamina));
    renderer.render_text(10.0, 50.0, &format!("Time: {:?}", game_world.time_of_day));
    
    renderer.end_frame();
}
