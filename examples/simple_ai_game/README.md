# Simple AI Game Example

This example demonstrates the core AI features of the UNIA Operating System in a simple game context.

## Features Demonstrated

- **NPC Behavior Trees**: NPCs with different behavior patterns based on time of day and player proximity
- **Day/Night Cycle**: Time system that affects NPC behavior
- **Player Stats and Inventory**: Basic player character with health, stamina, and inventory
- **Procedural Terrain Generation**: Simple terrain generation using noise functions

## NPCs in the Game

### Villager
- Wanders around the village during the day
- Goes home and sleeps at night
- Has simple conversations with other NPCs

### Guard
- Patrols a fixed route
- Approaches and greets the player when nearby
- Maintains vigilance at all times

### Merchant
- Manages inventory and calls for customers during the day
- Offers goods to the player when approached
- Closes shop and counts earnings at night

## Running the Example

```bash
# From the project root
cd examples/simple_ai_game
cargo run
```

## Code Structure

- `main.rs`: Main game loop and initialization
- `npc.rs`: NPC implementation with behavior trees
- `player.rs`: Player character implementation
- `inventory.rs`: Inventory system for items
- `world.rs`: Game world with terrain generation and time system

## Extending the Example

This example is designed to be easily extended. Here are some ideas:

1. **Add More NPCs**: Create new NPC types with different behavior trees
2. **Implement Combat**: Add combat mechanics between player and enemies
3. **Add Quests**: Implement a simple quest system with objectives
4. **Enhance Terrain**: Add more terrain features and biomes
5. **Add Buildings**: Create buildings that NPCs can enter and exit

## Learning from the Example

This example demonstrates several key concepts in UNIA:

1. **Behavior Trees**: How to create and use behavior trees for NPC AI
2. **World Model**: How to maintain and update a world model for AI decision making
3. **Entity Management**: How to manage game entities and their properties
4. **Game Loop**: How to structure a game loop with time-based updates
5. **Procedural Generation**: How to generate terrain procedurally

## Performance Considerations

The example is designed to be lightweight and run on minimal hardware. However, when scaling up:

- Consider optimizing behavior tree execution frequency for large numbers of NPCs
- Use spatial partitioning for efficient entity queries
- Implement level-of-detail systems for terrain rendering
- Consider multithreading for AI processing when dealing with many NPCs
