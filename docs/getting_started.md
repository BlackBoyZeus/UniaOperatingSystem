# Getting Started with UNIA Operating System

This guide will help you set up a development environment for UNIA OS and run your first AI-powered game.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Operating System**: Linux (Ubuntu 22.04 LTS recommended), macOS 12+, or Windows 11 with WSL2
- **Development Tools**:
  - GCC 12+ or Clang 15+
  - CMake 3.24+
  - Ninja build system
  - Git 2.34+
- **GPU Development**:
  - CUDA 12.0+ (for NVIDIA GPUs)
  - ROCm 5.4+ (for AMD GPUs)
  - Vulkan SDK 1.3+
- **Additional Dependencies**:
  - Python 3.10+
  - Rust 1.68+
  - Node.js 18+ (for UI development)

## Step 1: Clone the Repository

```bash
git clone https://github.com/BlackBoyZeus/UniaOperatingSystem.git
cd UniaOperatingSystem
```

## Step 2: Initialize Submodules

UNIA uses several submodules for external dependencies:

```bash
git submodule update --init --recursive
```

## Step 3: Configure Build Environment

Create a build directory and configure the project:

```bash
mkdir build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=ON -DENABLE_VULKAN=ON ..
```

Available configuration options:
- `ENABLE_CUDA`: Enable NVIDIA CUDA support (ON/OFF)
- `ENABLE_ROCM`: Enable AMD ROCm support (ON/OFF)
- `ENABLE_VULKAN`: Enable Vulkan support (ON/OFF)
- `ENABLE_LIDAR`: Enable LiDAR support (ON/OFF)
- `ENABLE_TESTS`: Build tests (ON/OFF)
- `ENABLE_EXAMPLES`: Build example games (ON/OFF)

## Step 4: Build the System

Build the UNIA operating system:

```bash
ninja
```

This will build the core OS, AI framework, graphics engine, and other components.

## Step 5: Run Tests

Verify your build by running the test suite:

```bash
ninja test
```

## Step 6: Run the Simple AI Game Example

UNIA comes with example games to demonstrate its capabilities:

```bash
cd examples/simple_ai_game
./run_game.sh
```

This will launch a simple game that demonstrates:
- NPC behavior using behavior trees
- Day/night cycle affecting NPC behavior
- Basic player stats and inventory system

## Step 7: Explore the Advanced AI Game

For a more complex example:

```bash
cd examples/advanced_ai_game
./run_game.sh
```

This demonstrates:
- Advanced NPC behaviors with memory and relationships
- Procedurally generated world with AI-driven features
- Player modeling with machine learning adaptation
- Multiplayer support with CRDT-based synchronization

## Development Workflow

### Creating a New Game Project

1. Create a new directory for your project:
   ```bash
   mkdir my_unia_game
   cd my_unia_game
   ```

2. Initialize the project structure:
   ```bash
   unia-cli init --template basic_game
   ```

3. Configure your game settings in `game_config.yaml`

4. Build your game:
   ```bash
   unia-cli build
   ```

5. Run your game:
   ```bash
   unia-cli run
   ```

### Working with AI Components

UNIA provides several AI components you can use in your games:

1. **Behavior Trees**: Define NPC behaviors
   ```cpp
   #include <unia/ai/behavior_tree.h>
   
   // Create a behavior tree
   auto tree = unia::ai::BehaviorTree::create();
   
   // Add nodes
   tree->addSequence("main_sequence")
       ->addAction("move_to_target", [](auto& context) {
           // Implementation
           return unia::ai::Status::Success;
       })
       ->addAction("perform_action", [](auto& context) {
           // Implementation
           return unia::ai::Status::Success;
       });
   
   // Run the behavior tree
   tree->execute(context);
   ```

2. **Procedural Generation**: Create game worlds
   ```cpp
   #include <unia/ai/procedural/terrain_generator.h>
   
   // Create terrain generator
   auto generator = unia::ai::procedural::TerrainGenerator::create();
   
   // Configure generator
   generator->setSize(1024, 1024);
   generator->setSeed(12345);
   generator->addFeature("mountains", 0.2f);
   generator->addFeature("forests", 0.3f);
   generator->addFeature("rivers", 0.1f);
   
   // Generate terrain
   auto terrain = generator->generate();
   ```

3. **Player Modeling**: Adapt to player behavior
   ```cpp
   #include <unia/ai/player_model.h>
   
   // Create player model
   auto model = unia::ai::PlayerModel::create();
   
   // Add player action
   model->addObservation("player_killed_enemy", {
       {"enemy_type", "boss"},
       {"weapon_used", "sword"},
       {"time_taken", 45.2f}
   });
   
   // Get player profile
   auto profile = model->generateProfile();
   
   // Adapt game based on profile
   if (profile.getTraitValue("aggression") > 0.7f) {
       // Spawn more enemies
   }
   ```

### Working with Graphics

UNIA provides a Vulkan-based graphics engine:

```cpp
#include <unia/graphics/renderer.h>

// Initialize renderer
auto renderer = unia::graphics::Renderer::create();

// Load model
auto model = renderer->loadModel("assets/character.glb");

// Create material
auto material = renderer->createMaterial("assets/shaders/pbr.vert", "assets/shaders/pbr.frag");

// Render frame
renderer->beginFrame();
renderer->drawModel(model, transform, material);
renderer->endFrame();
```

### Working with Networking

UNIA provides a mesh networking system:

```cpp
#include <unia/networking/mesh_network.h>

// Create mesh network
auto network = unia::networking::MeshNetwork::create();

// Connect to peer
network->connectToPeer("peer-id-123", signaling_data);

// Send message
network->sendMessage("peer-id-123", "game_state", game_state_data);

// Register message handler
network->registerHandler("game_state", [](const auto& message) {
    // Handle game state update
});
```

## Next Steps

- Explore the [API Reference](api/README.md) for detailed documentation
- Check out the [Architecture Overview](architecture/README.md) to understand the system
- Join our [Discord Server](https://discord.com/channels/1279930185856188446/1381339248166441042) to connect with other developers
- Contribute to the project by following our [Contribution Guidelines](../CONTRIBUTING.md)

## Troubleshooting

### Common Issues

1. **Build Fails with CUDA Errors**
   - Ensure you have the correct CUDA version installed
   - Check that your GPU is supported by the installed CUDA version
   - Try building with `-DENABLE_CUDA=OFF` if you don't need CUDA support

2. **Vulkan Initialization Fails**
   - Ensure your GPU supports Vulkan 1.3
   - Update your GPU drivers to the latest version
   - Check that the Vulkan SDK is properly installed

3. **LiDAR Support Not Working**
   - Ensure your LiDAR device is connected and recognized by the system
   - Check that you have the necessary drivers installed
   - Try running with `-DENABLE_LIDAR=OFF` if you don't have a LiDAR device

4. **Performance Issues**
   - Check that hardware acceleration is properly configured
   - Ensure your system meets the minimum requirements
   - Use the performance profiling tools to identify bottlenecks

For more help, please open an issue on GitHub or ask in our Discord server.
