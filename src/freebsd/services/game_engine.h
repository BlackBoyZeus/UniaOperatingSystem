/**
 * @file game_engine.h
 * @version 1.0.0
 * @brief Core game engine service coordinating physics, graphics, and mixed reality for TALD UNIA platform
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_GAME_ENGINE_H
#define TALD_GAME_ENGINE_H

// External dependencies with versions
#include <cuda.h>                    // CUDA 12.0
#include <vulkan/vulkan.h>          // v1.3

// Internal dependencies
#include "game/engine/physics_engine.h"
#include "game/engine/vulkan_renderer.h"
#include "game/engine/scene_manager.h"

#include <memory>
#include <atomic>
#include <mutex>
#include <string>

// Version and configuration constants
#define GAME_ENGINE_VERSION "1.0.0"
#define TARGET_FRAME_RATE 60
#define PHYSICS_UPDATE_RATE 120
#define LIDAR_UPDATE_RATE 30
#define MAX_POWER_STATES 4
#define ERROR_BUFFER_SIZE 1024

namespace tald {
namespace engine {

/**
 * @brief Power state enumeration for dynamic performance scaling
 */
enum class PowerState {
    LOW_POWER,
    BALANCED,
    HIGH_PERFORMANCE,
    TURBO
};

/**
 * @brief Performance metrics for game engine monitoring
 */
struct PerformanceMetrics {
    float frame_time_ms{0.0f};
    float physics_time_ms{0.0f};
    float render_time_ms{0.0f};
    float lidar_time_ms{0.0f};
    uint32_t fps{0};
    PowerState current_power_state{PowerState::BALANCED};
    float gpu_utilization{0.0f};
    float memory_usage_mb{0.0f};
};

/**
 * @brief Game engine configuration parameters
 */
struct game_engine_config_t {
    VkExtent2D resolution{1920, 1080};
    bool enable_vsync{true};
    bool enable_dynamic_resolution{true};
    PowerState initial_power_state{PowerState::BALANCED};
    uint32_t msaa_samples{VK_SAMPLE_COUNT_4_BIT};
    float target_frame_time{1.0f / TARGET_FRAME_RATE};
    size_t gpu_memory_limit_mb{2048};
};

/**
 * @brief Error callback function type
 */
using error_callback_t = void(*)(const char* message, void* user_data);

/**
 * @brief Result structure for game engine operations
 */
struct GameEngineResult {
    bool success{false};
    std::string error_message;
    void* engine_instance{nullptr};
};

/**
 * @brief Core game engine class coordinating all gaming subsystems
 */
class GameEngine final {
public:
    /**
     * @brief Creates a new game engine instance
     * @param config Engine configuration parameters
     * @param error_handler Error callback function
     * @throws std::runtime_error if initialization fails
     */
    GameEngine(const game_engine_config_t* config, error_callback_t error_handler);

    // Prevent copying
    GameEngine(const GameEngine&) = delete;
    GameEngine& operator=(const GameEngine&) = delete;

    /**
     * @brief Updates all game engine subsystems
     * @param delta_time Time since last update
     * @return Success status of update
     */
    [[nodiscard]]
    bool update(float delta_time);

    /**
     * @brief Processes new LiDAR scan data
     * @param points Array of 3D points
     * @param count Number of points
     * @return Success status of processing
     */
    [[nodiscard]]
    bool process_lidar_scan(const float3* points, size_t count);

    /**
     * @brief Sets current power mode
     * @param state New power state
     */
    void set_power_mode(PowerState state);

    /**
     * @brief Gets current performance metrics
     * @return const reference to performance metrics
     */
    [[nodiscard]]
    const PerformanceMetrics& get_metrics() const { return metrics; }

private:
    // Core components
    std::unique_ptr<physics::PhysicsWorld> physics;
    std::unique_ptr<render::VulkanRenderer> renderer;
    std::unique_ptr<scene::SceneManager> scene_manager;
    std::unique_ptr<cuda::CudaWrapper> cuda_wrapper;

    // State management
    std::atomic<PowerState> power_state;
    PerformanceMetrics metrics;
    game_engine_config_t config;
    error_callback_t error_handler;

    // Synchronization
    std::mutex engine_mutex;
    std::atomic<bool> is_updating{false};

    // Internal methods
    bool initialize_subsystems();
    void update_performance_metrics(float delta_time);
    void adjust_power_state();
    void handle_error(const char* message);
    void cleanup_resources();
};

/**
 * @brief Initializes the game engine service
 * @param config Engine configuration parameters
 * @param error_handler Error callback function
 * @return GameEngineResult containing engine instance or error details
 */
[[nodiscard]]
GameEngineResult init_game_engine(const game_engine_config_t* config,
                                error_callback_t error_handler);

} // namespace engine
} // namespace tald

#endif // TALD_GAME_ENGINE_H