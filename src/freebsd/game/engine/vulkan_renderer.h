/**
 * @file vulkan_renderer.h
 * @version 1.0.0
 * @brief High-performance Vulkan-based rendering system for TALD UNIA platform
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_VULKAN_RENDERER_H
#define TALD_VULKAN_RENDERER_H

// External dependencies with versions
#include <vulkan/vulkan.h>  // v1.3
#include <vk_mem_alloc.h>   // v3.0.1
#include <glm/glm.hpp>      // v0.9.9.8

// Internal dependencies
#include "vulkan_driver.h"
#include "scene_manager.h"

#include <memory>
#include <vector>
#include <array>

// Version and capability constants
#define VULKAN_RENDERER_VERSION "1.0.0"
#define MAX_FRAMES_IN_FLIGHT 2
#define MAX_MESHES 1024
#define MAX_POINT_CLOUDS 32
#define TARGET_FRAME_RATE 60
#define MIN_POWER_MODE_FPS 30
#define MAX_LOD_LEVELS 4
#define POINT_CLOUD_BATCH_SIZE 65536

namespace tald {
namespace render {

/**
 * @brief Power mode enumeration for dynamic performance scaling
 */
enum class PowerMode {
    LOW_POWER,
    BALANCED,
    HIGH_PERFORMANCE
};

/**
 * @brief Dynamic resolution scaling configuration
 */
struct DynamicResolution {
    float current_scale{1.0f};
    float min_scale{0.5f};
    float max_scale{1.0f};
    float target_frame_time{1.0f / TARGET_FRAME_RATE};
};

/**
 * @brief Configuration for Vulkan renderer initialization
 */
struct vulkan_renderer_config_t {
    VkExtent2D resolution;
    bool enable_vsync{true};
    bool enable_dynamic_resolution{true};
    PowerMode initial_power_mode{PowerMode::BALANCED};
    uint32_t msaa_samples{VK_SAMPLE_COUNT_1_BIT};
    float target_frame_time{1.0f / TARGET_FRAME_RATE};
};

/**
 * @brief Core Vulkan renderer class with enhanced features
 */
class VulkanRenderer {
public:
    /**
     * @brief Creates a new Vulkan renderer instance
     * @param config Renderer configuration parameters
     * @param scene_manager Scene management instance
     * @throws std::runtime_error if initialization fails
     */
    VulkanRenderer(const vulkan_renderer_config_t* config,
                  scene::SceneManager* scene_manager);

    // Prevent copying
    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    /**
     * @brief Renders a complete frame with all components
     * @param delta_time Time since last frame
     * @return VkResult indicating render success
     */
    [[nodiscard]]
    VkResult render_frame(float delta_time);

    /**
     * @brief Updates point cloud visualization data
     * @param points Array of 3D points
     * @param count Number of points
     * @param desired_lod Desired level of detail
     * @return Success status of update
     */
    [[nodiscard]]
    bool update_point_cloud(const glm::vec3* points,
                          size_t count,
                          uint32_t desired_lod = 0);

    /**
     * @brief Sets current power mode
     * @param mode New power mode
     */
    void set_power_mode(PowerMode mode);

private:
    // Core components
    VulkanDriver* driver{nullptr};
    scene::SceneManager* scene_manager{nullptr};

    // Render passes
    VkRenderPass main_render_pass{VK_NULL_HANDLE};
    VkRenderPass overlay_render_pass{VK_NULL_HANDLE};

    // Graphics pipelines
    VkPipeline mesh_pipeline{VK_NULL_HANDLE};
    VkPipeline point_cloud_pipeline{VK_NULL_HANDLE};
    VkPipeline occlusion_pipeline{VK_NULL_HANDLE};

    // Frame resources
    std::vector<VkFramebuffer> framebuffers;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> command_buffers;
    uint32_t current_frame{0};

    // Dynamic state
    PowerMode current_power_mode{PowerMode::BALANCED};
    DynamicResolution resolution_scaler;
    float frame_times[60]{};
    uint32_t frame_time_index{0};

    // Internal methods
    bool initialize_render_passes();
    bool create_graphics_pipelines();
    bool setup_frame_resources();
    void update_dynamic_resolution(float frame_time);
    void cleanup_resources();

    // Pipeline creation helpers
    [[nodiscard]]
    VkPipeline create_mesh_pipeline();
    [[nodiscard]]
    VkPipeline create_point_cloud_pipeline();
    [[nodiscard]]
    VkPipeline create_occlusion_pipeline();

    // Render pass recording
    void record_main_pass(VkCommandBuffer cmd_buffer);
    void record_overlay_pass(VkCommandBuffer cmd_buffer);
    void record_point_cloud_commands(VkCommandBuffer cmd_buffer);
};

/**
 * @brief Initializes the Vulkan renderer subsystem
 * @param config Renderer configuration
 * @param scene_manager Scene management instance
 * @return Initialized renderer or nullptr if failed
 */
[[nodiscard]]
VulkanRenderer* init_vulkan_renderer(const vulkan_renderer_config_t* config,
                                   scene::SceneManager* scene_manager);

/**
 * @brief Destroys Vulkan renderer instance
 * @param renderer Renderer to destroy
 */
void destroy_vulkan_renderer(VulkanRenderer* renderer);

} // namespace render
} // namespace tald

#endif // TALD_VULKAN_RENDERER_H