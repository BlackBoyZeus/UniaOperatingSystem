/**
 * @file vulkan_driver.h
 * @version 1.0.0
 * @brief Power-aware Vulkan driver implementation with LiDAR visualization support for TALD UNIA platform
 */

#ifndef TALD_UNIA_VULKAN_DRIVER_H
#define TALD_UNIA_VULKAN_DRIVER_H

// External dependencies with versions
#include <vulkan/vulkan.h>  // v1.3
#include <vk_mem_alloc.h>   // v3.0.1
#include "power_management/power_manager.h"  // v2.1.0

// Internal dependencies
#include "shader_compiler.h"

#include <memory>
#include <array>
#include <vector>

// Version and capability constants
#define VULKAN_DRIVER_VERSION "1.0.0"
#define MAX_FRAMES_IN_FLIGHT 2
#define MAX_DESCRIPTOR_SETS 1024
#define MAX_VERTEX_BUFFERS 8
#define MAX_COMMAND_BUFFERS 32
#define LIDAR_POINT_CLOUD_BUFFER_SIZE 1048576  // 1MB for point cloud data
#define FRAME_PACING_INTERVAL_60FPS 16666667   // 16.67ms in nanoseconds

// Power state enumeration
enum class PowerState {
    LOW_POWER,
    BALANCED,
    HIGH_PERFORMANCE
};

// Forward declarations
struct vulkan_driver_config_t;
struct power_profile_t;
struct frame_pacing_controller_t;
struct lidar_visualization_context_t;
struct lidar_buffer_config_t;

/**
 * @brief Enhanced Vulkan driver management class with power awareness and LiDAR support
 */
class VulkanDriver {
public:
    /**
     * @brief Initializes Vulkan driver with power management and LiDAR support
     * @param config Driver configuration parameters
     * @param power_profile Power management profile
     * @throws std::runtime_error on initialization failure
     */
    VulkanDriver(const vulkan_driver_config_t* config,
                 power_profile_t* power_profile);

    /**
     * @brief Destructor ensures proper cleanup of resources
     */
    ~VulkanDriver();

    /**
     * @brief Creates an optimized buffer for LiDAR point cloud data
     * @param config Buffer configuration parameters
     * @param buffer Output buffer handle
     * @param allocation Memory allocation handle
     * @return VkResult indicating buffer creation success
     */
    [[nodiscard]]
    VkResult create_lidar_buffer(const lidar_buffer_config_t* config,
                                VkBuffer* buffer,
                                VmaAllocation* allocation);

    /**
     * @brief Updates GPU power state based on system conditions
     * @param new_state Desired power state
     * @return VkResult indicating power state update success
     */
    [[nodiscard]]
    VkResult update_power_state(PowerState new_state);

    /**
     * @brief Creates a power-optimized graphics pipeline
     * @param create_info Pipeline creation parameters
     * @param pipeline Output pipeline handle
     * @return VkResult indicating pipeline creation success
     */
    [[nodiscard]]
    VkResult create_graphics_pipeline(const VkGraphicsPipelineCreateInfo* create_info,
                                    VkPipeline* pipeline);

    /**
     * @brief Updates frame pacing parameters for power efficiency
     * @param target_fps Desired frames per second
     * @return VkResult indicating update success
     */
    [[nodiscard]]
    VkResult update_frame_pacing(uint32_t target_fps);

private:
    // Core Vulkan handles
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VmaAllocator allocator;

    // Support systems
    std::unique_ptr<ShaderCompiler> shader_compiler;
    std::unique_ptr<PowerManager> power_manager;
    
    // Command management
    std::array<VkCommandPool, MAX_FRAMES_IN_FLIGHT> command_pools;
    std::vector<VkCommandBuffer> command_buffers;

    // Synchronization and pacing
    frame_pacing_controller_t frame_pacer;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> image_available_semaphores;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> render_finished_semaphores;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> in_flight_fences;

    // LiDAR visualization
    lidar_visualization_context_t lidar_context;
    VkBuffer lidar_staging_buffer;
    VmaAllocation lidar_staging_allocation;

    // Internal helper functions
    [[nodiscard]]
    VkResult create_instance(const vulkan_driver_config_t* config);
    
    [[nodiscard]]
    VkResult select_physical_device();
    
    [[nodiscard]]
    VkResult create_logical_device(const vulkan_driver_config_t* config);
    
    [[nodiscard]]
    VkResult init_memory_allocator();
    
    [[nodiscard]]
    VkResult setup_command_pools();
    
    [[nodiscard]]
    VkResult create_synchronization_objects();
    
    [[nodiscard]]
    VkResult init_lidar_context();

    // Prevent copying
    VulkanDriver(const VulkanDriver&) = delete;
    VulkanDriver& operator=(const VulkanDriver&) = delete;
};

/**
 * @brief Initializes the Vulkan driver subsystem
 * @param config Driver configuration parameters
 * @param power_profile Power management profile
 * @return VkResult indicating initialization success
 */
[[nodiscard]]
VkResult init_vulkan_driver(const vulkan_driver_config_t* config,
                           power_profile_t* power_profile);

/**
 * @brief Cleans up Vulkan driver resources
 */
void cleanup_vulkan_driver();

#endif // TALD_UNIA_VULKAN_DRIVER_H