/**
 * @file gpu_driver.h
 * @version 1.0.0
 * @brief Kernel-level GPU driver interfaces with power optimization for TALD UNIA platform
 */

#ifndef TALD_UNIA_KERNEL_GPU_DRIVER_H
#define TALD_UNIA_KERNEL_GPU_DRIVER_H

// External dependencies with versions
#include <vulkan/vulkan.h>  // v1.3
#include <vk_mem_alloc.h>   // v3.0.1

// Internal dependencies
#include "drivers/gpu/vulkan_driver.h"
#include "drivers/gpu/shader_compiler.h"

// Version and capability constants
#define GPU_DRIVER_VERSION "1.0.0"
#define MAX_GPU_MEMORY_REGIONS 16
#define GPU_MONITORING_INTERVAL_MS 100

// Power management enums and constants
enum gpu_power_state_t {
    GPU_POWER_PERFORMANCE,
    GPU_POWER_BALANCED,
    GPU_POWER_SAVE
};

#define GPU_POWER_STATE_DEFAULT GPU_POWER_BALANCED

// Forward declarations
struct gpu_init_config_t;
struct gpu_power_profile_t;
struct gpu_memory_manager_t;
struct gpu_power_manager_t;
struct gpu_performance_monitor_t;
struct gpu_frequency_curve_t;
struct gpu_driver_config_t;

/**
 * @brief Enhanced kernel-level GPU driver management class with power optimization
 */
class GPUDriver {
public:
    /**
     * @brief Constructs GPU driver with power management capabilities
     * @param config Driver configuration parameters
     * @param power_profile Power management profile
     * @throws std::runtime_error on initialization failure
     */
    GPUDriver(const gpu_driver_config_t* config, 
              const gpu_power_profile_t* power_profile);

    /**
     * @brief Destructor ensures proper cleanup of GPU resources
     */
    ~GPUDriver();

    /**
     * @brief Sets GPU power state with dynamic frequency control
     * @param state Target power state
     * @param freq_curve Frequency curve for dynamic scaling
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int set_power_state(gpu_power_state_t state, 
                       const gpu_frequency_curve_t* freq_curve);

    /**
     * @brief Allocates GPU memory with power-aware placement
     * @param size Memory size in bytes
     * @param flags Memory allocation flags
     * @param out_allocation Pointer to allocated memory
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int allocate_memory(size_t size, 
                       uint32_t flags, 
                       void** out_allocation);

    /**
     * @brief Retrieves current GPU power metrics
     * @param metrics Output power metrics structure
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int get_power_metrics(gpu_power_metrics_t* metrics);

private:
    // Core components
    VulkanDriver* vulkan_driver;
    ShaderCompiler* shader_compiler;
    gpu_memory_manager_t* memory_manager;
    gpu_power_manager_t* power_manager;
    gpu_performance_monitor_t* perf_monitor;
    gpu_power_profile_t current_profile;

    // Internal helper functions
    [[nodiscard]]
    int init_power_management(const gpu_power_profile_t* profile);
    
    [[nodiscard]]
    int setup_memory_manager();
    
    [[nodiscard]]
    int configure_performance_monitoring();
    
    [[nodiscard]]
    int initialize_power_states();

    // Prevent copying
    GPUDriver(const GPUDriver&) = delete;
    GPUDriver& operator=(const GPUDriver&) = delete;
};

/**
 * @brief Initializes the kernel GPU driver subsystem
 * @param config Driver configuration parameters
 * @param power_profile Power management profile
 * @return 0 on success, error code on failure
 */
[[nodiscard]]
int gpu_init(const gpu_init_config_t* config, 
            const gpu_power_profile_t* power_profile);

/**
 * @brief GPU driver configuration structure
 */
struct gpu_driver_config_t {
    uint32_t version;                    // Driver version
    uint32_t flags;                      // Configuration flags
    uint32_t max_memory_regions;         // Maximum memory regions
    uint32_t monitoring_interval_ms;     // Performance monitoring interval
    VkPhysicalDeviceFeatures features;   // Required Vulkan features
    gpu_power_profile_t power_profile;   // Initial power profile
};

/**
 * @brief GPU power profile configuration
 */
struct gpu_power_profile_t {
    gpu_power_state_t initial_state;     // Initial power state
    uint32_t performance_target_fps;     // Target FPS for performance state
    uint32_t balanced_target_fps;        // Target FPS for balanced state
    uint32_t power_save_target_fps;      // Target FPS for power save state
    float max_power_consumption_watts;    // Maximum power consumption
    float thermal_limit_celsius;         // Thermal limit
    gpu_frequency_curve_t freq_curves[3]; // Frequency curves for each state
};

/**
 * @brief GPU frequency curve configuration
 */
struct gpu_frequency_curve_t {
    uint32_t min_freq_mhz;              // Minimum frequency
    uint32_t max_freq_mhz;              // Maximum frequency
    uint32_t step_size_mhz;             // Frequency step size
    float voltage_curve[32];            // Voltage levels for frequencies
    float thermal_limits[32];           // Thermal limits per frequency
};

/**
 * @brief GPU power metrics structure
 */
struct gpu_power_metrics_t {
    float current_power_watts;           // Current power consumption
    float current_temp_celsius;          // Current temperature
    uint32_t current_freq_mhz;          // Current frequency
    gpu_power_state_t current_state;     // Current power state
    uint32_t current_fps;               // Current FPS
    float memory_bandwidth_gbps;         // Current memory bandwidth
    uint32_t gpu_utilization_percent;    // GPU utilization
    uint32_t memory_utilization_percent; // Memory utilization
};

#endif // TALD_UNIA_KERNEL_GPU_DRIVER_H