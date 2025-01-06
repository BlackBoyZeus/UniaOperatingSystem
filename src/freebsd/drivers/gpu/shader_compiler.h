/**
 * @file shader_compiler.h
 * @version 1.0.0
 * @brief Power-aware shader compilation and optimization interfaces for TALD UNIA platform
 * 
 * Provides SPIR-V compilation, shader optimization, runtime shader management capabilities,
 * and power-aware optimization features for the Vulkan-based graphics system.
 */

#ifndef TALD_UNIA_SHADER_COMPILER_H
#define TALD_UNIA_SHADER_COMPILER_H

// External dependencies with versions
#include <vulkan/vulkan.h> // v1.3
#include <SPIRV/SPIRV.h>   // v2023.2
#include <shaderc/shaderc.h> // v2023.7

#include <memory>
#include <vector>
#include <string>

// Global constants
#define SHADER_COMPILER_VERSION "1.0.0"
#define MAX_SHADER_STAGES 6
#define MAX_SHADER_SIZE 1048576  // 1MB
#define OPTIMIZATION_LEVEL_DEFAULT 2
#define POWER_EFFICIENT_OPTIMIZATION 1
#define MAX_CACHE_SIZE 268435456  // 256MB
#define SHADER_VALIDATION_ENABLED 1
#define PIPELINE_CACHE_VERSION "1.0.0"

// Forward declarations
struct shader_compiler_config_t;
struct power_profile_t;
struct power_state_t;
struct shader_source_t;
struct shader_binary_t;
struct optimization_config_t;
struct shader_cache_t;
struct optimization_passes_t;
struct power_manager_t;
struct thread_pool_t;
struct memory_pool_t;
struct validation_context_t;

/**
 * @brief Advanced shader compilation and optimization management class with power awareness
 */
class ShaderCompiler {
public:
    /**
     * @brief Constructs a new ShaderCompiler instance
     * @param config Shader compiler configuration
     * @param device Vulkan logical device handle
     * @param power_profile Power management profile
     * @throws std::runtime_error on initialization failure
     */
    ShaderCompiler(const shader_compiler_config_t* config,
                  VkDevice device,
                  const power_profile_t* power_profile);

    /**
     * @brief Destructor ensures proper cleanup of resources
     */
    ~ShaderCompiler();

    /**
     * @brief Compiles GLSL/HLSL shader to SPIR-V with power optimization
     * @param source Source shader code and metadata
     * @param output Compiled shader binary output
     * @param power_state Current power state for optimization decisions
     * @return VkResult indicating compilation success
     */
    [[nodiscard]]
    VkResult compile_shader(const shader_source_t* source,
                          shader_binary_t* output,
                          const power_state_t* power_state);

    /**
     * @brief Optimizes compiled SPIR-V shader code with power awareness
     * @param shader SPIR-V shader binary to optimize
     * @param config Optimization configuration parameters
     * @param power_state Current power state for optimization strategy
     * @return VkResult indicating optimization success
     */
    [[nodiscard]]
    VkResult shader_optimize(shader_binary_t* shader,
                           const optimization_config_t* config,
                           const power_state_t* power_state);

    /**
     * @brief Creates an optimized Vulkan graphics pipeline from compiled shaders
     * @param create_info Pipeline creation parameters
     * @param pipeline Output pipeline handle
     * @param power_profile Power profile for pipeline optimization
     * @return VkResult indicating pipeline creation success
     */
    [[nodiscard]]
    VkResult create_shader_pipeline(const VkGraphicsPipelineCreateInfo* create_info,
                                  VkPipeline* pipeline,
                                  const power_profile_t* power_profile);

private:
    shaderc_compiler_t* compiler;
    spv_context spv_context;
    shader_cache_t* cache;
    optimization_passes_t* opt_passes;
    VkDevice device;
    VkPipelineCache pipeline_cache;
    power_manager_t* power_manager;
    thread_pool_t* compile_threads;
    memory_pool_t* memory_pool;
    validation_context_t* validator;

    // Prevent copying
    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;
};

/**
 * @brief Initializes the shader compiler subsystem with power-aware configuration
 * @param config Compiler configuration parameters
 * @param power_profile Power management profile
 * @return VkResult indicating initialization success
 */
[[nodiscard]]
VkResult init_shader_compiler(const shader_compiler_config_t* config,
                            const power_profile_t* power_profile);

/**
 * @brief Cleans up shader compiler resources with optional cache persistence
 * @param persist_cache Whether to persist the shader cache to disk
 */
void cleanup_shader_compiler(bool persist_cache);

#endif // TALD_UNIA_SHADER_COMPILER_H