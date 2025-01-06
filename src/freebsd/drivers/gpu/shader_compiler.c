/**
 * @file shader_compiler.c
 * @version 1.0.0
 * @brief Power-aware shader compilation and optimization implementation for TALD UNIA platform
 */

#include "shader_compiler.h"
#include <vulkan/vulkan.h>  // v1.3
#include <SPIRV/SPIRV.h>    // v2023.2
#include <shaderc/shaderc.h> // v2023.7
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// Global constants from specification
#define SHADER_CACHE_SIZE 1024
#define COMPILER_THREAD_COUNT 4
#define COMPILATION_RETRY_COUNT 3
#define CACHE_VERSION "2023.2"

// Power-aware optimization pass configuration
static const char* DEFAULT_OPTIMIZATION_PASSES[] = {
    "eliminate-dead-code",
    "merge-return",
    "inline-functions",
    "optimize-power"
};

// Power profile thresholds
static const struct {
    int high;
    int medium;
    int low;
} POWER_PROFILE_THRESHOLDS = {
    .high = 90,
    .medium = 50,
    .low = 20
};

// Internal state tracking
static struct {
    bool initialized;
    shaderc_compiler_t* compiler;
    spv_context spv_context;
    shader_cache_t* cache;
    thread_pool_t* compile_threads;
    pthread_mutex_t compiler_mutex;
    power_monitor_t* power_monitor;
} g_compiler_state = {0};

/**
 * @brief Initialize power-aware optimization passes
 */
static VkResult init_optimization_passes(optimization_passes_t* passes, 
                                      const power_profile_t* power_profile) {
    assert(passes && power_profile);
    
    passes->count = sizeof(DEFAULT_OPTIMIZATION_PASSES) / sizeof(char*);
    passes->passes = malloc(passes->count * sizeof(optimization_pass_t));
    if (!passes->passes) return VK_ERROR_OUT_OF_HOST_MEMORY;

    for (size_t i = 0; i < passes->count; i++) {
        passes->passes[i].name = strdup(DEFAULT_OPTIMIZATION_PASSES[i]);
        passes->passes[i].power_threshold = power_profile->optimization_threshold;
    }

    return VK_SUCCESS;
}

/**
 * @brief Initialize shader cache with power profile considerations
 */
static VkResult init_shader_cache(shader_cache_t* cache, size_t size) {
    assert(cache);
    
    cache->entries = calloc(size, sizeof(shader_cache_entry_t));
    if (!cache->entries) return VK_ERROR_OUT_OF_HOST_MEMORY;
    
    cache->size = size;
    cache->version = strdup(CACHE_VERSION);
    pthread_mutex_init(&cache->mutex, NULL);
    
    return VK_SUCCESS;
}

/**
 * @brief Initialize the shader compiler subsystem
 */
__attribute__((visibility("default")))
VkResult init_shader_compiler(const shader_compiler_config_t* config,
                            const power_profile_t* power_profile) {
    assert(config && power_profile);
    
    if (g_compiler_state.initialized) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Initialize compiler mutex
    if (pthread_mutex_init(&g_compiler_state.compiler_mutex, NULL) != 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Create shaderc compiler
    g_compiler_state.compiler = shaderc_compiler_initialize();
    if (!g_compiler_state.compiler) {
        goto cleanup_mutex;
    }

    // Initialize SPIR-V context
    spv_target_env target_env = SPV_ENV_VULKAN_1_3;
    g_compiler_state.spv_context = spvContextCreate(target_env);
    if (!g_compiler_state.spv_context) {
        goto cleanup_compiler;
    }

    // Initialize shader cache
    g_compiler_state.cache = malloc(sizeof(shader_cache_t));
    if (!g_compiler_state.cache || 
        init_shader_cache(g_compiler_state.cache, SHADER_CACHE_SIZE) != VK_SUCCESS) {
        goto cleanup_spv;
    }

    // Initialize thread pool
    g_compiler_state.compile_threads = thread_pool_create(COMPILER_THREAD_COUNT);
    if (!g_compiler_state.compile_threads) {
        goto cleanup_cache;
    }

    // Initialize power monitor
    g_compiler_state.power_monitor = power_monitor_create(power_profile);
    if (!g_compiler_state.power_monitor) {
        goto cleanup_threads;
    }

    g_compiler_state.initialized = true;
    return VK_SUCCESS;

cleanup_threads:
    thread_pool_destroy(g_compiler_state.compile_threads);
cleanup_cache:
    free(g_compiler_state.cache);
cleanup_spv:
    spvContextDestroy(g_compiler_state.spv_context);
cleanup_compiler:
    shaderc_compiler_release(g_compiler_state.compiler);
cleanup_mutex:
    pthread_mutex_destroy(&g_compiler_state.compiler_mutex);
    return VK_ERROR_INITIALIZATION_FAILED;
}

/**
 * @brief Clean up shader compiler resources
 */
__attribute__((visibility("default")))
void cleanup_shader_compiler(void) {
    if (!g_compiler_state.initialized) return;

    pthread_mutex_lock(&g_compiler_state.compiler_mutex);

    power_monitor_destroy(g_compiler_state.power_monitor);
    thread_pool_destroy(g_compiler_state.compile_threads);
    
    if (g_compiler_state.cache) {
        pthread_mutex_destroy(&g_compiler_state.cache->mutex);
        free(g_compiler_state.cache->entries);
        free(g_compiler_state.cache);
    }

    spvContextDestroy(g_compiler_state.spv_context);
    shaderc_compiler_release(g_compiler_state.compiler);

    pthread_mutex_unlock(&g_compiler_state.compiler_mutex);
    pthread_mutex_destroy(&g_compiler_state.compiler_mutex);

    g_compiler_state.initialized = false;
}

/**
 * @brief ShaderCompiler implementation
 */
ShaderCompiler::ShaderCompiler(const shader_compiler_config_t* config,
                             VkDevice device,
                             const power_profile_t* power_profile)
    : device(device) {
    
    assert(config && power_profile);
    
    // Initialize compiler components
    compiler = shaderc_compiler_initialize();
    spv_context = spvContextCreate(SPV_ENV_VULKAN_1_3);
    cache = new shader_cache_t();
    opt_passes = new optimization_passes_t();
    
    // Initialize power monitoring
    power_monitor = power_monitor_create(power_profile);
    
    // Create pipeline cache
    VkPipelineCacheCreateInfo cache_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .initialDataSize = 0
    };
    vkCreatePipelineCache(device, &cache_info, nullptr, &pipeline_cache);
    
    // Initialize thread pool
    compile_threads = thread_pool_create(COMPILER_THREAD_COUNT);
}

VkResult ShaderCompiler::compile_shader(const shader_source_t* source,
                                      shader_binary_t* output,
                                      const power_state_t* power_state) {
    assert(source && output && power_state);

    // Check shader cache first
    if (cache->lookup(source->hash, output)) {
        return VK_SUCCESS;
    }

    // Configure compilation based on power state
    shaderc_compile_options_t options;
    shaderc_compile_options_initialize(&options);
    shaderc_compile_options_set_optimization_level(
        options,
        power_state->level >= POWER_PROFILE_THRESHOLDS.high ?
            shaderc_optimization_level_performance :
            shaderc_optimization_level_size
    );

    // Compile shader
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        compiler,
        source->code,
        source->code_size,
        source->stage,
        source->name,
        "main",
        options
    );

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Copy result to output
    output->size = shaderc_result_get_length(result);
    output->code = malloc(output->size);
    memcpy(output->code, shaderc_result_get_bytes(result), output->size);

    // Optimize based on power state
    shader_optimize(output, nullptr, power_state);

    // Update cache
    cache->insert(source->hash, output);

    shaderc_result_release(result);
    return VK_SUCCESS;
}

VkResult ShaderCompiler::shader_optimize(shader_binary_t* shader,
                                       const optimization_config_t* config,
                                       const power_state_t* power_state) {
    assert(shader && power_state);

    spv_optimizer_options options = spvOptimizerOptionsCreate();
    spv_optimizer optimizer = spvOptimizerCreate(SPV_ENV_VULKAN_1_3);

    // Configure optimization passes based on power state
    for (size_t i = 0; i < opt_passes->count; i++) {
        if (power_state->level >= opt_passes->passes[i].power_threshold) {
            spvOptimizerRegisterPass(optimizer, opt_passes->passes[i].name);
        }
    }

    // Run optimization
    spv_binary optimized;
    spv_result_t result = spvOptimizerOptimize(
        optimizer,
        shader->code,
        shader->size,
        &optimized
    );

    if (result != SPV_SUCCESS) {
        spvOptimizerDestroy(optimizer);
        spvOptimizerOptionsDestroy(options);
        return VK_ERROR_OPTIMIZATION_FAILED;
    }

    // Update shader with optimized code
    free(shader->code);
    shader->code = malloc(optimized->size);
    shader->size = optimized->size;
    memcpy(shader->code, optimized->code, optimized->size);

    spvBinaryDestroy(optimized);
    spvOptimizerDestroy(optimizer);
    spvOptimizerOptionsDestroy(options);

    return VK_SUCCESS;
}

ShaderCompiler::~ShaderCompiler() {
    vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    thread_pool_destroy(compile_threads);
    power_monitor_destroy(power_monitor);
    delete cache;
    delete opt_passes;
    spvContextDestroy(spv_context);
    shaderc_compiler_release(compiler);
}