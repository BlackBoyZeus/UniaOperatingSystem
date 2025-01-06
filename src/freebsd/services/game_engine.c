/**
 * @file game_engine.c
 * @version 1.0.0
 * @brief Core implementation of the TALD UNIA game engine service
 * @copyright TALD UNIA Platform
 */

#include "game_engine.h"
#include "game/engine/physics_engine.h"
#include "game/engine/vulkan_renderer.h"
#include <cuda.h>          // v12.0
#include <vulkan/vulkan.h> // v1.3
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

// Thread pool configuration
static pthread_t thread_pool[THREAD_POOL_SIZE];
static pthread_mutex_t thread_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t thread_pool_cond = PTHREAD_COND_INITIALIZER;

// Performance monitoring
static struct {
    float frame_times[60];
    uint32_t frame_index;
    float avg_frame_time;
    float min_frame_time;
    float max_frame_time;
} perf_monitor;

/**
 * @brief Initializes the game engine service
 * @param config Engine configuration parameters
 * @return Initialized GameEngine instance or NULL on failure
 */
GameEngine* init_game_engine(const game_engine_config_t* config) {
    if (!config) {
        errno = EINVAL;
        return NULL;
    }

    GameEngine* engine = (GameEngine*)calloc(1, sizeof(GameEngine));
    if (!engine) {
        return NULL;
    }

    // Initialize mutex
    if (pthread_mutex_init(&engine->state_mutex, NULL) != 0) {
        free(engine);
        return NULL;
    }

    // Copy configuration
    memcpy(&engine->config, config, sizeof(game_engine_config_t));
    engine->current_power_state = POWER_STATE_BALANCED;
    engine->running = true;

    // Initialize CUDA resources
    cuda_wrapper_config_t cuda_config = {
        .device_id = 0,
        .enable_p2p = true,
        .reserved_memory = 1024 * 1024 * 1024  // 1GB reserved
    };
    
    engine->cuda_wrapper = init_cuda_wrapper(&cuda_config);
    if (!engine->cuda_wrapper) {
        cleanup_game_engine(engine);
        return NULL;
    }

    // Initialize physics engine
    physics_config_t physics_config = {
        .max_bodies = 10000,
        .max_constraints = 20000,
        .timestep = 1.0f / PHYSICS_UPDATE_RATE,
        .cuda_wrapper = engine->cuda_wrapper
    };
    
    engine->physics = init_physics_engine(&physics_config);
    if (!engine->physics) {
        cleanup_game_engine(engine);
        return NULL;
    }

    // Initialize Vulkan renderer
    vulkan_renderer_config_t renderer_config = {
        .resolution = config->resolution,
        .enable_vsync = config->enable_vsync,
        .enable_dynamic_resolution = true,
        .initial_power_mode = POWER_MODE_BALANCED,
        .msaa_samples = VK_SAMPLE_COUNT_4_BIT,
        .target_frame_time = 1.0f / TARGET_FRAME_RATE
    };
    
    engine->renderer = init_vulkan_renderer(&renderer_config, engine->scene_manager);
    if (!engine->renderer) {
        cleanup_game_engine(engine);
        return NULL;
    }

    // Initialize scene manager
    scene_manager_config_t scene_config = {
        .max_objects = MAX_SCENE_OBJECTS,
        .max_env_meshes = MAX_ENVIRONMENT_MESHES,
        .update_frequency = UPDATE_FREQUENCY_HZ,
        .physics = engine->physics,
        .cuda_wrapper = engine->cuda_wrapper
    };
    
    engine->scene_manager = init_scene_manager(&scene_config);
    if (!engine->scene_manager) {
        cleanup_game_engine(engine);
        return NULL;
    }

    // Initialize thread pool
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        if (pthread_create(&thread_pool[i], NULL, thread_pool_worker, engine) != 0) {
            cleanup_game_engine(engine);
            return NULL;
        }
    }

    return engine;
}

/**
 * @brief Main update function for the game engine
 * @param engine Game engine instance
 * @param delta_time Time since last update
 * @return Success status
 */
bool game_engine_update(GameEngine* engine, float delta_time) {
    if (!engine || delta_time <= 0) {
        return false;
    }

    pthread_mutex_lock(&engine->state_mutex);

    // Update physics
    if (!physics_world_simulate(engine->physics, delta_time)) {
        pthread_mutex_unlock(&engine->state_mutex);
        return false;
    }

    // Update scene
    if (!scene_manager_update(engine->scene_manager, delta_time)) {
        pthread_mutex_unlock(&engine->state_mutex);
        return false;
    }

    // Render frame
    VkResult render_result = vulkan_renderer_render_frame(engine->renderer, delta_time);
    if (render_result != VK_SUCCESS) {
        pthread_mutex_unlock(&engine->state_mutex);
        return false;
    }

    // Update performance metrics
    update_performance_metrics(engine, delta_time);

    // Adjust power state if needed
    adjust_power_state(engine);

    pthread_mutex_unlock(&engine->state_mutex);
    return true;
}

/**
 * @brief Processes new LiDAR scan data
 * @param engine Game engine instance
 * @param points Array of 3D points
 * @param count Number of points
 * @return Success status
 */
bool process_lidar_scan(GameEngine* engine, const float3* points, size_t count) {
    if (!engine || !points || count == 0) {
        return false;
    }

    pthread_mutex_lock(&engine->state_mutex);

    // Update scene with new points
    if (!scene_manager_add_lidar_points(engine->scene_manager, points, count)) {
        pthread_mutex_unlock(&engine->state_mutex);
        return false;
    }

    // Update physics collision mesh
    if (!physics_world_update_collision_mesh(engine->physics, 
        scene_manager_get_environment_mesh(engine->scene_manager))) {
        pthread_mutex_unlock(&engine->state_mutex);
        return false;
    }

    // Update renderer point cloud visualization
    if (!vulkan_renderer_update_point_cloud(engine->renderer, points, count)) {
        pthread_mutex_unlock(&engine->state_mutex);
        return false;
    }

    pthread_mutex_unlock(&engine->state_mutex);
    return true;
}

/**
 * @brief Updates engine power state
 * @param engine Game engine instance
 * @param state New power state
 */
void set_power_state(GameEngine* engine, power_state_t state) {
    if (!engine) {
        return;
    }

    pthread_mutex_lock(&engine->state_mutex);
    
    engine->current_power_state = state;

    // Update subsystem power states
    physics_world_set_power_state(engine->physics, state);
    vulkan_renderer_set_power_mode(engine->renderer, convert_power_state(state));
    scene_manager_set_power_mode(engine->scene_manager, state);

    pthread_mutex_unlock(&engine->state_mutex);
}

/**
 * @brief Thread pool worker function
 * @param arg Game engine instance
 * @return NULL
 */
static void* thread_pool_worker(void* arg) {
    GameEngine* engine = (GameEngine*)arg;
    
    while (engine->running) {
        pthread_mutex_lock(&thread_pool_mutex);
        
        while (engine->running && !has_pending_work(engine)) {
            pthread_cond_wait(&thread_pool_cond, &thread_pool_mutex);
        }
        
        if (!engine->running) {
            pthread_mutex_unlock(&thread_pool_mutex);
            break;
        }

        // Get next work item
        work_item_t work = get_next_work_item(engine);
        pthread_mutex_unlock(&thread_pool_mutex);

        // Process work item
        process_work_item(engine, &work);
    }

    return NULL;
}

/**
 * @brief Updates performance metrics
 * @param engine Game engine instance
 * @param delta_time Time since last update
 */
static void update_performance_metrics(GameEngine* engine, float delta_time) {
    perf_monitor.frame_times[perf_monitor.frame_index] = delta_time;
    perf_monitor.frame_index = (perf_monitor.frame_index + 1) % 60;

    float sum = 0.0f;
    perf_monitor.min_frame_time = delta_time;
    perf_monitor.max_frame_time = delta_time;

    for (int i = 0; i < 60; i++) {
        sum += perf_monitor.frame_times[i];
        if (perf_monitor.frame_times[i] < perf_monitor.min_frame_time) {
            perf_monitor.min_frame_time = perf_monitor.frame_times[i];
        }
        if (perf_monitor.frame_times[i] > perf_monitor.max_frame_time) {
            perf_monitor.max_frame_time = perf_monitor.frame_times[i];
        }
    }

    perf_monitor.avg_frame_time = sum / 60.0f;
}

/**
 * @brief Adjusts power state based on performance metrics
 * @param engine Game engine instance
 */
static void adjust_power_state(GameEngine* engine) {
    float target_frame_time = 1.0f / TARGET_FRAME_RATE;
    
    if (perf_monitor.avg_frame_time > target_frame_time * 1.2f) {
        // Performance below target, increase power state
        if (engine->current_power_state < POWER_STATE_HIGH_PERFORMANCE) {
            set_power_state(engine, engine->current_power_state + 1);
        }
    } else if (perf_monitor.avg_frame_time < target_frame_time * 0.8f) {
        // Performance well above target, decrease power state
        if (engine->current_power_state > POWER_STATE_LOW_POWER) {
            set_power_state(engine, engine->current_power_state - 1);
        }
    }
}

/**
 * @brief Cleans up game engine resources
 * @param engine Game engine instance
 */
void cleanup_game_engine(GameEngine* engine) {
    if (!engine) {
        return;
    }

    engine->running = false;
    pthread_cond_broadcast(&thread_pool_cond);

    // Wait for threads to finish
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        pthread_join(thread_pool[i], NULL);
    }

    // Cleanup subsystems
    if (engine->renderer) {
        destroy_vulkan_renderer(engine->renderer);
    }
    if (engine->physics) {
        cleanup_physics_engine(engine->physics);
    }
    if (engine->scene_manager) {
        cleanup_scene_manager(engine->scene_manager);
    }
    if (engine->cuda_wrapper) {
        cleanup_cuda_wrapper(engine->cuda_wrapper);
    }

    pthread_mutex_destroy(&engine->state_mutex);
    free(engine);
}