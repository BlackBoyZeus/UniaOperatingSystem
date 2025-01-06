/**
 * @file gpu_driver_test.c
 * @version 1.0.0
 * @brief Comprehensive GPU driver test suite with power and thermal validation
 */

#include <unity.h>
#include <vulkan/vulkan.h>
#include "kernel/gpu_driver.h"
#include "drivers/gpu/vulkan_driver.h"

// Test configuration constants
#define TEST_GPU_MEMORY_SIZE (1024 * 1024 * 1024)  // 1GB test allocation
#define TEST_FRAME_COUNT 1000
#define MAX_THERMAL_THRESHOLD 85.0f
#define TARGET_FRAME_TIME_MS 16.67f
#define POWER_MEASUREMENT_INTERVAL_MS 100

// Test state variables
static GPUDriver* gpu_driver = NULL;
static VulkanDriver* vulkan_driver = NULL;
static gpu_power_metrics_t baseline_metrics;
static gpu_power_profile_t test_power_profile;
static gpu_driver_config_t test_config;

/**
 * @brief Test suite setup with power and thermal monitoring initialization
 */
void setUp(void) {
    // Initialize test power profile
    test_power_profile = (gpu_power_profile_t) {
        .initial_state = GPU_POWER_BALANCED,
        .performance_target_fps = 60,
        .balanced_target_fps = 45,
        .power_save_target_fps = 30,
        .max_power_consumption_watts = 15.0f,
        .thermal_limit_celsius = MAX_THERMAL_THRESHOLD,
        .freq_curves = {
            // Performance state curve
            {
                .min_freq_mhz = 800,
                .max_freq_mhz = 1200,
                .step_size_mhz = 50
            },
            // Balanced state curve
            {
                .min_freq_mhz = 600,
                .max_freq_mhz = 1000,
                .step_size_mhz = 50
            },
            // Power save curve
            {
                .min_freq_mhz = 400,
                .max_freq_mhz = 800,
                .step_size_mhz = 50
            }
        }
    };

    // Initialize test configuration
    test_config = (gpu_driver_config_t) {
        .version = 1,
        .flags = 0,
        .max_memory_regions = MAX_GPU_MEMORY_REGIONS,
        .monitoring_interval_ms = GPU_MONITORING_INTERVAL_MS,
        .power_profile = test_power_profile
    };

    // Initialize GPU driver
    gpu_driver = new GPUDriver(&test_config, &test_power_profile);
    TEST_ASSERT_NOT_NULL(gpu_driver);

    // Get baseline metrics
    int result = gpu_driver->get_power_metrics(&baseline_metrics);
    TEST_ASSERT_EQUAL_INT(0, result);
}

/**
 * @brief Test suite cleanup and resource deallocation
 */
void tearDown(void) {
    if (gpu_driver) {
        delete gpu_driver;
        gpu_driver = NULL;
    }
}

/**
 * @brief Tests GPU driver initialization with power management validation
 */
void test_gpu_initialization(void) {
    gpu_power_metrics_t init_metrics;
    int result;

    // Verify initial power state
    result = gpu_driver->get_power_metrics(&init_metrics);
    TEST_ASSERT_EQUAL_INT(0, result);
    TEST_ASSERT_EQUAL_INT(GPU_POWER_BALANCED, init_metrics.current_state);
    TEST_ASSERT_LESS_THAN(test_power_profile.max_power_consumption_watts, init_metrics.current_power_watts);
    TEST_ASSERT_LESS_THAN(test_power_profile.thermal_limit_celsius, init_metrics.current_temp_celsius);
}

/**
 * @brief Tests GPU memory allocation with power efficiency validation
 */
void test_memory_allocation(void) {
    void* allocation = NULL;
    gpu_power_metrics_t alloc_metrics;
    int result;

    // Record pre-allocation metrics
    result = gpu_driver->get_power_metrics(&baseline_metrics);
    TEST_ASSERT_EQUAL_INT(0, result);

    // Perform test allocation
    result = gpu_driver->allocate_memory(TEST_GPU_MEMORY_SIZE, 0, &allocation);
    TEST_ASSERT_EQUAL_INT(0, result);
    TEST_ASSERT_NOT_NULL(allocation);

    // Verify power impact
    result = gpu_driver->get_power_metrics(&alloc_metrics);
    TEST_ASSERT_EQUAL_INT(0, result);
    TEST_ASSERT_LESS_THAN(test_power_profile.max_power_consumption_watts, alloc_metrics.current_power_watts);
    TEST_ASSERT_LESS_THAN(baseline_metrics.current_power_watts + 5.0f, alloc_metrics.current_power_watts);
}

/**
 * @brief Tests power state transitions and thermal management
 */
void test_power_state_transitions(void) {
    gpu_power_metrics_t transition_metrics;
    gpu_frequency_curve_t test_curve = {
        .min_freq_mhz = 400,
        .max_freq_mhz = 1200,
        .step_size_mhz = 50
    };
    int result;

    // Test each power state
    const gpu_power_state_t test_states[] = {
        GPU_POWER_SAVE,
        GPU_POWER_BALANCED,
        GPU_POWER_PERFORMANCE
    };

    for (size_t i = 0; i < sizeof(test_states)/sizeof(test_states[0]); i++) {
        // Transition to test state
        result = gpu_driver->set_power_state(test_states[i], &test_curve);
        TEST_ASSERT_EQUAL_INT(0, result);

        // Allow state to stabilize
        usleep(POWER_MEASUREMENT_INTERVAL_MS * 1000);

        // Verify state transition
        result = gpu_driver->get_power_metrics(&transition_metrics);
        TEST_ASSERT_EQUAL_INT(0, result);
        TEST_ASSERT_EQUAL_INT(test_states[i], transition_metrics.current_state);

        // Verify power and thermal constraints
        TEST_ASSERT_LESS_THAN(test_power_profile.max_power_consumption_watts, transition_metrics.current_power_watts);
        TEST_ASSERT_LESS_THAN(test_power_profile.thermal_limit_celsius, transition_metrics.current_temp_celsius);
    }
}

/**
 * @brief Tests Vulkan integration with performance monitoring
 */
void test_vulkan_integration(void) {
    VkBuffer test_buffer;
    VmaAllocation test_allocation;
    gpu_power_metrics_t render_metrics;
    float frame_times[TEST_FRAME_COUNT];
    int result;

    // Initialize Vulkan test resources
    vulkan_driver = new VulkanDriver(&test_config, &test_power_profile);
    TEST_ASSERT_NOT_NULL(vulkan_driver);

    // Create test buffer
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = TEST_GPU_MEMORY_SIZE,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    };

    result = vulkan_driver->create_buffer(&buffer_info, &test_buffer, &test_allocation);
    TEST_ASSERT_EQUAL_INT(VK_SUCCESS, result);

    // Test frame rendering
    for (int i = 0; i < TEST_FRAME_COUNT; i++) {
        uint64_t frame_start = get_time_ns();
        
        result = vulkan_driver->begin_frame();
        TEST_ASSERT_EQUAL_INT(VK_SUCCESS, result);

        // Simulate frame workload
        result = vulkan_driver->end_frame();
        TEST_ASSERT_EQUAL_INT(VK_SUCCESS, result);

        uint64_t frame_end = get_time_ns();
        frame_times[i] = (float)(frame_end - frame_start) / 1000000.0f; // Convert to ms

        // Verify frame time
        TEST_ASSERT_LESS_THAN(TARGET_FRAME_TIME_MS * 1.1f, frame_times[i]);

        // Check power and thermal state every 100 frames
        if (i % 100 == 0) {
            result = gpu_driver->get_power_metrics(&render_metrics);
            TEST_ASSERT_EQUAL_INT(0, result);
            TEST_ASSERT_LESS_THAN(test_power_profile.max_power_consumption_watts, render_metrics.current_power_watts);
            TEST_ASSERT_LESS_THAN(test_power_profile.thermal_limit_celsius, render_metrics.current_temp_celsius);
        }
    }

    // Cleanup Vulkan resources
    if (vulkan_driver) {
        delete vulkan_driver;
        vulkan_driver = NULL;
    }
}

/**
 * @brief Main test runner
 */
int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_gpu_initialization);
    RUN_TEST(test_memory_allocation);
    RUN_TEST(test_power_state_transitions);
    RUN_TEST(test_vulkan_integration);
    
    return UNITY_END();
}