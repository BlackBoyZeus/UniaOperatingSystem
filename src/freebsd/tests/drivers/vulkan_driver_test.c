/**
 * @file vulkan_driver_test.c
 * @version 1.0.0
 * @brief Unit test suite for power-aware Vulkan driver implementation
 */

#include "unity.h"
#include <vulkan/vulkan.h>  // v1.3
#include "../../drivers/gpu/vulkan_driver.h"
#include "../../drivers/gpu/shader_compiler.h"
#include <string.h>
#include <stdlib.h>

// Test configuration constants
#define TEST_VERTEX_SHADER "test_shaders/basic.vert"
#define TEST_FRAGMENT_SHADER "test_shaders/basic.frag"
#define TEST_BUFFER_SIZE (1024 * 1024)  // 1MB test buffer
#define POWER_PROFILE_HIGH 0
#define POWER_PROFILE_BALANCED 1
#define POWER_PROFILE_POWER_SAVE 2
#define TARGET_FRAME_TIME_MS 16.67f  // ~60 FPS
#define MAX_POWER_CONSUMPTION_MW 5000 // 5W max power consumption

// Test context structure
typedef struct {
    VulkanDriver* driver;
    ShaderCompiler* shader_compiler;
    vulkan_driver_config_t config;
    power_profile_t power_profile;
    VkBuffer test_buffer;
    VmaAllocation test_allocation;
    uint32_t current_frame;
    float frame_times[100];  // Rolling frame time measurements
    uint32_t power_measurements[100];  // Rolling power measurements
} TestContext;

static TestContext test_ctx;

/**
 * @brief Set up test environment before each test
 */
void setUp(void) {
    memset(&test_ctx, 0, sizeof(TestContext));
    
    // Initialize driver configuration
    test_ctx.config.app_name = "VulkanDriverTest";
    test_ctx.config.engine_name = "TALD_UNIA_TEST";
    test_ctx.config.app_version = VK_MAKE_VERSION(1, 0, 0);
    test_ctx.config.validation_enabled = true;
    test_ctx.config.debug_utils_enabled = true;

    // Initialize power profile
    test_ctx.power_profile.target_fps = 60;
    test_ctx.power_profile.power_state = PowerState::BALANCED;
    test_ctx.power_profile.max_power_consumption = MAX_POWER_CONSUMPTION_MW;
    test_ctx.power_profile.adaptive_power_management = true;

    // Create driver instance
    test_ctx.driver = new VulkanDriver(&test_ctx.config, &test_ctx.power_profile);
    TEST_ASSERT_NOT_NULL(test_ctx.driver);

    // Initialize shader compiler
    shader_compiler_config_t shader_config = {
        .optimization_level = OPTIMIZATION_LEVEL_DEFAULT,
        .enable_power_optimization = true,
        .cache_size = MAX_CACHE_SIZE,
        .validation_enabled = true
    };
    test_ctx.shader_compiler = new ShaderCompiler(&shader_config, 
                                                test_ctx.driver->get_device(),
                                                &test_ctx.power_profile);
    TEST_ASSERT_NOT_NULL(test_ctx.shader_compiler);
}

/**
 * @brief Clean up test environment after each test
 */
void tearDown(void) {
    if (test_ctx.test_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(test_ctx.driver->get_allocator(), 
                        test_ctx.test_buffer, 
                        test_ctx.test_allocation);
    }

    delete test_ctx.shader_compiler;
    delete test_ctx.driver;
}

/**
 * @brief Test Vulkan driver initialization with power management
 */
void test_vulkan_driver_initialization(void) {
    // Verify driver initialization
    TEST_ASSERT_NOT_NULL(test_ctx.driver);
    
    // Test power profile configuration
    PowerState current_state;
    VkResult result = test_ctx.driver->get_power_state(&current_state);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);
    TEST_ASSERT_EQUAL(PowerState::BALANCED, current_state);

    // Verify device capabilities
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(test_ctx.driver->get_physical_device(), &props);
    TEST_ASSERT_GREATER_OR_EQUAL(VK_API_VERSION_1_3, props.apiVersion);
}

/**
 * @brief Test power-aware buffer creation and management
 */
void test_power_aware_buffer_creation(void) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = TEST_BUFFER_SIZE,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VkResult result = test_ctx.driver->create_buffer(&buffer_info,
                                                   &test_ctx.test_buffer,
                                                   &test_ctx.test_allocation);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);
    TEST_ASSERT_NOT_EQUAL(VK_NULL_HANDLE, test_ctx.test_buffer);

    // Verify power-efficient memory allocation
    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(test_ctx.driver->get_allocator(),
                        test_ctx.test_allocation,
                        &alloc_info);
    TEST_ASSERT_EQUAL(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                     alloc_info.memoryType & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

/**
 * @brief Test power-aware shader compilation and optimization
 */
void test_power_aware_shader_compilation(void) {
    shader_source_t vertex_source = {
        .filename = TEST_VERTEX_SHADER,
        .stage = VK_SHADER_STAGE_VERTEX_BIT
    };

    shader_binary_t vertex_binary;
    power_state_t power_state = {
        .current_state = PowerState::BALANCED,
        .target_fps = 60,
        .current_power_draw = 3000  // 3W
    };

    VkResult result = test_ctx.shader_compiler->compile_shader(&vertex_source,
                                                             &vertex_binary,
                                                             &power_state);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);
    TEST_ASSERT_NOT_NULL(vertex_binary.code);
    TEST_ASSERT_GREATER_THAN(0, vertex_binary.code_size);

    // Verify power-optimized compilation
    optimization_config_t opt_config = {
        .level = POWER_EFFICIENT_OPTIMIZATION,
        .enable_power_saving = true
    };

    result = test_ctx.shader_compiler->shader_optimize(&vertex_binary,
                                                     &opt_config,
                                                     &power_state);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);
}

/**
 * @brief Test power-aware rendering performance
 */
void test_power_aware_rendering(void) {
    // Configure high performance mode
    VkResult result = test_ctx.driver->update_power_state(PowerState::HIGH_PERFORMANCE);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);

    // Test rendering loop
    for (int i = 0; i < 100; i++) {
        uint64_t start_time = test_ctx.driver->get_timestamp();
        
        result = test_ctx.driver->begin_frame();
        TEST_ASSERT_EQUAL(VK_SUCCESS, result);

        // Simulate rendering work
        test_ctx.driver->submit_commands();

        result = test_ctx.driver->end_frame();
        TEST_ASSERT_EQUAL(VK_SUCCESS, result);

        uint64_t end_time = test_ctx.driver->get_timestamp();
        float frame_time = (end_time - start_time) * 0.000001f; // Convert to ms
        
        // Verify frame time meets target
        TEST_ASSERT_LESS_OR_EQUAL(TARGET_FRAME_TIME_MS, frame_time);

        // Check power consumption
        power_stats_t power_stats;
        test_ctx.driver->get_power_stats(&power_stats);
        TEST_ASSERT_LESS_OR_EQUAL(MAX_POWER_CONSUMPTION_MW, power_stats.current_power_draw);
    }
}

/**
 * @brief Test power profile switching
 */
void test_power_profile_switching(void) {
    // Test high performance mode
    VkResult result = test_ctx.driver->update_power_state(PowerState::HIGH_PERFORMANCE);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);
    
    PowerState current_state;
    result = test_ctx.driver->get_power_state(&current_state);
    TEST_ASSERT_EQUAL(PowerState::HIGH_PERFORMANCE, current_state);

    // Verify performance characteristics
    power_stats_t power_stats;
    test_ctx.driver->get_power_stats(&power_stats);
    TEST_ASSERT_GREATER_THAN(0, power_stats.gpu_frequency);
    TEST_ASSERT_LESS_OR_EQUAL(MAX_POWER_CONSUMPTION_MW, power_stats.current_power_draw);

    // Test power save mode
    result = test_ctx.driver->update_power_state(PowerState::LOW_POWER);
    TEST_ASSERT_EQUAL(VK_SUCCESS, result);
    
    result = test_ctx.driver->get_power_state(&current_state);
    TEST_ASSERT_EQUAL(PowerState::LOW_POWER, current_state);

    // Verify reduced power consumption
    test_ctx.driver->get_power_stats(&power_stats);
    TEST_ASSERT_LESS_THAN(MAX_POWER_CONSUMPTION_MW / 2, power_stats.current_power_draw);
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_vulkan_driver_initialization);
    RUN_TEST(test_power_aware_buffer_creation);
    RUN_TEST(test_power_aware_shader_compilation);
    RUN_TEST(test_power_aware_rendering);
    RUN_TEST(test_power_profile_switching);
    
    return UNITY_END();
}