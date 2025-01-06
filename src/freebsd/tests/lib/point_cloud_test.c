/**
 * @file point_cloud_test.c
 * @version 1.0.0
 * @brief Comprehensive unit test suite for point cloud processing functionality
 * @copyright TALD UNIA Platform
 */

#include "unity.h"                // v2.5.2
#include "point_cloud.h"          // Internal
#include "cuda_wrapper.h"         // Internal
#include <cuda.h>                 // v12.0
#include <sys/types.h>           // FreeBSD 9.0
#include <time.h>
#include <stdlib.h>
#include <string.h>

// Test configuration constants
#define TEST_POINT_CLOUD_SIZE 100000
#define TEST_OPTIMIZATION_DISTANCE 0.01f
#define TEST_TIMEOUT_MS 50
#define MAX_RETRY_ATTEMPTS 3
#define GPU_MEMORY_THRESHOLD 0.9f
#define TEST_DATA_RESOLUTION 0.01f

// Test context structure
typedef struct {
    tald::cuda::CudaWrapper* cuda_wrapper;
    tald::lidar::PointCloud* point_cloud;
    float3* test_points;
    float* test_confidences;
    struct timespec start_time;
    struct timespec end_time;
} TestContext;

static TestContext ctx;

// Helper functions
static float3* generate_test_points(size_t count) {
    float3* points = (float3*)malloc(count * sizeof(float3));
    TEST_ASSERT_NOT_NULL(points);
    
    for (size_t i = 0; i < count; i++) {
        points[i].x = ((float)rand() / RAND_MAX) * 5000.0f;
        points[i].y = ((float)rand() / RAND_MAX) * 5000.0f;
        points[i].z = ((float)rand() / RAND_MAX) * 5000.0f;
    }
    return points;
}

static float* generate_test_confidences(size_t count) {
    float* confidences = (float*)malloc(count * sizeof(float));
    TEST_ASSERT_NOT_NULL(confidences);
    
    for (size_t i = 0; i < count; i++) {
        confidences[i] = ((float)rand() / RAND_MAX);
    }
    return confidences;
}

static float measure_elapsed_ms(struct timespec* start, struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0f + 
           (end->tv_nsec - start->tv_nsec) / 1000000.0f;
}

void setUp(void) {
    // Initialize CUDA environment
    tald::cuda::CudaStatus status = tald::cuda::initialize_cuda(0, false, 0);
    TEST_ASSERT_EQUAL(tald::cuda::CudaStatus::SUCCESS, status);
    
    // Create CUDA wrapper with test configuration
    tald::cuda::StreamConfig stream_config;
    stream_config.stream_count = 2;
    stream_config.enable_priority = true;
    
    tald::cuda::MemoryConfig memory_config;
    memory_config.enable_tracking = true;
    memory_config.reserved_size = TEST_POINT_CLOUD_SIZE * sizeof(float3);
    
    ctx.cuda_wrapper = new tald::cuda::CudaWrapper(0, stream_config, memory_config);
    TEST_ASSERT_NOT_NULL(ctx.cuda_wrapper);
    
    // Create point cloud instance
    ctx.point_cloud = tald::lidar::create_point_cloud(TEST_POINT_CLOUD_SIZE, ctx.cuda_wrapper);
    TEST_ASSERT_NOT_NULL(ctx.point_cloud);
    
    // Generate test data
    ctx.test_points = generate_test_points(TEST_POINT_CLOUD_SIZE);
    ctx.test_confidences = generate_test_confidences(TEST_POINT_CLOUD_SIZE);
    
    // Verify GPU memory availability
    auto [total, available] = ctx.cuda_wrapper->get_memory_stats();
    TEST_ASSERT_GREATER_THAN(memory_config.reserved_size, available);
}

void tearDown(void) {
    // Cleanup test data
    free(ctx.test_points);
    free(ctx.test_confidences);
    
    // Destroy point cloud
    tald::lidar::destroy_point_cloud(ctx.point_cloud);
    
    // Cleanup CUDA resources
    delete ctx.cuda_wrapper;
    tald::cuda::cleanup_cuda(true);
}

void test_point_cloud_creation(void) {
    TEST_ASSERT_EQUAL(0, ctx.point_cloud->get_point_count());
    TEST_ASSERT_EQUAL(TEST_POINT_CLOUD_SIZE, ctx.point_cloud->get_capacity());
    
    auto bounds = ctx.point_cloud->get_bounds();
    TEST_ASSERT_EQUAL_FLOAT(0.0f, bounds.first.x);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, bounds.first.y);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, bounds.first.z);
}

void test_point_addition(void) {
    const size_t test_size = 1000;
    bool result = ctx.point_cloud->add_points(ctx.test_points, test_size, ctx.test_confidences);
    TEST_ASSERT_TRUE(result);
    TEST_ASSERT_EQUAL(test_size, ctx.point_cloud->get_point_count());
}

void test_point_removal(void) {
    const size_t test_size = 1000;
    ctx.point_cloud->add_points(ctx.test_points, test_size, ctx.test_confidences);
    
    bool* removal_mask = (bool*)calloc(test_size, sizeof(bool));
    for (size_t i = 0; i < test_size/2; i++) {
        removal_mask[i] = true;
    }
    
    size_t removed = ctx.point_cloud->remove_points(removal_mask, test_size);
    TEST_ASSERT_EQUAL(test_size/2, removed);
    TEST_ASSERT_EQUAL(test_size/2, ctx.point_cloud->get_point_count());
    
    free(removal_mask);
}

void test_point_optimization(void) {
    const size_t test_size = 10000;
    ctx.point_cloud->add_points(ctx.test_points, test_size, ctx.test_confidences);
    
    bool result = ctx.point_cloud->optimize(TEST_OPTIMIZATION_DISTANCE, 0.5f);
    TEST_ASSERT_TRUE(result);
    TEST_ASSERT_LESS_THAN(test_size, ctx.point_cloud->get_point_count());
}

void test_performance(void) {
    clock_gettime(CLOCK_MONOTONIC, &ctx.start_time);
    
    // Test 30Hz processing requirement
    for (int i = 0; i < 30; i++) {
        bool result = ctx.point_cloud->add_points(ctx.test_points, TEST_POINT_CLOUD_SIZE, ctx.test_confidences);
        TEST_ASSERT_TRUE(result);
        
        result = ctx.point_cloud->optimize(TEST_OPTIMIZATION_DISTANCE, 0.5f);
        TEST_ASSERT_TRUE(result);
        
        float3 output_buffer[TEST_POINT_CLOUD_SIZE];
        size_t points_retrieved = ctx.point_cloud->get_points(output_buffer, TEST_POINT_CLOUD_SIZE, 0.0f);
        TEST_ASSERT_GREATER_THAN(0, points_retrieved);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &ctx.end_time);
    float elapsed = measure_elapsed_ms(&ctx.start_time, &ctx.end_time);
    
    // Verify 50ms latency requirement
    TEST_ASSERT_LESS_OR_EQUAL(TEST_TIMEOUT_MS, elapsed/30.0f);
}

void test_gpu_resource_management(void) {
    auto [total, initial_available] = ctx.cuda_wrapper->get_memory_stats();
    
    // Test memory management under load
    for (int i = 0; i < 10; i++) {
        ctx.point_cloud->add_points(ctx.test_points, TEST_POINT_CLOUD_SIZE/10, ctx.test_confidences);
    }
    
    auto [_, final_available] = ctx.cuda_wrapper->get_memory_stats();
    TEST_ASSERT_GREATER_THAN(total * (1.0f - GPU_MEMORY_THRESHOLD), final_available);
}

void test_error_handling(void) {
    // Test invalid point addition
    bool result = ctx.point_cloud->add_points(nullptr, TEST_POINT_CLOUD_SIZE, ctx.test_confidences);
    TEST_ASSERT_FALSE(result);
    
    // Test capacity overflow
    result = ctx.point_cloud->add_points(ctx.test_points, TEST_POINT_CLOUD_SIZE + 1, ctx.test_confidences);
    TEST_ASSERT_FALSE(result);
    
    // Test invalid optimization parameters
    result = ctx.point_cloud->optimize(-1.0f, 2.0f);
    TEST_ASSERT_FALSE(result);
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_point_cloud_creation);
    RUN_TEST(test_point_addition);
    RUN_TEST(test_point_removal);
    RUN_TEST(test_point_optimization);
    RUN_TEST(test_performance);
    RUN_TEST(test_gpu_resource_management);
    RUN_TEST(test_error_handling);
    
    return UNITY_END();
}