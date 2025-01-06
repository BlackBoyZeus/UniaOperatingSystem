/**
 * @file lidar_processing_test.c
 * @version 1.0.0
 * @brief Comprehensive unit test suite for TALD UNIA LiDAR processing library
 * @copyright TALD UNIA Platform
 */

// External dependencies with versions
#include <unity.h>           // Unity 2.5.2
#include <cuda_runtime.h>    // CUDA 12.0
#include <NvInfer.h>         // TensorRT 8.6

// Internal dependencies
#include "lidar_processing.h"
#include "point_cloud.h"
#include "cuda_wrapper.h"
#include "tensorrt_wrapper.h"

// Standard includes
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace tald::lidar;
using namespace tald::cuda;
using namespace tald::ai;

// Test configuration constants
static const size_t TEST_POINT_CLOUD_SIZE = 32768;
static const uint32_t MAX_PROCESSING_TIME_MS = 50;
static const float MIN_CLASSIFICATION_CONFIDENCE = 0.85f;
static const size_t TEST_ITERATIONS = 1000;
static const uint32_t TIMING_PRECISION_US = 1;
static const size_t GPU_MEMORY_THRESHOLD = 2048 * 1024 * 1024; // 2GB

// Global test fixtures
static CudaWrapper* cuda_wrapper = nullptr;
static TensorRTWrapper* tensorrt_wrapper = nullptr;
static LidarProcessor* processor = nullptr;
static float3* test_points = nullptr;
static uint8_t* test_raw_data = nullptr;
static struct timespec timer_start, timer_end;

void setUp(void) {
    // Initialize CUDA wrapper with error checking
    try {
        cuda_wrapper = new CudaWrapper(0); // Use primary GPU
        TEST_ASSERT_NOT_NULL(cuda_wrapper);
        
        auto [total_mem, available_mem] = cuda_wrapper->get_memory_stats();
        TEST_ASSERT_GREATER_OR_EQUAL(GPU_MEMORY_THRESHOLD, available_mem);
    } catch (const std::exception& e) {
        TEST_FAIL_MESSAGE("CUDA initialization failed");
    }

    // Initialize TensorRT wrapper
    try {
        TensorRTConfig trt_config;
        trt_config.max_batch_size = TEST_POINT_CLOUD_SIZE;
        trt_config.enable_fp16 = true;
        tensorrt_wrapper = new TensorRTWrapper(cuda_wrapper, "models/lidar_classifier.trt", trt_config);
        TEST_ASSERT_NOT_NULL(tensorrt_wrapper);
    } catch (const std::exception& e) {
        TEST_FAIL_MESSAGE("TensorRT initialization failed");
    }

    // Initialize LidarProcessor with test configuration
    LidarConfig config;
    config.scan_frequency = SCAN_FREQUENCY_HZ;
    config.min_resolution = MIN_RESOLUTION_MM;
    config.max_range = MAX_RANGE_MM;
    config.batch_size = TEST_POINT_CLOUD_SIZE;
    config.enable_optimization = true;
    config.enable_monitoring = true;

    auto result = initialize_lidar_processor(
        cuda_wrapper,
        tensorrt_wrapper,
        config,
        [](const std::string& msg, int code) {
            TEST_FAIL_MESSAGE(msg.c_str());
        }
    );

    TEST_ASSERT_TRUE(result.success);
    processor = result.value;
    TEST_ASSERT_NOT_NULL(processor);

    // Allocate test data
    test_points = (float3*)malloc(TEST_POINT_CLOUD_SIZE * sizeof(float3));
    test_raw_data = (uint8_t*)malloc(TEST_POINT_CLOUD_SIZE * sizeof(float3));
    TEST_ASSERT_NOT_NULL(test_points);
    TEST_ASSERT_NOT_NULL(test_raw_data);
}

void tearDown(void) {
    // Clean up test data
    free(test_points);
    free(test_raw_data);

    // Clean up processor
    delete processor;
    delete tensorrt_wrapper;
    delete cuda_wrapper;

    processor = nullptr;
    tensorrt_wrapper = nullptr;
    cuda_wrapper = nullptr;
    test_points = nullptr;
    test_raw_data = nullptr;
}

void test_lidar_processor_initialization(void) {
    // Verify processor state
    TEST_ASSERT_NOT_NULL(processor);
    
    // Verify health status
    auto health = processor->get_health_status();
    TEST_ASSERT_TRUE(health.is_healthy);
    TEST_ASSERT_TRUE(health.gpu_available);
    TEST_ASSERT_TRUE(health.inference_available);
    TEST_ASSERT_EQUAL_STRING("", health.error_message.c_str());
    
    // Verify performance metrics initialization
    auto metrics = processor->get_performance_metrics();
    TEST_ASSERT_EQUAL(0.0f, metrics.processing_latency_ms);
    TEST_ASSERT_EQUAL(0, metrics.points_processed);
    TEST_ASSERT_EQUAL(0, metrics.processing_errors);
}

void test_point_cloud_processing(void) {
    // Generate test point cloud data
    for (size_t i = 0; i < TEST_POINT_CLOUD_SIZE; i++) {
        test_points[i].x = (float)(rand() % 5000) / 100.0f; // 0-50m range
        test_points[i].y = (float)(rand() % 5000) / 100.0f;
        test_points[i].z = (float)(rand() % 5000) / 100.0f;
    }
    memcpy(test_raw_data, test_points, TEST_POINT_CLOUD_SIZE * sizeof(float3));

    // Measure processing time
    clock_gettime(CLOCK_MONOTONIC, &timer_start);
    
    auto result = processor->process_scan(test_raw_data, TEST_POINT_CLOUD_SIZE * sizeof(float3));
    
    clock_gettime(CLOCK_MONOTONIC, &timer_end);
    
    // Verify processing success
    TEST_ASSERT_TRUE(result.success);
    TEST_ASSERT_EQUAL_STRING("", result.error_message.c_str());

    // Verify processing time meets requirements
    uint64_t processing_time_us = 
        (timer_end.tv_sec - timer_start.tv_sec) * 1000000 +
        (timer_end.tv_nsec - timer_start.tv_nsec) / 1000;
    TEST_ASSERT_LESS_OR_EQUAL(MAX_PROCESSING_TIME_MS * 1000, processing_time_us);

    // Verify performance metrics
    auto metrics = processor->get_performance_metrics();
    TEST_ASSERT_GREATER_THAN(0, metrics.points_processed);
    TEST_ASSERT_EQUAL(0, metrics.processing_errors);
}

void test_environment_classification(void) {
    // Process multiple test iterations for classification accuracy
    size_t successful_classifications = 0;
    
    for (size_t i = 0; i < TEST_ITERATIONS; i++) {
        // Generate varied test environments
        for (size_t j = 0; j < TEST_POINT_CLOUD_SIZE; j++) {
            test_points[j].x = (float)(rand() % 5000) / 100.0f;
            test_points[j].y = (float)(rand() % 5000) / 100.0f;
            test_points[j].z = (float)(rand() % 5000) / 100.0f;
        }
        memcpy(test_raw_data, test_points, TEST_POINT_CLOUD_SIZE * sizeof(float3));

        auto result = processor->process_scan(test_raw_data, TEST_POINT_CLOUD_SIZE * sizeof(float3));
        if (result.success) {
            successful_classifications++;
        }
    }

    // Verify classification success rate
    float success_rate = (float)successful_classifications / TEST_ITERATIONS;
    TEST_ASSERT_GREATER_OR_EQUAL(MIN_CLASSIFICATION_CONFIDENCE, success_rate);
}

void test_processing_error_handling(void) {
    // Test null input
    auto result = processor->process_scan(nullptr, TEST_POINT_CLOUD_SIZE);
    TEST_ASSERT_FALSE(result.success);
    TEST_ASSERT_NOT_EQUAL_STRING("", result.error_message.c_str());

    // Test invalid size
    result = processor->process_scan(test_raw_data, 0);
    TEST_ASSERT_FALSE(result.success);
    TEST_ASSERT_NOT_EQUAL_STRING("", result.error_message.c_str());

    // Test oversized input
    result = processor->process_scan(test_raw_data, SIZE_MAX);
    TEST_ASSERT_FALSE(result.success);
    TEST_ASSERT_NOT_EQUAL_STRING("", result.error_message.c_str());

    // Verify error metrics
    auto metrics = processor->get_performance_metrics();
    TEST_ASSERT_GREATER_THAN(0, metrics.processing_errors);
}

int main(void) {
    srand(time(NULL));
    
    UNITY_BEGIN();
    
    RUN_TEST(test_lidar_processor_initialization);
    RUN_TEST(test_point_cloud_processing);
    RUN_TEST(test_environment_classification);
    RUN_TEST(test_processing_error_handling);
    
    return UNITY_END();
}