/**
 * @file ai_classifier_test.c
 * @version 1.0.0
 * @brief Comprehensive unit test suite for AI-driven environment classification system
 * @copyright TALD UNIA Platform
 */

#include <unity.h>  // Unity Test Framework v2.5.2
#include <stdlib.h> // FreeBSD 9.0
#include "game/ai/environment_classifier.h"
#include "lib/libtald/tensorrt_wrapper.h"
#include "lib/libtald/cuda_wrapper.h"

using namespace tald::ai;
using namespace tald::cuda;
using namespace tald::lidar;

// Test configuration constants
#define TEST_MODEL_PATH "/usr/local/share/tald/models/classifier.engine"
#define TEST_BATCH_SIZE 32
#define TEST_CONFIDENCE_THRESHOLD 0.85f
#define MAX_PROCESSING_TIME_MS 50
#define TEST_MEMORY_LIMIT_MB 512

// Global test fixtures
static EnvironmentClassifier* classifier = nullptr;
static TensorRTWrapper* tensorrt = nullptr;
static CudaWrapper* cuda_wrapper = nullptr;
static PointCloud* test_point_cloud = nullptr;

/**
 * @brief Enhanced test fixture setup with comprehensive initialization
 */
void setUp(void) {
    // Initialize CUDA wrapper with test configuration
    cuda_wrapper = new CudaWrapper(0, StreamConfig{
        .stream_count = 2,
        .enable_priority = true
    });
    TEST_ASSERT_NOT_NULL(cuda_wrapper);

    // Initialize TensorRT with test configuration
    TensorRTConfig tensorrt_config{
        .max_batch_size = TEST_BATCH_SIZE,
        .enable_fp16 = true,
        .enable_profiling = true
    };
    tensorrt = new TensorRTWrapper(cuda_wrapper, TEST_MODEL_PATH, tensorrt_config);
    TEST_ASSERT_NOT_NULL(tensorrt);

    // Initialize classifier with test configuration
    ClassifierConfig classifier_config{
        .confidence_threshold = TEST_CONFIDENCE_THRESHOLD,
        .batch_size = TEST_BATCH_SIZE,
        .enable_profiling = true
    };
    classifier = new EnvironmentClassifier(std::unique_ptr<TensorRTWrapper>(tensorrt), classifier_config);
    TEST_ASSERT_NOT_NULL(classifier);

    // Initialize test point cloud
    test_point_cloud = create_point_cloud(1000, cuda_wrapper);
    TEST_ASSERT_NOT_NULL(test_point_cloud);
}

/**
 * @brief Enhanced test fixture cleanup with resource verification
 */
void tearDown(void) {
    // Clean up resources in reverse order of creation
    if (test_point_cloud) {
        destroy_point_cloud(test_point_cloud);
        test_point_cloud = nullptr;
    }

    if (classifier) {
        delete classifier;
        classifier = nullptr;
    }

    // TensorRT is managed by classifier's unique_ptr
    tensorrt = nullptr;

    if (cuda_wrapper) {
        delete cuda_wrapper;
        cuda_wrapper = nullptr;
    }
}

/**
 * @brief Tests classifier initialization with comprehensive validation
 */
void test_classifier_initialization(void) {
    // Verify classifier instance
    TEST_ASSERT_NOT_NULL(classifier);

    // Verify TensorRT model loading
    TEST_ASSERT_EQUAL(tensorrt->getStatus(), TensorRTStatus::SUCCESS);

    // Verify hardware compatibility
    const auto& device_props = cuda_wrapper->get_device_properties();
    TEST_ASSERT_TRUE(device_props.major >= 7);

    // Verify memory allocation
    auto [total_mem, available_mem] = cuda_wrapper->get_memory_stats();
    TEST_ASSERT_TRUE(available_mem >= (TEST_MEMORY_LIMIT_MB * 1024 * 1024));
}

/**
 * @brief Tests scene classification with performance validation
 */
void test_scene_classification(void) {
    // Prepare test point cloud data
    float3 test_points[100];
    for (int i = 0; i < 100; i++) {
        test_points[i] = {float(i), float(i), float(i)};
    }
    TEST_ASSERT_TRUE(test_point_cloud->add_points(test_points, 100));

    // Perform classification
    auto result = classifier->classify_scene(test_point_cloud);
    TEST_ASSERT_TRUE(result.has_value());

    // Verify performance requirements
    const auto stats = classifier->get_performance_stats();
    TEST_ASSERT_TRUE(stats.average_inference_time_ms <= MAX_PROCESSING_TIME_MS);
    TEST_ASSERT_TRUE(stats.successful_inferences > 0);
}

/**
 * @brief Tests batch processing with strict performance requirements
 */
void test_batch_processing(void) {
    // Prepare multiple point cloud batches
    const int num_batches = 3;
    std::vector<PointCloud*> test_clouds;
    
    for (int b = 0; b < num_batches; b++) {
        auto cloud = create_point_cloud(1000, cuda_wrapper);
        TEST_ASSERT_NOT_NULL(cloud);
        
        float3 batch_points[100];
        for (int i = 0; i < 100; i++) {
            batch_points[i] = {float(i), float(i), float(b)};
        }
        TEST_ASSERT_TRUE(cloud->add_points(batch_points, 100));
        test_clouds.push_back(cloud);
    }

    // Process all batches
    for (auto cloud : test_clouds) {
        auto result = classifier->classify_scene(cloud);
        TEST_ASSERT_TRUE(result.has_value());
        
        const auto& objects = result.value();
        TEST_ASSERT_TRUE(objects.size() > 0);
    }

    // Verify batch processing performance
    const auto stats = classifier->get_performance_stats();
    TEST_ASSERT_TRUE(stats.average_inference_time_ms <= MAX_PROCESSING_TIME_MS);
    TEST_ASSERT_EQUAL(stats.total_inferences, num_batches);

    // Cleanup test clouds
    for (auto cloud : test_clouds) {
        destroy_point_cloud(cloud);
    }
}

/**
 * @brief Tests confidence threshold validation with boundary testing
 */
void test_confidence_thresholds(void) {
    // Prepare test point cloud
    float3 test_points[100];
    float confidences[100];
    for (int i = 0; i < 100; i++) {
        test_points[i] = {float(i), float(i), float(i)};
        confidences[i] = float(i) / 100.0f;
    }
    TEST_ASSERT_TRUE(test_point_cloud->add_points(test_points, 100, confidences));

    // Test different confidence thresholds
    const float thresholds[] = {0.5f, 0.75f, 0.85f, 0.95f};
    for (float threshold : thresholds) {
        ClassifierConfig config{
            .confidence_threshold = threshold,
            .batch_size = TEST_BATCH_SIZE
        };
        
        auto result = classifier->classify_scene(test_point_cloud);
        TEST_ASSERT_TRUE(result.has_value());
        
        const auto& objects = result.value();
        for (const auto& obj : objects) {
            TEST_ASSERT_TRUE(obj.confidence >= threshold);
        }
    }
}

/**
 * @brief Main test runner with comprehensive reporting
 */
int runAllTests(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_classifier_initialization);
    RUN_TEST(test_scene_classification);
    RUN_TEST(test_batch_processing);
    RUN_TEST(test_confidence_thresholds);
    
    return UNITY_END();
}

/**
 * @brief Main entry point for test execution
 */
int main(void) {
    return runAllTests();
}