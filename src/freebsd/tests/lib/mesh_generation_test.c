/**
 * @file mesh_generation_test.c
 * @version 1.0.0
 * @brief Unit test suite for TALD UNIA platform's mesh generation system
 * @copyright TALD UNIA Platform
 */

#include <unity.h>
#include <cuda.h>
#include "mesh_generation.h"
#include "point_cloud.h"

// Test configuration constants
#define TEST_POINT_CLOUD_SIZE 1000000
#define TEST_MESH_RESOLUTION_MM 1.0f
#define PERFORMANCE_TEST_ITERATIONS 100
#define MAX_GENERATION_TIME_MS 33.3f
#define MAX_PROCESSING_TIME_MS 50.0f
#define LOD_LEVELS 4
#define CONCURRENT_STREAMS 4
#define GPU_MEMORY_THRESHOLD 0.9f

// Test fixtures
static tald::cuda::CudaWrapper* cuda_wrapper = nullptr;
static tald::lidar::PointCloud* test_point_cloud = nullptr;
static tald::mesh::MeshGenerator* mesh_generator = nullptr;
static float3* test_points = nullptr;
static cudaEvent_t start_event, stop_event;

void setUp(void) {
    // Initialize CUDA runtime
    cudaDeviceProp prop;
    TEST_ASSERT_EQUAL(cudaSuccess, cudaGetDeviceProperties(&prop, 0));
    TEST_ASSERT_TRUE_MESSAGE(prop.major >= 7, "GPU compute capability too low");

    // Create CUDA wrapper with test configuration
    tald::cuda::StreamConfig stream_config;
    stream_config.stream_count = CONCURRENT_STREAMS;
    stream_config.enable_priority = true;
    
    tald::cuda::MemoryConfig memory_config;
    memory_config.enable_tracking = true;
    memory_config.reserved_size = TEST_POINT_CLOUD_SIZE * sizeof(float3);

    cuda_wrapper = new tald::cuda::CudaWrapper(0, stream_config, memory_config);
    TEST_ASSERT_NOT_NULL(cuda_wrapper);

    // Initialize test point cloud
    test_point_cloud = new tald::lidar::PointCloud(cuda_wrapper, TEST_POINT_CLOUD_SIZE);
    TEST_ASSERT_NOT_NULL(test_point_cloud);

    // Setup mesh generator with test configuration
    tald::mesh::MeshQualitySettings quality_settings;
    quality_settings.target_edge_length_mm = TEST_MESH_RESOLUTION_MM;
    quality_settings.optimization_iterations = 3;
    quality_settings.enable_hole_filling = true;

    tald::mesh::PhysicsConfig physics_config;
    physics_config.generate_convex_decomposition = true;
    physics_config.max_convex_pieces = 32;

    mesh_generator = new tald::mesh::MeshGenerator(cuda_wrapper, quality_settings, physics_config);
    TEST_ASSERT_NOT_NULL(mesh_generator);

    // Allocate and initialize test points
    TEST_ASSERT_EQUAL(cudaSuccess, 
        cudaMallocHost(&test_points, TEST_POINT_CLOUD_SIZE * sizeof(float3)));
    
    // Create timing events
    TEST_ASSERT_EQUAL(cudaSuccess, cudaEventCreate(&start_event));
    TEST_ASSERT_EQUAL(cudaSuccess, cudaEventCreate(&stop_event));
}

void tearDown(void) {
    // Clean up mesh generator
    if (mesh_generator) {
        tald::mesh::destroy_mesh_generator(mesh_generator);
        mesh_generator = nullptr;
    }

    // Clean up point cloud
    if (test_point_cloud) {
        tald::lidar::destroy_point_cloud(test_point_cloud);
        test_point_cloud = nullptr;
    }

    // Clean up CUDA resources
    if (test_points) {
        cudaFreeHost(test_points);
        test_points = nullptr;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    delete cuda_wrapper;
    cuda_wrapper = nullptr;

    // Verify no memory leaks
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    TEST_ASSERT_GREATER_THAN(total_mem * GPU_MEMORY_THRESHOLD, free_mem);
}

void test_mesh_generation_performance(void) {
    // Generate test point cloud data
    for (size_t i = 0; i < TEST_POINT_CLOUD_SIZE; i++) {
        test_points[i].x = static_cast<float>(rand()) / RAND_MAX * 5000.0f;
        test_points[i].y = static_cast<float>(rand()) / RAND_MAX * 5000.0f;
        test_points[i].z = static_cast<float>(rand()) / RAND_MAX * 5000.0f;
    }

    TEST_ASSERT_TRUE(test_point_cloud->add_points(test_points, TEST_POINT_CLOUD_SIZE));

    // Performance test over multiple iterations
    float total_generation_time = 0.0f;
    float total_processing_time = 0.0f;
    
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS; i++) {
        float generation_time, processing_time;
        
        cudaEventRecord(start_event);
        tald::mesh::MeshResult result = mesh_generator->generate_mesh(
            test_point_cloud, nullptr);
        cudaEventRecord(stop_event);
        
        TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_message.c_str());
        
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&generation_time, start_event, stop_event);
        
        // Verify generation time meets 30Hz requirement
        TEST_ASSERT_LESS_OR_EQUAL_FLOAT(MAX_GENERATION_TIME_MS, generation_time);
        
        // Verify total processing time including optimization
        processing_time = result.metrics.generation_time_ms + 
                         result.metrics.optimization_time_ms;
        TEST_ASSERT_LESS_OR_EQUAL_FLOAT(MAX_PROCESSING_TIME_MS, processing_time);
        
        total_generation_time += generation_time;
        total_processing_time += processing_time;
        
        // Verify mesh quality
        TEST_ASSERT_GREATER_THAN_FLOAT(0.8f, result.quality_score);
    }

    // Verify average performance
    float avg_generation_time = total_generation_time / PERFORMANCE_TEST_ITERATIONS;
    float avg_processing_time = total_processing_time / PERFORMANCE_TEST_ITERATIONS;
    
    TEST_ASSERT_LESS_OR_EQUAL_FLOAT(MAX_GENERATION_TIME_MS, avg_generation_time);
    TEST_ASSERT_LESS_OR_EQUAL_FLOAT(MAX_PROCESSING_TIME_MS, avg_processing_time);
}

void test_physics_mesh_generation(void) {
    // Generate base mesh first
    tald::mesh::MeshResult result = mesh_generator->generate_mesh(
        test_point_cloud, nullptr);
    TEST_ASSERT_TRUE(result.success);

    // Get physics mesh and verify properties
    const thrust::device_vector<float3>* physics_mesh = 
        mesh_generator->get_physics_mesh(0);
    TEST_ASSERT_NOT_NULL(physics_mesh);

    // Verify physics mesh size is appropriate for collision detection
    TEST_ASSERT_LESS_THAN(result.metrics.vertex_count, physics_mesh->size());
    
    // Verify physics mesh generation time
    TEST_ASSERT_LESS_OR_EQUAL_FLOAT(
        MAX_PROCESSING_TIME_MS, 
        result.metrics.physics_generation_time_ms);
}

void test_lod_mesh_generation(void) {
    // Generate base mesh
    tald::mesh::MeshResult result = mesh_generator->generate_mesh(
        test_point_cloud, nullptr);
    TEST_ASSERT_TRUE(result.success);

    // Test each LOD level
    size_t previous_vertex_count = result.metrics.vertex_count;
    
    for (uint32_t level = 1; level < LOD_LEVELS; level++) {
        const tald::mesh::LODMesh* lod_mesh = mesh_generator->get_lod_mesh(level);
        TEST_ASSERT_NOT_NULL(lod_mesh);

        // Verify decreasing complexity
        TEST_ASSERT_LESS_THAN(lod_mesh->vertex_count, previous_vertex_count);
        previous_vertex_count = lod_mesh->vertex_count;

        // Verify LOD mesh quality
        TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, lod_mesh->detail_level);
        TEST_ASSERT_LESS_OR_EQUAL_FLOAT(1.0f, lod_mesh->detail_level);
    }
}

void test_concurrent_mesh_generation(void) {
    const size_t points_per_cloud = TEST_POINT_CLOUD_SIZE / CONCURRENT_STREAMS;
    std::vector<tald::lidar::PointCloud*> point_clouds;
    std::vector<tald::mesh::MeshResult> results;

    // Create multiple point clouds
    for (int i = 0; i < CONCURRENT_STREAMS; i++) {
        auto cloud = new tald::lidar::PointCloud(cuda_wrapper, points_per_cloud);
        TEST_ASSERT_NOT_NULL(cloud);
        
        // Add offset points to each cloud
        for (size_t j = 0; j < points_per_cloud; j++) {
            test_points[j].x += i * 5000.0f;
        }
        TEST_ASSERT_TRUE(cloud->add_points(test_points, points_per_cloud));
        point_clouds.push_back(cloud);
    }

    // Test concurrent mesh generation
    float max_generation_time = 0.0f;
    cudaEventRecord(start_event);

    for (auto cloud : point_clouds) {
        results.push_back(mesh_generator->generate_mesh(cloud, nullptr));
    }

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&max_generation_time, start_event, stop_event);

    // Verify all generations succeeded within time constraints
    TEST_ASSERT_LESS_OR_EQUAL_FLOAT(
        MAX_PROCESSING_TIME_MS * CONCURRENT_STREAMS, 
        max_generation_time);

    for (const auto& result : results) {
        TEST_ASSERT_TRUE(result.success);
    }

    // Cleanup
    for (auto cloud : point_clouds) {
        tald::lidar::destroy_point_cloud(cloud);
    }
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_mesh_generation_performance);
    RUN_TEST(test_physics_mesh_generation);
    RUN_TEST(test_lod_mesh_generation);
    RUN_TEST(test_concurrent_mesh_generation);
    
    return UNITY_END();
}