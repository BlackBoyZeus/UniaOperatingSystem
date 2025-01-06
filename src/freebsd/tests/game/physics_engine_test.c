/**
 * @file physics_engine_test.c
 * @version 1.0.0
 * @brief Unit test suite for TALD UNIA physics engine system
 * @copyright TALD UNIA Platform
 */

#include <unity.h>  // v2.5.2
#include <cuda.h>   // v12.0
#include "game/engine/physics_engine.h"
#include "lib/libtald/cuda_wrapper.h"

// Test configuration constants
#define TEST_PHYSICS_TIMESTEP 0.016667f
#define TEST_ITERATIONS 100
#define TEST_TOLERANCE 0.0001f
#define MAX_FLEET_SIZE 32
#define GPU_MEMORY_THRESHOLD 4294967296  // 4GB
#define SIMULATION_LATENCY_THRESHOLD 0.050f

// Global test variables
static tald::cuda::CudaWrapper* cuda_wrapper = nullptr;
static tald::physics::PhysicsWorld* physics_world = nullptr;
static tald::mesh::MeshData* test_mesh = nullptr;
static float3* test_vertices = nullptr;
static uint32_t test_vertex_count = 0;

/**
 * @brief Enhanced test setup with GPU validation
 */
void setUp(void) {
    // Initialize CUDA wrapper with memory tracking
    tald::cuda::StreamConfig stream_config;
    stream_config.stream_count = 4;
    stream_config.enable_priority = true;

    tald::cuda::MemoryConfig memory_config;
    memory_config.enable_tracking = true;
    memory_config.reserved_size = GPU_MEMORY_THRESHOLD;

    cuda_wrapper = new tald::cuda::CudaWrapper(0, stream_config, memory_config);
    TEST_ASSERT_NOT_NULL(cuda_wrapper);

    // Validate GPU capabilities
    const cudaDeviceProp& props = cuda_wrapper->get_device_properties();
    TEST_ASSERT_TRUE(props.major >= 7);  // Ensure Volta or newer architecture
    TEST_ASSERT_TRUE(props.totalGlobalMem >= GPU_MEMORY_THRESHOLD);

    // Initialize physics world
    physics_world = new tald::physics::PhysicsWorld(cuda_wrapper);
    TEST_ASSERT_NOT_NULL(physics_world);

    // Setup test data
    test_vertex_count = 1000;
    auto memory_handle = cuda_wrapper->allocate_device_memory(
        test_vertex_count * sizeof(float3),
        tald::cuda::MemoryFlags::DEFAULT
    );
    test_vertices = static_cast<float3*>(memory_handle.ptr);
    TEST_ASSERT_NOT_NULL(test_vertices);
}

/**
 * @brief Enhanced teardown with resource validation
 */
void tearDown(void) {
    // Log performance metrics
    const auto& stats = physics_world->get_stats();
    printf("Physics Stats - Simulation: %.2fms, Memory: %.2fMB\n",
           stats.simulation_time_ms, stats.memory_usage_mb);

    // Validate GPU cleanup
    cuda_wrapper->synchronize();
    auto [total_mem, available_mem] = cuda_wrapper->get_memory_stats();
    TEST_ASSERT_TRUE(available_mem >= total_mem * 0.95f);  // Ensure 95% memory free

    // Cleanup resources
    delete physics_world;
    delete cuda_wrapper;
    physics_world = nullptr;
    cuda_wrapper = nullptr;
    test_vertices = nullptr;
}

/**
 * @brief Tests physics engine initialization with GPU validation
 */
void test_physics_initialization(void) {
    // Verify CUDA initialization
    TEST_ASSERT_NOT_NULL(cuda_wrapper);
    TEST_ASSERT_TRUE(cuda_wrapper->get_device_properties().major >= 7);

    // Verify physics world creation
    TEST_ASSERT_NOT_NULL(physics_world);
    const auto& stats = physics_world->get_stats();
    TEST_ASSERT_EQUAL_UINT32(0, stats.active_bodies);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.0f, stats.memory_usage_mb);

    // Test GPU resource allocation
    tald::physics::RigidBodyDesc desc;
    desc.mass = 1.0f;
    desc.is_static = false;
    auto body_handle = physics_world->add_rigid_body(&desc);
    TEST_ASSERT_NOT_EQUAL(0, body_handle);

    // Verify memory tracking
    auto [total_mem, available_mem] = cuda_wrapper->get_memory_stats();
    TEST_ASSERT_TRUE(available_mem < total_mem);
}

/**
 * @brief Tests physics synchronization across fleet devices
 */
void test_fleet_physics_sync(void) {
    // Initialize test bodies for fleet sync
    std::vector<tald::physics::body_handle_t> fleet_bodies;
    for (uint32_t i = 0; i < MAX_FLEET_SIZE; ++i) {
        tald::physics::RigidBodyDesc desc;
        desc.mass = 1.0f + i;
        desc.is_static = false;
        auto handle = physics_world->add_rigid_body(&desc);
        TEST_ASSERT_NOT_EQUAL(0, handle);
        fleet_bodies.push_back(handle);
    }

    // Run distributed simulation
    float total_latency = 0.0f;
    for (uint32_t i = 0; i < TEST_ITERATIONS; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        physics_world->simulate(TEST_PHYSICS_TIMESTEP);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        float iteration_latency = std::chrono::duration<float, std::milli>(
            end_time - start_time).count();
        total_latency += iteration_latency;
        
        TEST_ASSERT_TRUE(iteration_latency < SIMULATION_LATENCY_THRESHOLD);
    }

    // Verify performance
    float avg_latency = total_latency / TEST_ITERATIONS;
    TEST_ASSERT_TRUE(avg_latency < SIMULATION_LATENCY_THRESHOLD);

    // Verify fleet state
    const auto& stats = physics_world->get_stats();
    TEST_ASSERT_EQUAL_UINT32(MAX_FLEET_SIZE, stats.active_bodies);
}

/**
 * @brief Tests physics interaction with LiDAR-generated meshes
 */
void test_lidar_mesh_physics(void) {
    // Generate test mesh data
    test_mesh = new tald::mesh::MeshData();
    TEST_ASSERT_NOT_NULL(test_mesh);

    // Initialize test vertices
    for (uint32_t i = 0; i < test_vertex_count; ++i) {
        test_vertices[i] = make_float3(
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX
        );
    }

    // Update collision mesh
    auto start_time = std::chrono::high_resolution_clock::now();
    bool update_success = physics_world->update_collision_mesh(test_mesh);
    auto end_time = std::chrono::high_resolution_clock::now();

    TEST_ASSERT_TRUE(update_success);

    // Verify mesh integration performance
    float update_time = std::chrono::duration<float, std::milli>(
        end_time - start_time).count();
    TEST_ASSERT_TRUE(update_time < SIMULATION_LATENCY_THRESHOLD);

    // Run physics simulation with mesh
    for (uint32_t i = 0; i < TEST_ITERATIONS; ++i) {
        physics_world->simulate(TEST_PHYSICS_TIMESTEP);
        cuda_wrapper->synchronize();
        
        const auto& stats = physics_world->get_stats();
        TEST_ASSERT_TRUE(stats.collision_time_ms < SIMULATION_LATENCY_THRESHOLD);
    }

    delete test_mesh;
}

/**
 * @brief Main test runner
 */
int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_physics_initialization);
    RUN_TEST(test_fleet_physics_sync);
    RUN_TEST(test_lidar_mesh_physics);
    
    return UNITY_END();
}