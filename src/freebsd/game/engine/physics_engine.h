/**
 * @file physics_engine.h
 * @version 1.0.0
 * @brief GPU-accelerated physics engine system for TALD UNIA platform with fleet support
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_PHYSICS_ENGINE_H
#define TALD_PHYSICS_ENGINE_H

// External dependencies with versions
#include <cuda.h>                    // CUDA 12.0
#include <thrust/device_vector.h>    // CUDA 12.0

// Internal dependencies
#include "mesh_generation.h"

// Global constants for physics configuration
#define PHYSICS_MAX_BODIES 10000      // Maximum number of rigid bodies supported across 32-device fleet
#define PHYSICS_MAX_CONSTRAINTS 20000  // Maximum number of physics constraints for complex interactions
#define PHYSICS_TIMESTEP 0.016667f    // Fixed timestep for 60 FPS physics simulation
#define PHYSICS_ITERATIONS 8          // Number of constraint solver iterations per physics step

namespace tald {
namespace physics {

/**
 * @brief Structure for rigid body physical properties
 */
struct RigidBodyDesc {
    float mass{1.0f};
    float3 inertia{1.0f, 1.0f, 1.0f};
    float restitution{0.5f};
    float friction{0.5f};
    bool is_static{false};
    bool enable_sleeping{true};
};

/**
 * @brief Performance statistics for physics simulation
 */
struct PhysicsStats {
    float simulation_time_ms{0.0f};
    float collision_time_ms{0.0f};
    float constraint_time_ms{0.0f};
    uint32_t active_bodies{0};
    uint32_t active_constraints{0};
    float memory_usage_mb{0.0f};
};

/**
 * @brief Handle type for rigid bodies
 */
using body_handle_t = uint32_t;

/**
 * @brief Core physics simulation class with GPU acceleration and fleet support
 */
class PhysicsWorld {
public:
    /**
     * @brief Initializes physics world with GPU acceleration
     * @param cuda_wrapper Pointer to CUDA wrapper instance
     * @throws std::runtime_error if initialization fails
     */
    explicit PhysicsWorld(cuda::CudaWrapper* cuda_wrapper);

    // Prevent copying to avoid GPU resource conflicts
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    /**
     * @brief Advances physics simulation by one timestep
     * @param dt Time step in seconds
     */
    void simulate(float dt);

    /**
     * @brief Adds a new rigid body to the physics world
     * @param desc Rigid body description
     * @return Handle to the created rigid body
     */
    [[nodiscard]]
    body_handle_t add_rigid_body(const RigidBodyDesc* desc);

    /**
     * @brief Updates collision geometry from LiDAR scan
     * @param mesh Mesh data from LiDAR scan
     * @return Success status of update
     */
    [[nodiscard]]
    bool update_collision_mesh(const mesh::MeshData* mesh);

    /**
     * @brief Gets current performance statistics
     * @return const reference to physics statistics
     */
    [[nodiscard]]
    const PhysicsStats& get_stats() const { return performance_stats; }

private:
    cuda::CudaWrapper* cuda_wrapper;
    thrust::device_vector<RigidBody> bodies;
    thrust::device_vector<Constraint> constraints;
    CollisionDetector* collision_detector;
    PhysicsStats performance_stats;

    bool initialize_gpu_resources();
    void update_broad_phase();
    void solve_constraints();
    void integrate_bodies();
    void update_performance_stats();
};

/**
 * @brief Initializes the physics engine system
 * @param cuda_wrapper CUDA wrapper instance
 * @return Status code indicating initialization success/failure
 */
[[nodiscard]]
int init_physics_engine(cuda::CudaWrapper* cuda_wrapper);

/**
 * @brief Cleans up physics engine resources
 */
void cleanup_physics_engine();

} // namespace physics
} // namespace tald

#endif // TALD_PHYSICS_ENGINE_H