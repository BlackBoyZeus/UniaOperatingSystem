/**
 * @file mesh_generation.h
 * @version 1.0.0
 * @brief Real-time 3D mesh generation system for TALD UNIA platform's LiDAR data
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_MESH_GENERATION_H
#define TALD_MESH_GENERATION_H

// External dependencies with versions
#include <cuda.h>                    // CUDA 12.0
#include <thrust/device_vector.h>    // CUDA 12.0
#include <sys/types.h>              // FreeBSD 9.0

// Internal dependencies
#include "point_cloud.h"
#include "cuda_wrapper.h"

// Version and configuration constants
#define MESH_GENERATION_VERSION "1.0.0"
#define MAX_VERTICES_PER_MESH 1000000
#define MAX_TRIANGLES_PER_MESH 2000000
#define MIN_TRIANGLE_AREA_MM2 1.0
#define MAX_LOD_LEVELS 4
#define VERTEX_CACHE_SIZE 32
#define MESH_UPDATE_FREQUENCY_HZ 30
#define MAX_GPU_MEMORY_MB 2048

namespace tald {
namespace mesh {

/**
 * @brief Configuration for mesh generation quality settings
 */
struct MeshQualitySettings {
    float target_edge_length_mm{10.0f};
    float max_deviation_mm{1.0f};
    float smoothing_factor{0.5f};
    uint32_t optimization_iterations{3};
    bool enable_hole_filling{true};
    float hole_filling_threshold_mm{20.0f};
};

/**
 * @brief Configuration for physics mesh generation
 */
struct PhysicsConfig {
    float collision_margin_mm{2.0f};
    uint32_t max_convex_pieces{32};
    bool generate_convex_decomposition{true};
    float simplification_threshold{0.1f};
};

/**
 * @brief Level of Detail mesh structure
 */
struct LODMesh {
    thrust::device_vector<float3> vertices;
    thrust::device_vector<float3> normals;
    thrust::device_vector<uint3> triangles;
    float detail_level;
    uint32_t vertex_count{0};
    uint32_t triangle_count{0};
};

/**
 * @brief Performance metrics for mesh generation
 */
struct PerformanceMetrics {
    float generation_time_ms{0.0f};
    float optimization_time_ms{0.0f};
    float physics_generation_time_ms{0.0f};
    uint32_t vertex_count{0};
    uint32_t triangle_count{0};
    float memory_usage_mb{0.0f};
};

/**
 * @brief Result structure for mesh generation operations
 */
struct MeshResult {
    bool success{false};
    std::string error_message;
    PerformanceMetrics metrics;
    float quality_score{0.0f};
};

/**
 * @brief Advanced class for real-time 3D mesh generation with physics support
 */
class MeshGenerator final {
public:
    /**
     * @brief Initializes mesh generator with advanced configuration
     * @param cuda_wrapper Pointer to CUDA wrapper instance
     * @param config Mesh generation configuration
     * @param physics_config Physics mesh configuration
     * @throws std::runtime_error if initialization fails
     */
    MeshGenerator(cuda::CudaWrapper* cuda_wrapper,
                 const MeshQualitySettings& config,
                 const PhysicsConfig& physics_config);

    // Prevent copying to avoid GPU resource conflicts
    MeshGenerator(const MeshGenerator&) = delete;
    MeshGenerator& operator=(const MeshGenerator&) = delete;

    /**
     * @brief Generates optimized 3D mesh from point cloud with physics support
     * @param point_cloud Input point cloud data
     * @param quality_settings Mesh quality configuration
     * @return MeshResult containing generation results and metrics
     */
    [[nodiscard]]
    MeshResult generate_mesh(const lidar::PointCloud* point_cloud,
                           const MeshQualitySettings* quality_settings);

    /**
     * @brief Retrieves physics collision mesh
     * @param lod_level LOD level for physics mesh
     * @return Pointer to physics mesh data
     */
    [[nodiscard]]
    const thrust::device_vector<float3>* get_physics_mesh(uint32_t lod_level = 0) const;

    /**
     * @brief Retrieves LOD mesh at specified level
     * @param level LOD level to retrieve
     * @return Pointer to LOD mesh data
     */
    [[nodiscard]]
    const LODMesh* get_lod_mesh(uint32_t level) const;

    /**
     * @brief Gets current performance metrics
     * @return const reference to performance metrics
     */
    [[nodiscard]]
    const PerformanceMetrics& get_metrics() const { return metrics; }

private:
    cuda::CudaWrapper* cuda_wrapper;
    thrust::device_vector<float3> vertices;
    thrust::device_vector<float3> normals;
    thrust::device_vector<uint3> triangles;
    thrust::device_vector<float3> physics_vertices;
    std::array<LODMesh, MAX_LOD_LEVELS> lod_meshes;
    PerformanceMetrics metrics;
    uint32_t num_vertices{0};
    uint32_t num_triangles{0};
    float mesh_resolution_mm{1.0f};

    bool initialize_gpu_resources();
    bool generate_initial_mesh(const lidar::PointCloud* point_cloud);
    void optimize_mesh_topology();
    void generate_physics_mesh();
    void create_lod_hierarchy();
    void update_performance_metrics();
    bool validate_mesh_quality();
    void cleanup_resources();
};

/**
 * @brief Initializes mesh generation system
 * @param cuda_wrapper CUDA wrapper instance
 * @param config Mesh generation configuration
 * @param physics_config Physics configuration
 * @return Initialized mesh generator instance
 */
[[nodiscard]]
MeshGenerator* init_mesh_generator(cuda::CudaWrapper* cuda_wrapper,
                                 const MeshQualitySettings* config,
                                 const PhysicsConfig* physics_config);

/**
 * @brief Destroys mesh generator instance
 * @param generator Mesh generator to destroy
 */
void destroy_mesh_generator(MeshGenerator* generator);

} // namespace mesh
} // namespace tald

#endif // TALD_MESH_GENERATION_H