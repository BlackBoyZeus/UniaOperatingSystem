/**
 * @file point_cloud.h
 * @version 1.0.0
 * @brief Core point cloud data structure and processing interfaces for TALD UNIA platform's LiDAR system
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_POINT_CLOUD_H
#define TALD_POINT_CLOUD_H

// External dependencies with versions
#include <cuda.h>                    // CUDA 12.0
#include <thrust/device_vector.h>    // CUDA 12.0
#include <sys/types.h>              // FreeBSD 9.0

// Internal dependencies
#include "cuda_wrapper.h"

// Version and configuration constants
#define POINT_CLOUD_VERSION "1.0.0"
#define MAX_POINTS_PER_CLOUD 1000000
#define MIN_POINT_DISTANCE_MM 0.1f
#define MAX_POINT_DISTANCE_MM 5000.0f
#define DEFAULT_BATCH_SIZE 32768
#define OPTIMIZATION_THRESHOLD 0.5f
#define MAX_CONCURRENT_OPERATIONS 4

namespace tald {
namespace lidar {

/**
 * @brief Core class for managing and processing 3D point cloud data with GPU acceleration
 */
class PointCloud {
public:
    /**
     * @brief Initializes point cloud with GPU resources and optimization settings
     * @param cuda_wrapper Pointer to CUDA wrapper instance
     * @param initial_capacity Initial point capacity
     * @throws std::runtime_error if initialization fails
     */
    PointCloud(cuda::CudaWrapper* cuda_wrapper, size_t initial_capacity);

    // Prevent copying to avoid GPU resource conflicts
    PointCloud(const PointCloud&) = delete;
    PointCloud& operator=(const PointCloud&) = delete;

    /**
     * @brief Adds new points to the cloud with GPU acceleration
     * @param new_points Array of 3D points to add
     * @param count Number of points to add
     * @param confidence_values Optional confidence values for points
     * @return Success status of operation
     */
    [[nodiscard]]
    bool add_points(const float3* new_points, 
                   size_t count,
                   const float* confidence_values = nullptr);

    /**
     * @brief Removes points based on mask with GPU acceleration
     * @param removal_mask Boolean mask indicating points to remove
     * @param mask_size Size of the mask array
     * @return Number of points removed
     */
    [[nodiscard]]
    size_t remove_points(const bool* removal_mask, size_t mask_size);

    /**
     * @brief Optimizes point cloud using GPU acceleration
     * @param min_distance Minimum distance between points
     * @param confidence_threshold Minimum confidence value
     * @return Success status of optimization
     */
    [[nodiscard]]
    bool optimize(float min_distance = MIN_POINT_DISTANCE_MM,
                 float confidence_threshold = OPTIMIZATION_THRESHOLD);

    /**
     * @brief Retrieves points from GPU memory
     * @param output_buffer Buffer for point data
     * @param buffer_size Size of output buffer
     * @param min_confidence Minimum confidence threshold
     * @return Number of points copied
     */
    [[nodiscard]]
    size_t get_points(float3* output_buffer,
                     size_t buffer_size,
                     float min_confidence = 0.0f);

    /**
     * @brief Gets current point count
     * @return Number of points in cloud
     */
    [[nodiscard]]
    size_t get_point_count() const { return num_points; }

    /**
     * @brief Gets cloud capacity
     * @return Maximum number of points supported
     */
    [[nodiscard]]
    size_t get_capacity() const { return capacity; }

    /**
     * @brief Gets cloud bounds
     * @return Pair of min and max bounds
     */
    [[nodiscard]]
    std::pair<float3, float3> get_bounds() const {
        return {bounds_min, bounds_max};
    }

private:
    thrust::device_vector<float3> points;
    thrust::device_vector<float3> normals;
    thrust::device_vector<float> confidences;
    cuda::CudaWrapper* cuda_wrapper;
    size_t num_points{0};
    size_t capacity{0};
    float3 bounds_min{};
    float3 bounds_max{};
    cudaStream_t processing_stream{};
    bool optimization_enabled{true};

    bool resize_storage(size_t new_capacity);
    void update_bounds(const float3* new_points, size_t count);
    bool validate_points(const float3* points, size_t count);
    void calculate_normals(size_t start_idx, size_t count);
};

/**
 * @brief Creates a new point cloud instance
 * @param initial_capacity Initial point capacity
 * @param cuda_wrapper CUDA wrapper instance
 * @return Newly created point cloud instance
 */
[[nodiscard]]
PointCloud* create_point_cloud(size_t initial_capacity,
                             cuda::CudaWrapper* cuda_wrapper);

/**
 * @brief Destroys a point cloud instance
 * @param cloud Point cloud to destroy
 */
void destroy_point_cloud(PointCloud* cloud);

} // namespace lidar
} // namespace tald

#endif // TALD_POINT_CLOUD_H