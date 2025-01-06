/**
 * @file lidar_processing.h
 * @version 1.0.0
 * @brief Core header file for real-time LiDAR data processing in TALD UNIA platform
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_LIDAR_PROCESSING_H
#define TALD_LIDAR_PROCESSING_H

// External dependencies with versions
#include <cuda.h>                    // CUDA 12.0
#include <thrust/device_vector.h>    // CUDA 12.0
#include <NvInfer.h>                 // TensorRT 8.6

// Standard library includes
#include <atomic>
#include <mutex>
#include <functional>
#include <string>
#include <memory>

// Internal dependencies
#include "point_cloud.h"
#include "cuda_wrapper.h"
#include "tensorrt_wrapper.h"

namespace tald {
namespace lidar {

// Version and configuration constants
constexpr const char* LIDAR_PROCESSING_VERSION = "1.0.0";
constexpr int SCAN_FREQUENCY_HZ = 30;
constexpr float MIN_RESOLUTION_MM = 0.1f;
constexpr float MAX_RANGE_MM = 5000.0f;
constexpr size_t PROCESSING_BATCH_SIZE = 32768;
constexpr int MAX_ERROR_RETRIES = 3;
constexpr uint32_t PERFORMANCE_MONITORING_INTERVAL_MS = 1000;
constexpr uint32_t HEALTH_CHECK_INTERVAL_MS = 5000;

/**
 * @brief Configuration for LiDAR processing
 */
struct LidarConfig {
    float scan_frequency{SCAN_FREQUENCY_HZ};
    float min_resolution{MIN_RESOLUTION_MM};
    float max_range{MAX_RANGE_MM};
    size_t batch_size{PROCESSING_BATCH_SIZE};
    bool enable_optimization{true};
    bool enable_monitoring{true};
    uint32_t monitoring_interval{PERFORMANCE_MONITORING_INTERVAL_MS};
};

/**
 * @brief Performance metrics for monitoring
 */
struct PerformanceMetrics {
    std::atomic<float> processing_latency_ms{0.0f};
    std::atomic<size_t> points_processed{0};
    std::atomic<size_t> processing_errors{0};
    std::atomic<float> gpu_utilization{0.0f};
    std::atomic<float> memory_utilization{0.0f};
    std::atomic<uint32_t> frames_processed{0};
};

/**
 * @brief Health status of the LiDAR processor
 */
struct HealthStatus {
    bool is_healthy{true};
    bool gpu_available{true};
    bool inference_available{true};
    std::string error_message;
    float processing_load{0.0f};
};

/**
 * @brief Error callback type for error handling
 */
using ErrorCallback = std::function<void(const std::string&, int)>;

/**
 * @brief Result template for error handling
 */
template<typename T>
class Result {
public:
    T value;
    bool success;
    std::string error_message;

    static Result<T> ok(T val) {
        return Result<T>{std::move(val), true, ""};
    }

    static Result<T> error(std::string msg) {
        return Result<T>{T{}, false, std::move(msg)};
    }
};

/**
 * @brief Core class for LiDAR processing with comprehensive error handling
 */
class LidarProcessor {
public:
    /**
     * @brief Constructor with enhanced error handling
     * @param cuda_wrapper CUDA wrapper instance
     * @param tensorrt_wrapper TensorRT wrapper instance
     * @param config Processing configuration
     * @param error_callback Error handling callback
     * @throws std::runtime_error if initialization fails
     */
    LidarProcessor(cuda::CudaWrapper* cuda_wrapper,
                  ai::TensorRTWrapper* tensorrt_wrapper,
                  const LidarConfig& config,
                  ErrorCallback error_callback);

    // Prevent copying
    LidarProcessor(const LidarProcessor&) = delete;
    LidarProcessor& operator=(const LidarProcessor&) = delete;

    /**
     * @brief Thread-safe processing of raw LiDAR data
     * @param raw_data Pointer to raw LiDAR data
     * @param data_size Size of raw data
     * @return Processing result with error handling
     */
    [[nodiscard]]
    Result<bool> process_scan(const uint8_t* raw_data, size_t data_size);

    /**
     * @brief Retrieves current health status
     * @return Current health state
     */
    [[nodiscard]]
    HealthStatus get_health_status() const;

    /**
     * @brief Retrieves current performance metrics
     * @return Copy of performance metrics
     */
    [[nodiscard]]
    PerformanceMetrics get_performance_metrics() const;

private:
    std::mutex mutex;
    cuda::CudaWrapper* cuda_wrapper;
    ai::TensorRTWrapper* tensorrt_wrapper;
    std::unique_ptr<PointCloud> current_scan;
    LidarConfig config;
    thrust::device_vector<float3> raw_points;
    PerformanceMetrics metrics;
    ErrorCallback error_handler;
    std::atomic<bool> is_healthy{true};

    bool validate_input(const uint8_t* data, size_t size);
    bool process_point_cloud();
    bool run_inference();
    void update_metrics();
    void check_health();
    void handle_error(const std::string& message, int error_code);
};

/**
 * @brief Initializes LiDAR processor with error handling
 * @param cuda_wrapper CUDA wrapper instance
 * @param tensorrt_wrapper TensorRT wrapper instance
 * @param config Processing configuration
 * @param error_callback Error handling callback
 * @return Result containing processor instance or error
 */
[[nodiscard]]
Result<LidarProcessor*> initialize_lidar_processor(
    cuda::CudaWrapper* cuda_wrapper,
    ai::TensorRTWrapper* tensorrt_wrapper,
    const LidarConfig& config,
    ErrorCallback error_callback);

} // namespace lidar
} // namespace tald

#endif // TALD_LIDAR_PROCESSING_H