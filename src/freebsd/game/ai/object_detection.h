/**
 * @file object_detection.h
 * @version 1.0.0
 * @brief Real-time object detection system for TALD UNIA platform using GPU-accelerated TensorRT inference
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_OBJECT_DETECTION_H
#define TALD_OBJECT_DETECTION_H

// External dependencies with versions
#include <NvInfer.h>           // TensorRT 8.6
#include <cuda_runtime.h>      // CUDA 12.0
#include <future>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

// Internal dependencies
#include "tensorrt_wrapper.h"
#include "point_cloud.h"

namespace tald {
namespace ai {

// Version and configuration constants
constexpr const char* OBJECT_DETECTOR_VERSION = "1.0.0";
constexpr uint32_t MAX_OBJECTS = 256;
constexpr float MIN_DETECTION_CONFIDENCE = 0.90f;
constexpr uint32_t DETECTION_BATCH_SIZE = 32;
constexpr uint32_t DETECTION_TIMEOUT_MS = 50;
constexpr uint32_t MAX_TRACKING_HISTORY = 30;
constexpr uint32_t CACHE_LIFETIME_MS = 100;

/**
 * @brief Unique identifier for tracked objects
 */
using ObjectId = uint64_t;

/**
 * @brief Configuration for object detector initialization
 */
struct DetectorConfig {
    uint32_t batch_size{DETECTION_BATCH_SIZE};
    float min_confidence{MIN_DETECTION_CONFIDENCE};
    uint32_t max_objects{MAX_OBJECTS};
    bool enable_tracking{true};
    uint32_t tracking_history{MAX_TRACKING_HISTORY};
    uint32_t cache_lifetime_ms{CACHE_LIFETIME_MS};
};

/**
 * @brief Represents a detected object in 3D space
 */
struct DetectedObject {
    float3 position;           // 3D position in world space
    float3 dimensions;         // Width, height, depth
    float3 velocity;          // Current velocity vector
    float confidence;         // Detection confidence [0,1]
    uint32_t class_id;        // Object class identifier
    uint64_t timestamp;       // Detection timestamp
};

/**
 * @brief Extended information for tracked objects
 */
struct TrackedObject : DetectedObject {
    ObjectId id;              // Unique tracking identifier
    std::vector<float3> trajectory;  // Position history
    float3 predicted_position;  // Next predicted position
    uint32_t tracking_age;    // Number of frames tracked
    bool is_occluded;        // Currently occluded flag
};

/**
 * @brief Cache for recent detection results
 */
class DetectionCache {
public:
    DetectionCache(uint32_t lifetime_ms) : cache_lifetime_ms(lifetime_ms) {}
    
    void update(const std::vector<DetectedObject>& objects);
    std::optional<std::vector<DetectedObject>> get_recent();

private:
    std::mutex cache_mutex;
    std::vector<DetectedObject> cached_objects;
    uint64_t last_update{0};
    uint32_t cache_lifetime_ms;
};

/**
 * @brief Thread-safe object detection and tracking system
 */
class ObjectDetector {
public:
    /**
     * @brief Initializes detector with TensorRT backend
     * @param tensorrt Pointer to TensorRT wrapper
     * @param config Detector configuration
     * @throws std::runtime_error if initialization fails
     */
    ObjectDetector(TensorRTWrapper* tensorrt, const DetectorConfig& config);
    
    // Prevent copying
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    
    /**
     * @brief Performs object detection on point cloud
     * @param point_cloud Input point cloud data
     * @return Optional vector of detected objects
     */
    [[nodiscard]]
    std::optional<std::vector<DetectedObject>> detect_objects(
        const lidar::PointCloud* point_cloud);
    
    /**
     * @brief Updates object tracking with new detections
     * @param previous_objects Previously detected objects
     * @param current_objects Currently detected objects
     * @return Future containing tracked objects
     */
    [[nodiscard]]
    std::future<std::vector<TrackedObject>> track_objects(
        const std::vector<DetectedObject>& previous_objects,
        const std::vector<DetectedObject>& current_objects);

private:
    std::mutex detector_mutex;
    TensorRTWrapper* tensorrt;
    std::vector<DetectedObject> detected_objects;
    std::unique_ptr<float[]> detection_buffer;
    uint32_t current_batch_size;
    std::unordered_map<ObjectId, TrackedObject> tracked_objects;
    DetectionCache result_cache;
    std::atomic<bool> is_processing;

    bool preprocess_point_cloud(const lidar::PointCloud* point_cloud);
    bool run_inference();
    void postprocess_detections();
    void update_tracking_history(const TrackedObject& object);
    float3 predict_next_position(const TrackedObject& object);
};

/**
 * @brief Initializes object detector with model and configuration
 * @param model_path Path to TensorRT model file
 * @param tensorrt Pointer to TensorRT wrapper
 * @param config Detector configuration
 * @return Pointer to initialized detector, nullptr on failure
 */
[[nodiscard]]
ObjectDetector* init_object_detector(
    const char* model_path,
    TensorRTWrapper* tensorrt,
    const DetectorConfig& config);

} // namespace ai
} // namespace tald

#endif // TALD_OBJECT_DETECTION_H