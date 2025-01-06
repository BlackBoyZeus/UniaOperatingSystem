/**
 * @file environment_classifier.h
 * @version 1.0.0
 * @brief AI-driven environment classification system for real-time scene understanding
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_ENVIRONMENT_CLASSIFIER_H
#define TALD_ENVIRONMENT_CLASSIFIER_H

// External dependencies with versions
#include <NvInfer.h>           // TensorRT 8.6
#include <cuda_runtime.h>      // CUDA 12.0
#include <expected>            // C++23
#include <memory>
#include <mutex>
#include <atomic>
#include <vector>

// Internal dependencies
#include "tensorrt_wrapper.h"
#include "point_cloud.h"

namespace tald {
namespace ai {

// Version and configuration constants
constexpr const char* ENV_CLASSIFIER_VERSION = "1.0.0";
constexpr uint32_t MAX_SCENE_OBJECTS = 1024;
constexpr float MIN_CONFIDENCE_THRESHOLD = 0.85f;
constexpr uint32_t INFERENCE_BATCH_SIZE = 32;
constexpr uint32_t MAX_RETRY_ATTEMPTS = 3;
constexpr uint32_t INFERENCE_TIMEOUT_MS = 50;

/**
 * @brief Configuration for environment classifier
 */
struct ClassifierConfig {
    float confidence_threshold{MIN_CONFIDENCE_THRESHOLD};
    uint32_t batch_size{INFERENCE_BATCH_SIZE};
    bool enable_profiling{false};
    uint32_t max_objects{MAX_SCENE_OBJECTS};
    uint32_t inference_timeout_ms{INFERENCE_TIMEOUT_MS};
};

/**
 * @brief Performance statistics for classifier
 */
struct ClassifierStats {
    float average_inference_time_ms{0.0f};
    float average_preprocessing_time_ms{0.0f};
    uint32_t total_inferences{0};
    uint32_t successful_inferences{0};
    uint32_t retry_count{0};
    float gpu_memory_usage_mb{0.0f};
};

/**
 * @brief Detected object in the scene
 */
struct SceneObject {
    uint32_t class_id;
    float confidence;
    float3 position;
    float3 dimensions;
    float3 orientation;
};

/**
 * @brief Error codes for classifier operations
 */
enum class ClassifierError {
    SUCCESS,
    MODEL_LOAD_ERROR,
    INFERENCE_ERROR,
    TIMEOUT_ERROR,
    MEMORY_ERROR,
    INVALID_INPUT,
    GPU_ERROR
};

/**
 * @brief Thread-safe environment classifier using TensorRT
 */
class EnvironmentClassifier {
public:
    /**
     * @brief Initializes classifier with TensorRT engine
     * @param tensorrt Unique pointer to TensorRT wrapper
     * @param config Classifier configuration
     * @throws std::runtime_error if initialization fails
     */
    EnvironmentClassifier(std::unique_ptr<TensorRTWrapper> tensorrt,
                         const ClassifierConfig& config);

    // Prevent copying
    EnvironmentClassifier(const EnvironmentClassifier&) = delete;
    EnvironmentClassifier& operator=(const EnvironmentClassifier&) = delete;

    /**
     * @brief Thread-safe scene classification
     * @param point_cloud Input point cloud data
     * @return Expected vector of detected objects or error
     */
    [[nodiscard]]
    std::expected<std::vector<SceneObject>, ClassifierError> 
    classify_scene(const lidar::PointCloud* point_cloud);

    /**
     * @brief Retrieves current performance statistics
     * @return Current classifier statistics
     */
    [[nodiscard]]
    ClassifierStats get_performance_stats() const noexcept;

private:
    std::unique_ptr<TensorRTWrapper> tensorrt;
    std::vector<SceneObject> detected_objects;
    std::unique_ptr<float[]> inference_buffer;
    uint32_t current_batch_size{0};
    std::mutex classification_mutex;
    std::atomic<bool> is_processing{false};
    ClassifierStats performance_stats;
    ClassifierConfig config;

    bool preprocess_point_cloud(const lidar::PointCloud* point_cloud);
    bool run_inference();
    bool postprocess_results();
    void update_performance_stats(float inference_time);
    ClassifierError handle_inference_error(TensorRTStatus status);
};

/**
 * @brief Initializes environment classifier
 * @param model_path Path to TensorRT model
 * @param tensorrt TensorRT wrapper instance
 * @param config Classifier configuration
 * @return Expected pointer to classifier or error
 */
[[nodiscard]]
std::expected<EnvironmentClassifier*, ClassifierError>
init_environment_classifier(const char* model_path,
                          TensorRTWrapper* tensorrt,
                          const ClassifierConfig& config = ClassifierConfig{});

} // namespace ai
} // namespace tald

#endif // TALD_ENVIRONMENT_CLASSIFIER_H