/**
 * @file environment_classifier.c
 * @version 1.0.0
 * @brief Production implementation of AI-driven environment classification system
 * @copyright TALD UNIA Platform
 */

#include "environment_classifier.h"
#include <string.h>
#include <time.h>
#include <syslog.h>

// Version: TensorRT 8.6
#include <NvInfer.h>
// Version: CUDA 12.0
#include <cuda_runtime.h>
// Version: 1.2.0
#include "performance_monitor.h"

// Internal constants
static const char* const ENV_CLASSIFIER_LOG_PREFIX = "EnvironmentClassifier";
static const uint32_t WARMUP_ITERATIONS = 3;
static const uint32_t PERFORMANCE_WINDOW_SIZE = 100;
static const float GPU_MEMORY_WARNING_THRESHOLD = 0.85f;

// Thread-safe singleton instance
static std::atomic<EnvironmentClassifier*> g_classifier_instance{nullptr};
static std::mutex g_initialization_mutex;

EnvironmentClassifier::EnvironmentClassifier(TensorRTWrapper* tensorrt, 
                                           const ClassifierConfig& config) 
    : tensorrt(tensorrt),
      current_batch_size(0),
      is_processing(false),
      timeout_duration(std::chrono::milliseconds(config.inference_timeout_ms)),
      retry_count(0),
      total_processed_frames(0),
      error_count(0) {
    
    if (!tensorrt) {
        throw std::runtime_error("Invalid TensorRT wrapper");
    }

    // Initialize CUDA inference buffers
    size_t buffer_size = config.batch_size * sizeof(float) * 3; // XYZ coordinates
    inference_buffer = cuda_malloc_wrapper(buffer_size);
    if (!inference_buffer) {
        throw std::runtime_error("Failed to allocate inference buffer");
    }

    // Initialize performance monitoring
    perf_monitor.initialize(PERFORMANCE_WINDOW_SIZE);
}

EnvironmentClassifier::~EnvironmentClassifier() {
    std::lock_guard<std::mutex> lock(classifier_mutex);
    if (inference_buffer) {
        cuda_free_wrapper(inference_buffer);
    }
}

std::expected<std::vector<SceneObject>, std::error_code> 
EnvironmentClassifier::classify_scene(const PointCloud* point_cloud) {
    if (!point_cloud || !point_cloud->validate()) {
        return std::unexpected(make_error_code(ClassifierError::INVALID_INPUT));
    }

    // Acquire classification mutex with timeout
    std::unique_lock<std::mutex> lock(classifier_mutex, std::defer_lock);
    if (!lock.try_lock_for(timeout_duration)) {
        error_count++;
        return std::unexpected(make_error_code(ClassifierError::TIMEOUT_ERROR));
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    is_processing = true;

    try {
        // Preprocess point cloud data
        if (!preprocess_point_cloud(point_cloud)) {
            error_count++;
            return std::unexpected(make_error_code(ClassifierError::PREPROCESSING_ERROR));
        }

        // Execute TensorRT inference with retry logic
        bool inference_success = false;
        for (uint32_t attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
            auto status = tensorrt->infer(inference_buffer, 
                                        current_batch_size,
                                        timeout_duration.count());
            if (status == TensorRTStatus::SUCCESS) {
                inference_success = true;
                break;
            }
            retry_count++;
            // Reset device on critical errors
            if (status == TensorRTStatus::GPU_ERROR) {
                tensorrt->resetDevice();
            }
        }

        if (!inference_success) {
            error_count++;
            return std::unexpected(make_error_code(ClassifierError::INFERENCE_ERROR));
        }

        // Post-process results
        detected_objects.clear();
        if (!postprocess_results()) {
            error_count++;
            return std::unexpected(make_error_code(ClassifierError::POSTPROCESSING_ERROR));
        }

        // Update performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        perf_monitor.update_metrics(duration.count(), current_batch_size);
        total_processed_frames++;

        return detected_objects;

    } catch (const std::exception& e) {
        syslog(LOG_ERR, "%s: Classification error: %s", 
               ENV_CLASSIFIER_LOG_PREFIX, e.what());
        error_count++;
        return std::unexpected(make_error_code(ClassifierError::RUNTIME_ERROR));
    }

    is_processing = false;
}

ClassifierStats EnvironmentClassifier::get_performance_stats() const noexcept {
    ClassifierStats stats;
    stats.average_inference_time_ms = perf_monitor.get_average_inference_time();
    stats.total_processed_frames = total_processed_frames.load();
    stats.error_rate = error_count.load() / 
                      static_cast<float>(total_processed_frames.load());
    stats.retry_rate = retry_count / static_cast<float>(total_processed_frames.load());
    return stats;
}

bool EnvironmentClassifier::preprocess_point_cloud(const PointCloud* point_cloud) {
    const auto num_points = point_cloud->num_points;
    if (num_points == 0 || num_points > MAX_SCENE_OBJECTS) {
        return false;
    }

    // Prepare batch for inference
    current_batch_size = std::min(num_points, 
                                static_cast<size_t>(INFERENCE_BATCH_SIZE));
    
    // Copy points to inference buffer with CUDA
    auto cuda_status = cudaMemcpyAsync(
        inference_buffer,
        point_cloud->points,
        current_batch_size * sizeof(float) * 3,
        cudaMemcpyDeviceToDevice,
        cudaStreamDefault
    );

    return cuda_status == cudaSuccess;
}

bool EnvironmentClassifier::postprocess_results() {
    // Apply confidence thresholding
    for (size_t i = 0; i < current_batch_size; i++) {
        float confidence = get_object_confidence(i);
        if (confidence >= MIN_CONFIDENCE_THRESHOLD) {
            SceneObject obj;
            obj.class_id = get_object_class(i);
            obj.confidence = confidence;
            obj.position = get_object_position(i);
            obj.dimensions = get_object_dimensions(i);
            obj.orientation = get_object_orientation(i);
            detected_objects.push_back(obj);
        }
    }
    return true;
}

std::expected<EnvironmentClassifier*, std::error_code>
init_environment_classifier(const char* model_path,
                          TensorRTWrapper* tensorrt,
                          const ClassifierConfig& config) {
    if (!model_path || !tensorrt) {
        return std::unexpected(make_error_code(ClassifierError::INVALID_INPUT));
    }

    std::lock_guard<std::mutex> lock(g_initialization_mutex);

    try {
        // Check for existing instance
        auto existing = g_classifier_instance.load();
        if (existing) {
            return existing;
        }

        // Load TensorRT model
        auto status = tensorrt->loadModel(model_path, config);
        if (status != TensorRTStatus::SUCCESS) {
            syslog(LOG_ERR, "%s: Failed to load model: %s", 
                   ENV_CLASSIFIER_LOG_PREFIX, tensorrt->getLastError().c_str());
            return std::unexpected(make_error_code(ClassifierError::MODEL_LOAD_ERROR));
        }

        // Create new classifier instance
        auto classifier = new EnvironmentClassifier(tensorrt, config);

        // Perform warmup iterations
        PointCloud dummy_cloud;
        for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
            classifier->classify_scene(&dummy_cloud);
        }

        // Store instance atomically
        g_classifier_instance.store(classifier);
        
        syslog(LOG_INFO, "%s: Successfully initialized classifier", 
               ENV_CLASSIFIER_LOG_PREFIX);
        return classifier;

    } catch (const std::exception& e) {
        syslog(LOG_ERR, "%s: Initialization error: %s", 
               ENV_CLASSIFIER_LOG_PREFIX, e.what());
        return std::unexpected(make_error_code(ClassifierError::INITIALIZATION_ERROR));
    }
}