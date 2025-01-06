/**
 * @file object_detection.c
 * @version 1.0.0
 * @brief Implementation of real-time object detection system for TALD UNIA platform
 * @copyright TALD UNIA Platform
 */

#include "object_detection.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

// Version: TensorRT 8.6
#include <NvInfer.h>
// Version: CUDA 12.0
#include <cuda_runtime.h>

namespace tald {
namespace ai {

namespace {
    // Internal helper functions
    uint64_t get_timestamp_ms() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
    }

    float3 calculate_velocity(const float3& prev_pos, const float3& curr_pos, float dt) {
        return {
            (curr_pos.x - prev_pos.x) / dt,
            (curr_pos.y - prev_pos.y) / dt,
            (curr_pos.z - prev_pos.z) / dt
        };
    }
}

// DetectionCache implementation
void DetectionCache::update(const std::vector<DetectedObject>& objects) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cached_objects = objects;
    last_update = get_timestamp_ms();
}

std::optional<std::vector<DetectedObject>> DetectionCache::get_recent() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    uint64_t current_time = get_timestamp_ms();
    if (current_time - last_update <= cache_lifetime_ms) {
        return cached_objects;
    }
    return std::nullopt;
}

// ObjectDetector implementation
ObjectDetector::ObjectDetector(TensorRTWrapper* tensorrt, const DetectorConfig& config)
    : tensorrt(tensorrt),
      current_batch_size(config.batch_size),
      result_cache(config.cache_lifetime_ms),
      is_processing(false) {
    
    // Allocate detection buffer with proper alignment
    detection_buffer = std::make_unique<float[]>(
        config.batch_size * config.max_objects * 7); // x,y,z,w,h,d,conf
    
    if (!tensorrt) {
        throw std::runtime_error("Invalid TensorRT wrapper");
    }
}

std::optional<std::vector<DetectedObject>> ObjectDetector::detect_objects(
    const lidar::PointCloud* point_cloud) {
    
    if (!point_cloud || point_cloud->get_point_count() == 0) {
        return std::nullopt;
    }

    // Check cache first
    if (auto cached = result_cache.get_recent()) {
        return cached;
    }

    std::lock_guard<std::mutex> lock(detector_mutex);
    bool expected = false;
    if (!is_processing.compare_exchange_strong(expected, true)) {
        return std::nullopt;
    }

    try {
        if (!preprocess_point_cloud(point_cloud)) {
            is_processing = false;
            return std::nullopt;
        }

        if (!run_inference()) {
            is_processing = false;
            return std::nullopt;
        }

        postprocess_detections();
        result_cache.update(detected_objects);
        is_processing = false;
        return detected_objects;

    } catch (const std::exception& e) {
        is_processing = false;
        return std::nullopt;
    }
}

std::future<std::vector<TrackedObject>> ObjectDetector::track_objects(
    const std::vector<DetectedObject>& previous_objects,
    const std::vector<DetectedObject>& current_objects) {
    
    return std::async(std::launch::async, [this, previous_objects, current_objects]() {
        std::vector<TrackedObject> tracked_results;
        const float dt = 0.033f; // 30Hz update rate

        for (const auto& curr_obj : current_objects) {
            TrackedObject tracked_obj;
            tracked_obj.position = curr_obj.position;
            tracked_obj.dimensions = curr_obj.dimensions;
            tracked_obj.confidence = curr_obj.confidence;
            tracked_obj.class_id = curr_obj.class_id;
            tracked_obj.timestamp = curr_obj.timestamp;

            // Find matching object in previous frame
            auto prev_it = std::find_if(previous_objects.begin(), previous_objects.end(),
                [&curr_obj](const DetectedObject& prev) {
                    float dist = std::hypot(
                        curr_obj.position.x - prev.position.x,
                        curr_obj.position.y - prev.position.y,
                        curr_obj.position.z - prev.position.z
                    );
                    return dist < 0.5f && curr_obj.class_id == prev.class_id;
                });

            if (prev_it != previous_objects.end()) {
                tracked_obj.velocity = calculate_velocity(
                    prev_it->position, curr_obj.position, dt);
                tracked_obj.tracking_age++;
            } else {
                tracked_obj.velocity = {0.0f, 0.0f, 0.0f};
                tracked_obj.tracking_age = 1;
            }

            tracked_obj.predicted_position = {
                curr_obj.position.x + tracked_obj.velocity.x * dt,
                curr_obj.position.y + tracked_obj.velocity.y * dt,
                curr_obj.position.z + tracked_obj.velocity.z * dt
            };

            tracked_results.push_back(tracked_obj);
        }

        return tracked_results;
    });
}

bool ObjectDetector::preprocess_point_cloud(const lidar::PointCloud* point_cloud) {
    float3* points;
    size_t num_points = point_cloud->get_points(
        points, point_cloud->get_point_count(), MIN_DETECTION_CONFIDENCE);
    
    if (num_points == 0) {
        return false;
    }

    // Convert point cloud to detection input format using CUDA
    auto status = tensorrt->infer(points, detection_buffer.get(), 
        current_batch_size, DETECTION_TIMEOUT_MS);
    
    return status == TensorRTStatus::SUCCESS;
}

bool ObjectDetector::run_inference() {
    auto status = tensorrt->infer(detection_buffer.get(), 
        detection_buffer.get(), current_batch_size, DETECTION_TIMEOUT_MS);
    
    return status == TensorRTStatus::SUCCESS;
}

void ObjectDetector::postprocess_detections() {
    detected_objects.clear();
    const float* results = detection_buffer.get();
    
    for (uint32_t i = 0; i < current_batch_size; i++) {
        const float* obj_data = results + i * 7;
        float confidence = obj_data[6];
        
        if (confidence >= MIN_DETECTION_CONFIDENCE) {
            DetectedObject obj;
            obj.position = {obj_data[0], obj_data[1], obj_data[2]};
            obj.dimensions = {obj_data[3], obj_data[4], obj_data[5]};
            obj.confidence = confidence;
            obj.timestamp = get_timestamp_ms();
            detected_objects.push_back(obj);
        }
    }
}

ObjectDetector* init_object_detector(
    const char* model_path,
    TensorRTWrapper* tensorrt,
    const DetectorConfig& config) {
    
    if (!model_path || !tensorrt) {
        return nullptr;
    }

    try {
        auto status = tensorrt->loadModel(model_path, {
            .max_batch_size = config.batch_size,
            .enable_fp16 = true,
            .enable_int8 = false,
            .strict_types = true
        });

        if (status != TensorRTStatus::SUCCESS) {
            return nullptr;
        }

        return new ObjectDetector(tensorrt, config);

    } catch (const std::exception&) {
        return nullptr;
    }
}

} // namespace ai
} // namespace tald