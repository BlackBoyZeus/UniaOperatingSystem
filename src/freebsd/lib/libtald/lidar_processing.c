/**
 * @file lidar_processing.c
 * @version 1.0.0
 * @brief Implementation of core LiDAR processing functionality for TALD UNIA platform
 * @copyright TALD UNIA Platform
 */

#include "lidar_processing.h"
#include "point_cloud.h"
#include <cuda.h>                    // CUDA 12.0
#include <thrust/device_vector.h>    // CUDA 12.0
#include <NvInfer.h>                 // TensorRT 8.6
#include <pthread.h>                 // FreeBSD 9.0
#include <monitoring/system.h>       // System Monitoring 2.1
#include <string.h>
#include <errno.h>

// Constants from specification
#define SCAN_BATCH_SIZE 32768
#define MAX_POINTS_PER_SCAN 1000000
#define MIN_POINT_DISTANCE 0.1f
#define PROCESSING_THREADS 1024
#define MAX_PROCESSING_TIME_MS 50
#define HEALTH_CHECK_INTERVAL_MS 1000
#define ERROR_RETRY_COUNT 3
#define MEMORY_ALERT_THRESHOLD 0.9f

using namespace tald::lidar;
using namespace tald::cuda;
using namespace tald::ai;

// Local function declarations
static bool validate_processor_health(LidarProcessor* processor);
static void update_performance_metrics(LidarProcessor* processor);
static void handle_processing_error(LidarProcessor* processor, const char* error_msg);

LidarProcessor::LidarProcessor(
    CudaWrapper* cuda_wrapper,
    TensorRTWrapper* tensorrt_wrapper,
    const LidarConfig& config,
    MonitoringSystem* monitor) 
    : cuda_wrapper(cuda_wrapper),
      tensorrt_wrapper(tensorrt_wrapper),
      config(config),
      monitor(monitor),
      healthy(true) {
    
    try {
        // Initialize CUDA resources with error checking
        if (!cuda_wrapper) {
            throw std::runtime_error("Invalid CUDA wrapper");
        }

        // Allocate point cloud with safety checks
        current_scan = std::make_unique<PointCloud>(cuda_wrapper, MAX_POINTS_PER_SCAN);
        if (!current_scan) {
            throw std::runtime_error("Failed to allocate point cloud");
        }

        // Initialize GPU memory for raw points
        raw_points.resize(MAX_POINTS_PER_SCAN);

        // Initialize metrics
        metrics = {};
        metrics.processing_latency_ms = 0.0f;
        metrics.points_processed = 0;
        metrics.processing_errors = 0;

        // Setup monitoring
        if (monitor) {
            monitor->register_component("LidarProcessor", HEALTH_CHECK_INTERVAL_MS);
        }

    } catch (const std::exception& e) {
        handle_processing_error(this, e.what());
        throw;
    }
}

Result<bool> LidarProcessor::process_scan(const uint8_t* raw_data, size_t data_size) {
    std::lock_guard<std::mutex> lock(processing_mutex);
    
    if (!validate_processor_health(this)) {
        return Result<bool>::error("Processor unhealthy");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Validate input data
        if (!raw_data || data_size == 0 || data_size > MAX_POINTS_PER_SCAN * sizeof(float3)) {
            throw std::runtime_error("Invalid input data");
        }

        // Convert raw data to point cloud with GPU acceleration
        auto kernel_status = cuda_wrapper->launch_kernel(
            reinterpret_cast<void*>(&convert_raw_to_points),
            dim3((data_size + PROCESSING_THREADS - 1) / PROCESSING_THREADS),
            dim3(PROCESSING_THREADS),
            reinterpret_cast<void**>(&raw_data),
            0
        );

        if (!kernel_status.success) {
            throw std::runtime_error("Point conversion failed: " + kernel_status.error_message);
        }

        // Filter and optimize point cloud
        if (!current_scan->optimize(MIN_POINT_DISTANCE)) {
            throw std::runtime_error("Point cloud optimization failed");
        }

        // Run environment classification with TensorRT
        auto inference_status = tensorrt_wrapper->infer(
            current_scan->get_points(nullptr, 0),
            nullptr,
            1,
            MAX_PROCESSING_TIME_MS
        );

        if (inference_status != TensorRTStatus::SUCCESS) {
            throw std::runtime_error("Environment classification failed");
        }

        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics.processing_latency_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        metrics.points_processed += current_scan->get_point_count();

        // Update monitoring
        if (monitor) {
            monitor->update_metric("processing_latency", metrics.processing_latency_ms);
            monitor->update_metric("points_processed", metrics.points_processed);
        }

        return Result<bool>::ok(true);

    } catch (const std::exception& e) {
        handle_processing_error(this, e.what());
        metrics.processing_errors++;
        return Result<bool>::error(e.what());
    }
}

HealthStatus LidarProcessor::get_health_status() const {
    HealthStatus status;
    status.is_healthy = healthy.load();
    
    // Check GPU health
    auto [total_mem, available_mem] = cuda_wrapper->get_memory_stats();
    status.gpu_available = available_mem > (total_mem * MEMORY_ALERT_THRESHOLD);
    
    // Check inference engine
    status.inference_available = 
        tensorrt_wrapper->getStatus() == TensorRTStatus::SUCCESS;
    
    // Check processing load
    status.processing_load = metrics.processing_latency_ms / MAX_PROCESSING_TIME_MS;
    
    return status;
}

ProcessingMetrics LidarProcessor::get_metrics() const {
    return metrics;
}

static bool validate_processor_health(LidarProcessor* processor) {
    if (!processor->healthy.load()) {
        return false;
    }

    auto status = processor->get_health_status();
    return status.is_healthy && 
           status.gpu_available && 
           status.inference_available &&
           status.processing_load < 1.0f;
}

static void update_performance_metrics(LidarProcessor* processor) {
    auto [total_mem, available_mem] = processor->cuda_wrapper->get_memory_stats();
    processor->metrics.gpu_utilization = 
        static_cast<float>(total_mem - available_mem) / total_mem;
    
    if (processor->monitor) {
        processor->monitor->update_metric("gpu_utilization", 
            processor->metrics.gpu_utilization);
    }
}

static void handle_processing_error(LidarProcessor* processor, const char* error_msg) {
    processor->healthy.store(false);
    
    if (processor->monitor) {
        processor->monitor->report_error("LidarProcessor", error_msg);
    }
    
    // Log error for debugging
    fprintf(stderr, "LidarProcessor error: %s\n", error_msg);
}

Result<LidarProcessor*> initialize_lidar_processor(
    CudaWrapper* cuda_wrapper,
    TensorRTWrapper* tensorrt_wrapper,
    const LidarConfig& config,
    MonitoringSystem* monitor) {
    
    try {
        auto processor = new LidarProcessor(cuda_wrapper, tensorrt_wrapper, config, monitor);
        return Result<LidarProcessor*>::ok(processor);
    } catch (const std::exception& e) {
        return Result<LidarProcessor*>::error(e.what());
    }
}