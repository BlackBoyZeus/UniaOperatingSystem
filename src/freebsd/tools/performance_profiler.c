/**
 * @file performance_profiler.c
 * @version 1.0.0
 * @brief Performance profiling tool for TALD UNIA platform with comprehensive metrics
 * @copyright TALD UNIA Platform
 */

#include <sys/time.h>
#include <sys/resource.h>
#include <nvml.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdbool.h>

#include "cuda_wrapper.h"
#include "lidar_processing.h"

/* Version and configuration constants */
#define PROFILER_VERSION "1.0.0"
#define MAX_SAMPLES 1000
#define SAMPLING_INTERVAL_MS 100
#define METRICS_BUFFER_SIZE 8192
#define GPU_METRICS_COUNT 12
#define LIDAR_METRICS_COUNT 8
#define MEMORY_METRICS_COUNT 6
#define ALERT_THRESHOLD_COUNT 10

/* Data structures */
typedef struct {
    float gpu_utilization;
    float memory_utilization;
    float temperature;
    float power_usage;
    float memory_bandwidth;
    float compute_queue_depth;
    uint64_t memory_total;
    uint64_t memory_used;
    uint64_t memory_free;
    float pcie_bandwidth;
    uint32_t sm_active;
    uint32_t memory_ecc_errors;
} GpuMetrics;

typedef struct {
    float scan_latency_ms;
    uint32_t points_per_scan;
    float processing_time_ms;
    float point_cloud_density;
    uint32_t frames_processed;
    uint32_t processing_errors;
    float classification_accuracy;
    float mesh_quality;
} LidarMetrics;

typedef struct {
    uint64_t total_physical;
    uint64_t used_physical;
    uint64_t total_virtual;
    uint64_t used_virtual;
    uint64_t page_faults;
    float memory_bandwidth;
} MemoryMetrics;

typedef struct {
    pthread_mutex_t mutex;
    size_t write_index;
    size_t read_index;
    size_t count;
    GpuMetrics gpu_samples[MAX_SAMPLES];
    LidarMetrics lidar_samples[MAX_SAMPLES];
    MemoryMetrics memory_samples[MAX_SAMPLES];
} MetricsBuffer;

typedef struct {
    float gpu_util_threshold;
    float memory_util_threshold;
    float temperature_threshold;
    float latency_threshold_ms;
    uint32_t error_count_threshold;
} AlertThresholds;

typedef struct {
    MetricsBuffer* metrics_buffer;
    nvmlDevice_t gpu_handle;
    pthread_t sampling_thread;
    pthread_mutex_t metrics_mutex;
    AlertThresholds alert_thresholds;
    bool running;
    uint32_t sampling_interval;
} PerformanceProfiler;

/* Static function declarations */
static void* sampling_thread_func(void* arg);
static GpuMetrics* collect_gpu_metrics(PerformanceProfiler* profiler);
static LidarMetrics* collect_lidar_metrics(PerformanceProfiler* profiler);
static MemoryMetrics* collect_memory_metrics(void);
static void check_alert_thresholds(PerformanceProfiler* profiler, const GpuMetrics* gpu, const LidarMetrics* lidar);
static bool add_metrics_sample(MetricsBuffer* buffer, const GpuMetrics* gpu, const LidarMetrics* lidar, const MemoryMetrics* memory);

/* Function implementations */
__attribute__((visibility("default")))
bool initialize_profiler(PerformanceProfiler** profiler, nvmlDevice_t gpu_handle, uint32_t flags) {
    PerformanceProfiler* p = calloc(1, sizeof(PerformanceProfiler));
    if (!p) {
        return false;
    }

    p->metrics_buffer = calloc(1, sizeof(MetricsBuffer));
    if (!p->metrics_buffer) {
        free(p);
        return false;
    }

    if (pthread_mutex_init(&p->metrics_mutex, NULL) != 0 ||
        pthread_mutex_init(&p->metrics_buffer->mutex, NULL) != 0) {
        free(p->metrics_buffer);
        free(p);
        return false;
    }

    p->gpu_handle = gpu_handle;
    p->sampling_interval = SAMPLING_INTERVAL_MS;
    p->running = true;

    // Initialize alert thresholds
    p->alert_thresholds = (AlertThresholds){
        .gpu_util_threshold = 90.0f,
        .memory_util_threshold = 85.0f,
        .temperature_threshold = 85.0f,
        .latency_threshold_ms = 50.0f,
        .error_count_threshold = 100
    };

    // Start sampling thread
    if (pthread_create(&p->sampling_thread, NULL, sampling_thread_func, p) != 0) {
        pthread_mutex_destroy(&p->metrics_mutex);
        pthread_mutex_destroy(&p->metrics_buffer->mutex);
        free(p->metrics_buffer);
        free(p);
        return false;
    }

    *profiler = p;
    return true;
}

static void* sampling_thread_func(void* arg) {
    PerformanceProfiler* profiler = (PerformanceProfiler*)arg;
    struct timespec sleep_time = {
        .tv_sec = 0,
        .tv_nsec = profiler->sampling_interval * 1000000
    };

    while (profiler->running) {
        GpuMetrics* gpu = collect_gpu_metrics(profiler);
        LidarMetrics* lidar = collect_lidar_metrics(profiler);
        MemoryMetrics* memory = collect_memory_metrics();

        if (gpu && lidar && memory) {
            check_alert_thresholds(profiler, gpu, lidar);
            add_metrics_sample(profiler->metrics_buffer, gpu, lidar, memory);
        }

        free(gpu);
        free(lidar);
        free(memory);

        nanosleep(&sleep_time, NULL);
    }

    return NULL;
}

static __attribute__((hot))
GpuMetrics* collect_gpu_metrics(PerformanceProfiler* profiler) {
    GpuMetrics* metrics = calloc(1, sizeof(GpuMetrics));
    if (!metrics) {
        return NULL;
    }

    nvmlUtilization_t utilization;
    nvmlMemory_t memory;
    unsigned int temperature;

    if (nvmlDeviceGetUtilizationRates(profiler->gpu_handle, &utilization) == NVML_SUCCESS) {
        metrics->gpu_utilization = (float)utilization.gpu;
        metrics->memory_utilization = (float)utilization.memory;
    }

    if (nvmlDeviceGetTemperature(profiler->gpu_handle, NVML_TEMPERATURE_GPU, &temperature) == NVML_SUCCESS) {
        metrics->temperature = (float)temperature;
    }

    if (nvmlDeviceGetMemoryInfo(profiler->gpu_handle, &memory) == NVML_SUCCESS) {
        metrics->memory_total = memory.total;
        metrics->memory_used = memory.used;
        metrics->memory_free = memory.free;
    }

    unsigned int power;
    if (nvmlDeviceGetPowerUsage(profiler->gpu_handle, &power) == NVML_SUCCESS) {
        metrics->power_usage = (float)power / 1000.0f; // Convert to watts
    }

    return metrics;
}

static LidarMetrics* collect_lidar_metrics(PerformanceProfiler* profiler) {
    LidarMetrics* metrics = calloc(1, sizeof(LidarMetrics));
    if (!metrics) {
        return NULL;
    }

    pthread_mutex_lock(&profiler->metrics_mutex);
    // Collect metrics from LidarProcessor
    // Implementation would interface with LidarProcessor instance
    pthread_mutex_unlock(&profiler->metrics_mutex);

    return metrics;
}

static MemoryMetrics* collect_memory_metrics(void) {
    MemoryMetrics* metrics = calloc(1, sizeof(MemoryMetrics));
    if (!metrics) {
        return NULL;
    }

    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        metrics->page_faults = usage.ru_majflt + usage.ru_minflt;
    }

    // Get system memory information
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (meminfo) {
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            unsigned long value;
            if (sscanf(line, "MemTotal: %lu kB", &value) == 1) {
                metrics->total_physical = value * 1024;
            } else if (sscanf(line, "MemAvailable: %lu kB", &value) == 1) {
                metrics->used_physical = metrics->total_physical - (value * 1024);
            }
        }
        fclose(meminfo);
    }

    return metrics;
}

static void check_alert_thresholds(PerformanceProfiler* profiler, const GpuMetrics* gpu, const LidarMetrics* lidar) {
    if (gpu->gpu_utilization > profiler->alert_thresholds.gpu_util_threshold) {
        fprintf(stderr, "ALERT: GPU utilization above threshold: %.1f%%\n", gpu->gpu_utilization);
    }

    if (gpu->temperature > profiler->alert_thresholds.temperature_threshold) {
        fprintf(stderr, "ALERT: GPU temperature above threshold: %.1f°C\n", gpu->temperature);
    }

    if (lidar->scan_latency_ms > profiler->alert_thresholds.latency_threshold_ms) {
        fprintf(stderr, "ALERT: LiDAR scan latency above threshold: %.1fms\n", lidar->scan_latency_ms);
    }
}

static bool add_metrics_sample(MetricsBuffer* buffer, const GpuMetrics* gpu, const LidarMetrics* lidar, const MemoryMetrics* memory) {
    pthread_mutex_lock(&buffer->mutex);

    if (buffer->count >= MAX_SAMPLES) {
        buffer->read_index = (buffer->read_index + 1) % MAX_SAMPLES;
        buffer->count--;
    }

    memcpy(&buffer->gpu_samples[buffer->write_index], gpu, sizeof(GpuMetrics));
    memcpy(&buffer->lidar_samples[buffer->write_index], lidar, sizeof(LidarMetrics));
    memcpy(&buffer->memory_samples[buffer->write_index], memory, sizeof(MemoryMetrics));

    buffer->write_index = (buffer->write_index + 1) % MAX_SAMPLES;
    buffer->count++;

    pthread_mutex_unlock(&buffer->mutex);
    return true;
}