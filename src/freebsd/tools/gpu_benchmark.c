/**
 * @file gpu_benchmark.c
 * @version 1.0.0
 * @brief GPU benchmarking tool for TALD UNIA platform performance validation
 */

#include <vulkan/vulkan.h>  // v1.3
#include <stdio.h>          // C11
#include <stdlib.h>         // C11
#include <time.h>           // C11
#include "../drivers/gpu/vulkan_driver.h"
#include "../drivers/gpu/shader_compiler.h"

// Version and configuration constants
#define BENCHMARK_VERSION "1.0.0"
#define MAX_TEST_ITERATIONS 1000
#define WARMUP_ITERATIONS 100
#define TARGET_FRAME_TIME_MS 16.6
#define THERMAL_THRESHOLD_C 85.0
#define POWER_SAMPLE_INTERVAL_MS 100

// Benchmark result structures
typedef struct {
    double min_frame_time;
    double max_frame_time;
    double avg_frame_time;
    double frame_time_variance;
    double percentile_95;
    double percentile_99;
    uint32_t frames_below_target;
    uint32_t total_frames;
} timing_stats_t;

typedef struct {
    double avg_power_draw;
    double peak_power_draw;
    double total_energy_consumed;
} power_stats_t;

typedef struct {
    double avg_temperature;
    double peak_temperature;
    double thermal_throttling_events;
} thermal_stats_t;

typedef struct {
    timing_stats_t timing;
    power_stats_t power;
    thermal_stats_t thermal;
    char benchmark_version[16];
    char driver_version[16];
    time_t timestamp;
} benchmark_result_t;

// BenchmarkManager class implementation
typedef struct BenchmarkManager {
    VulkanDriver* driver;
    ShaderCompiler* compiler;
    benchmark_result_t results;
    double* frame_times;
    double* power_samples;
    double* thermal_samples;
    uint32_t sample_count;
} BenchmarkManager;

/**
 * @brief Creates a new BenchmarkManager instance
 */
BenchmarkManager* create_benchmark_manager(VulkanDriver* driver, ShaderCompiler* compiler) {
    BenchmarkManager* manager = (BenchmarkManager*)malloc(sizeof(BenchmarkManager));
    if (!manager) return NULL;

    manager->driver = driver;
    manager->compiler = compiler;
    manager->frame_times = (double*)malloc(MAX_TEST_ITERATIONS * sizeof(double));
    manager->power_samples = (double*)malloc(MAX_TEST_ITERATIONS * sizeof(double));
    manager->thermal_samples = (double*)malloc(MAX_TEST_ITERATIONS * sizeof(double));
    manager->sample_count = 0;

    strncpy(manager->results.benchmark_version, BENCHMARK_VERSION, sizeof(manager->results.benchmark_version) - 1);
    manager->results.timestamp = time(NULL);

    return manager;
}

/**
 * @brief Performs statistical analysis on collected frame times
 */
static void analyze_frame_times(BenchmarkManager* manager) {
    qsort(manager->frame_times, manager->sample_count, sizeof(double), compare_doubles);

    timing_stats_t* timing = &manager->results.timing;
    timing->min_frame_time = manager->frame_times[0];
    timing->max_frame_time = manager->frame_times[manager->sample_count - 1];
    
    double sum = 0.0;
    uint32_t frames_below = 0;
    for (uint32_t i = 0; i < manager->sample_count; i++) {
        sum += manager->frame_times[i];
        if (manager->frame_times[i] <= TARGET_FRAME_TIME_MS) {
            frames_below++;
        }
    }
    
    timing->avg_frame_time = sum / manager->sample_count;
    timing->frames_below_target = frames_below;
    timing->total_frames = manager->sample_count;
    
    // Calculate percentiles
    timing->percentile_95 = manager->frame_times[(uint32_t)(manager->sample_count * 0.95)];
    timing->percentile_99 = manager->frame_times[(uint32_t)(manager->sample_count * 0.99)];
}

/**
 * @brief Runs comprehensive draw call benchmark with monitoring
 */
benchmark_result_t run_draw_call_benchmark(VulkanDriver* driver, const benchmark_config_t* config) {
    BenchmarkManager* manager = create_benchmark_manager(driver, NULL);
    struct timespec start, end;
    
    // Warmup phase
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
        driver->begin_frame();
        // Execute standard draw calls
        driver->end_frame();
    }
    
    // Main benchmark loop
    for (uint32_t i = 0; i < MAX_TEST_ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        driver->begin_frame();
        // Execute benchmark draw calls
        driver->end_frame();
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        // Record metrics
        double frame_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                          (end.tv_nsec - start.tv_nsec) / 1000000.0;
        manager->frame_times[i] = frame_time;
        
        // Sample power and thermal data
        if (i % (POWER_SAMPLE_INTERVAL_MS / TARGET_FRAME_TIME_MS) == 0) {
            manager->power_samples[manager->sample_count] = driver->get_power_state();
            manager->thermal_samples[manager->sample_count] = driver->get_temperature();
            manager->sample_count++;
        }
    }
    
    // Analyze results
    analyze_frame_times(manager);
    
    // Calculate power and thermal statistics
    power_stats_t* power = &manager->results.power;
    thermal_stats_t* thermal = &manager->results.thermal;
    
    double power_sum = 0.0, temp_sum = 0.0;
    power->peak_power_draw = 0.0;
    thermal->peak_temperature = 0.0;
    thermal->thermal_throttling_events = 0;
    
    for (uint32_t i = 0; i < manager->sample_count; i++) {
        power_sum += manager->power_samples[i];
        temp_sum += manager->thermal_samples[i];
        
        if (manager->power_samples[i] > power->peak_power_draw) {
            power->peak_power_draw = manager->power_samples[i];
        }
        
        if (manager->thermal_samples[i] > thermal->peak_temperature) {
            thermal->peak_temperature = manager->thermal_samples[i];
        }
        
        if (manager->thermal_samples[i] > THERMAL_THRESHOLD_C) {
            thermal->thermal_throttling_events++;
        }
    }
    
    power->avg_power_draw = power_sum / manager->sample_count;
    thermal->avg_temperature = temp_sum / manager->sample_count;
    
    benchmark_result_t results = manager->results;
    
    // Cleanup
    free(manager->frame_times);
    free(manager->power_samples);
    free(manager->thermal_samples);
    free(manager);
    
    return results;
}

/**
 * @brief Helper function for qsort comparison
 */
static int compare_doubles(const void* a, const void* b) {
    double diff = *(const double*)a - *(const double*)b;
    return (diff > 0) - (diff < 0);
}

/**
 * @brief Writes benchmark results to file
 */
static void write_benchmark_report(const benchmark_result_t* results, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    
    fprintf(file, "TALD UNIA GPU Benchmark Report\n");
    fprintf(file, "Version: %s\n", results->benchmark_version);
    fprintf(file, "Timestamp: %s\n", ctime(&results->timestamp));
    fprintf(file, "\nTiming Statistics:\n");
    fprintf(file, "Min Frame Time: %.2f ms\n", results->timing.min_frame_time);
    fprintf(file, "Max Frame Time: %.2f ms\n", results->timing.max_frame_time);
    fprintf(file, "Average Frame Time: %.2f ms\n", results->timing.avg_frame_time);
    fprintf(file, "95th Percentile: %.2f ms\n", results->timing.percentile_95);
    fprintf(file, "99th Percentile: %.2f ms\n", results->timing.percentile_99);
    fprintf(file, "Frames Below Target: %u/%u\n", 
            results->timing.frames_below_target, results->timing.total_frames);
    
    fprintf(file, "\nPower Statistics:\n");
    fprintf(file, "Average Power Draw: %.2f W\n", results->power.avg_power_draw);
    fprintf(file, "Peak Power Draw: %.2f W\n", results->power.peak_power_draw);
    fprintf(file, "Total Energy Consumed: %.2f J\n", results->power.total_energy_consumed);
    
    fprintf(file, "\nThermal Statistics:\n");
    fprintf(file, "Average Temperature: %.2f °C\n", results->thermal.avg_temperature);
    fprintf(file, "Peak Temperature: %.2f °C\n", results->thermal.peak_temperature);
    fprintf(file, "Thermal Throttling Events: %.0f\n", results->thermal.thermal_throttling_events);
    
    fclose(file);
}