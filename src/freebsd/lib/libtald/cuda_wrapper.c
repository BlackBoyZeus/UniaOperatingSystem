/**
 * @file cuda_wrapper.c
 * @version 1.0.0
 * @brief Production implementation of CUDA wrapper for TALD UNIA platform
 */

#include "cuda_wrapper.h" // v1.0.0
#include <cuda.h>         // v12.0
#include <cuda_runtime.h> // v12.0
#include <device_launch_parameters.h> // v12.0
#include <thrust/device_vector.h>     // v12.0

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); \
        return CUDA_STATUS_ERROR; \
    } \
} while(0)

// Constants for resource management
static const size_t DEFAULT_BLOCK_SIZE = 256;
static const int MAX_GRID_SIZE = 65535;
static const int MAX_STREAMS = 8;
static const size_t MEMORY_POOL_SIZE = 1024 * 1024 * 1024; // 1GB
static const int WATCHDOG_TIMEOUT_MS = 5000;
static const int MAX_RETRY_ATTEMPTS = 3;

// Internal tracking structures
typedef struct {
    cudaStream_t stream;
    bool in_use;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
} StreamInfo;

typedef struct {
    void* ptr;
    size_t size;
    int stream_id;
} MemoryAllocation;

// Global state
static struct {
    bool initialized;
    int device_id;
    cudaDeviceProp device_props;
    StreamInfo streams[MAX_STREAMS];
    MemoryAllocation* allocations;
    size_t allocation_count;
    size_t total_memory;
    size_t available_memory;
    cudaEvent_t watchdog_event;
} g_cuda_state = {0};

CudaInitResult initialize_cuda(int device_id, size_t memory_pool_size, bool enable_monitoring) {
    CudaInitResult result = {0};
    int driver_version = 0;
    int runtime_version = 0;

    // Check CUDA versions
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Verify device availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_id >= device_count) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Set and verify device
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&g_cuda_state.device_props, device_id));

    // Check compute capability
    if (g_cuda_state.device_props.major < 7) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Initialize streams
    for (int i = 0; i < MAX_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&g_cuda_state.streams[i].stream));
        CUDA_CHECK(cudaEventCreate(&g_cuda_state.streams[i].start_event));
        CUDA_CHECK(cudaEventCreate(&g_cuda_state.streams[i].stop_event));
        g_cuda_state.streams[i].in_use = false;
    }

    // Setup memory tracking
    g_cuda_state.allocations = (MemoryAllocation*)malloc(
        sizeof(MemoryAllocation) * 1024);
    if (!g_cuda_state.allocations) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Initialize memory pool if requested
    if (memory_pool_size > 0) {
        cudaMemPool_t memPool;
        CUDA_CHECK(cudaMemPoolCreate(&memPool));
        CUDA_CHECK(cudaMemPoolSetAttribute(memPool, 
            cudaMemPoolAttrReleaseThreshold, &memory_pool_size));
    }

    // Setup monitoring
    if (enable_monitoring) {
        CUDA_CHECK(cudaEventCreate(&g_cuda_state.watchdog_event));
    }

    // Set global state
    g_cuda_state.initialized = true;
    g_cuda_state.device_id = device_id;
    g_cuda_state.total_memory = g_cuda_state.device_props.totalGlobalMem;
    g_cuda_state.available_memory = g_cuda_state.total_memory;

    // Populate result
    result.status = CUDA_STATUS_SUCCESS;
    result.device_id = device_id;
    result.total_memory = g_cuda_state.total_memory;
    result.compute_capability_major = g_cuda_state.device_props.major;
    result.compute_capability_minor = g_cuda_state.device_props.minor;

    return result;
}

CleanupResult cleanup_cuda(void) {
    CleanupResult result = {0};
    
    if (!g_cuda_state.initialized) {
        result.status = CUDA_STATUS_SUCCESS;
        return result;
    }

    // Synchronize all streams
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (cudaStreamSynchronize(g_cuda_state.streams[i].stream) != cudaSuccess) {
            result.status = CUDA_STATUS_ERROR;
            return result;
        }
    }

    // Free all tracked allocations
    for (size_t i = 0; i < g_cuda_state.allocation_count; i++) {
        if (g_cuda_state.allocations[i].ptr) {
            cudaFree(g_cuda_state.allocations[i].ptr);
        }
    }

    // Destroy streams and events
    for (int i = 0; i < MAX_STREAMS; i++) {
        cudaStreamDestroy(g_cuda_state.streams[i].stream);
        cudaEventDestroy(g_cuda_state.streams[i].start_event);
        cudaEventDestroy(g_cuda_state.streams[i].stop_event);
    }

    // Cleanup monitoring
    if (g_cuda_state.watchdog_event) {
        cudaEventDestroy(g_cuda_state.watchdog_event);
    }

    // Free tracking structures
    free(g_cuda_state.allocations);

    // Reset device
    cudaDeviceReset();

    // Clear state
    memset(&g_cuda_state, 0, sizeof(g_cuda_state));

    result.status = CUDA_STATUS_SUCCESS;
    return result;
}

AllocationResult CudaWrapper::allocate_device_memory(size_t size, MemoryFlags flags) {
    AllocationResult result = {0};
    void* ptr = NULL;
    
    if (!g_cuda_state.initialized || size == 0) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Check available memory
    if (size > g_cuda_state.available_memory) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Attempt allocation with retry logic
    for (int attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
        cudaError_t err;
        
        if (flags & MEMORY_FLAG_PINNED) {
            err = cudaMallocHost(&ptr, size);
        } else if (flags & MEMORY_FLAG_UNIFIED) {
            err = cudaMallocManaged(&ptr, size);
        } else {
            err = cudaMalloc(&ptr, size);
        }

        if (err == cudaSuccess && ptr) {
            break;
        }

        // Wait before retry
        cudaDeviceSynchronize();
    }

    if (!ptr) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Track allocation
    if (g_cuda_state.allocation_count < 1024) {
        g_cuda_state.allocations[g_cuda_state.allocation_count].ptr = ptr;
        g_cuda_state.allocations[g_cuda_state.allocation_count].size = size;
        g_cuda_state.allocation_count++;
    }

    // Update memory tracking
    g_cuda_state.available_memory -= size;

    result.status = CUDA_STATUS_SUCCESS;
    result.ptr = ptr;
    result.size = size;
    return result;
}

LaunchResult CudaWrapper::launch_kernel(void* kernel_function, dim3 grid_dim, 
    dim3 block_dim, void** arguments, size_t shared_memory_size, 
    cudaStream_t stream, LaunchConfig config) {
    
    LaunchResult result = {0};
    
    if (!g_cuda_state.initialized || !kernel_function || !arguments) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Validate dimensions
    if (block_dim.x * block_dim.y * block_dim.z > MAX_THREADS_PER_BLOCK) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Find available stream if none specified
    if (stream == NULL) {
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (!g_cuda_state.streams[i].in_use) {
                stream = g_cuda_state.streams[i].stream;
                g_cuda_state.streams[i].in_use = true;
                break;
            }
        }
    }

    if (!stream) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Record start event
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Launch kernel
    cudaError_t launch_error = cudaLaunchKernel(
        kernel_function,
        grid_dim,
        block_dim,
        arguments,
        shared_memory_size,
        stream
    );

    if (launch_error != cudaSuccess) {
        result.status = CUDA_STATUS_ERROR;
        return result;
    }

    // Record stop event and calculate timing
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Release stream
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (g_cuda_state.streams[i].stream == stream) {
            g_cuda_state.streams[i].in_use = false;
            break;
        }
    }

    result.status = CUDA_STATUS_SUCCESS;
    result.execution_time_ms = milliseconds;
    result.grid_dim = grid_dim;
    result.block_dim = block_dim;
    return result;
}