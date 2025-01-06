/**
 * @file cuda_wrapper.h
 * @version 1.0.0
 * @brief Production-ready C++ wrapper for CUDA runtime operations in TALD UNIA platform
 * @copyright TALD UNIA Platform
 * 
 * Provides comprehensive CUDA runtime management with:
 * - Enhanced error handling and monitoring
 * - Multi-stream support for concurrent operations
 * - Optimized memory management for LiDAR processing
 * - Performance tracking and resource management
 */

#ifndef TALD_CUDA_WRAPPER_H
#define TALD_CUDA_WRAPPER_H

// External dependencies with versions
#include <cuda.h>              // CUDA 12.0
#include <cuda_runtime.h>      // CUDA 12.0
#include <device_launch_parameters.h> // CUDA 12.0
#include <sys/types.h>         // FreeBSD 9.0

#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <memory>

// Global constants
#define CUDA_WRAPPER_VERSION "1.0.0"
#define MAX_THREADS_PER_BLOCK 1024
#define DEFAULT_STREAM_COUNT 4
#define MAX_SHARED_MEMORY_SIZE (48 * 1024)
#define MIN_COMPUTE_CAPABILITY 7.0
#define MAX_ERROR_STRING_LENGTH 256
#define MEMORY_ALIGNMENT 256
#define MAX_CONCURRENT_KERNELS 32

namespace tald {
namespace cuda {

// Forward declarations
class MemoryTracker;
class ErrorLogger;

/**
 * @brief Configuration for CUDA stream management
 */
struct StreamConfig {
    uint32_t stream_count{DEFAULT_STREAM_COUNT};
    bool enable_priority{true};
    int priority_high{cudaStreamDefault};
    int priority_low{cudaStreamDefault};
};

/**
 * @brief Configuration for memory management
 */
struct MemoryConfig {
    size_t reserved_size{0};
    bool enable_tracking{true};
    bool enable_unified_memory{false};
    size_t alignment{MEMORY_ALIGNMENT};
};

/**
 * @brief Status codes for CUDA operations
 */
enum class CudaStatus {
    SUCCESS,
    DRIVER_ERROR,
    MEMORY_ERROR,
    INITIALIZATION_ERROR,
    LAUNCH_ERROR,
    CAPABILITY_ERROR,
    RESOURCE_ERROR
};

/**
 * @brief Memory allocation flags
 */
enum class MemoryFlags {
    DEFAULT = 0,
    PINNED = 1,
    UNIFIED = 2,
    WRITE_COMBINED = 4
};

/**
 * @brief Handle for managed GPU memory
 */
class MemoryHandle {
public:
    void* ptr{nullptr};
    size_t size{0};
    MemoryFlags flags{MemoryFlags::DEFAULT};
    uint32_t stream_id{0};
    
    bool isValid() const { return ptr != nullptr; }
};

/**
 * @brief Status information for kernel launches
 */
struct KernelStatus {
    bool success{false};
    float execution_time_ms{0.0f};
    size_t shared_memory_used{0};
    std::string error_message;
};

/**
 * @brief Primary CUDA wrapper class
 */
class CudaWrapper {
public:
    /**
     * @brief Constructor with enhanced configuration
     * @param device_id GPU device identifier
     * @param stream_config Stream configuration parameters
     * @param memory_config Memory management configuration
     * @throws std::runtime_error if initialization fails
     */
    CudaWrapper(int device_id, 
                const StreamConfig& stream_config = StreamConfig{},
                const MemoryConfig& memory_config = MemoryConfig{});
    
    /**
     * @brief Destructor ensures proper cleanup
     */
    ~CudaWrapper();

    // Prevent copying
    CudaWrapper(const CudaWrapper&) = delete;
    CudaWrapper& operator=(const CudaWrapper&) = delete;

    /**
     * @brief Allocates device memory with tracking
     * @param size Size in bytes
     * @param flags Memory allocation flags
     * @param stream_id Stream identifier
     * @return MemoryHandle containing allocation details
     */
    [[nodiscard]]
    MemoryHandle allocate_device_memory(size_t size, 
                                      MemoryFlags flags = MemoryFlags::DEFAULT,
                                      uint32_t stream_id = 0);

    /**
     * @brief Launches CUDA kernel with comprehensive error checking
     * @param kernel_function Pointer to kernel function
     * @param grid_dim Grid dimensions
     * @param block_dim Block dimensions
     * @param arguments Kernel arguments
     * @param shared_memory_size Shared memory size
     * @param stream_id Stream identifier
     * @param config Launch configuration
     * @return KernelStatus containing launch results
     */
    [[nodiscard]]
    KernelStatus launch_kernel(void* kernel_function,
                             dim3 grid_dim,
                             dim3 block_dim,
                             void** arguments,
                             size_t shared_memory_size = 0,
                             uint32_t stream_id = 0);

    /**
     * @brief Retrieves device properties
     * @return const reference to device properties
     */
    [[nodiscard]]
    const cudaDeviceProp& get_device_properties() const;

    /**
     * @brief Retrieves memory statistics
     * @return Pair of total and available memory
     */
    [[nodiscard]]
    std::pair<size_t, size_t> get_memory_stats() const;

private:
    cudaStream_t* streams{nullptr};
    int device_id{-1};
    size_t available_memory{0};
    cudaDeviceProp device_properties{};
    std::atomic<size_t> active_allocations{0};
    std::vector<cudaEvent_t> events;
    std::mutex resource_mutex;
    std::unique_ptr<MemoryTracker> memory_tracker;
    std::unique_ptr<ErrorLogger> error_logger;

    void initialize_streams(const StreamConfig& config);
    void initialize_memory(const MemoryConfig& config);
    bool validate_device_capabilities();
    void cleanup_resources();
};

/**
 * @brief Initializes CUDA environment
 * @param device_id GPU device identifier
 * @param enable_p2p Enable peer-to-peer access
 * @param reserved_memory Memory to reserve
 * @return CudaStatus initialization status
 */
[[nodiscard]]
CudaStatus initialize_cuda(int device_id, 
                         bool enable_p2p = false,
                         size_t reserved_memory = 0);

/**
 * @brief Cleanup CUDA resources
 * @param force_cleanup Force cleanup of all resources
 */
void cleanup_cuda(bool force_cleanup = false);

} // namespace cuda
} // namespace tald

#endif // TALD_CUDA_WRAPPER_H