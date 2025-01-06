/**
 * @file tensorrt_wrapper.h
 * @version 1.1.0
 * @brief Production-ready C++ wrapper for NVIDIA TensorRT with enhanced DLA support
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_TENSORRT_WRAPPER_H
#define TALD_TENSORRT_WRAPPER_H

// External dependencies with versions
#include <NvInfer.h>           // TensorRT 8.6
#include <cuda_runtime.h>      // CUDA 12.0
#include <sys/types.h>         // FreeBSD 9.0
#include <mutex>               // C++17

// Internal dependencies
#include "cuda_wrapper.h"

namespace tald {
namespace ai {

// Version and configuration constants
constexpr const char* TENSORRT_WRAPPER_VERSION = "1.1.0";
constexpr int MAX_BATCH_SIZE = 32;
constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30;  // 1GB
constexpr int DEFAULT_DLA_CORE = -1;
constexpr size_t MAX_ERROR_STRING_SIZE = 1024;
constexpr uint32_t INFERENCE_TIMEOUT_MS = 50;

/**
 * @brief Configuration for TensorRT initialization
 */
struct TensorRTConfig {
    int max_batch_size{MAX_BATCH_SIZE};
    size_t workspace_size{MAX_WORKSPACE_SIZE};
    int dla_core{DEFAULT_DLA_CORE};
    bool enable_fp16{true};
    bool enable_int8{false};
    bool strict_types{true};
    bool enable_profiling{false};
};

/**
 * @brief Status codes for TensorRT operations
 */
enum class TensorRTStatus {
    SUCCESS,
    MODEL_LOAD_ERROR,
    RUNTIME_ERROR,
    MEMORY_ERROR,
    INFERENCE_ERROR,
    TIMEOUT_ERROR,
    DLA_ERROR,
    INVALID_CONFIG
};

/**
 * @brief Custom TensorRT logger with severity control
 */
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
private:
    std::mutex log_mutex;
};

/**
 * @brief Thread-safe wrapper class for TensorRT operations
 */
class TensorRTWrapper {
public:
    /**
     * @brief Constructor with enhanced configuration
     * @param cuda_wrapper Pointer to CUDA wrapper instance
     * @param model_path Path to serialized TensorRT model
     * @param config TensorRT configuration parameters
     * @throws std::runtime_error if initialization fails
     */
    TensorRTWrapper(cuda::CudaWrapper* cuda_wrapper,
                   const char* model_path,
                   const TensorRTConfig& config = TensorRTConfig{});

    /**
     * @brief Destructor ensures proper cleanup
     */
    ~TensorRTWrapper();

    // Prevent copying
    TensorRTWrapper(const TensorRTWrapper&) = delete;
    TensorRTWrapper& operator=(const TensorRTWrapper&) = delete;

    /**
     * @brief Loads and prepares TensorRT model
     * @param model_path Path to serialized model
     * @param config Configuration parameters
     * @return Status of load operation
     */
    [[nodiscard]]
    TensorRTStatus loadModel(const char* model_path,
                           const TensorRTConfig& config);

    /**
     * @brief Performs thread-safe inference
     * @param input_data Pointer to input data
     * @param output_data Pointer to output buffer
     * @param batch_size Number of samples in batch
     * @param timeout_ms Maximum inference time
     * @return Status of inference operation
     */
    [[nodiscard]]
    TensorRTStatus infer(void* input_data,
                        void* output_data,
                        size_t batch_size = 1,
                        uint32_t timeout_ms = INFERENCE_TIMEOUT_MS);

    /**
     * @brief Configures DLA core usage
     * @param dla_core DLA core to use (-1 for GPU)
     * @param strict_mode Enforce DLA usage
     * @return Status of DLA configuration
     */
    [[nodiscard]]
    TensorRTStatus setDLACore(int dla_core, bool strict_mode = false);

    /**
     * @brief Retrieves last error message
     * @return Const reference to error message
     */
    [[nodiscard]]
    const std::string& getLastError() const noexcept { return error_message; }

    /**
     * @brief Retrieves last operation status
     * @return Last recorded status
     */
    [[nodiscard]]
    TensorRTStatus getStatus() const noexcept { return last_status; }

private:
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    cuda::CudaWrapper* cuda_wrapper{nullptr};
    void** binding_buffers{nullptr};
    size_t workspace_size{0};
    std::mutex inference_mutex;
    std::string error_message;
    TensorRTStatus last_status{TensorRTStatus::SUCCESS};
    int active_dla_core{DEFAULT_DLA_CORE};
    Logger logger;

    bool allocateBuffers();
    void freeBuffers();
    bool validateConfig(const TensorRTConfig& config);
    void setErrorMessage(const char* msg);
};

/**
 * @brief Initializes TensorRT environment
 * @param model_path Path to serialized model
 * @param cuda_wrapper Pointer to CUDA wrapper
 * @param config TensorRT configuration
 * @return Status of initialization
 */
[[nodiscard]]
TensorRTStatus initialize_tensorrt(const char* model_path,
                                 cuda::CudaWrapper* cuda_wrapper,
                                 TensorRTConfig* config = nullptr) noexcept;

/**
 * @brief Cleanup TensorRT resources
 */
void cleanup_tensorrt() noexcept;

} // namespace ai
} // namespace tald

#endif // TALD_TENSORRT_WRAPPER_H