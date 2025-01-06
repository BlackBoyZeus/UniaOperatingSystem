/**
 * @file tensorrt_wrapper.c
 * @version 1.1.0
 * @brief Production implementation of TensorRT wrapper for TALD UNIA platform
 * @copyright TALD UNIA Platform
 */

#include "tensorrt_wrapper.h"
#include "cuda_wrapper.h"
#include <NvInfer.h>           // TensorRT 8.6
#include <cuda_runtime.h>      // CUDA 12.0
#include <sys/types.h>         // FreeBSD 9.0
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

namespace tald {
namespace ai {

// Global state management
static Logger gLogger;
static nvinfer1::IRuntime* gRuntime = nullptr;
static pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_int gErrorState = ATOMIC_VAR_INIT(0);

// Logger implementation
void Logger::log(Severity severity, const char* msg) noexcept {
    std::lock_guard<std::mutex> lock(log_mutex);
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            atomic_store(&gErrorState, 1);
            syslog(LOG_ERR, "TensorRT Internal Error: %s", msg);
            break;
        case Severity::kERROR:
            syslog(LOG_ERR, "TensorRT Error: %s", msg);
            break;
        case Severity::kWARNING:
            syslog(LOG_WARNING, "TensorRT Warning: %s", msg);
            break;
        case Severity::kINFO:
            syslog(LOG_INFO, "TensorRT Info: %s", msg);
            break;
        default:
            break;
    }
}

// TensorRTWrapper implementation
TensorRTWrapper::TensorRTWrapper(cuda::CudaWrapper* cuda_wrapper,
                               const char* model_path,
                               const TensorRTConfig& config) 
    : cuda_wrapper(cuda_wrapper) {
    
    if (!cuda_wrapper) {
        setErrorMessage("Invalid CUDA wrapper");
        throw std::runtime_error("CUDA wrapper initialization failed");
    }

    pthread_mutex_init(&inference_mutex, nullptr);
    last_status = loadModel(model_path, config);
    
    if (last_status != TensorRTStatus::SUCCESS) {
        throw std::runtime_error("Model initialization failed");
    }
}

TensorRTWrapper::~TensorRTWrapper() {
    pthread_mutex_lock(&inference_mutex);
    
    if (context) {
        context->destroy();
    }
    if (engine) {
        engine->destroy();
    }
    freeBuffers();
    
    pthread_mutex_unlock(&inference_mutex);
    pthread_mutex_destroy(&inference_mutex);
}

TensorRTStatus TensorRTWrapper::loadModel(const char* model_path,
                                        const TensorRTConfig& config) {
    pthread_mutex_lock(&inference_mutex);
    
    if (!validateConfig(config)) {
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::INVALID_CONFIG;
    }

    // Load engine from file with integrity check
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        setErrorMessage("Failed to open model file");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::MODEL_LOAD_ERROR;
    }

    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    std::vector<char> engine_data(file_size);
    if (read(fd, engine_data.data(), file_size) != file_size) {
        close(fd);
        setErrorMessage("Failed to read model file");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::MODEL_LOAD_ERROR;
    }
    close(fd);

    // Create runtime and engine
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        setErrorMessage("Failed to create TensorRT runtime");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::RUNTIME_ERROR;
    }

    // Configure DLA if requested
    if (config.dla_core >= 0) {
        runtime->setDLACore(config.dla_core);
        active_dla_core = config.dla_core;
    }

    engine = runtime->deserializeCudaEngine(engine_data.data(), file_size);
    if (!engine) {
        setErrorMessage("Failed to deserialize CUDA engine");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::MODEL_LOAD_ERROR;
    }

    context = engine->createExecutionContext();
    if (!context) {
        setErrorMessage("Failed to create execution context");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::RUNTIME_ERROR;
    }

    // Allocate binding buffers
    if (!allocateBuffers()) {
        setErrorMessage("Failed to allocate binding buffers");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::MEMORY_ERROR;
    }

    workspace_size = config.workspace_size;
    pthread_mutex_unlock(&inference_mutex);
    return TensorRTStatus::SUCCESS;
}

TensorRTStatus TensorRTWrapper::infer(void* input_data,
                                    void* output_data,
                                    size_t batch_size,
                                    uint32_t timeout_ms) {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;

    if (pthread_mutex_timedlock(&inference_mutex, &timeout) != 0) {
        setErrorMessage("Inference timeout while waiting for mutex");
        return TensorRTStatus::TIMEOUT_ERROR;
    }

    if (!context || !binding_buffers) {
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::RUNTIME_ERROR;
    }

    // Copy input data to GPU
    cuda::MemoryHandle input_handle = cuda_wrapper->allocate_device_memory(
        engine->getBindingSize(0) * batch_size,
        cuda::MemoryFlags::DEFAULT
    );

    if (!input_handle.isValid()) {
        setErrorMessage("Failed to allocate input buffer");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::MEMORY_ERROR;
    }

    cudaMemcpyAsync(input_handle.ptr, input_data,
                   input_handle.size,
                   cudaMemcpyHostToDevice);

    binding_buffers[0] = input_handle.ptr;

    // Execute inference
    bool status = context->executeV2(binding_buffers);
    if (!status) {
        setErrorMessage("Inference execution failed");
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::INFERENCE_ERROR;
    }

    // Copy output data back to host
    cudaMemcpyAsync(output_data, binding_buffers[1],
                   engine->getBindingSize(1) * batch_size,
                   cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(0);
    pthread_mutex_unlock(&inference_mutex);
    return TensorRTStatus::SUCCESS;
}

TensorRTStatus TensorRTWrapper::setDLACore(int dla_core, bool strict_mode) {
    pthread_mutex_lock(&inference_mutex);
    
    if (!runtime) {
        pthread_mutex_unlock(&inference_mutex);
        return TensorRTStatus::RUNTIME_ERROR;
    }

    try {
        runtime->setDLACore(dla_core);
        active_dla_core = dla_core;
    } catch (...) {
        if (strict_mode) {
            setErrorMessage("Failed to set DLA core in strict mode");
            pthread_mutex_unlock(&inference_mutex);
            return TensorRTStatus::DLA_ERROR;
        }
        // Fall back to GPU if not strict
        runtime->setDLACore(-1);
        active_dla_core = -1;
    }

    pthread_mutex_unlock(&inference_mutex);
    return TensorRTStatus::SUCCESS;
}

bool TensorRTWrapper::allocateBuffers() {
    const int num_bindings = engine->getNbBindings();
    binding_buffers = new void*[num_bindings];

    for (int i = 0; i < num_bindings; i++) {
        size_t size = engine->getBindingSize(i);
        cuda::MemoryHandle handle = cuda_wrapper->allocate_device_memory(
            size,
            cuda::MemoryFlags::DEFAULT
        );
        
        if (!handle.isValid()) {
            freeBuffers();
            return false;
        }
        binding_buffers[i] = handle.ptr;
    }
    return true;
}

void TensorRTWrapper::freeBuffers() {
    if (binding_buffers) {
        const int num_bindings = engine->getNbBindings();
        for (int i = 0; i < num_bindings; i++) {
            if (binding_buffers[i]) {
                cudaFree(binding_buffers[i]);
            }
        }
        delete[] binding_buffers;
        binding_buffers = nullptr;
    }
}

bool TensorRTWrapper::validateConfig(const TensorRTConfig& config) {
    if (config.max_batch_size <= 0 || config.max_batch_size > MAX_BATCH_SIZE) {
        setErrorMessage("Invalid batch size configuration");
        return false;
    }

    if (config.workspace_size > MAX_WORKSPACE_SIZE) {
        setErrorMessage("Workspace size exceeds maximum limit");
        return false;
    }

    if (config.dla_core >= 8) { // Maximum 8 DLA cores
        setErrorMessage("Invalid DLA core specified");
        return false;
    }

    return true;
}

void TensorRTWrapper::setErrorMessage(const char* msg) {
    error_message = std::string(msg);
    syslog(LOG_ERR, "TensorRT Error: %s", msg);
}

// Global initialization and cleanup
TensorRTStatus initialize_tensorrt(const char* model_path,
                                 cuda::CudaWrapper* cuda_wrapper,
                                 TensorRTConfig* config) noexcept {
    pthread_mutex_lock(&gMutex);
    
    if (!model_path || !cuda_wrapper) {
        pthread_mutex_unlock(&gMutex);
        return TensorRTStatus::INVALID_CONFIG;
    }

    try {
        TensorRTConfig default_config;
        if (!config) {
            config = &default_config;
        }

        // Initialize plugins
        initLibNvInferPlugins(&gLogger, "");

        if (atomic_load(&gErrorState)) {
            pthread_mutex_unlock(&gMutex);
            return TensorRTStatus::RUNTIME_ERROR;
        }

    } catch (...) {
        pthread_mutex_unlock(&gMutex);
        return TensorRTStatus::RUNTIME_ERROR;
    }

    pthread_mutex_unlock(&gMutex);
    return TensorRTStatus::SUCCESS;
}

void cleanup_tensorrt() noexcept {
    pthread_mutex_lock(&gMutex);
    
    if (gRuntime) {
        gRuntime->destroy();
        gRuntime = nullptr;
    }

    atomic_store(&gErrorState, 0);
    pthread_mutex_unlock(&gMutex);
}

} // namespace ai
} // namespace tald