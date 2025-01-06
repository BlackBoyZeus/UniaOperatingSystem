# TALD UNIA Game Engine Makefile
# Version: 1.0.0

# Compiler and tool configurations
CC = clang
NVCC = nvcc
CMAKE = cmake
GLSLC = glslc

# Base directories
SRC_DIR = .
BUILD_DIR = build
INSTALL_DIR = /usr/local/lib/tald

# External dependency versions and paths
CUDA_VERSION = 12.0
VULKAN_VERSION = 1.3
THRUST_VERSION = 12.0
CMAKE_VERSION = 3.26

# Include paths
INCLUDES = \
    -I/usr/local/cuda-$(CUDA_VERSION)/include \
    -I/usr/local/vulkan/include \
    -I/usr/local/include \
    -I$(SRC_DIR)/../lib/libtald \
    -I$(SRC_DIR)/../drivers/gpu

# Library paths
LIB_PATHS = \
    -L/usr/local/cuda-$(CUDA_VERSION)/lib64 \
    -L/usr/local/vulkan/lib \
    -L/usr/local/lib

# Libraries
LIBS = \
    -lcuda \
    -lcudart \
    -lvulkan \
    -lthrust \
    -lpthread \
    -lm

# Compiler flags
CFLAGS = \
    -O3 \
    -march=native \
    -flto \
    -fprofile-use \
    -Wall \
    -Wextra \
    -std=c++20 \
    -D_FORTIFY_SOURCE=2 \
    -fstack-protector-strong \
    $(INCLUDES)

# CUDA flags
NVCCFLAGS = \
    -O3 \
    -arch=sm_80 \
    -std=c++20 \
    --use_fast_math \
    --compiler-options "$(CFLAGS)"

# Linker flags
LDFLAGS = \
    $(LIB_PATHS) \
    $(LIBS) \
    -Wl,-z,now \
    -Wl,-z,relro \
    -Wl,--as-needed

# Source files
CPP_SOURCES = \
    physics_engine.c \
    vulkan_renderer.c \
    scene_manager.c

CUDA_SOURCES = \
    physics_cuda_kernels.cu \
    scene_cuda_kernels.cu

SHADER_SOURCES = \
    shaders/mesh.vert \
    shaders/mesh.frag \
    shaders/point_cloud.comp

# Object files
CPP_OBJECTS = $(CPP_SOURCES:%.c=$(BUILD_DIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=$(BUILD_DIR)/%.o)
SHADER_OBJECTS = $(SHADER_SOURCES:%.vert=$(BUILD_DIR)/%.spv)
SHADER_OBJECTS += $(SHADER_SOURCES:%.frag=$(BUILD_DIR)/%.spv)
SHADER_OBJECTS += $(SHADER_SOURCES:%.comp=$(BUILD_DIR)/%.spv)

# Output library
TARGET = $(BUILD_DIR)/libgame_engine.so

# Default target
all: directories $(TARGET)

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/shaders

# Build shared library
$(TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS) $(SHADER_OBJECTS)
	@echo "Linking $(TARGET)..."
	@$(CC) -shared -o $@ $(CPP_OBJECTS) $(CUDA_OBJECTS) $(LDFLAGS)
	@echo "Build complete."

# Compile C++ sources
$(BUILD_DIR)/%.o: %.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Compile CUDA sources
$(BUILD_DIR)/%.o: %.cu
	@echo "Compiling CUDA $<..."
	@$(NVCC) $(NVCCFLAGS) -Xcompiler "-fPIC" -c $< -o $@

# Compile shaders
$(BUILD_DIR)/%.spv: %.vert
	@echo "Compiling vertex shader $<..."
	@$(GLSLC) -O -mfmt=num $< -o $@

$(BUILD_DIR)/%.spv: %.frag
	@echo "Compiling fragment shader $<..."
	@$(GLSLC) -O -mfmt=num $< -o $@

$(BUILD_DIR)/%.spv: %.comp
	@echo "Compiling compute shader $<..."
	@$(GLSLC) -O -mfmt=num $< -o $@

# Profile-guided optimization
profile:
	@echo "Building with instrumentation..."
	@$(MAKE) clean
	@$(MAKE) CFLAGS="$(CFLAGS) -fprofile-generate"
	@echo "Running test suite for profile data collection..."
	@./run_tests.sh
	@echo "Rebuilding with profile data..."
	@$(MAKE) clean
	@$(MAKE) CFLAGS="$(CFLAGS) -fprofile-use"

# Installation
install: all
	@echo "Installing to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)
	@cp $(TARGET) $(INSTALL_DIR)
	@mkdir -p $(INSTALL_DIR)/shaders
	@cp $(SHADER_OBJECTS) $(INSTALL_DIR)/shaders/
	@ldconfig

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f *.gcda *.gcno

# Help target
help:
	@echo "TALD UNIA Game Engine Build System"
	@echo "Available targets:"
	@echo "  all        - Build game engine library (default)"
	@echo "  profile    - Build with profile-guided optimization"
	@echo "  install    - Install library and shaders"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help message"

.PHONY: all directories profile install clean help