# TALD UNIA Platform - Diagnostic and Calibration Tools Makefile
# Version: 1.0.0

# Compiler and flags
CC=clang
CFLAGS=-O2 -pipe -fstack-protector-strong -fno-strict-aliasing -I../include -DNDEBUG -D_FORTIFY_SOURCE=2 -fPIE
CXXFLAGS=$(CFLAGS) -std=c++17
LDFLAGS=-L../lib -Wl,-z,relro,-z,now -pie

# External dependencies with versions
CUDA_LIBS=-lcudart -lcuda              # CUDA 12.0
TENSORRT_LIBS=-lnvinfer -lnvonnxparser # TensorRT 8.6
VULKAN_LIBS=-lvulkan                   # Vulkan 1.3
NVML_LIBS=-lnvidia-ml                  # NVML
WEBRTC_LIBS=-lwebrtc                   # WebRTC M98

# Internal library dependencies
TALD_LIBS=-ltald

# Tools to build
TOOLS=gpu_benchmark lidar_calibration_tool mesh_network_analyzer performance_profiler

# Object files
GPU_BENCH_OBJS=gpu_benchmark.o
LIDAR_CAL_OBJS=lidar_calibration_tool.o
MESH_ANALYZER_OBJS=mesh_network_analyzer.o
PERF_PROFILER_OBJS=performance_profiler.o

# Include paths
INC_PATHS=-I../include \
          -I../drivers/gpu \
          -I../drivers/lidar \
          -I../drivers/network \
          -I../lib/libtald

# Default target
.PHONY: all
all: $(TOOLS)

# GPU Benchmark tool
gpu_benchmark: $(GPU_BENCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(VULKAN_LIBS) $(CUDA_LIBS) $(TALD_LIBS)

# LiDAR Calibration tool
lidar_calibration_tool: $(LIDAR_CAL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(TALD_LIBS)

# Mesh Network Analyzer
mesh_network_analyzer: $(MESH_ANALYZER_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(WEBRTC_LIBS) $(TALD_LIBS)

# Performance Profiler
performance_profiler: $(PERF_PROFILER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) $(NVML_LIBS) $(TENSORRT_LIBS) $(TALD_LIBS)

# Generic object file compilation
%.o: %.c
	$(CC) $(CFLAGS) $(INC_PATHS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_PATHS) -c $< -o $@

# Installation target
.PHONY: install
install: all
	@echo "Installing tools to /usr/local/bin..."
	@for tool in $(TOOLS); do \
		install -s -m 0755 $$tool /usr/local/bin/$$tool; \
	done
	@echo "Installing debug symbols..."
	@for tool in $(TOOLS); do \
		install -m 0644 $$tool.debug /usr/local/lib/debug/$$tool.debug; \
	done

# Clean target
.PHONY: clean
clean:
	rm -f $(TOOLS) *.o *.core *.debug
	rm -f *.pdb *.ilk
	rm -rf build/*
	rm -f .depend

# Generate dependencies
.PHONY: depend
depend: .depend

.depend: $(wildcard *.c *.cpp)
	rm -f ./.depend
	$(CC) $(CFLAGS) $(INC_PATHS) -MM $^ > ./.depend

# Security checks
.PHONY: security-check
security-check: all
	@echo "Running security checks..."
	@for tool in $(TOOLS); do \
		readelf -s $$tool | grep -i stack; \
		readelf -d $$tool | grep RELRO; \
		readelf -d $$tool | grep BIND_NOW; \
	done

# Include dependency rules
include .depend

# Help target
.PHONY: help
help:
	@echo "TALD UNIA Platform - Diagnostic and Calibration Tools"
	@echo "Available targets:"
	@echo "  all             - Build all tools (default)"
	@echo "  install         - Install tools to system"
	@echo "  clean           - Remove built files"
	@echo "  security-check  - Run security checks on binaries"
	@echo "  help            - Show this help message"
	@echo "Tools:"
	@echo "  gpu_benchmark   - GPU performance testing tool"
	@echo "  lidar_calibration_tool - LiDAR calibration utility"
	@echo "  mesh_network_analyzer  - Mesh network analysis tool"
	@echo "  performance_profiler   - System performance profiler"