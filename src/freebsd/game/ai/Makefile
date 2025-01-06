# TALD UNIA AI Subsystem Makefile
# Requires CUDA 12.0 and TensorRT 8.6
# Version: 1.0.0

# Installation directories
PREFIX ?= /usr/local
INSTALL_INCLUDE_DIR = $(PREFIX)/include/tald/ai
INSTALL_LIB_DIR = $(PREFIX)/lib

# Compiler settings
CC = clang
CXX = clang++
CFLAGS = -O3 -march=native -Wall -Wextra -fPIC \
         -I/usr/local/cuda/include \
         -I/usr/local/tensorrt/include \
         -MMD -MP
CXXFLAGS = $(CFLAGS) -std=c++20
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/local/tensorrt/lib \
          -Wl,-rpath,$(INSTALL_LIB_DIR)
LIBS = -lcudart -lnvinfer -lnvonnxparser

# Dependency paths
CUDA_PATH = /usr/local/cuda
TENSORRT_PATH = /usr/local/tensorrt

# Library settings
SONAME = libai.so.1
VERSION = 1.0.0
BUILD_TYPE ?= release

# Source files
SRCS = environment_classifier.c \
       object_detection.c

OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.d)

# Library targets
LIB = libai.so.$(VERSION)
LIB_SONAME = libai.so.1
LIB_LINK = libai.so

# Debug symbols
DEBUG_SYMBOLS = $(LIB).debug

# Default target
.PHONY: all
all: check-deps $(LIB)

# Dependency checking
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@test -d $(CUDA_PATH) || (echo "CUDA 12.0 not found in $(CUDA_PATH)" && exit 1)
	@test -d $(TENSORRT_PATH) || (echo "TensorRT 8.6 not found in $(TENSORRT_PATH)" && exit 1)
	@$(CC) --version >/dev/null 2>&1 || (echo "Clang compiler not found" && exit 1)
	@nvcc --version >/dev/null 2>&1 || (echo "NVCC compiler not found" && exit 1)

# Compilation rules
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Library build
$(LIB): $(OBJS)
	$(CC) -shared -Wl,-soname,$(SONAME) \
	      $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
	@if [ "$(BUILD_TYPE)" = "debug" ]; then \
		objcopy --only-keep-debug $@ $(DEBUG_SYMBOLS); \
		strip -s $@; \
		objcopy --add-gnu-debuglink=$(DEBUG_SYMBOLS) $@; \
	fi

# Installation
.PHONY: install
install: all
	install -d $(DESTDIR)$(INSTALL_INCLUDE_DIR)
	install -d $(DESTDIR)$(INSTALL_LIB_DIR)
	install -m 644 environment_classifier.h $(DESTDIR)$(INSTALL_INCLUDE_DIR)
	install -m 644 object_detection.h $(DESTDIR)$(INSTALL_INCLUDE_DIR)
	install -m 755 $(LIB) $(DESTDIR)$(INSTALL_LIB_DIR)
	ln -sf $(LIB) $(DESTDIR)$(INSTALL_LIB_DIR)/$(LIB_SONAME)
	ln -sf $(LIB_SONAME) $(DESTDIR)$(INSTALL_LIB_DIR)/$(LIB_LINK)
	ldconfig

# Cleanup
.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) $(LIB) $(LIB_SONAME) $(LIB_LINK) $(DEBUG_SYMBOLS)

# Include generated dependencies
-include $(DEPS)