# TALD UNIA FreeBSD Test Suite Makefile
# Version: 1.0.0

# Compiler and flags
CC?=		clang
CFLAGS+=	-O2 -pipe -fstack-protector-strong -fno-strict-aliasing \
		-g -DDEBUG \
		-I/usr/local/include \
		-I/usr/local/cuda/include \
		-I${.CURDIR}/../drivers \
		-I${.CURDIR}/../kernel \
		-I${.CURDIR}/../lib/libtald

LDFLAGS+=	-L/usr/local/lib \
		-L/usr/local/cuda/lib64 \
		-latf-c \
		-lunity \
		-lcuda \
		-lcudart \
		-Wl,-rpath,/usr/local/cuda/lib64

# Test directories
TEST_DIRS=	drivers game kernel

# Parallel execution
PARALLEL_JOBS?=	${shell sysctl -n hw.ncpu}

# Test logs directory
TEST_LOG_DIR=	${.CURDIR}/test_logs

# Default target
.PHONY: all
all: setup-env
	@echo "Building TALD UNIA test suites..."
	@for dir in ${TEST_DIRS}; do \
		${MAKE} -C $$dir -j${PARALLEL_JOBS} all || exit 1; \
	done
	@echo "Build complete. Run 'make test' to execute test suites."

# Setup test environment
.PHONY: setup-env
setup-env:
	@mkdir -p ${TEST_LOG_DIR}
	@echo "Initializing test environment..."
	@nvidia-smi -pm 1 > /dev/null 2>&1 || true
	@nvidia-smi -ac 5001,1590 > /dev/null 2>&1 || true

# Clean target
.PHONY: clean
clean:
	@echo "Cleaning test artifacts..."
	@rm -rf ${TEST_LOG_DIR}
	@for dir in ${TEST_DIRS}; do \
		${MAKE} -C $$dir clean; \
	done
	@nvidia-smi -rac > /dev/null 2>&1 || true

# Test execution target
.PHONY: test
test: all
	@echo "Running TALD UNIA test suites..."
	@mkdir -p ${TEST_LOG_DIR}
	@echo "=== Test Execution Start: $$(date) ===" > ${TEST_LOG_DIR}/test_summary.log
	
	@echo "\nRunning LiDAR driver tests..."
	@${MAKE} -C drivers test > ${TEST_LOG_DIR}/drivers_test.log 2>&1 || \
		(echo "LiDAR tests failed, check ${TEST_LOG_DIR}/drivers_test.log"; exit 1)
	
	@echo "Running game engine tests..."
	@${MAKE} -C game test > ${TEST_LOG_DIR}/game_test.log 2>&1 || \
		(echo "Game engine tests failed, check ${TEST_LOG_DIR}/game_test.log"; exit 1)
	
	@echo "Running kernel tests..."
	@${MAKE} -C kernel test > ${TEST_LOG_DIR}/kernel_test.log 2>&1 || \
		(echo "Kernel tests failed, check ${TEST_LOG_DIR}/kernel_test.log"; exit 1)
	
	@echo "=== Test Execution End: $$(date) ===" >> ${TEST_LOG_DIR}/test_summary.log
	@echo "All tests completed successfully."

# Individual test targets
.PHONY: test-drivers
test-drivers: setup-env
	@echo "Running LiDAR driver tests..."
	@${MAKE} -C drivers test

.PHONY: test-game
test-game: setup-env
	@echo "Running game engine tests..."
	@${MAKE} -C game test

.PHONY: test-kernel
test-kernel: setup-env
	@echo "Running kernel tests..."
	@${MAKE} -C kernel test

# GPU resource cleanup on exit
.PHONY: gpu-cleanup
gpu-cleanup:
	@nvidia-smi -rac > /dev/null 2>&1 || true
	@nvidia-smi -pm 0 > /dev/null 2>&1 || true

# Error handling
.error?= exit 1

# Include local configuration if exists
.-include "Makefile.local"