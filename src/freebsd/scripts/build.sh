#!/bin/sh

# TALD UNIA Build System
# Version: 1.0.0
# Dependencies:
# - FreeBSD make v9.0
# - Clang v15.0
# - Ninja v1.11.1

# Enable strict error handling
set -euo pipefail

# Global build configuration
BUILD_ROOT=$(dirname $0)/..
LOG_DIR=/var/log/tald/build
MAKE="make -j$(nproc)"
CC="clang"
CFLAGS="-O3 -pipe -flto -fno-strict-aliasing -march=native -mtune=native -Wall -Wextra -Werror"
LDFLAGS="-flto -Wl,-O3"
BUILD_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUILD_LOG="${LOG_DIR}/build_${BUILD_TIMESTAMP}.log"

# Signal handlers for clean termination
trap 'echo "Build interrupted. Cleaning up..."; exit 1' INT TERM

# Logging function with rotation
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "${BUILD_LOG}"
}

# Environment setup and validation
setup_build_env() {
    # Check for root/sudo permissions
    if [ "$(id -u)" -ne 0 ]; then
        log "ERROR" "Build script must be run as root"
        exit 1
    }

    # Create and secure log directory
    install -d -m 750 "${LOG_DIR}"
    chown root:wheel "${LOG_DIR}"

    # Verify build tools
    local required_tools="make clang ninja"
    for tool in ${required_tools}; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            log "ERROR" "Required tool not found: ${tool}"
            exit 1
        fi
    done

    # Verify tool versions
    if ! make -v | grep -q "FreeBSD Make 9.0"; then
        log "ERROR" "Incorrect make version. Required: FreeBSD Make 9.0+"
        exit 1
    fi

    if ! clang --version | grep -q "clang version 15.0"; then
        log "ERROR" "Incorrect clang version. Required: 15.0+"
        exit 1
    fi

    if ! ninja --version | grep -q "1.11"; then
        log "ERROR" "Incorrect ninja version. Required: 1.11.1+"
        exit 1
    fi

    # Check system resources
    if [ "$(df -k "${BUILD_ROOT}" | awk 'NR==2 {print $4}')" -lt 5242880 ]; then
        log "ERROR" "Insufficient disk space. Required: 5GB+"
        exit 1
    fi

    if [ "$(sysctl -n hw.physmem)" -lt 8589934592 ]; then
        log "ERROR" "Insufficient memory. Required: 8GB+"
        exit 1
    }

    log "INFO" "Build environment validated successfully"
}

# Kernel module build function
build_kernel_modules() {
    log "INFO" "Building kernel modules..."
    
    cd "${BUILD_ROOT}/kernel" || exit 1
    
    local modules="gpu_module lidar_module memory_protection mesh_network tald_core"
    for module in ${modules}; do
        log "INFO" "Building module: ${module}"
        ${MAKE} -C "${module}" \
            CFLAGS="${CFLAGS}" \
            LDFLAGS="${LDFLAGS}" || {
            log "ERROR" "Failed to build module: ${module}"
            exit 1
        }
        
        # Verify module signature
        if ! kldxref "${module}/${module}.ko"; then
            log "ERROR" "Module signature verification failed: ${module}"
            exit 1
        }
    done
}

# Driver build function
build_drivers() {
    log "INFO" "Building device drivers..."
    
    cd "${BUILD_ROOT}/drivers" || exit 1
    
    local drivers="vulkan_driver lidar_driver network_driver"
    for driver in ${drivers}; do
        log "INFO" "Building driver: ${driver}"
        ${MAKE} -C "${driver}" \
            CFLAGS="${CFLAGS} -DNDEBUG" \
            LDFLAGS="${LDFLAGS}" || {
            log "ERROR" "Failed to build driver: ${driver}"
            exit 1
        }
    done
}

# Main build orchestration
main() {
    log "INFO" "Starting TALD UNIA build process..."
    
    # Setup and validate build environment
    setup_build_env
    
    # Create clean build directory
    rm -rf "${BUILD_ROOT}/build"
    mkdir -p "${BUILD_ROOT}/build"
    
    # Build components in optimal order
    build_kernel_modules
    build_drivers
    
    # Build core system components
    cd "${BUILD_ROOT}" || exit 1
    ${MAKE} all || {
        log "ERROR" "Core system build failed"
        exit 1
    }
    
    # Run performance validation tests
    ${MAKE} test || {
        log "ERROR" "Performance validation failed"
        exit 1
    }
    
    # Install built components
    ${MAKE} install || {
        log "ERROR" "Installation failed"
        exit 1
    }
    
    # Verify installation
    if ! ${MAKE} verify-install; then
        log "ERROR" "Installation verification failed"
        exit 1
    }
    
    log "INFO" "Build completed successfully"
    log "INFO" "Build log available at: ${BUILD_LOG}"
}

# Execute main build process
main "$@"