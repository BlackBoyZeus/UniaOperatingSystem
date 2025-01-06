#!/bin/sh

# TALD UNIA Deployment Script
# Version: 1.0.0
# Secure deployment script for TALD UNIA FreeBSD gaming platform components

# Enable strict error handling
set -euo pipefail

# Source build environment
. "$(dirname "$0")/build.sh"

# Global deployment configuration
DEPLOY_ROOT="$(dirname "$0")/.."
LOG_DIR="/var/log/tald/deploy"
AUDIT_DIR="/var/log/tald/audit"
TARGET_USER="tald"
TARGET_GROUP="tald"
DEPLOY_TIMEOUT=300
TPM_DEVICE="/dev/tpm0"
SECURE_CHANNEL_TIMEOUT=60

# Deployment timestamp
DEPLOY_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Signal handlers
trap 'echo "Deployment interrupted. Cleaning up..."; exit 1' INT TERM

# Logging function with audit support
log() {
    local level="$1"
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${timestamp} [${level}] $*" | tee -a "${LOG_DIR}/deploy_${DEPLOY_TIMESTAMP}.log"
    
    # Audit logging for security events
    if [ "$level" = "SECURITY" ] || [ "$level" = "ERROR" ]; then
        echo "${timestamp} [${level}] $*" >> "${AUDIT_DIR}/security_${DEPLOY_TIMESTAMP}.log"
    fi
}

# Setup deployment environment with enhanced security
setup_deploy_env() {
    log "INFO" "Setting up secure deployment environment..."

    # Verify root privileges
    if [ "$(id -u)" -ne 0 ]; then
        log "ERROR" "Deployment requires root privileges"
        exit 1
    fi

    # Create and secure log directories
    install -d -m 750 "${LOG_DIR}"
    install -d -m 750 "${AUDIT_DIR}"
    chown root:wheel "${LOG_DIR}" "${AUDIT_DIR}"

    # Verify TPM availability
    if [ ! -c "${TPM_DEVICE}" ]; then
        log "ERROR" "TPM device not available"
        exit 1
    fi

    # Initialize TPM for deployment
    if ! tpm2_startup -c || ! tpm2_selftest --full; then
        log "ERROR" "TPM initialization failed"
        exit 1
    }

    # Verify SSH keys and setup secure channel
    if ! ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key >/dev/null 2>&1; then
        log "ERROR" "SSH host keys not found or invalid"
        exit 1
    }

    log "INFO" "Deployment environment setup complete"
    return 0
}

# Deploy kernel modules with signature verification
deploy_kernel_modules() {
    local target_device="$1"
    log "INFO" "Deploying kernel modules to ${target_device}..."

    # Verify module signatures
    local modules="gpu_module lidar_module memory_protection mesh_network tald_core"
    for module in ${modules}; do
        if ! kldxref "${BUILD_ROOT}/kernel/${module}/${module}.ko"; then
            log "ERROR" "Module signature verification failed: ${module}"
            return 1
        fi
    done

    # Create secure transfer channel
    ssh -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" "${target_device}" true || {
        log "ERROR" "Failed to establish secure channel to ${target_device}"
        return 1
    }

    # Transfer and load modules
    for module in ${modules}; do
        log "INFO" "Deploying module: ${module}"
        
        # Secure copy with encryption
        scp -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" \
            "${BUILD_ROOT}/kernel/${module}/${module}.ko" \
            "${target_device}:/boot/modules/" || {
            log "ERROR" "Failed to transfer module: ${module}"
            return 1
        }

        # Load module with verification
        ssh -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" "${target_device}" \
            "kldload -n /boot/modules/${module}.ko && \
             kldstat -n ${module}.ko" || {
            log "ERROR" "Failed to load module: ${module}"
            return 1
        }
    done

    log "INFO" "Kernel modules deployed successfully"
    return 0
}

# Deploy system libraries with security validation
deploy_system_libraries() {
    local target_device="$1"
    log "INFO" "Deploying system libraries to ${target_device}..."

    # Verify library signatures and versions
    local libraries="libtald.so libmesh.so libgpu.so"
    for lib in ${libraries}; do
        if ! readelf -n "${BUILD_ROOT}/lib/${lib}" | grep -q "GNU_BUILD_ID"; then
            log "ERROR" "Library signature verification failed: ${lib}"
            return 1
        fi
    done

    # Setup secure library paths
    ssh -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" "${target_device}" \
        "install -d -m 755 /usr/local/lib/tald" || {
        log "ERROR" "Failed to create library directory"
        return 1
    }

    # Transfer libraries with encryption
    for lib in ${libraries}; do
        log "INFO" "Deploying library: ${lib}"
        
        scp -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" \
            "${BUILD_ROOT}/lib/${lib}" \
            "${target_device}:/usr/local/lib/tald/" || {
            log "ERROR" "Failed to transfer library: ${lib}"
            return 1
        }

        # Verify library installation
        ssh -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" "${target_device}" \
            "ldconfig -r /usr/local/lib/tald && \
             ldd /usr/local/lib/tald/${lib}" || {
            log "ERROR" "Library verification failed: ${lib}"
            return 1
        }
    done

    log "INFO" "System libraries deployed successfully"
    return 0
}

# Validate deployment with comprehensive checks
validate_deployment() {
    local target_device="$1"
    log "INFO" "Validating deployment on ${target_device}..."

    # Verify system integrity
    ssh -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" "${target_device}" \
        "kldstat && sysctl -a | grep tald" || {
        log "ERROR" "System integrity verification failed"
        return 1
    }

    # Run performance tests
    if ! run_performance_tests "${target_device}"; then
        log "ERROR" "Performance validation failed"
        return 1
    }

    # Verify security configuration
    ssh -o ConnectTimeout="${SECURE_CHANNEL_TIMEOUT}" "${target_device}" \
        "sysctl kern.securelevel && \
         kenv | grep tald && \
         ls -l /dev/tald* && \
         grep tald /etc/rc.conf" || {
        log "ERROR" "Security configuration verification failed"
        return 1
    }

    # Generate validation report
    {
        echo "=== Deployment Validation Report ==="
        echo "Timestamp: $(date)"
        echo "Target: ${target_device}"
        echo "Kernel Modules: $(ssh ${target_device} kldstat | grep tald)"
        echo "Libraries: $(ssh ${target_device} ldconfig -r | grep tald)"
        echo "Security Level: $(ssh ${target_device} sysctl kern.securelevel)"
        echo "=================================="
    } > "${LOG_DIR}/validation_${DEPLOY_TIMESTAMP}.log"

    log "INFO" "Deployment validation complete"
    return 0
}

# Main deployment function
main() {
    local target_device="$1"
    
    log "INFO" "Starting TALD UNIA deployment to ${target_device}..."
    
    # Setup deployment environment
    setup_deploy_env || exit 1
    
    # Deploy components
    deploy_kernel_modules "${target_device}" || exit 1
    deploy_system_libraries "${target_device}" || exit 1
    
    # Validate deployment
    validate_deployment "${target_device}" || exit 1
    
    log "INFO" "Deployment completed successfully"
    return 0
}

# Execute main function with command line arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <target_device>"
    exit 1
fi

main "$@"