#!/usr/bin/env bash

# TALD UNIA FreeBSD Test Suite Runner
# Version: 1.0.0
# Executes comprehensive test suites for hardware drivers, kernel modules,
# and system integration with parallel execution and resource management.

# Strict error handling
set -euo pipefail
IFS=$'\n\t'

# Import build configuration
source "${TEST_ROOT:-${PWD}/../tests}/Makefile" 2>/dev/null || {
    echo "Error: Failed to source Makefile configuration"
    exit 1
}

# Global constants
readonly TEST_ROOT="${PWD}/../tests"
readonly TEST_REPORT_DIR="${PWD}/../tests/reports"
readonly TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-INFO}"
readonly TEST_TIMEOUT=1200  # 20 minutes
readonly MAX_PARALLEL_TESTS=4
readonly RETRY_COUNT=3
readonly RESOURCE_MONITOR_INTERVAL=5

# Test categories
declare -A TEST_SUITES=(
    ["drivers"]="lidar_hw_test mesh_network_test"
    ["kernel"]="tald_core_test"
    ["game"]="physics_engine_test"
)

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Logging functions
log() {
    local level=$1
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] $*" | tee -a "${TEST_REPORT_DIR}/test.log"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# Setup test environment with comprehensive validation
setup_test_environment() {
    log_info "Setting up test environment..."

    # Verify script execution as root
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    }

    # Create test report directory with proper permissions
    mkdir -p "${TEST_REPORT_DIR}"
    chmod 755 "${TEST_REPORT_DIR}"

    # Initialize logging with rotation
    local log_file="${TEST_REPORT_DIR}/test.log"
    touch "${log_file}"
    if [[ -f "${log_file}.5" ]]; then
        rm "${log_file}.5"
        for i in {4..1}; do
            [[ -f "${log_file}.$i" ]] && mv "${log_file}.$i" "${log_file}.$((i+1))"
        done
        [[ -f "${log_file}" ]] && mv "${log_file}" "${log_file}.1"
    fi

    # Verify hardware availability
    check_hardware_requirements || {
        log_error "Hardware verification failed"
        exit 1
    }

    # Initialize resource monitoring
    setup_resource_monitoring

    log_info "Test environment setup complete"
    return 0
}

# Hardware verification
check_hardware_requirements() {
    log_info "Verifying hardware requirements..."

    # Check LiDAR hardware
    if ! kldstat -n lidar_hw.ko >/dev/null 2>&1; then
        log_error "LiDAR kernel module not loaded"
        return 1
    fi

    # Verify GPU availability
    if ! nvidia-smi >/dev/null 2>&1; then
        log_warn "NVIDIA GPU not detected, some tests may be skipped"
    fi

    # Check memory requirements
    local total_mem=$(sysctl -n hw.physmem)
    if [[ ${total_mem} -lt 4294967296 ]]; then  # 4GB
        log_error "Insufficient system memory"
        return 1
    }

    return 0
}

# Resource monitoring
setup_resource_monitoring() {
    log_info "Initializing resource monitoring..."
    
    # Start background resource monitoring
    (
        while true; do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local mem_usage=$(vmstat -H | tail -1 | awk '{print $4}')
            local cpu_usage=$(ps -ax -o pcpu | awk '{sum+=$1} END {print sum}')
            echo "${timestamp},${mem_usage},${cpu_usage}" >> "${TEST_REPORT_DIR}/resource_usage.csv"
            sleep ${RESOURCE_MONITOR_INTERVAL}
        done
    ) &
    MONITOR_PID=$!
    
    # Ensure cleanup on exit
    trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT
}

# Execute driver test suites with parallel execution
run_driver_tests() {
    log_info "Executing driver test suites..."
    local failed_tests=()
    
    # Run tests in parallel with resource limits
    for test in ${TEST_SUITES["drivers"]}; do
        (
            local retry=0
            while [[ ${retry} -lt ${RETRY_COUNT} ]]; do
                if timeout ${TEST_TIMEOUT} make -C "${TEST_ROOT}/drivers" test-${test} \
                    >> "${TEST_REPORT_DIR}/${test}.log" 2>&1; then
                    log_info "Test ${test} passed"
                    exit 0
                else
                    log_warn "Test ${test} failed, attempt $((retry+1))/${RETRY_COUNT}"
                    retry=$((retry+1))
                    sleep 2
                fi
            done
            log_error "Test ${test} failed after ${RETRY_COUNT} attempts"
            failed_tests+=("${test}")
            exit 1
        ) &
        
        # Limit parallel execution
        while [[ $(jobs -r | wc -l) -ge ${MAX_PARALLEL_TESTS} ]]; do
            sleep 1
        done
    done
    
    # Wait for all tests to complete
    wait
    
    return ${#failed_tests[@]}
}

# Execute kernel test suites with state validation
run_kernel_tests() {
    log_info "Executing kernel test suites..."
    
    # Verify kernel state before tests
    if ! sysctl kern.securelevel >/dev/null 2>&1; then
        log_error "Unable to verify kernel state"
        return 1
    }
    
    for test in ${TEST_SUITES["kernel"]}; do
        log_info "Running kernel test: ${test}"
        if ! timeout ${TEST_TIMEOUT} make -C "${TEST_ROOT}/kernel" test-${test} \
            >> "${TEST_REPORT_DIR}/${test}.log" 2>&1; then
            log_error "Kernel test ${test} failed"
            return 1
        fi
    done
    
    return 0
}

# Generate detailed test reports
generate_test_report() {
    local report_format=${1:-"html"}
    log_info "Generating test report in ${report_format} format..."
    
    # Aggregate test results
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    for suite in "${!TEST_SUITES[@]}"; do
        for test in ${TEST_SUITES[${suite}]}; do
            total_tests=$((total_tests+1))
            if grep -q "PASS" "${TEST_REPORT_DIR}/${test}.log" 2>/dev/null; then
                passed_tests=$((passed_tests+1))
            else
                failed_tests=$((failed_tests+1))
            fi
        done
    done
    
    # Generate HTML report
    cat > "${TEST_REPORT_DIR}/report.html" << EOF
<!DOCTYPE html>
<html>
<head><title>TALD UNIA Test Report</title></head>
<body>
<h1>Test Execution Report</h1>
<p>Date: $(date)</p>
<p>Total Tests: ${total_tests}</p>
<p>Passed: ${passed_tests}</p>
<p>Failed: ${failed_tests}</p>
<hr>
<h2>Test Details</h2>
<pre>
$(cat "${TEST_REPORT_DIR}/test.log")
</pre>
</body>
</html>
EOF
    
    # Compress test artifacts
    tar czf "${TEST_REPORT_DIR}/test_artifacts.tar.gz" \
        -C "${TEST_REPORT_DIR}" \
        --exclude="*.tar.gz" \
        .
    
    log_info "Test report generated: ${TEST_REPORT_DIR}/report.html"
    return 0
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    setup_test_environment || exit 1
    
    log_info "Starting TALD UNIA test execution..."
    
    # Execute test suites
    run_driver_tests
    local driver_status=$?
    
    run_kernel_tests
    local kernel_status=$?
    
    # Generate report
    generate_test_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Test execution completed in ${duration} seconds"
    
    # Return overall status
    [[ ${driver_status} -eq 0 && ${kernel_status} -eq 0 ]]
    return $?
}

# Execute main function
main "$@"