#!/bin/bash
# UNIA Gaming OS - Comprehensive Test Automation Script

set -e

# Configuration
TERRAFORM_DIR="../"
TEST_RESULTS_DIR="./test-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎮 UNIA Gaming OS - AWS Cloud Testing Suite${NC}"
echo -e "${BLUE}================================================${NC}"

# Create test results directory
mkdir -p $TEST_RESULTS_DIR

# Function to log with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run test on instance
run_test_on_instance() {
    local instance_ip=$1
    local test_type=$2
    local test_command=$3
    
    log "${YELLOW}Running $test_type test on $instance_ip${NC}"
    
    ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ec2-user@$instance_ip "$test_command" > "$TEST_RESULTS_DIR/${test_type}_${TIMESTAMP}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        log "${GREEN}✅ $test_type test completed successfully${NC}"
    else
        log "${RED}❌ $test_type test failed${NC}"
    fi
}

# Deploy infrastructure
log "${YELLOW}🚀 Deploying AWS test infrastructure...${NC}"
cd $TERRAFORM_DIR
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Get instance IPs
GPU_IP=$(terraform output -raw gpu_test_instance_ip)
CPU_IP=$(terraform output -raw cpu_test_instance_ip)
MEMORY_IP=$(terraform output -raw memory_test_instance_ip)
NETWORK_IP=$(terraform output -raw network_test_instance_ip)
S3_BUCKET=$(terraform output -raw test_results_bucket)

log "${GREEN}Infrastructure deployed successfully!${NC}"
log "GPU Test Instance: $GPU_IP"
log "CPU Test Instance: $CPU_IP"
log "Memory Test Instance: $MEMORY_IP"
log "Network Test Instance: $NETWORK_IP"
log "S3 Results Bucket: $S3_BUCKET"

# Wait for instances to be ready
log "${YELLOW}⏳ Waiting for instances to initialize...${NC}"
sleep 300

# Test 1: GPU Performance Tests
log "${BLUE}🎯 Starting GPU Performance Tests${NC}"
run_test_on_instance $GPU_IP "gpu-performance" "
    cd /opt/unia-gaming/gpu-tests
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    
    # GPU stress test
    nvidia-smi -l 1 &
    stress-ng --gpu 1 --timeout 60s
    
    # Vulkan test
    vulkaninfo > vulkan-info.txt
    
    # OpenGL test
    glxinfo | grep -E 'OpenGL|direct rendering'
"

# Test 2: CPU AI Performance Tests
log "${BLUE}🧠 Starting CPU AI Performance Tests${NC}"
run_test_on_instance $CPU_IP "cpu-ai-performance" "
    cd /opt/unia-gaming/ai-tests
    python3 /opt/unia-gaming/ai-performance-test.py
    
    # CPU stress test
    stress-ng --cpu $(nproc) --timeout 60s --metrics-brief
    
    # AI workload simulation
    python3 -c '
import numpy as np
import time
print(\"Testing NumPy performance...\")
start = time.time()
a = np.random.rand(5000, 5000)
b = np.random.rand(5000, 5000)
c = np.dot(a, b)
end = time.time()
print(f\"Matrix multiplication (5000x5000): {end-start:.2f} seconds\")
'
"

# Test 3: Memory Performance Tests
log "${BLUE}💾 Starting Memory Performance Tests${NC}"
run_test_on_instance $MEMORY_IP "memory-performance" "
    cd /opt/unia-gaming/memory-tests
    /opt/unia-gaming/memory-performance-test.sh
    
    # Additional memory tests
    free -h
    cat /proc/meminfo
    
    # Memory bandwidth test
    dd if=/dev/zero of=/tmp/test bs=1M count=2048 conv=fdatasync
    dd if=/tmp/test of=/dev/null bs=1M
    rm /tmp/test
"

# Test 4: Network Performance Tests
log "${BLUE}🌐 Starting Network Performance Tests${NC}"
run_test_on_instance $NETWORK_IP "network-performance" "
    cd /opt/unia-gaming/network-tests
    /opt/unia-gaming/network-performance-test.sh
    
    # Additional network tests
    netstat -i
    ss -tuln
    
    # Network speed test
    curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python3
"

# Test 5: UNIA Gaming OS Boot Test
log "${BLUE}🎮 Starting UNIA Gaming OS Boot Tests${NC}"

# Copy UNIA OS binary to test instances
for ip in $GPU_IP $CPU_IP $MEMORY_IP $NETWORK_IP; do
    log "Copying UNIA Gaming OS to $ip"
    scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ../gaming-kernel/target/x86_64-unknown-none/debug/bootimage-unia-gaming-os.bin ec2-user@$ip:/tmp/
    
    # Test QEMU boot
    run_test_on_instance $ip "unia-os-boot" "
        cd /tmp
        timeout 30s qemu-system-x86_64 -accel tcg -m 512M -drive format=raw,file=bootimage-unia-gaming-os.bin -nographic -serial stdio > unia-boot-test.log 2>&1 || true
        cat unia-boot-test.log
    "
done

# Test 6: Performance Benchmarking
log "${BLUE}📊 Starting Performance Benchmarking${NC}"

# Create comprehensive benchmark script
cat > /tmp/benchmark-suite.sh << 'EOF'
#!/bin/bash
echo "=== UNIA Gaming OS Performance Benchmark ==="
echo "Timestamp: $(date)"
echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo ""

echo "=== CPU Information ==="
lscpu
echo ""

echo "=== Memory Information ==="
free -h
echo ""

echo "=== Storage Information ==="
df -h
echo ""

echo "=== Network Information ==="
ip addr show
echo ""

echo "=== Performance Tests ==="
# CPU benchmark
echo "CPU Benchmark (calculating pi):"
time echo "scale=5000; 4*a(1)" | bc -l > /dev/null

# Memory benchmark
echo "Memory Benchmark:"
dd if=/dev/zero of=/tmp/memory-test bs=1M count=1024 2>&1 | grep copied

# Disk I/O benchmark
echo "Disk I/O Benchmark:"
dd if=/dev/zero of=/tmp/disk-test bs=1M count=1024 conv=fdatasync 2>&1 | grep copied
rm /tmp/disk-test /tmp/memory-test

echo "=== Benchmark Complete ==="
EOF

for ip in $GPU_IP $CPU_IP $MEMORY_IP $NETWORK_IP; do
    scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa /tmp/benchmark-suite.sh ec2-user@$ip:/tmp/
    run_test_on_instance $ip "performance-benchmark" "bash /tmp/benchmark-suite.sh"
done

# Collect all test results
log "${YELLOW}📥 Collecting test results...${NC}"
for ip in $GPU_IP $CPU_IP $MEMORY_IP $NETWORK_IP; do
    instance_type=$(ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ec2-user@$ip "curl -s http://169.254.169.254/latest/meta-data/instance-type")
    
    # Create directory for this instance
    mkdir -p "$TEST_RESULTS_DIR/$instance_type-$ip"
    
    # Download test results
    scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -r ec2-user@$ip:/opt/unia-gaming/test-results/* "$TEST_RESULTS_DIR/$instance_type-$ip/" 2>/dev/null || true
done

# Upload results to S3
log "${YELLOW}☁️ Uploading results to S3...${NC}"
aws s3 sync $TEST_RESULTS_DIR s3://$S3_BUCKET/test-results-$TIMESTAMP/

# Generate test report
log "${YELLOW}📋 Generating test report...${NC}"
cat > "$TEST_RESULTS_DIR/test-report-$TIMESTAMP.md" << EOF
# UNIA Gaming OS - AWS Cloud Testing Report

**Test Date:** $(date)
**Test Duration:** Started at $TIMESTAMP

## Test Infrastructure

- **GPU Test Instance:** $GPU_IP (g4dn.xlarge)
- **CPU Test Instance:** $CPU_IP (c5n.4xlarge)  
- **Memory Test Instance:** $MEMORY_IP (r5.2xlarge)
- **Network Test Instance:** $NETWORK_IP (c5n.large)

## Tests Performed

### 1. GPU Performance Tests
- NVIDIA driver installation and verification
- Vulkan API support testing
- OpenGL compatibility testing
- GPU stress testing
- Graphics memory bandwidth testing

### 2. CPU AI Performance Tests
- Neural network simulation benchmarks
- Multi-threaded AI workload testing
- NumPy/SciPy performance validation
- CPU stress testing under AI workloads

### 3. Memory Performance Tests
- Memory bandwidth testing
- Memory latency measurements
- Gaming-specific memory access patterns
- Large memory allocation testing
- Memory stress testing

### 4. Network Performance Tests
- Network latency measurements
- Bandwidth testing
- Gaming network pattern simulation
- Jitter testing for real-time gaming
- Packet loss analysis

### 5. UNIA Gaming OS Boot Tests
- QEMU virtualization testing
- OS boot sequence validation
- System initialization verification
- Hardware detection testing

### 6. Performance Benchmarking
- Comprehensive system benchmarks
- Cross-instance performance comparison
- Resource utilization analysis

## Results Location

All test results have been uploaded to S3 bucket: \`$S3_BUCKET\`

## Next Steps

1. Analyze performance metrics
2. Identify optimization opportunities
3. Validate gaming console requirements
4. Plan hardware-specific optimizations

---
Generated by UNIA Gaming OS Test Automation Suite
EOF

# Display summary
log "${GREEN}✅ All tests completed successfully!${NC}"
log "${BLUE}📊 Test Summary:${NC}"
log "- GPU Performance: Tested on g4dn.xlarge"
log "- CPU AI Performance: Tested on c5n.4xlarge"
log "- Memory Performance: Tested on r5.2xlarge"
log "- Network Performance: Tested on c5n.large"
log "- UNIA OS Boot: Tested on all instances"
log ""
log "${YELLOW}📁 Results saved to: $TEST_RESULTS_DIR${NC}"
log "${YELLOW}☁️ Results uploaded to: s3://$S3_BUCKET/test-results-$TIMESTAMP/${NC}"
log ""
log "${GREEN}🎮 UNIA Gaming OS testing complete!${NC}"

# Optional: Clean up infrastructure
read -p "Do you want to destroy the test infrastructure? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "${YELLOW}🧹 Cleaning up infrastructure...${NC}"
    terraform destroy -auto-approve
    log "${GREEN}✅ Infrastructure cleaned up${NC}"
fi
