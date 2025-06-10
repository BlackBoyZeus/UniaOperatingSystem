#!/bin/bash
# CPU AI Test Instance Setup Script

# Update system
yum update -y
yum groupinstall -y "Development Tools"

# Install performance monitoring tools
yum install -y htop iotop perf sysstat

# Install Python for AI testing
yum install -y python3 python3-pip python3-devel

# Install AI/ML libraries
pip3 install numpy scipy scikit-learn tensorflow torch

# Install Rust with AI features
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly

# Install QEMU for OS testing
yum install -y qemu-kvm

# Create test directories
mkdir -p /opt/unia-gaming/cpu-tests
mkdir -p /opt/unia-gaming/ai-tests
mkdir -p /opt/unia-gaming/test-results

# Setup CPU stress testing tools
yum install -y stress-ng

# Create AI performance test script
cat > /opt/unia-gaming/ai-performance-test.py << 'EOF'
#!/usr/bin/env python3
import time
import numpy as np
import multiprocessing
import json
from datetime import datetime

def neural_network_simulation(iterations=1000):
    """Simulate neural network processing"""
    start_time = time.time()
    
    # Create random neural network weights
    weights = np.random.rand(1000, 1000)
    
    for i in range(iterations):
        # Matrix multiplication (common in neural networks)
        result = np.dot(weights, weights.T)
        # Activation function simulation
        result = np.tanh(result)
    
    end_time = time.time()
    return end_time - start_time

def ai_workload_test():
    """Test AI workload performance"""
    cpu_count = multiprocessing.cpu_count()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'cpu_cores': cpu_count,
        'tests': {}
    }
    
    # Single-threaded AI test
    print("Running single-threaded AI test...")
    single_time = neural_network_simulation(500)
    results['tests']['single_thread'] = {
        'duration': single_time,
        'operations_per_second': 500 / single_time
    }
    
    # Multi-threaded AI test
    print("Running multi-threaded AI test...")
    start_time = time.time()
    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(neural_network_simulation, [100] * cpu_count)
    multi_time = time.time() - start_time
    
    results['tests']['multi_thread'] = {
        'duration': multi_time,
        'operations_per_second': (100 * cpu_count) / multi_time,
        'speedup': single_time / multi_time
    }
    
    # Save results
    with open('/opt/unia-gaming/test-results/ai-performance.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"AI Performance Test Results:")
    print(f"Single-threaded: {results['tests']['single_thread']['operations_per_second']:.2f} ops/sec")
    print(f"Multi-threaded: {results['tests']['multi_thread']['operations_per_second']:.2f} ops/sec")
    print(f"Speedup: {results['tests']['multi_thread']['speedup']:.2f}x")

if __name__ == "__main__":
    ai_workload_test()
EOF

chmod +x /opt/unia-gaming/ai-performance-test.py

# Create CPU monitoring script
cat > /opt/unia-gaming/cpu-monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date),$(cat /proc/loadavg),$(cat /proc/stat | head -1)" >> /opt/unia-gaming/test-results/cpu-metrics.csv
    sleep 1
done
EOF

chmod +x /opt/unia-gaming/cpu-monitor.sh

# Setup test environment variables
echo "export UNIA_TEST_ENV=cpu" >> /etc/environment
echo "export OMP_NUM_THREADS=$(nproc)" >> /etc/environment

# Install CloudWatch agent
yum install -y amazon-cloudwatch-agent

# Start CPU monitoring
nohup /opt/unia-gaming/cpu-monitor.sh &

echo "CPU AI test instance setup complete"
