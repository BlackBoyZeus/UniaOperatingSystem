#!/bin/bash
# GPU Test Instance Setup Script

# Update system
yum update -y
yum groupinstall -y "Development Tools"

# Install NVIDIA drivers and CUDA
yum install -y gcc kernel-devel-$(uname -r) kernel-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
rpm -i cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
yum clean all
yum -y install cuda

# Install Vulkan SDK
yum install -y vulkan-tools vulkan-loader-devel

# Install OpenGL development libraries
yum install -y mesa-libGL-devel mesa-libGLU-devel

# Install X11 and VNC for GUI testing
yum groupinstall -y "X Window System"
yum install -y tigervnc-server

# Install monitoring tools
yum install -y htop iotop nvidia-smi

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly

# Install QEMU for OS testing
yum install -y qemu-kvm

# Create test directories
mkdir -p /opt/unia-gaming/gpu-tests
mkdir -p /opt/unia-gaming/test-results

# Setup GPU monitoring script
cat > /opt/unia-gaming/gpu-monitor.sh << 'EOF'
#!/bin/bash
while true; do
    nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> /opt/unia-gaming/test-results/gpu-metrics.csv
    sleep 1
done
EOF

chmod +x /opt/unia-gaming/gpu-monitor.sh

# Setup test environment variables
echo "export UNIA_TEST_ENV=gpu" >> /etc/environment
echo "export VULKAN_SDK=/usr/local/vulkan" >> /etc/environment
echo "export LD_LIBRARY_PATH=/usr/local/vulkan/lib" >> /etc/environment

# Setup CloudWatch agent for metrics
yum install -y amazon-cloudwatch-agent

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "metrics": {
    "metrics_collected": {
      "nvidia_gpu": {
        "measurement": [
          "utilization_gpu",
          "temperature_gpu",
          "memory_total",
          "memory_used",
          "memory_free"
        ]
      }
    }
  }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Start GPU monitoring
nohup /opt/unia-gaming/gpu-monitor.sh &

echo "GPU test instance setup complete"
