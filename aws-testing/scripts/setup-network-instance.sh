#!/bin/bash
# Network Performance Test Instance Setup Script

# Update system
yum update -y
yum groupinstall -y "Development Tools"

# Install network testing tools
yum install -y iperf3 nmap tcpdump wireshark nethogs iftop

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly

# Install QEMU for OS testing
yum install -y qemu-kvm

# Create test directories
mkdir -p /opt/unia-gaming/network-tests
mkdir -p /opt/unia-gaming/test-results

# Create network performance test script
cat > /opt/unia-gaming/network-performance-test.sh << 'EOF'
#!/bin/bash

echo "Starting network performance tests..."

# Test 1: Network latency test
echo "Testing network latency..."
for server in 8.8.8.8 1.1.1.1; do
    ping -c 100 $server | tee -a /opt/unia-gaming/test-results/latency-test.log
done

# Test 2: Network bandwidth test
echo "Testing network bandwidth..."
iperf3 -c iperf.he.net -t 30 | tee -a /opt/unia-gaming/test-results/bandwidth-test.log

# Test 3: Gaming-specific network patterns
cat > /tmp/gaming-network-test.c << 'EOC'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>

#define PACKET_SIZE 128
#define NUM_PACKETS 1000
#define GAME_PORT 7777

typedef struct {
    uint32_t sequence;
    uint64_t timestamp;
    char payload[PACKET_SIZE - 16];
} GamePacket;

void simulate_game_network() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        return;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(GAME_PORT);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    GamePacket packet;
    clock_t start = clock();

    // Simulate game network traffic
    for (uint32_t i = 0; i < NUM_PACKETS; i++) {
        packet.sequence = i;
        packet.timestamp = time(NULL);
        memset(packet.payload, 'G', sizeof(packet.payload));

        sendto(sock, &packet, sizeof(packet), 0,
               (struct sockaddr*)&server_addr, sizeof(server_addr));

        usleep(16000); // Simulate 60 FPS game loop
    }

    clock_t end = clock();
    double duration = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Gaming network simulation completed in %.3f seconds\n", duration);
    printf("Average packet rate: %.1f packets/sec\n", NUM_PACKETS / duration);

    close(sock);
}

int main() {
    printf("Gaming network pattern test\n");
    simulate_game_network();
    return 0;
}
EOC

gcc -O2 /tmp/gaming-network-test.c -o /tmp/gaming-network-test
/tmp/gaming-network-test >> /opt/unia-gaming/test-results/gaming-network.log

# Test 4: Network jitter test
echo "Testing network jitter..."
ping -c 1000 -i 0.01 8.8.8.8 | tee -a /opt/unia-gaming/test-results/jitter-test.log

echo "Network performance tests completed"
EOF

chmod +x /opt/unia-gaming/network-performance-test.sh

# Create network monitoring script
cat > /opt/unia-gaming/network-monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date),$(netstat -s | grep -E 'packets|errors|dropped')" >> /opt/unia-gaming/test-results/network-metrics.csv
    sleep 1
done
EOF

chmod +x /opt/unia-gaming/network-monitor.sh

# Setup test environment variables
echo "export UNIA_TEST_ENV=network" >> /etc/environment

# Install CloudWatch agent
yum install -y amazon-cloudwatch-agent

# Configure CloudWatch agent for network metrics
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "metrics": {
    "metrics_collected": {
      "net": {
        "resources": [
          "eth0"
        ],
        "measurement": [
          "bytes_sent",
          "bytes_recv",
          "packets_sent",
          "packets_recv",
          "drop_in",
          "drop_out",
          "err_in",
          "err_out"
        ]
      }
    }
  }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Start network monitoring
nohup /opt/unia-gaming/network-monitor.sh &

echo "Network test instance setup complete"
