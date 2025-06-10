#!/bin/bash
# Memory Test Instance Setup Script

# Update system
yum update -y
yum groupinstall -y "Development Tools"

# Install memory testing tools
yum install -y htop iotop valgrind

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly

# Install QEMU for OS testing
yum install -y qemu-kvm

# Create test directories
mkdir -p /opt/unia-gaming/memory-tests
mkdir -p /opt/unia-gaming/test-results

# Install memory stress testing tools
yum install -y stress-ng memtester

# Create memory performance test script
cat > /opt/unia-gaming/memory-performance-test.sh << 'EOF'
#!/bin/bash

echo "Starting memory performance tests..."

# Get system memory info
TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
echo "Total Memory: ${TOTAL_MEM}MB"

# Test 1: Memory bandwidth test
echo "Testing memory bandwidth..."
dd if=/dev/zero of=/tmp/memory-test bs=1M count=1024 2>&1 | grep copied >> /opt/unia-gaming/test-results/memory-bandwidth.log

# Test 2: Memory latency test using custom program
cat > /tmp/memory-latency-test.c << 'EOC'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define ARRAY_SIZE (1024 * 1024 * 100) // 100MB array
#define ITERATIONS 1000

int main() {
    int *array = malloc(ARRAY_SIZE * sizeof(int));
    if (!array) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = i;
    }
    
    // Sequential access test
    clock_t start = clock();
    long sum = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            sum += array[i];
        }
    }
    clock_t end = clock();
    
    double sequential_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Sequential access time: %.3f seconds\n", sequential_time);
    
    // Random access test
    srand(time(NULL));
    start = clock();
    sum = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            int index = rand() % ARRAY_SIZE;
            sum += array[index];
        }
    }
    end = clock();
    
    double random_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Random access time: %.3f seconds\n", random_time);
    printf("Random/Sequential ratio: %.2f\n", random_time / sequential_time);
    
    free(array);
    return 0;
}
EOC

gcc -O2 /tmp/memory-latency-test.c -o /tmp/memory-latency-test
/tmp/memory-latency-test >> /opt/unia-gaming/test-results/memory-latency.log

# Test 3: Memory allocation stress test
echo "Testing memory allocation patterns..."
stress-ng --vm 4 --vm-bytes 75% --timeout 60s --metrics-brief >> /opt/unia-gaming/test-results/memory-stress.log

# Test 4: Gaming-specific memory patterns
cat > /tmp/gaming-memory-test.c << 'EOC'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Simulate game object allocation patterns
typedef struct {
    float x, y, z;
    float velocity_x, velocity_y, velocity_z;
    int health;
    int type;
    char name[32];
} GameObject;

#define MAX_OBJECTS 100000

int main() {
    printf("Gaming memory pattern test\n");
    
    GameObject *objects = malloc(MAX_OBJECTS * sizeof(GameObject));
    if (!objects) {
        printf("Failed to allocate game objects\n");
        return 1;
    }
    
    clock_t start = clock();
    
    // Simulate game loop memory access patterns
    for (int frame = 0; frame < 1000; frame++) {
        // Update all objects (typical game loop)
        for (int i = 0; i < MAX_OBJECTS; i++) {
            objects[i].x += objects[i].velocity_x;
            objects[i].y += objects[i].velocity_y;
            objects[i].z += objects[i].velocity_z;
        }
        
        // Simulate object creation/destruction
        if (frame % 10 == 0) {
            // Reallocate some objects
            for (int i = 0; i < 1000; i++) {
                int index = rand() % MAX_OBJECTS;
                memset(&objects[index], 0, sizeof(GameObject));
                objects[index].health = 100;
                objects[index].type = rand() % 10;
            }
        }
    }
    
    clock_t end = clock();
    double game_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Gaming memory pattern test completed in %.3f seconds\n", game_time);
    printf("Simulated FPS: %.1f\n", 1000.0 / game_time);
    
    free(objects);
    return 0;
}
EOC

gcc -O2 /tmp/gaming-memory-test.c -o /tmp/gaming-memory-test
/tmp/gaming-memory-test >> /opt/unia-gaming/test-results/gaming-memory.log

echo "Memory performance tests completed"
EOF

chmod +x /opt/unia-gaming/memory-performance-test.sh

# Create memory monitoring script
cat > /opt/unia-gaming/memory-monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date),$(free -m | awk 'NR==2{printf "%s,%s,%s", $2,$3,$4}')" >> /opt/unia-gaming/test-results/memory-metrics.csv
    sleep 1
done
EOF

chmod +x /opt/unia-gaming/memory-monitor.sh

# Setup test environment variables
echo "export UNIA_TEST_ENV=memory" >> /etc/environment

# Install CloudWatch agent
yum install -y amazon-cloudwatch-agent

# Start memory monitoring
nohup /opt/unia-gaming/memory-monitor.sh &

echo "Memory test instance setup complete"
