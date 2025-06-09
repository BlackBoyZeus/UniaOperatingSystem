#!/bin/bash
# Updated UNIA OS QEMU Boot Script with proper cross-architecture support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOT_DIR="$SCRIPT_DIR/src/boot"
TARGET_DIR="$BOOT_DIR/target/x86_64-unia/debug"
BINARY_NAME="bootimage-unia-os-bootable.bin"
BINARY_PATH="$TARGET_DIR/$BINARY_NAME"
FALLBACK_BINARY="$BOOT_DIR/$BINARY_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}      UNIA OS QEMU Boot Script v2     ${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if QEMU is installed
if ! command -v qemu-system-x86_64 &> /dev/null; then
    echo -e "${RED}Error: qemu-system-x86_64 is not installed.${NC}"
    echo -e "${YELLOW}Please install QEMU before continuing:${NC}"
    echo -e "  - macOS: brew install qemu"
    echo -e "  - Ubuntu: sudo apt-get install qemu-system-x86"
    echo -e "  - Fedora: sudo dnf install qemu-system-x86"
    exit 1
fi

# Check if we're on Apple Silicon
if [ "$(uname -m)" = "arm64" ]; then
    echo -e "${YELLOW}Detected Apple Silicon (ARM64) architecture.${NC}"
    echo -e "${YELLOW}Using x86_64 emulation mode with TCG acceleration.${NC}"
    QEMU_ARGS="-accel tcg -cpu qemu64"
else
    echo -e "${GREEN}Detected x86_64 architecture.${NC}"
    if [ -r /dev/kvm ]; then
        echo -e "${GREEN}Using KVM hardware acceleration.${NC}"
        QEMU_ARGS="-enable-kvm"
    else
        echo -e "${YELLOW}KVM not available, using TCG acceleration.${NC}"
        QEMU_ARGS="-accel tcg"
    fi
fi

# Check if the binary exists
if [ ! -f "$BINARY_PATH" ] && [ ! -f "$FALLBACK_BINARY" ]; then
    echo -e "${YELLOW}Binary not found. Attempting to build...${NC}"
    
    # Navigate to boot directory
    cd "$BOOT_DIR"
    
    # Build the bootable image with the enhanced cross-compilation script
    echo -e "${YELLOW}Building with enhanced cross-compilation v2...${NC}"
    chmod +x build_cross_fixed_v2.sh
    ./build_cross_fixed_v2.sh
    
    # Check if the build was successful
    if [ ! -f "$BINARY_PATH" ] && [ ! -f "$FALLBACK_BINARY" ]; then
        echo -e "${RED}Error: Failed to build bootable image.${NC}"
        echo -e "${RED}Binary not found at: $BINARY_PATH${NC}"
        echo -e "${RED}Fallback binary not found at: $FALLBACK_BINARY${NC}"
        echo -e "${YELLOW}Falling back to web simulation...${NC}"
        
        # Run web simulation instead
        cd "$SCRIPT_DIR"
        ./run_web_simulation.sh
        exit 1
    fi
fi

# Determine which binary to use
if [ -f "$BINARY_PATH" ]; then
    FINAL_BINARY="$BINARY_PATH"
    echo -e "${GREEN}Using binary: $BINARY_PATH${NC}"
elif [ -f "$FALLBACK_BINARY" ]; then
    FINAL_BINARY="$FALLBACK_BINARY"
    echo -e "${GREEN}Using fallback binary: $FALLBACK_BINARY${NC}"
fi

# Run in QEMU with appropriate settings
echo -e "${GREEN}Starting UNIA OS in QEMU...${NC}"
echo -e "${YELLOW}Press Ctrl+Alt+G to release mouse, Ctrl+Alt+2 for QEMU console, Ctrl+Alt+X to exit${NC}"

qemu-system-x86_64 $QEMU_ARGS \
    -m 1G \
    -smp 4 \
    -drive format=raw,file="$FINAL_BINARY" \
    -vga std \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
    -serial stdio \
    -monitor telnet:127.0.0.1:55555,server,nowait \
    -netdev user,id=net0 \
    -device rtl8139,netdev=net0

echo -e "${GREEN}UNIA OS session ended.${NC}"
