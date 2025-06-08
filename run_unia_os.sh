#!/bin/bash
# UNIA OS Runner Script
# This script builds and runs the UNIA OS bootable experience

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOT_DIR="$SCRIPT_DIR/src/boot"
TARGET_DIR="$BOOT_DIR/target/x86_64-unia/debug"
BINARY_NAME="bootimage-unia-os-bootable.bin"
BINARY_PATH="$TARGET_DIR/$BINARY_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}      UNIA OS Runner Script          ${NC}"
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

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}Error: Rust is not installed.${NC}"
    echo -e "${YELLOW}Please install Rust before continuing:${NC}"
    echo -e "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check if bootimage is installed
if ! command -v bootimage &> /dev/null; then
    echo -e "${YELLOW}Installing bootimage...${NC}"
    cargo install bootimage
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install bootimage.${NC}"
        exit 1
    fi
fi

# Check if cargo-xbuild is installed
if ! command -v cargo-xbuild &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-xbuild...${NC}"
    cargo install cargo-xbuild
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install cargo-xbuild.${NC}"
        exit 1
    fi
fi

# Install required Rust components
echo -e "${YELLOW}Installing required Rust components...${NC}"
rustup component add rust-src
rustup component add llvm-tools-preview

# Navigate to boot directory
cd "$BOOT_DIR"

# Build the bootable image
echo -e "${YELLOW}Building UNIA OS bootable image...${NC}"
./build.sh

# Check if the binary was created
if [ ! -f "$BINARY_PATH" ]; then
    echo -e "${RED}Error: Failed to build bootable image.${NC}"
    echo -e "${RED}Binary not found at: $BINARY_PATH${NC}"
    exit 1
fi

# Run in QEMU
echo -e "${GREEN}Starting UNIA OS in QEMU...${NC}"
echo -e "${YELLOW}Press Ctrl+Alt+G to release mouse, Ctrl+Alt+2 for QEMU console, Ctrl+Alt+X to exit${NC}"
qemu-system-x86_64 -drive format=raw,file="$BINARY_PATH" -m 1G -smp 4 -vga std

echo -e "${GREEN}UNIA OS session ended.${NC}"
