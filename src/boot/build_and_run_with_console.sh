#!/bin/bash
# Enhanced build and run script for UNIA OS with serial output to console

set -e

echo "Building UNIA OS with critical allocator..."

# Ensure nightly toolchain is installed
rustup toolchain install nightly

# Build the kernel
cargo +nightly build --target x86_64-unia.json

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Run in QEMU with serial output to console
    echo "Running in QEMU with serial output to console..."
    qemu-system-x86_64 \
        -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin \
        -serial stdio \
        -display none \
        -d int \
        -no-reboot
    
    echo "QEMU session ended."
else
    echo "Build failed."
    exit 1
fi
