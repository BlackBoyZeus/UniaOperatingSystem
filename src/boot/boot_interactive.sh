#!/bin/bash
# Interactive boot script for UNIA OS

set -e

echo "Building UNIA OS with critical allocator..."

# Ensure nightly toolchain is installed
rustup toolchain install nightly

# Build the kernel
cargo +nightly build --target x86_64-unia.json

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Run in QEMU with graphical display for interaction
    echo "Running UNIA OS in QEMU with interactive display..."
    qemu-system-x86_64 \
        -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin \
        -serial stdio \
        -monitor stdio \
        -vga std \
        -m 256M \
        -no-reboot
    
    echo "QEMU session ended."
else
    echo "Build failed."
    exit 1
fi
