#!/bin/bash
# Interactive boot script for UNIA OS

set -e

echo "Building UNIA OS with enhanced critical allocator..."

# Ensure nightly toolchain is installed
rustup toolchain install nightly

# Build the kernel
cargo +nightly build --target x86_64-unia.json

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Kill any existing QEMU processes
    pkill -f qemu || true
    
    # Run in QEMU with graphical display for interaction
    echo "Running UNIA OS in QEMU with interactive display..."
    qemu-system-x86_64 \
        -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin \
        -serial file:serial_output.log \
        -vga std \
        -m 256M \
        -no-reboot
    
    echo "QEMU session ended."
    echo "Check serial_output.log for debug information."
    
    # Display the serial output
    echo "Serial output:"
    cat serial_output.log
else
    echo "Build failed."
    exit 1
fi
