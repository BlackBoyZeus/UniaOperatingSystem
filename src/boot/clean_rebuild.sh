#!/bin/bash
# Complete clean rebuild script for UNIA OS

set -e

echo "Performing complete clean rebuild of UNIA OS..."

# Clean the project completely
echo "Cleaning project..."
cd ..
cargo clean
cd boot

# Remove any cached bootloader artifacts
echo "Removing bootloader cache..."
rm -rf target/bootimage-*.bin || true
rm -rf target/x86_64-unia || true

# Ensure nightly toolchain is installed
echo "Setting up nightly toolchain..."
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly

# Kill any running QEMU processes
echo "Killing any running QEMU instances..."
pkill -f qemu || true

# Rebuild from scratch
echo "Rebuilding from scratch..."
cargo +nightly build --target x86_64-unia.json

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Clear previous serial output
    echo "" > serial_output.log
    
    # Run in QEMU with graphical display for interaction
    echo "Running UNIA OS in QEMU with interactive display..."
    qemu-system-x86_64 \
        -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin \
        -serial file:serial_output.log \
        -vga std \
        -m 256M \
        -no-reboot \
        -no-shutdown
    
    echo "QEMU session ended."
    echo "Check serial_output.log for debug information."
    
    # Display the serial output
    echo "Serial output:"
    cat serial_output.log
else
    echo "Build failed."
    exit 1
fi
