#!/bin/bash
# Build script for UNIA OS bootable image with x86_64 cross-compilation

set -e

echo "Building UNIA OS bootable image for x86_64..."

# Add x86_64 target if not already added
rustup target add x86_64-unknown-none

# Build the bootable image with explicit target
cargo build --target x86_64-unia

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary location: target/x86_64-unia/debug/unia-os-bootable"
    
    # Create bootable image
    echo "Creating bootable image..."
    cargo bootimage --target x86_64-unia
    
    echo "Bootable image created at: target/x86_64-unia/debug/bootimage-unia-os-bootable.bin"
else
    echo "Build failed."
    exit 1
fi
