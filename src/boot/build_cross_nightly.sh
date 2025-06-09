#!/bin/bash
# Enhanced cross-compilation build script for UNIA OS using nightly Rust

set -e

echo "Building UNIA OS bootable image with cross-compilation (nightly)..."

# Install nightly Rust if not already installed
if ! rustup toolchain list | grep -q nightly; then
    echo "Installing nightly Rust toolchain..."
    rustup toolchain install nightly
fi

# Ensure we have the required Rust components
rustup +nightly component add rust-src llvm-tools-preview
rustup +nightly target add x86_64-unknown-none

# Install required cargo tools if not already installed
if ! command -v bootimage &> /dev/null; then
    echo "Installing bootimage..."
    cargo +nightly install bootimage
fi

# Create .cargo/config.toml
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[unstable]
build-std = ["core", "compiler_builtins", "alloc"]
build-std-features = ["compiler-builtins-mem"]

[build]
target = "x86_64-unia.json"

[target.'cfg(target_os = "none")']
runner = "bootimage runner"
EOF

# Build the bootable image
echo "Building kernel with nightly toolchain..."
cargo +nightly bootimage --target x86_64-unia.json

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary location: target/x86_64-unia/debug/bootimage-unia-os-bootable.bin"
else
    echo "Build failed."
    exit 1
fi
