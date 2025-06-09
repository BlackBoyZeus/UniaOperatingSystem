#!/bin/bash
# Enhanced cross-compilation build script for UNIA OS

set -e

echo "Building UNIA OS bootable image with cross-compilation..."

# Ensure we have the required Rust components
rustup component add rust-src llvm-tools-preview
rustup target add x86_64-unknown-none

# Install required cargo tools if not already installed
if ! command -v bootimage &> /dev/null; then
    echo "Installing bootimage..."
    cargo install bootimage
fi

if ! command -v cargo-xbuild &> /dev/null; then
    echo "Installing cargo-xbuild..."
    cargo install cargo-xbuild
fi

# Create .cargo/config.toml
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[unstable]
build-std = ["core", "compiler_builtins", "alloc"]
build-std-features = ["compiler-builtins-mem"]

[build]
target = "x86_64-unknown-none"

[target.'cfg(target_os = "none")']
runner = "bootimage runner"
EOF

# Build the bootable image
echo "Building kernel..."
RUSTFLAGS="-C link-arg=-nostartfiles" cargo bootimage --target x86_64-unknown-none

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary location: target/x86_64-unknown-none/debug/bootimage-unia-os-bootable.bin"
else
    echo "Build failed."
    exit 1
fi
