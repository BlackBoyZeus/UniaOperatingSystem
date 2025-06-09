#!/bin/bash
# Minimal build script for UNIA OS that avoids complex memory management

set -e

echo "Building minimal UNIA OS bootable image..."

# Switch to nightly toolchain
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly
rustup component add llvm-tools-preview --toolchain nightly
rustup target add x86_64-unknown-none

# Install required cargo tools if not already installed
if ! command -v bootimage &> /dev/null; then
    echo "Installing bootimage..."
    cargo install bootimage
fi

# Create .cargo/config.toml with minimal settings
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[unstable]
build-std = ["core", "compiler_builtins"]
build-std-features = ["compiler-builtins-mem"]

[build]
target = "x86_64-unknown-none"

[target.'cfg(target_os = "none")']
runner = "bootimage runner"

[target.x86_64-unknown-none]
rustflags = [
    "-C", "link-arg=--nmagic",
    "-C", "link-arg=-static",
    "-C", "link-arg=-zmax-page-size=0x1000",
    "-C", "link-arg=-no-pie"
]
EOF

# Create minimal Cargo.toml
cat > Cargo.toml << EOF
[package]
name = "unia-os-bootable"
version = "0.1.0"
edition = "2021"

[dependencies]
bootloader = { version = "0.9.23", features = ["map_physical_memory"] }
x86_64 = "0.14.10"

[profile.dev]
panic = "abort"

[profile.release]
panic = "abort"
EOF

# Copy minimal main.rs into place
cp src/main_minimal.rs src/main.rs

# Clean previous build artifacts
echo "Cleaning previous build..."
cargo clean

# Build the bootable image with nightly toolchain
echo "Building kernel with nightly toolchain..."
RUSTFLAGS="-C target-cpu=x86-64" cargo +nightly bootimage --target x86_64-unknown-none

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary location: target/x86_64-unknown-none/debug/bootimage-unia-os-bootable.bin"
    
    # Create symbolic link for QEMU
    ln -sf target/x86_64-unknown-none/debug/bootimage-unia-os-bootable.bin bootimage-unia-os-bootable.bin
    
    echo "Ready to boot with QEMU!"
else
    echo "Build failed."
    exit 1
fi
