#!/bin/bash
# Final build script for UNIA OS with nightly toolchain

set -e

echo "Building UNIA OS bootable image..."

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

# Create .cargo/config.toml with updated settings
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[unstable]
build-std = ["core", "compiler_builtins", "alloc"]
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

# Copy final files into place
cp src/main_final.rs src/main.rs
cp src/lib_clean.rs src/lib.rs

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
