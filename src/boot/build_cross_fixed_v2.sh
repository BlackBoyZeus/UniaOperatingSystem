#!/bin/bash
# Enhanced cross-compilation build script for UNIA OS with nightly toolchain

set -e

echo "Building UNIA OS bootable image with cross-compilation..."

# Switch to nightly toolchain
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly
rustup component add llvm-tools-preview --toolchain nightly

# Install required cargo tools if not already installed
if ! command -v bootimage &> /dev/null; then
    echo "Installing bootimage..."
    cargo install bootimage
fi

if ! command -v cargo-xbuild &> /dev/null; then
    echo "Installing cargo-xbuild..."
    cargo install cargo-xbuild
fi

# Create custom target JSON for x86_64
cat > x86_64-unia.json << EOF
{
    "llvm-target": "x86_64-unknown-none",
    "data-layout": "e-m:e-i64:64-f80:128-n8:16:32:64-S128",
    "arch": "x86_64",
    "target-endian": "little",
    "target-pointer-width": "64",
    "target-c-int-width": "32",
    "os": "none",
    "executables": true,
    "linker-flavor": "ld.lld",
    "linker": "rust-lld",
    "panic-strategy": "abort",
    "disable-redzone": true,
    "features": "-mmx,-sse,+soft-float"
}
EOF

# Create .cargo/config.toml with updated settings
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[unstable]
build-std = ["core", "compiler_builtins", "alloc"]
build-std-features = ["compiler-builtins-mem"]

[build]
target = "x86_64-unia.json"

[target.'cfg(target_os = "none")']
runner = "bootimage runner"

[target.x86_64-unia]
rustflags = [
    "-C", "link-arg=-nostartfiles",
    "-C", "link-arg=-static",
    "-C", "link-arg=-zmax-page-size=0x1000",
    "-C", "link-arg=-no-pie"
]
EOF

# Clean previous build artifacts
echo "Cleaning previous build..."
cargo clean

# Build the bootable image with nightly toolchain
echo "Building kernel with nightly toolchain..."
RUSTFLAGS="-C link-arg=-nostartfiles" cargo +nightly bootimage --target x86_64-unia.json

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary location: target/x86_64-unia/debug/bootimage-unia-os-bootable.bin"
    
    # Create symbolic link for QEMU
    ln -sf target/x86_64-unia/debug/bootimage-unia-os-bootable.bin bootimage-unia-os-bootable.bin
else
    echo "Build failed."
    exit 1
fi
