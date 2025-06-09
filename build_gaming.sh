#!/bin/bash
# Build script for UNIA Gaming OS

set -e

echo "Building UNIA Gaming OS..."

# Setup Rust toolchain for gaming optimizations
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly
rustup component add llvm-tools-preview --toolchain nightly
rustup target add x86_64-unknown-none

# Install required tools
cargo install bootimage
cargo install cargo-xbuild

# Create optimized kernel configuration
mkdir -p gaming-kernel/.cargo
cat > gaming-kernel/.cargo/config.toml << EOF
[unstable]
build-std = ["core", "compiler_builtins", "alloc"]
build-std-features = ["compiler-builtins-mem"]

[build]
target = "x86_64-unknown-none"
rustflags = [
    "-C", "target-cpu=native",
    "-C", "link-arg=--nmagic",
    "-C", "opt-level=3",
    "-C", "lto=fat",
    "-C", "codegen-units=1",
    "-C", "debuginfo=0",
    "-C", "panic=abort"
]

[target.x86_64-unknown-none]
runner = "bootimage runner"
EOF

# Build gaming kernel with optimizations
echo "Building gaming kernel..."
cd gaming-kernel
RUSTFLAGS="-C target-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2" cargo bootimage --target x86_64-unknown-none --release

# Create QEMU launch script with gaming optimizations
cat > ../run_gaming.sh << 'EOF'
#!/bin/bash

# QEMU options optimized for gaming performance
qemu-system-x86_64 \
    -accel tcg \
    -cpu max \
    -smp cores=4,threads=2 \
    -m 4G \
    -device virtio-gpu-pci \
    -device virtio-net-pci \
    -device virtio-tablet-pci \
    -device virtio-keyboard-pci \
    -device ich9-intel-hda \
    -device hda-duplex \
    -drive format=raw,file=gaming-kernel/target/x86_64-unknown-none/release/bootimage-unia-gaming-os.bin \
    -vga std \
    -display default \
    -serial stdio
EOF

chmod +x ../run_gaming.sh

echo "Build complete! Run ./run_gaming.sh to start UNIA Gaming OS"
