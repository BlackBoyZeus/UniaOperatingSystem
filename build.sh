#!/bin/bash
# Modern build script for UNIA OS v2.0

set -e

echo "Building UNIA OS v2.0..."

# Ensure we have the right Rust toolchain
rustup default nightly
rustup update nightly
rustup component add rust-src --toolchain nightly
rustup component add llvm-tools-preview --toolchain nightly
rustup target add x86_64-unknown-none

# Install required tools
cargo install bootimage
cargo install cargo-xbuild

# Build kernel
echo "Building kernel..."
cd kernel
RUSTFLAGS="-C link-arg=-nostartfiles" cargo bootimage --target x86_64-unknown-none

# Create bootable image
echo "Creating bootable image..."
cp target/x86_64-unknown-none/debug/bootimage-unia-kernel.bin ../unia-os.bin

# Create QEMU launch script
cat > ../run_unia.sh << 'EOF'
#!/bin/bash
qemu-system-x86_64 \
    -accel tcg \
    -cpu qemu64 \
    -m 512M \
    -smp cores=4 \
    -drive format=raw,file=unia-os.bin \
    -vga std \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
    -device e1000,netdev=net0 \
    -netdev user,id=net0 \
    -audiodev coreaudio,id=snd0 \
    -device intel-hda \
    -device hda-duplex,audiodev=snd0 \
    -serial stdio \
    -no-reboot \
    -no-shutdown
EOF

chmod +x ../run_unia.sh

echo "Build complete! Run ./run_unia.sh to start UNIA OS"
