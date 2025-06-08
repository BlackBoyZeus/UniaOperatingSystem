#!/bin/bash
# Build script for UNIA OS bootable image

set -e

echo "Building UNIA OS bootable image..."

# Install required dependencies if not already installed
if ! command -v cargo-bootimage &> /dev/null; then
    echo "Installing cargo-bootimage..."
    cargo install cargo-bootimage
fi

if ! command -v cargo-xbuild &> /dev/null; then
    echo "Installing cargo-xbuild..."
    cargo install cargo-xbuild
fi

# Build the bootable image
echo "Building bootable image..."
cargo bootimage

# Create ISO image
echo "Creating ISO image..."
mkdir -p isofiles/boot/grub
cp target/x86_64-unia/debug/bootimage-unia-os-bootable.bin isofiles/boot/kernel.bin

cat > isofiles/boot/grub/grub.cfg << EOF
set timeout=0
set default=0

menuentry "UNIA OS" {
    multiboot /boot/kernel.bin
    boot
}
EOF

if command -v grub-mkrescue &> /dev/null; then
    grub-mkrescue -o unia-os.iso isofiles
    echo "ISO image created: unia-os.iso"
else
    echo "Warning: grub-mkrescue not found. ISO image not created."
    echo "You can still use the binary image: target/x86_64-unia/debug/bootimage-unia-os-bootable.bin"
fi

echo "Build complete!"
echo ""
echo "To run in QEMU:"
echo "qemu-system-x86_64 -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin"
echo ""
echo "To create a bootable USB drive:"
echo "sudo dd if=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin of=/dev/sdX bs=4M status=progress"
echo "(Replace /dev/sdX with your USB drive device)"
