#!/bin/bash

echo "Launching UNIA Gaming OS v3.0..."

# QEMU options optimized for gaming performance
qemu-system-x86_64 \
    -accel tcg \
    -cpu max \
    -smp cores=4,threads=2 \
    -m 2G \
    -device virtio-gpu-pci \
    -device virtio-net-pci \
    -device virtio-tablet-pci \
    -device virtio-keyboard-pci \
    -device ich9-intel-hda \
    -device hda-duplex \
    -drive format=raw,file=gaming-kernel/target/x86_64-unknown-none/debug/bootimage-unia-gaming-os.bin \
    -vga std \
    -display default \
    -serial stdio \
    -no-reboot \
    -no-shutdown
