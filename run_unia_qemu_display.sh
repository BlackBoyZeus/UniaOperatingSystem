#!/bin/bash

# Set display environment variables for QEMU
export QEMU_OPTS="-display cocoa"
export QEMU_AUDIO_DRV=none

# Run QEMU with optimized settings for Apple Silicon
qemu-system-x86_64 \
    -accel tcg \
    -cpu qemu64 \
    -m 1G \
    -smp 2 \
    -drive format=raw,file=src/boot/target/x86_64-unknown-none/debug/bootimage-unia-os-bootable.bin \
    -vga virtio \
    -display default,show-cursor=on \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
    -serial stdio \
    -no-reboot \
    -no-shutdown
