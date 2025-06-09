#!/bin/bash
# Minimal space build script for UNIA OS

set -e

echo "Building minimal UNIA OS bootable image..."

# Clean any existing artifacts
cargo clean

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
opt-level = "z"  # Optimize for size
lto = true       # Enable link-time optimization
codegen-units = 1

[profile.release]
panic = "abort"
opt-level = "z"  # Optimize for size
lto = true       # Enable link-time optimization
codegen-units = 1
EOF

# Create minimal main.rs
cat > src/main.rs << EOF
#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{entry_point, BootInfo};

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    let vga_buffer = 0xb8000 as *mut u8;
    let msg = b"UNIA OS v0.1.0 - Minimal Boot";
    
    for (i, &byte) in msg.iter().enumerate() {
        unsafe {
            *vga_buffer.add(i * 2) = byte;
            *vga_buffer.add(i * 2 + 1) = 0x0F;
        }
    }
    
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}
EOF

# Create minimal config
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
rustflags = ["-C", "link-arg=--nmagic"]
EOF

# Build with minimal settings
RUSTFLAGS="-C target-cpu=x86-64 -C opt-level=z" cargo +nightly bootimage --target x86_64-unknown-none

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
