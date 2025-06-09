# UNIA OS Bootable Experience

This directory contains the necessary components to build a bootable version of UNIA OS, providing users with a console-like experience to interact with the operating system.

## Overview

The UNIA OS bootable experience allows users to:

1. Boot directly into the UNIA OS environment
2. Interact with the system through a modern UI dashboard
3. Access and test AI gaming features
4. Experiment with mesh networking capabilities
5. Visualize system performance metrics

## Building the Bootable Image

### Prerequisites

- Rust 1.70 or later
- QEMU for testing the bootable image
- limine bootloader
- xorriso for creating ISO images

### Build Instructions

```bash
# Install dependencies
cargo install bootimage
cargo install cargo-xbuild

# Build the bootable image
cd src/boot
cargo bootimage

# The bootable image will be available at target/x86_64-unia/debug/bootimage-unia.bin
```

### Running in QEMU

```bash
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia.bin
```

### Creating a bootable USB drive

```bash
dd if=target/x86_64-unia/debug/bootimage-unia.bin of=/dev/sdX bs=4M status=progress
```

Replace `/dev/sdX` with your USB drive device.

## Architecture

The bootable UNIA OS consists of:

1. **Bootloader**: Uses limine to boot the system
2. **Kernel**: Minimal Rust kernel that initializes hardware
3. **UI Layer**: Modern dashboard interface built with embedded web technologies
4. **Game Engine**: Direct access to UNIA's game engine capabilities
5. **AI Subsystem**: Interface to test AI features
6. **Networking Stack**: Tools to test mesh networking

## UI Dashboard

The UI dashboard provides:

- System performance metrics
- Game engine testing tools
- AI behavior visualization
- Network configuration and testing
- Hardware acceleration settings

## Customization

The bootable image can be customized by modifying:

- `config/boot_config.toml`: Boot configuration
- `ui/theme.css`: UI appearance
- `hardware/profiles.yaml`: Hardware acceleration settings
