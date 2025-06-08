# UNIA OS Bootable Experience

The UNIA OS Bootable Experience provides a console-like environment for users to interact with the UNIA Operating System directly. This document explains how to build, run, and customize the bootable experience.

## Overview

The bootable experience allows users to:

1. Boot directly into UNIA OS without requiring a host operating system
2. Interact with a modern UI dashboard similar to gaming consoles
3. Test AI gaming features in a controlled environment
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
# Navigate to the boot directory
cd src/boot

# Run the build script
./build.sh
```

This will create:
- A bootable binary image at `target/x86_64-unia/debug/bootimage-unia-os-bootable.bin`
- An ISO image at `unia-os.iso` (if grub-mkrescue is available)

### Running in QEMU

To test the bootable image in QEMU:

```bash
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin
```

### Creating a Bootable USB Drive

To create a bootable USB drive:

```bash
sudo dd if=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin of=/dev/sdX bs=4M status=progress
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

- System performance metrics (CPU, memory, network usage)
- Game engine testing tools
- AI behavior visualization
- Network configuration and testing
- Hardware acceleration settings

## Web-Based Dashboard

For users who prefer not to boot into the OS directly, we also provide a web-based version of the dashboard that simulates the UNIA OS experience:

```bash
# Navigate to the web directory
cd src/boot/web

# Serve the web dashboard (requires Python)
python -m http.server 8000
```

Then open your browser to `http://localhost:8000` to see the simulated dashboard.

## Customization

The bootable image can be customized by modifying:

- `src/boot/src/ui/theme.rs`: UI appearance
- `src/boot/config/boot_config.toml`: Boot configuration
- `src/boot/config/hardware_profiles.yaml`: Hardware acceleration settings

## Integration with Game Development

Game developers can use the bootable experience to:

1. Test games in a controlled environment
2. Benchmark performance on different hardware profiles
3. Experiment with AI-driven gameplay features
4. Test mesh networking capabilities with multiple instances

## Troubleshooting

If you encounter issues with the bootable experience:

1. **Boot Failures**: Ensure your hardware is compatible with UEFI booting
2. **Display Issues**: Try different display modes in the boot menu
3. **Performance Problems**: Check the hardware acceleration settings
4. **Networking Issues**: Verify that your network adapter is supported
