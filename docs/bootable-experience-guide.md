# UNIA OS Bootable Experience Guide

This guide provides detailed instructions for building, running, and customizing the UNIA OS bootable experience.

## What is the UNIA OS Bootable Experience?

The UNIA OS Bootable Experience is a standalone operating system that provides a console-like environment for interacting with the UNIA Operating System. It allows users to:

- Boot directly into UNIA OS without requiring a host operating system
- Interact with a modern UI dashboard similar to gaming consoles
- Test AI gaming features in a controlled environment
- Experiment with mesh networking capabilities
- Visualize system performance metrics

## Prerequisites

Before building the bootable experience, ensure you have the following installed:

- Rust 1.70 or later (`rustup update`)
- QEMU for testing (`brew install qemu` on macOS)
- Required Rust components:
  ```bash
  rustup component add rust-src
  rustup component add llvm-tools-preview
  ```
- Required Cargo tools:
  ```bash
  cargo install bootimage
  cargo install cargo-xbuild
  ```

## Building the Bootable Image

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/BlackBoyZeus/UniaOperatingSystem.git
   cd UniaOperatingSystem
   ```

2. **Navigate to the boot directory**:
   ```bash
   cd src/boot
   ```

3. **Run the build script**:
   ```bash
   ./build.sh
   ```

   This will create:
   - A bootable binary image at `target/x86_64-unia/debug/bootimage-unia-os-bootable.bin`
   - An ISO image at `unia-os.iso` (if grub-mkrescue is available)

## Running UNIA OS

### In QEMU (Virtual Machine)

To test the bootable image in QEMU:

```bash
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin
```

For a better experience with more memory and multiple cores:

```bash
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin -m 1G -smp 4
```

### On Real Hardware

To create a bootable USB drive:

```bash
sudo dd if=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin of=/dev/sdX bs=4M status=progress
```

Replace `/dev/sdX` with your USB drive device (be extremely careful to specify the correct device as this will erase all data on the target device).

## Using the Web Simulation

For users who prefer not to boot into the OS directly, we also provide a web-based version of the dashboard that simulates the UNIA OS experience:

```bash
# Navigate to the web directory
cd src/boot/web

# Serve the web dashboard (requires Python)
python -m http.server 8000
```

Then open your browser to `http://localhost:8000` to see the simulated dashboard.

## Customizing UNIA OS

### UI Appearance

The UI appearance can be customized by modifying:

```bash
src/boot/src/ui/theme.rs
```

This file contains color definitions, typography settings, spacing values, and other UI-related constants.

### Boot Configuration

Boot settings can be customized by modifying:

```bash
src/boot/config/boot_config.toml
```

This file contains settings for display resolution, boot options, hardware profiles, network settings, and more.

### Hardware Profiles

Hardware acceleration settings can be customized by modifying:

```bash
testing/cloud_simulation/config/hardware_profiles.yaml
```

This file defines different hardware profiles that simulate various console configurations.

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Ensure you have the latest Rust toolchain
   - Check that all required dependencies are installed
   - Try running `cargo clean` before rebuilding

2. **Boot Failures**:
   - Ensure your hardware is compatible with UEFI booting
   - Try different QEMU parameters
   - Check the serial output for error messages

3. **Display Issues**:
   - Try different display modes in the boot menu
   - Check if your graphics card is supported

4. **Performance Problems**:
   - Check the hardware acceleration settings
   - Increase the allocated memory in QEMU

### Getting Help

If you encounter issues that aren't covered in this guide:

1. Check the [GitHub Issues](https://github.com/BlackBoyZeus/UniaOperatingSystem/issues) for similar problems
2. Join our [Discord community](https://discord.gg/unia-os) for real-time support
3. Post a detailed description of your issue on our [forums](https://forums.unia-os.org)

## Contributing

We welcome contributions to improve the bootable experience! See our [Contributing Guide](../CONTRIBUTING.md) for details on how to get started.

## License

The UNIA OS Bootable Experience is licensed under the same license as the main UNIA Operating System. See the [LICENSE](../LICENSE) file for details.
