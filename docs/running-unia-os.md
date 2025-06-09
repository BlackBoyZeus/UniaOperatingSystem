# Running UNIA OS

This guide provides instructions for running the UNIA Operating System on your local machine.

## Prerequisites

Before running UNIA OS, ensure you have the following installed:

1. **Rust and Cargo**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **QEMU**
   - macOS: `brew install qemu`
   - Ubuntu: `sudo apt-get install qemu-system-x86`
   - Fedora: `sudo dnf install qemu-system-x86`
   - Windows: Download from [QEMU's website](https://www.qemu.org/download/)

3. **Required Rust Components**
   ```bash
   rustup component add rust-src
   rustup component add llvm-tools-preview
   ```

4. **Required Cargo Tools**
   ```bash
   cargo install bootimage
   cargo install cargo-xbuild
   ```

## Quick Start

The easiest way to run UNIA OS is using our runner script:

```bash
./run_unia_os.sh
```

This script will:
1. Check for required dependencies
2. Build the bootable image
3. Launch UNIA OS in QEMU

## Manual Steps

If you prefer to run UNIA OS manually, follow these steps:

### 1. Build the Bootable Image

```bash
# Navigate to the boot directory
cd src/boot

# Run the build script
./build.sh
```

This will create a bootable binary at `src/boot/target/x86_64-unia/debug/bootimage-unia-os-bootable.bin`.

### 2. Run in QEMU

```bash
qemu-system-x86_64 -drive format=raw,file=src/boot/target/x86_64-unia/debug/bootimage-unia-os-bootable.bin -m 1G -smp 4
```

Additional QEMU options:
- `-m 1G`: Allocate 1GB of memory (adjust as needed)
- `-smp 4`: Simulate 4 CPU cores (adjust as needed)
- `-vga std`: Use standard VGA graphics
- `-serial stdio`: Redirect serial output to terminal

## Creating a Bootable USB Drive

To run UNIA OS on real hardware, you can create a bootable USB drive:

```bash
# Replace /dev/sdX with your USB drive device
sudo dd if=src/boot/target/x86_64-unia/debug/bootimage-unia-os-bootable.bin of=/dev/sdX bs=4M status=progress
```

⚠️ **WARNING**: Be extremely careful with the `dd` command. Using the wrong device name can result in data loss. Double-check the device name before running the command.

## Using the Web Simulation

If you don't want to run the full OS, you can use our web-based simulation:

```bash
# Navigate to the web directory
cd src/boot/web

# Start a simple HTTP server
python -m http.server 8000
```

Then open your browser to `http://localhost:8000` to see the simulated dashboard.

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Ensure you have the latest Rust toolchain
   - Check that all required dependencies are installed
   - Try running `cargo clean` before rebuilding

2. **QEMU Errors**
   - Ensure QEMU is properly installed
   - Try different QEMU parameters
   - Check if virtualization is enabled in your BIOS

3. **Boot Failures**
   - Ensure your hardware is compatible with UEFI booting
   - Try different QEMU parameters
   - Check the serial output for error messages

### Getting Help

If you encounter issues that aren't covered in this guide:

1. Check the [GitHub Issues](https://github.com/BlackBoyZeus/UniaOperatingSystem/issues) for similar problems
2. Join our [Discord community](https://discord.gg/unia-os) for real-time support
3. Post a detailed description of your issue on our [forums](https://forums.unia-os.org)
