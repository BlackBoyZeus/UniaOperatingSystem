# UNIA OS QEMU Boot Guide

This guide explains how to boot UNIA OS in QEMU, providing a full console experience similar to what you would expect from a gaming console like PlayStation or Nintendo.

## Prerequisites

Before booting UNIA OS in QEMU, ensure you have the following installed:

1. **QEMU**
   - macOS: `brew install qemu`
   - Ubuntu: `sudo apt-get install qemu-system-x86`
   - Fedora: `sudo dnf install qemu-system-x86`

2. **Rust and Cargo**
   - Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - Install nightly toolchain: `rustup toolchain install nightly`

3. **Required Rust Components**
   - `rustup +nightly component add rust-src llvm-tools-preview`
   - `rustup +nightly target add x86_64-unknown-none`

4. **Required Cargo Tools**
   - `cargo +nightly install bootimage`
   - `cargo +nightly install cargo-xbuild`

## Quick Start

We provide several boot scripts to make running UNIA OS in QEMU easy:

### For Most Users

```bash
./boot_unia_qemu_nightly.sh
```

This script:
1. Detects your system architecture
2. Uses the nightly Rust toolchain to build UNIA OS
3. Launches QEMU with appropriate settings
4. Falls back to web simulation if building fails

### For Advanced Users

```bash
# If you want to manually build and run
cd src/boot
./build_cross_nightly.sh
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unknown-none/debug/bootimage-unia-os-bootable.bin
```

## Architecture Compatibility

UNIA OS is primarily designed for x86_64 architecture. If you're using:

- **x86_64 System**: Native performance with hardware acceleration
- **ARM64 System** (e.g., Apple Silicon): Uses QEMU's x86_64 emulation (slower)

## The Boot Experience

When you boot UNIA OS in QEMU, you'll experience:

1. **Boot Animation**: The UNIA logo animates during startup
2. **Loading Bar**: Shows boot progress
3. **Dashboard Interface**: Main system interface with:
   - System overview
   - Game library
   - AI subsystem status
   - Network configuration
   - System settings

4. **Command Console**: Press ESC to access the command-line interface

## Interacting with UNIA OS

Once booted:

- Use number keys (1-5) to navigate between dashboard sections
- Press ESC to access the command console
- Use keyboard for text input in the console

QEMU keyboard shortcuts:
- Ctrl+Alt+G: Release mouse capture
- Ctrl+Alt+2: Access QEMU monitor
- Ctrl+Alt+X: Exit QEMU

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Ensure you're using the nightly Rust toolchain
   - Check that all required components are installed
   - Try running `cargo clean` before rebuilding

2. **QEMU Errors**
   - Ensure QEMU is properly installed
   - Try different QEMU parameters
   - Check if virtualization is enabled in your BIOS (for x86_64 systems)

3. **Slow Performance on ARM64**
   - This is expected as QEMU is emulating x86_64 on ARM
   - Consider using the web simulation for better performance on ARM64

### Fallback Options

If QEMU boot fails:

1. **Web Simulation**
   ```bash
   ./run_web_simulation.sh
   ```

2. **Docker Container**
   ```bash
   # Coming soon
   ```

## Advanced Configuration

You can customize the QEMU experience by editing the boot scripts:

- Change memory allocation: Edit `-m 1G` parameter
- Change CPU cores: Edit `-smp 4` parameter
- Add network interfaces: Add `-net` parameters
- Add virtual devices: Add additional device parameters

## Next Steps

After successfully booting UNIA OS:

1. Explore the dashboard interface
2. Try the command console
3. Check out the AI subsystem demo
4. Test the mesh networking capabilities
5. Experiment with the game engine demo
