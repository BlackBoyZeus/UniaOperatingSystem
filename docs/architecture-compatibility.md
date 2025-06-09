# UNIA OS Architecture Compatibility Guide

This document provides information about running UNIA OS on different hardware architectures and offers solutions for cross-architecture compatibility.

## Architecture Support

UNIA OS is primarily designed for the x86_64 architecture, which is common in desktop computers, servers, and many gaming consoles. However, with the rise of ARM-based systems like Apple Silicon (M1/M2/M3), we need to address cross-architecture compatibility.

### Supported Architectures

- **x86_64**: Fully supported for both building and running
- **ARM64/AArch64**: Partial support (web simulation only)

## Running on Apple Silicon (ARM64)

If you're using an Apple Silicon Mac (M1, M2, or M3), you have several options to run UNIA OS:

### Option 1: Web Simulation (Recommended for Quick Testing)

The web simulation provides a visual representation of the UNIA OS interface without requiring architecture-specific code:

```bash
# Run the web simulation
./run_web_simulation.sh
```

This will start a local web server and open a browser window showing the UNIA OS dashboard interface.

### Option 2: x86_64 Emulation with QEMU

You can use QEMU to emulate x86_64 architecture on your ARM64 system:

```bash
# Install QEMU with x86_64 support
brew install qemu

# Run with x86_64 emulation (note: this will be significantly slower)
qemu-system-x86_64 -accel tcg -cpu qemu64 -drive format=raw,file=target/x86_64-unia/debug/bootimage-unia-os-bootable.bin
```

### Option 3: Use a Virtual Machine

Run an x86_64 Linux VM and build/run UNIA OS there:

1. Install UTM (https://mac.getutm.app/)
2. Set up an Ubuntu VM with x86_64 architecture
3. Clone the repository inside the VM
4. Build and run UNIA OS natively within the VM

## Cross-Compilation Setup

To build UNIA OS for x86_64 on an ARM64 system:

```bash
# Add x86_64 target
rustup target add x86_64-unknown-none

# Build with explicit target
cargo build --target x86_64-unia
```

Note: Cross-compilation may require additional configuration and might not work perfectly in all cases.

## Future Architecture Support

We are working on expanding architecture support for UNIA OS:

1. **Native ARM64 Support**: Future versions will include native support for ARM64 architecture
2. **RISC-V Support**: Planned for future releases
3. **WebAssembly**: Web-based full system simulation

## Troubleshooting

### Common Issues on Apple Silicon

1. **"Error loading target specification"**:
   - This occurs because the bootloader is specifically designed for x86_64
   - Solution: Use the web simulation or x86_64 emulation

2. **Slow Performance with QEMU Emulation**:
   - x86_64 emulation on ARM is significantly slower than native execution
   - Solution: Use the web simulation for interface testing, or a VM for better performance

3. **Missing Dependencies**:
   - Some dependencies may not be available for ARM64
   - Solution: Use Homebrew to install ARM64 versions of dependencies

## Getting Help

If you encounter issues related to architecture compatibility:

1. Check the [GitHub Issues](https://github.com/BlackBoyZeus/UniaOperatingSystem/issues) for similar problems
2. Join our [Discord community](https://discord.gg/unia-os) for real-time support
3. Post a detailed description of your issue on our [forums](https://forums.unia-os.org)
